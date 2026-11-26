"""Custom Recurrent PPO trainer from scratch for ESRHA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.models.doom_model import ESRHAAgentModel
from training.rollout import RolloutBuffer


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.015
    strat_ent_coef: float = 0.005
    icm_coef: float = 0.05
    icm_loss_weight: float = 0.2
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    seq_len: int = 32
    batch_seqs: int = 8
    target_kl: float | None = 0.05


class PPOTrainer:
    """Clipped PPO with truncated BPTT over GRU+Slot memory sequences and optional ICM."""

    def __init__(
        self,
        model: ESRHAAgentModel,
        config: PPOConfig,
        total_updates: int,
        device: torch.device | str,
        icm=None,
    ):
        self.model = model
        self.cfg = config
        self.device = device
        self.icm = icm
        params = list(model.parameters()) + (list(icm.parameters()) if icm is not None else [])
        self.optimizer = torch.optim.AdamW(params, lr=config.lr, eps=1e-5, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_updates), eta_min=config.lr * 0.1
        )

    def update(self, buf: RolloutBuffer, last_values: np.ndarray, last_dones: np.ndarray) -> dict[str, float]:
        intr_mean = 0.0
        if self.icm is not None:
            self.icm.eval()
            with torch.no_grad():
                flat_obs = {
                    k: torch.as_tensor(v.reshape(buf.T * buf.N, *v.shape[2:]), device=self.device)
                    for k, v in buf.obs.items()
                }
                flat_pa = torch.as_tensor(buf.prev_actions.reshape(buf.T * buf.N, -1), device=self.device)
                obs_emb = self.model.encode_obs(flat_obs, prev_action=flat_pa).reshape(buf.T, buf.N, -1)
                act_seq = torch.as_tensor(buf.actions, device=self.device)
                intr_r = self.icm.compute_intrinsic_rewards(obs_emb, act_seq, buf.dones, coef=self.cfg.icm_coef)
                buf.rewards = buf.rewards + intr_r
                intr_mean = float(intr_r.mean())
            self.icm.train()

        self.model.train()
        buf.compute_gae(last_values, last_dones, self.cfg.gamma, self.cfg.gae_lambda)

        # Normalize advantages globally across the rollout
        adv = buf.advantages
        buf.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        pg_losses, vf_losses, entropies, kls, clip_fracs, icm_losses = [], [], [], [], [], []
        early_stop = False

        for _ in range(self.cfg.ppo_epochs):
            for mb in buf.iter_sequences(self.cfg.seq_len, self.cfg.batch_seqs, self.device):
                b, t = mb["actions"].shape[:2]
                out = self.model.forward_sequence(
                    mb["obs"],
                    hidden=mb["init_hidden"],
                    prev_actions=mb["prev_actions"],
                    dones=mb["dones"],
                )
                actions_flat = mb["actions"].reshape(b * t, -1)
                new_log_probs = out["tactical_dist"].log_prob(actions_flat).reshape(b, t)
                tact_entropy = out["tactical_dist"].entropy().mean()
                strat_entropy = out["strategic_dist"].entropy().mean()

                log_ratio = new_log_probs - mb["old_log_probs"]
                ratio = torch.exp(log_ratio)

                mb_adv = mb["advantages"]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss = 0.5 * ((out["values"] - mb["returns"]) ** 2).mean()
                entropy = tact_entropy + (self.cfg.strat_ent_coef / max(1e-8, self.cfg.ent_coef)) * strat_entropy
                loss = pg_loss + self.cfg.vf_coef * vf_loss - self.cfg.ent_coef * entropy

                icm_loss_val = 0.0
                if self.icm is not None and t >= 2:
                    flat_obs = {k: v.reshape(b * t, *v.shape[2:]) for k, v in mb["obs"].items()}
                    flat_pa = mb["prev_actions"].reshape(b * t, -1)
                    emb_seq = self.model.encode_obs(flat_obs, prev_action=flat_pa).reshape(b, t, -1).detach()
                    _, inv_l, fwd_l = self.icm.forward_dynamics(
                        emb_seq[:, :-1].reshape(b * (t - 1), -1),
                        emb_seq[:, 1:].reshape(b * (t - 1), -1),
                        mb["actions"][:, :-1].reshape(b * (t - 1), -1),
                    )
                    icm_l = inv_l + fwd_l
                    loss = loss + self.cfg.icm_loss_weight * icm_l
                    icm_loss_val = float(icm_l.item())

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                all_params = list(self.model.parameters()) + (list(self.icm.parameters()) if self.icm is not None else [])
                nn.utils.clip_grad_norm_(all_params, self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_eps).float().mean().item()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                entropies.append(tact_entropy.item())
                kls.append(approx_kl)
                clip_fracs.append(clip_frac)
                icm_losses.append(icm_loss_val)

                if self.cfg.target_kl is not None and approx_kl > 1.5 * self.cfg.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        self.scheduler.step()
        return {
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(vf_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kls)),
            "clip_frac": float(np.mean(clip_fracs)),
            "icm_loss": float(np.mean(icm_losses)),
            "intr_rew": intr_mean,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def save_checkpoint(self, path: str | Path, update_idx: int, total_steps: int, extra: dict | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "update_idx": update_idx,
            "total_steps": total_steps,
            "model_config": {
                "hidden_dim": self.model.hidden_dim,
                "use_visual": self.model.use_visual,
                "use_depth": self.model.use_depth,
            },
            "extra": extra or {},
        }
        torch.save(payload, p)

    @staticmethod
    def load_checkpoint(
        path: str | Path,
        model: ESRHAAgentModel,
        trainer: PPOTrainer | None = None,
        device: torch.device | str = "cpu",
    ) -> dict:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if trainer is not None:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt
