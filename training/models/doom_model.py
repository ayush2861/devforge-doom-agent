"""Complete Ego-Spatial Recurrent Hierarchical Agent (ESRHA) model with Enemy Awareness Memory."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from doom_env import spaces as S
from training.models.enemy_memory import EnemyAwarenessMemory
from training.models.entity_encoder import EntityEncoder
from training.models.memory import RecurrentMemory
from training.models.policy import HierarchicalPolicy, ValueNetwork
from training.models.spatial_encoder import SpatialEncoder
from training.models.visual_encoder import VisualEncoder


def obs_to_tensors(obs: dict, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Convert a single or batched doom_env observation dict to PyTorch tensors on `device`."""
    single = np.asarray(obs["vitals"]).ndim == 1

    def _t(x, dtype: torch.dtype) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        return t.unsqueeze(0) if single else t

    return {
        "rgb": _t(obs["rgb"], torch.uint8),
        "depth": _t(obs["depth"], torch.uint8),
        "vitals": _t(obs["vitals"], torch.float32),
        "ray_distance": _t(obs["rays"]["distance"], torch.float32),
        "ray_hit_type": _t(obs["rays"]["hit_type"], torch.int64),
        "ent_features": _t(obs["visible_entities"]["features"], torch.float32),
        "ent_type": _t(obs["visible_entities"]["entity_type"], torch.int64),
        "ent_mask": _t(obs["visible_entities"]["mask"], torch.int8),
    }


class ESRHAAgentModel(nn.Module):
    """Ego-Spatial Recurrent Hierarchical Agent (ESRHA) with persistent EnemyAwarenessMemory."""

    def __init__(
        self,
        spatial_dim: int = 192,
        entity_dim: int = 128,
        visual_dim: int = 128,
        hidden_dim: int = 256,
        use_visual: bool = True,
        use_depth: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_visual = use_visual
        self.use_depth = use_depth

        self.spatial_encoder = SpatialEncoder(out_dim=spatial_dim, use_depth=use_depth)
        self.entity_encoder = EntityEncoder(out_dim=entity_dim)
        self.enemy_memory = EnemyAwarenessMemory(out_dim=entity_dim)
        self.visual_encoder = VisualEncoder(out_dim=visual_dim, enabled=use_visual)

        fused_dim = spatial_dim + entity_dim + entity_dim + visual_dim
        self.pre_memory = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.memory = RecurrentMemory(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.state_dim = self.memory.state_dim + self.enemy_memory.state_dim

        skip_dim = spatial_dim + entity_dim + entity_dim
        self.policy = HierarchicalPolicy(hidden_dim=hidden_dim, obs_skip_dim=skip_dim)
        self.value_net = ValueNetwork(hidden_dim=hidden_dim, obs_skip_dim=skip_dim)

        # Inference-time state (for single-agent act/reset)
        self._hidden: torch.Tensor | None = None
        self._prev_action: torch.Tensor | None = None

    def init_hidden(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        rec_h = self.memory.init_hidden(batch_size, device)
        em_s = self.enemy_memory.init_state(batch_size, device).unsqueeze(0)
        return torch.cat([rec_h, em_s], dim=-1)

    def reset_state(self, batch_size: int = 1, device: torch.device | str | None = None) -> None:
        dev = device if device is not None else next(self.parameters()).device
        self._hidden = self.init_hidden(batch_size, dev)
        self._prev_action = torch.zeros(batch_size, len(S.BUTTONS), device=dev, dtype=torch.float32)

    def encode_obs_parts(
        self,
        obs_tensors: dict[str, torch.Tensor],
        prev_action: torch.Tensor | None = None,
        mem_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_z = self.spatial_encoder(
            ray_distance=obs_tensors["ray_distance"],
            ray_hit_type=obs_tensors["ray_hit_type"],
            vitals=obs_tensors["vitals"],
            depth=obs_tensors["depth"] if self.use_depth else None,
            prev_action=prev_action,
        )
        entity_z = self.entity_encoder(
            features=obs_tensors["ent_features"],
            entity_type=obs_tensors["ent_type"],
            mask=obs_tensors["ent_mask"],
        )
        if mem_summary is None:
            mem_z = torch.zeros_like(entity_z)
        else:
            mem_z = self.enemy_memory(mem_summary)
        visual_z = self.visual_encoder(obs_tensors["rgb"])
        skip = torch.cat([spatial_z, entity_z, mem_z], dim=-1)
        fused = self.pre_memory(torch.cat([skip, visual_z], dim=-1))
        return fused, skip

    def encode_obs(
        self, obs_tensors: dict[str, torch.Tensor], prev_action: torch.Tensor | None = None
    ) -> torch.Tensor:
        fused, _ = self.encode_obs_parts(obs_tensors, prev_action=prev_action)
        return fused

    def forward_step(
        self,
        obs_tensors: dict[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
    ) -> dict:
        b = obs_tensors["vitals"].shape[0]
        if hidden is None:
            hidden = self.init_hidden(b, obs_tensors["vitals"].device)
        rec_h = hidden[:, :, : self.memory.state_dim]
        em_state = hidden.squeeze(0)[:, self.memory.state_dim :]

        next_em_state, mem_summary = self.enemy_memory.step_state(
            em_state,
            obs_tensors["ent_features"],
            obs_tensors["ent_type"],
            obs_tensors["ent_mask"],
            obs_tensors["ray_distance"],
            obs_tensors["vitals"],
            prev_action,
        )
        fused, skip = self.encode_obs_parts(obs_tensors, prev_action=prev_action, mem_summary=mem_summary)
        mem_out, next_rec_h = self.memory.forward_step(fused, rec_h)
        tactical_dist, strategic_dist, action_logits, strategy_probs = self.policy(mem_out, obs_skip=skip)
        value = self.value_net(mem_out, strategy_probs, obs_skip=skip)
        next_hidden = torch.cat([next_rec_h, next_em_state.unsqueeze(0)], dim=-1)
        return {
            "tactical_dist": tactical_dist,
            "strategic_dist": strategic_dist,
            "action_logits": action_logits,
            "strategy_probs": strategy_probs,
            "value": value,
            "next_hidden": next_hidden,
            "mem_summary": mem_summary,
            "enemy_mem_state": next_em_state,
        }

    def forward_sequence(
        self,
        obs_seq: dict[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        prev_actions: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> dict:
        """Batched (B, T, ...) sequence forward for truncated BPTT training."""
        b, t = obs_seq["vitals"].shape[:2]
        if hidden is None:
            hidden = self.init_hidden(b, obs_seq["vitals"].device)
        rec_h = hidden[:, :, : self.memory.state_dim]
        em_state = hidden.squeeze(0)[:, self.memory.state_dim :]

        summaries = []
        for step in range(t):
            if step > 0 and dones is not None:
                keep = (1.0 - dones[:, step - 1].float()).unsqueeze(-1)
                em_state = em_state * keep
            pa_step = None if prev_actions is None else prev_actions[:, step, :]
            em_state, summ = self.enemy_memory.step_state(
                em_state,
                obs_seq["ent_features"][:, step],
                obs_seq["ent_type"][:, step],
                obs_seq["ent_mask"][:, step],
                obs_seq["ray_distance"][:, step],
                obs_seq["vitals"][:, step],
                pa_step,
            )
            summaries.append(summ)
        mem_summary_flat = torch.stack(summaries, dim=1).reshape(b * t, -1)

        flat_obs = {k: v.reshape(b * t, *v.shape[2:]) for k, v in obs_seq.items()}
        flat_pa = None if prev_actions is None else prev_actions.reshape(b * t, -1)
        fused_flat, skip_flat = self.encode_obs_parts(flat_obs, prev_action=flat_pa, mem_summary=mem_summary_flat)
        fused_seq = fused_flat.reshape(b, t, -1)

        mem_seq, next_rec_h = self.memory.forward_sequence(fused_seq, hidden=rec_h, dones=dones)
        mem_flat = mem_seq.reshape(b * t, -1)
        tactical_dist, strategic_dist, action_logits, strategy_probs = self.policy(mem_flat, obs_skip=skip_flat)
        values = self.value_net(mem_flat, strategy_probs, obs_skip=skip_flat).reshape(b, t)
        next_hidden = torch.cat([next_rec_h, em_state.unsqueeze(0)], dim=-1)
        return {
            "tactical_dist": tactical_dist,
            "strategic_dist": strategic_dist,
            "action_logits": action_logits.reshape(b, t, -1),
            "strategy_probs": strategy_probs.reshape(b, t, -1),
            "values": values,
            "next_hidden": next_hidden,
        }

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        """Single-observation inference returning int8 (9,) binary action."""
        dev = next(self.parameters()).device
        if self._hidden is None or self._prev_action is None:
            self.reset_state(batch_size=1, device=dev)

        obs_t = obs_to_tensors(obs, device=dev)
        out = self.forward_step(obs_t, hidden=self._hidden, prev_action=self._prev_action)
        self._hidden = out["next_hidden"]

        if deterministic:
            action_t = (out["tactical_dist"].base_dist.probs > 0.5).to(torch.float32)
        else:
            action_t = out["tactical_dist"].sample()
        self._prev_action = action_t
        return action_t.squeeze(0).cpu().numpy().astype(np.int8)
