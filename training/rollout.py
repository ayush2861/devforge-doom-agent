"""Rollout collection and GAE computation for recurrent MultiBinary(9) PPO."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from doom_env import spaces as S
from training.models.doom_model import ESRHAAgentModel, obs_to_tensors
from training.models.policy import STRATEGIES
from training.rewards.reward import DoomRewardShaper, EpisodeStats


class RolloutBuffer:
    """Stores T steps across N environments and yields shuffled BPTT sequence chunks."""

    def __init__(self, num_steps: int, num_envs: int, hidden_dim: int):
        self.T = num_steps
        self.N = num_envs
        self.H = hidden_dim
        self.ptr = 0

        self.obs: dict[str, np.ndarray] = {
            "rgb": np.zeros((num_steps, num_envs, S.SCREEN_HEIGHT, S.SCREEN_WIDTH, 3), dtype=np.uint8),
            "depth": np.zeros((num_steps, num_envs, S.SCREEN_HEIGHT, S.SCREEN_WIDTH), dtype=np.uint8),
            "vitals": np.zeros((num_steps, num_envs, len(S.VITALS_FIELDS)), dtype=np.float32),
            "ray_distance": np.zeros((num_steps, num_envs, S.NUM_RAYS), dtype=np.float32),
            "ray_hit_type": np.zeros((num_steps, num_envs, S.NUM_RAYS), dtype=np.int64),
            "ent_features": np.zeros((num_steps, num_envs, S.MAX_ENTITIES, len(S.ENTITY_FEATURES)), dtype=np.float32),
            "ent_type": np.zeros((num_steps, num_envs, S.MAX_ENTITIES), dtype=np.int64),
            "ent_mask": np.zeros((num_steps, num_envs, S.MAX_ENTITIES), dtype=np.int8),
        }
        self.prev_actions = np.zeros((num_steps, num_envs, len(S.BUTTONS)), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, len(S.BUTTONS)), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.hiddens = np.zeros((num_steps, num_envs, hidden_dim), dtype=np.float32)

        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)

    def add(
        self,
        obs_list: list[dict],
        prev_action: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        hidden: np.ndarray,
    ) -> None:
        t = self.ptr
        for i, o in enumerate(obs_list):
            self.obs["rgb"][t, i] = o["rgb"]
            self.obs["depth"][t, i] = o["depth"]
            self.obs["vitals"][t, i] = o["vitals"]
            self.obs["ray_distance"][t, i] = o["rays"]["distance"]
            self.obs["ray_hit_type"][t, i] = o["rays"]["hit_type"]
            self.obs["ent_features"][t, i] = o["visible_entities"]["features"]
            self.obs["ent_type"][t, i] = o["visible_entities"]["entity_type"]
            self.obs["ent_mask"][t, i] = o["visible_entities"]["mask"]
        self.prev_actions[t] = prev_action
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done
        self.hiddens[t] = hidden
        self.ptr += 1

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray, gamma: float, gae_lambda: float) -> None:
        last_gae = np.zeros(self.N, dtype=np.float32)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def iter_sequences(
        self, seq_len: int, batch_seqs: int, device: torch.device | str
    ) -> Iterator[dict]:
        """Yield shuffled (B, seq_len, ...) minibatches on `device` for truncated BPTT."""
        n_chunks = self.T // seq_len
        usable_t = n_chunks * seq_len
        # Reshape (T, N, ...) -> (N, n_chunks, seq_len, ...) -> (N * n_chunks, seq_len, ...)
        total_seqs = self.N * n_chunks
        perm = np.random.permutation(total_seqs)

        def _seq(arr: np.ndarray) -> np.ndarray:
            a = arr[:usable_t].swapaxes(0, 1)  # (N, usable_t, ...)
            return a.reshape(total_seqs, seq_len, *arr.shape[2:])

        obs_seqs = {k: _seq(v) for k, v in self.obs.items()}
        pa_seqs = _seq(self.prev_actions)
        act_seqs = _seq(self.actions)
        lp_seqs = _seq(self.log_probs)
        val_seqs = _seq(self.values)
        adv_seqs = _seq(self.advantages)
        ret_seqs = _seq(self.returns)
        done_seqs = _seq(self.dones)
        hid_seqs = _seq(self.hiddens)[:, 0, :]  # initial hidden at t=0 of each chunk: (total_seqs, H)

        for start in range(0, total_seqs, batch_seqs):
            idx = perm[start : start + batch_seqs]
            obs_batch = {
                k: torch.as_tensor(v[idx], device=device) for k, v in obs_seqs.items()
            }
            yield {
                "obs": obs_batch,
                "prev_actions": torch.as_tensor(pa_seqs[idx], device=device),
                "actions": torch.as_tensor(act_seqs[idx], device=device),
                "old_log_probs": torch.as_tensor(lp_seqs[idx], device=device),
                "old_values": torch.as_tensor(val_seqs[idx], device=device),
                "advantages": torch.as_tensor(adv_seqs[idx], device=device),
                "returns": torch.as_tensor(ret_seqs[idx], device=device),
                "dones": torch.as_tensor(done_seqs[idx], device=device),
                "init_hidden": torch.as_tensor(hid_seqs[idx], device=device).unsqueeze(0),  # (1, B, H)
            }


def _batch_obs(obs_list: list[dict], device: torch.device | str) -> dict[str, torch.Tensor]:
    batched = {
        "rgb": np.stack([o["rgb"] for o in obs_list], axis=0),
        "depth": np.stack([o["depth"] for o in obs_list], axis=0),
        "vitals": np.stack([o["vitals"] for o in obs_list], axis=0),
        "rays": {
            "distance": np.stack([o["rays"]["distance"] for o in obs_list], axis=0),
            "hit_type": np.stack([o["rays"]["hit_type"] for o in obs_list], axis=0),
        },
        "visible_entities": {
            "features": np.stack([o["visible_entities"]["features"] for o in obs_list], axis=0),
            "entity_type": np.stack([o["visible_entities"]["entity_type"] for o in obs_list], axis=0),
            "mask": np.stack([o["visible_entities"]["mask"] for o in obs_list], axis=0),
        },
    }
    return obs_to_tensors(batched, device=device)


@torch.no_grad()
def collect_rollouts(
    envs: list,
    shapers: list[DoomRewardShaper],
    model: ESRHAAgentModel,
    obs_list: list[dict],
    hidden: torch.Tensor,
    prev_action: np.ndarray,
    num_steps: int,
    device: torch.device | str,
    seed: int | None = None,
    viewer=None,
) -> tuple[RolloutBuffer, list[dict], torch.Tensor, np.ndarray, np.ndarray, np.ndarray, list[EpisodeStats]]:
    """Collect `num_steps` transitions across `envs` and return buffer + updated state."""
    model.eval()
    n_envs = len(envs)
    buf = RolloutBuffer(num_steps, n_envs, model.state_dim)
    completed: list[EpisodeStats] = []

    for step_idx in range(num_steps):
        obs_t = _batch_obs(obs_list, device=device)
        pa_t = torch.as_tensor(prev_action, dtype=torch.float32, device=device)
        out = model.forward_step(obs_t, hidden=hidden, prev_action=pa_t)

        action_t = out["tactical_dist"].sample()
        log_prob_t = out["tactical_dist"].log_prob(action_t)
        value_t = out["value"]

        action_np = action_t.cpu().numpy().astype(np.float32)
        action_int = action_np.astype(np.int8)
        hid_in_np = hidden.squeeze(0).cpu().numpy()

        if viewer is not None and not viewer.closed:
            strat = STRATEGIES[int(out["strategy_probs"][0].argmax().item())]
            viewer.render(
                obs_list[0],
                step=step_idx,
                episode=len(completed) + 1,
                ep_reward=shapers[0]._stats.reward,
                strategy=strat,
                action=action_int[0],
                throttle=False,
            )

        rewards = np.zeros(n_envs, dtype=np.float32)
        dones = np.zeros(n_envs, dtype=np.float32)
        next_obs_list: list[dict] = []
        next_hidden = out["next_hidden"]
        next_prev_action = action_np.copy()

        for i, env in enumerate(envs):
            next_obs, _, term, trunc, _ = env.step(action_int[i])
            done = bool(term or trunc)
            shaped_r, ep_stats = shapers[i].step(env.privileged_state(), action=action_int[i], obs=next_obs)
            rewards[i] = shaped_r
            dones[i] = float(done)
            if done:
                if ep_stats is not None:
                    completed.append(ep_stats)
                next_obs, _ = env.reset(seed=None if seed is None else seed + len(completed))
                shapers[i].reset(env.privileged_state())
                next_hidden[:, i, :] = 0.0
                next_prev_action[i] = 0.0
            next_obs_list.append(next_obs)

        buf.add(
            obs_list=obs_list,
            prev_action=prev_action,
            action=action_np,
            log_prob=log_prob_t.cpu().numpy(),
            value=value_t.cpu().numpy(),
            reward=rewards,
            done=dones,
            hidden=hid_in_np,
        )
        obs_list = next_obs_list
        hidden = next_hidden
        prev_action = next_prev_action

    # Bootstrap value for last step
    obs_t = _batch_obs(obs_list, device=device)
    pa_t = torch.as_tensor(prev_action, dtype=torch.float32, device=device)
    last_out = model.forward_step(obs_t, hidden=hidden, prev_action=pa_t)
    last_values = last_out["value"].cpu().numpy()
    last_dones = np.zeros(n_envs, dtype=np.float32)  # freshly reset envs have done=0
    return buf, obs_list, hidden, prev_action, last_values, last_dones, completed
