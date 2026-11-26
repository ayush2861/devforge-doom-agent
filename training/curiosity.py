"""Intrinsic Curiosity Module (ICM) and running reward normalizer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from doom_env import spaces as S

NUM_ACTIONS = len(S.BUTTONS)


class RunningMeanStd:
    """Welford online mean/std tracker for normalizing intrinsic rewards."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float64).ravel()
        if arr.size == 0:
            return
        batch_mean = float(arr.mean())
        batch_var = float(arr.var())
        batch_count = arr.size

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-8)))


class IntrinsicCuriosityModule(nn.Module):
    """Prediction-error curiosity (ICM) over encoded ego-centric observations."""

    def __init__(self, obs_dim: int = 384, feat_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.feat_dim = feat_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.SiLU(),
        )
        # Inverse model: (phi_t, phi_{t+1}) -> action logits (9)
        self.inverse_net = nn.Sequential(
            nn.Linear(feat_dim * 2, 128),
            nn.SiLU(),
            nn.Linear(128, NUM_ACTIONS),
        )
        # Forward model: (phi_t, action) -> predicted phi_{t+1}
        self.forward_net = nn.Sequential(
            nn.Linear(feat_dim + NUM_ACTIONS, 128),
            nn.SiLU(),
            nn.Linear(128, feat_dim),
        )
        self.rms = RunningMeanStd()

    def forward_dynamics(
        self, obs_emb_t: torch.Tensor, obs_emb_next: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (per_step_intrinsic_error, inverse_loss, forward_loss)."""
        phi_t = self.encoder(obs_emb_t)
        phi_next = self.encoder(obs_emb_next)

        pred_action_logits = self.inverse_net(torch.cat([phi_t, phi_next], dim=-1))
        inv_loss = F.binary_cross_entropy_with_logits(pred_action_logits, actions.float())

        pred_phi_next = self.forward_net(torch.cat([phi_t.detach(), actions.float()], dim=-1))
        fwd_err = 0.5 * ((pred_phi_next - phi_next.detach()) ** 2).mean(dim=-1)
        fwd_loss = fwd_err.mean()
        return fwd_err, inv_loss, fwd_loss

    @torch.no_grad()
    def compute_intrinsic_rewards(
        self,
        obs_emb_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        dones_seq: torch.Tensor,
        coef: float = 0.1,
        clip_max: float = 0.2,
    ) -> np.ndarray:
        """Compute normalized intrinsic rewards (T, N) for a rollout (zeroed at episode boundaries)."""
        # obs_emb_seq: (T, N, D) -> pairs (0..T-2 -> 1..T-1)
        t, n, _ = obs_emb_seq.shape
        if t < 2:
            return np.zeros((t, n), dtype=np.float32)

        emb_t = obs_emb_seq[:-1].reshape((t - 1) * n, -1)
        emb_next = obs_emb_seq[1:].reshape((t - 1) * n, -1)
        act_t = actions_seq[:-1].reshape((t - 1) * n, -1)

        fwd_err, _, _ = self.forward_dynamics(emb_t, emb_next, act_t)
        raw = fwd_err.reshape(t - 1, n).cpu().numpy()
        # Mask out transitions across episode resets
        valid = 1.0 - dones_seq[:-1]
        raw = raw * valid

        self.rms.update(raw[valid > 0.5])
        normed = np.clip(coef * (raw / self.rms.std), 0.0, clip_max) * valid

        out = np.zeros((t, n), dtype=np.float32)
        out[:-1] = normed
        return out
