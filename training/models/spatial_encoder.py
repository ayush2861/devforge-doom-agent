"""Spatial encoder for 64 ego-centric rays, depth buffer, player vitals, and previous action."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from doom_env import spaces as S

NUM_HIT_TYPES = len(S.RAY_HIT_TYPES)  # 6
NUM_VITALS = len(S.VITALS_FIELDS)  # 17
NUM_ACTIONS = len(S.BUTTONS)  # 9
NUM_WEAPONS = 8  # weapon slots 0..7


def normalize_vitals(vitals: torch.Tensor) -> torch.Tensor:
    """Normalize (B, 17) vitals into a well-scaled (B, 24) tensor."""
    health = torch.clamp(vitals[:, 0:1] / 100.0, 0.0, 2.5)
    armor = torch.clamp(vitals[:, 1:2] / 100.0, 0.0, 2.5)
    weapon_idx = torch.clamp(vitals[:, 2].round().long(), 0, NUM_WEAPONS - 1)
    weapon_oh = F.one_hot(weapon_idx, num_classes=NUM_WEAPONS).float()
    ammo = torch.log1p(torch.clamp(vitals[:, 3:8], min=0.0)) / 5.0
    keys = torch.clamp(vitals[:, 8:14], 0.0, 1.0)
    vel = torch.clamp(vitals[:, 14:17] / 15.0, -3.0, 3.0)
    return torch.cat([health, armor, weapon_oh, ammo, keys, vel], dim=-1)


class SpatialEncoder(nn.Module):
    """Encodes 64-ray 360° telemetry, (120, 160) depth buffer, 17 vitals, and previous action."""

    def __init__(self, out_dim: int = 192, use_depth: bool = True, ray_embed_dim: int = 8):
        super().__init__()
        self.out_dim = out_dim
        self.use_depth = use_depth

        # 64-ray circular 1D CNN: [dist, proximity, hit_type_embed(8)] -> 10 channels
        self.hit_embed = nn.Embedding(NUM_HIT_TYPES, ray_embed_dim)
        ray_in_ch = 2 + ray_embed_dim
        self.ray_conv = nn.Sequential(
            nn.Conv1d(ray_in_ch, 32, kernel_size=5, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        self.ray_proj = nn.Sequential(
            nn.Linear(64 * 16, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )

        # Depth CNN (optional)
        depth_out_dim = 64 if use_depth else 0
        if use_depth:
            self.depth_conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=4, padding=2),
                nn.SiLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((3, 4)),
            )
            self.depth_proj = nn.Sequential(
                nn.Linear(32 * 3 * 4, depth_out_dim),
                nn.LayerNorm(depth_out_dim),
                nn.SiLU(),
            )

        # Vitals (24 normalized) + prev_action (9) = 33
        self.vitals_mlp = nn.Sequential(
            nn.Linear(24 + NUM_ACTIONS, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + depth_out_dim + 64, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        ray_distance: torch.Tensor,
        ray_hit_type: torch.Tensor,
        vitals: torch.Tensor,
        depth: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = ray_distance.shape[0]
        dist = torch.clamp(ray_distance.float(), 0.0, 1.0).unsqueeze(-1)  # (B, 64, 1)
        prox = 1.0 - dist
        hit_emb = self.hit_embed(torch.clamp(ray_hit_type.long(), 0, NUM_HIT_TYPES - 1))  # (B, 64, 8)
        ray_x = torch.cat([dist, prox, hit_emb], dim=-1).transpose(1, 2)  # (B, 10, 64)
        ray_feat = self.ray_proj(self.ray_conv(ray_x).flatten(1))  # (B, 128)

        v_norm = normalize_vitals(vitals.float())
        if prev_action is None:
            pa = torch.zeros(b, NUM_ACTIONS, device=vitals.device, dtype=torch.float32)
        else:
            pa = prev_action.float()
        vitals_feat = self.vitals_mlp(torch.cat([v_norm, pa], dim=-1))  # (B, 64)

        parts = [ray_feat, vitals_feat]
        if self.use_depth:
            if depth is None:
                depth_feat = torch.zeros(b, 64, device=vitals.device, dtype=torch.float32)
            else:
                d = depth.float()
                if d.max() > 1.0:
                    d = d / 255.0
                if d.ndim == 3:
                    d = d.unsqueeze(1)  # (B, 1, H, W)
                depth_feat = self.depth_proj(self.depth_conv(d).flatten(1))  # (B, 64)
            parts.append(depth_feat)

        return self.fusion(torch.cat(parts, dim=-1))
