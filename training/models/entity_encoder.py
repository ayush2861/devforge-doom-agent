"""Attention-based variable-count entity encoder with direct enemy aim pathway."""

from __future__ import annotations

import torch
import torch.nn as nn

from doom_env import spaces as S

NUM_ENTITY_TYPES = len(S.ENTITY_TYPES)  # 12 (0 = pad)


def normalize_entity_features(features: torch.Tensor) -> torch.Tensor:
    """Normalize (B, N, 5) entity features: [dist_norm, sin, cos, rel_vf, rel_vr] to O(1) scale.

    Note: sensors.py already normalizes distance by RAY_RANGE (2048.0) into [0, 1].
    """
    dist = torch.clamp(features[..., 0:1], 0.0, 1.0)
    angles = torch.clamp(features[..., 1:3], -1.0, 1.0)
    vel = torch.clamp(features[..., 3:5] / 15.0, -2.0, 2.0)
    return torch.cat([dist, angles, vel], dim=-1)


class EntityEncoder(nn.Module):
    """Encodes up to MAX_ENTITIES=32 visible entities using masked self-attention.

    Key design: a GATED direct-aim pathway that injects the nearest enemy's
    (dist, sin, cos, has_enemy) into the output with a learned gate, so the
    policy can directly read enemy angle without attention washing it out.
    """

    def __init__(self, out_dim: int = 128, d_model: int = 64, n_heads: int = 4, type_embed_dim: int = 8):
        super().__init__()
        self.out_dim = out_dim
        self.d_model = d_model

        self.type_embed = nn.Embedding(NUM_ENTITY_TYPES, type_embed_dim, padding_idx=0)
        self.entity_mlp = nn.Sequential(
            nn.Linear(len(S.ENTITY_FEATURES) + type_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

        # ---- GATED direct aim pathway ----
        # Input: [dist_norm, sin, cos, has_enemy, enemy_count_norm]
        aim_input_dim = 5
        self.aim_proj = nn.Sequential(
            nn.Linear(aim_input_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        # Learned gate: how much to trust the direct aim signal vs attention
        self.aim_gate = nn.Sequential(
            nn.Linear(aim_input_dim, 32),
            nn.SiLU(),
            nn.Linear(32, out_dim),
            nn.Sigmoid(),
        )
        # Initialize gate bias so it starts at ~0.5 (balanced)
        with torch.no_grad():
            self.aim_gate[-2].bias.fill_(0.0)

    def forward(
        self,
        features: torch.Tensor,
        entity_type: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = features.shape
        valid = mask.bool()  # (B, N)

        feat_norm = normalize_entity_features(features.float())
        type_emb = self.type_embed(torch.clamp(entity_type.long(), 0, NUM_ENTITY_TYPES - 1))
        tokens = self.entity_mlp(torch.cat([feat_norm, type_emb], dim=-1))  # (B, N, D)
        tokens = tokens * valid.unsqueeze(-1).float()

        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, D)
        seq = torch.cat([cls, tokens], dim=1)  # (B, N+1, D)

        # key_padding_mask: True = ignore. [CLS] at index 0 is never ignored.
        cls_unmasked = torch.zeros(b, 1, dtype=torch.bool, device=features.device)
        kpm = torch.cat([cls_unmasked, ~valid], dim=1)  # (B, N+1)

        attn_out, _ = self.attn(seq, seq, seq, key_padding_mask=kpm, need_weights=False)
        seq_out = self.attn_norm(seq + attn_out)

        cls_out = seq_out[:, 0, :]  # (B, D)
        ent_out = seq_out[:, 1:, :]  # (B, N, D)

        # Masked max pool over real entities (zeros when no entities are visible)
        any_valid = valid.any(dim=1, keepdim=True)  # (B, 1)
        ent_masked = ent_out.masked_fill(~valid.unsqueeze(-1), -1e9)
        max_pool = ent_masked.max(dim=1).values
        max_pool = torch.where(any_valid, max_pool, torch.zeros_like(max_pool))

        attn_emb = self.out_proj(torch.cat([cls_out, max_pool], dim=-1))  # (B, out_dim)

        # ---- Direct aim pathway: nearest enemy ----
        et = entity_type.long()
        is_enemy = valid & ((et == 1) | (et == 2) | (et == 3))
        enemy_dist = torch.where(is_enemy, feat_norm[..., 0], torch.full_like(feat_norm[..., 0], 1e6))
        best_idx = enemy_dist.argmin(dim=1)  # (B,)
        b_idx = torch.arange(b, device=features.device)
        nearest_feat = feat_norm[b_idx, best_idx, 0:3]  # (B, 3): [dist_norm, sin, cos]
        has_enemy = is_enemy.any(dim=1, keepdim=True).float()  # (B, 1)
        enemy_count = is_enemy.float().sum(dim=1, keepdim=True)  # (B, 1)
        enemy_count_norm = torch.clamp(enemy_count / 5.0, 0.0, 1.0)

        aim_input = torch.cat([
            nearest_feat * has_enemy,  # zero out if no enemy
            has_enemy,
            enemy_count_norm,
        ], dim=-1)  # (B, 5)

        aim_emb = self.aim_proj(aim_input)  # (B, out_dim)
        gate = self.aim_gate(aim_input)  # (B, out_dim) in [0, 1]

        # Gated combination: when enemy present, gate opens to let aim signal dominate
        return attn_emb * (1.0 - gate) + aim_emb * gate
