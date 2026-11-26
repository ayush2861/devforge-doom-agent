"""Observation-legal Ego-Centric Enemy Awareness & Cleared-Area Memory.

Tracks remembered enemies, confirmed kills, and cleared angular sectors using ONLY
legal agent observations (visible_entities, rays, vitals) and previous actions.
Never uses global map coordinates, privileged_state(), or engine entity IDs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from doom_env import spaces as S

MAX_MEM_SLOTS = 12
NUM_SECTORS = 8
# Per-slot state: [ego_x, ego_y, status(0=empty,1=alive,2=killed), confidence, etype, age, miss_fov, shot_at]
SLOT_DIM = 8
# Summary feature vector fed to the neural policy:
#   nearest alive (6): [dist, sin, cos, conf, is_vis, present]
#   second alive  (6): [dist, sin, cos, conf, is_vis, present]
#   8 sectors * 2 (16): [living_threat(8), cleared_score(8)]
#   global        (3): [alive_norm, killed_norm, any_offscreen_alive]
MEMORY_FEAT_DIM = 6 + 6 + NUM_SECTORS * 2 + 3  # 31

# Empirical ViZDoom turn step (FRAME_SKIP=4): 14.0625 deg = 0.245437 rad
TURN_STEP_RAD = math.radians(14.0625)
HALF_FOV_RAD = math.radians(45.0)
CENTER_FOV_RAD = math.radians(32.0)
MATCH_RADIUS = 0.18  # in RAY_RANGE=1024 normalized units (~184 map units)
_ENEMY_TYPES = (1, 2, 3)


class EnemyAwarenessMemory(nn.Module):
    """Stateful ego-centric enemy & cleared-area tracker with learned projection."""

    def __init__(self, out_dim: int = 128, max_slots: int = MAX_MEM_SLOTS, num_sectors: int = NUM_SECTORS):
        super().__init__()
        self.max_slots = max_slots
        self.num_sectors = num_sectors
        self.state_dim = max_slots * SLOT_DIM + num_sectors  # 12*8 + 8 = 104

        self.feat_net = nn.Sequential(
            nn.Linear(MEMORY_FEAT_DIM, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, self.state_dim, device=device, dtype=torch.float32)

    @torch.no_grad()
    def step_state(
        self,
        state: torch.Tensor,
        ent_features: torch.Tensor,
        ent_type: torch.Tensor,
        ent_mask: torch.Tensor,
        ray_distance: torch.Tensor,
        vitals: torch.Tensor,
        prev_action: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update ego-centric slots & cleared sectors for one step.

        Returns:
            next_state: (B, state_dim)
            summary_feat: (B, MEMORY_FEAT_DIM)
        """
        b = ent_features.shape[0]
        dev = ent_features.device
        slots = state[:, : self.max_slots * SLOT_DIM].view(b, self.max_slots, SLOT_DIM).clone()
        sectors = state[:, self.max_slots * SLOT_DIM :].clone()  # (B, 8) seen_recency

        # 1. Ego-motion compensation for all non-empty slots & sector recency
        if prev_action is not None:
            turn_l = prev_action[:, 4] > 0.5
            turn_r = prev_action[:, 5] > 0.5
            d_theta = torch.where(turn_l & ~turn_r, TURN_STEP_RAD, 0.0)
            d_theta = torch.where(turn_r & ~turn_l, -TURN_STEP_RAD, d_theta)
            fired = prev_action[:, 6] > 0.5
        else:
            d_theta = torch.zeros(b, device=dev)
            fired = torch.zeros(b, dtype=torch.bool, device=dev)

        # Ego displacement over FRAME_SKIP=4 tics, normalized by RAY_RANGE=1024
        dx = (vitals[:, 14].float() * S.FRAME_SKIP) / S.RAY_RANGE
        dy = -(vitals[:, 15].float() * S.FRAME_SKIP) / S.RAY_RANGE

        cos_dt = torch.cos(d_theta).unsqueeze(1)
        sin_dt = torch.sin(d_theta).unsqueeze(1)

        non_empty = slots[:, :, 2] > 0.5
        sx = slots[:, :, 0] - dx.unsqueeze(1)
        sy = slots[:, :, 1] - dy.unsqueeze(1)
        # Rotate world points by -d_theta when player rotates by +d_theta
        rot_x = sx * cos_dt + sy * sin_dt
        rot_y = -sx * sin_dt + sy * cos_dt
        slots[:, :, 0] = torch.where(non_empty, rot_x, 0.0)
        slots[:, :, 1] = torch.where(non_empty, rot_y, 0.0)

        # Rotate 8 sector recency values smoothly when turning
        # Sector k is centered at angle k * (2*pi/8) in [0, 2*pi); turning +d_theta shifts content by -d_theta
        if self.num_sectors == 8:
            frac = d_theta / (2.0 * math.pi / 8.0)  # ~0.3125 sectors per turn step
            for bi in range(b):
                f = float(frac[bi].item())
                if abs(f) > 1e-4:
                    # Linear circular interpolation
                    src_idx = (torch.arange(8, device=dev, dtype=torch.float32) + f) % 8.0
                    i0 = torch.floor(src_idx).long() % 8
                    i1 = (i0 + 1) % 8
                    w1 = src_idx - torch.floor(src_idx)
                    sectors[bi] = sectors[bi, i0] * (1.0 - w1) + sectors[bi, i1] * w1

        # Decay sector recency and mark FOV sectors (7, 0, 1 cover [-45, +45] deg) as seen=1.0
        sectors = sectors * 0.994
        sectors[:, 0] = 1.0
        sectors[:, 1] = torch.maximum(sectors[:, 1], torch.full((b,), 0.85, device=dev))
        sectors[:, 7] = torch.maximum(sectors[:, 7], torch.full((b,), 0.85, device=dev))

        # 2. Match visible enemies to existing ALIVE slots, and update kill / occlusion states
        summary = torch.zeros(b, MEMORY_FEAT_DIM, device=dev, dtype=torch.float32)
        valid_mask = ent_mask.bool()
        et = ent_type.long()
        is_enemy = valid_mask & ((et == 1) | (et == 2) | (et == 3))

        for bi in range(b):
            e_indices = torch.nonzero(is_enemy[bi], as_tuple=False).squeeze(-1)
            matched_slots: set[int] = set()
            matched_enemies: set[int] = set()

            # Compute visible enemy ego coords
            vis_coords: list[tuple[int, float, float, float]] = []
            for idx in e_indices.tolist():
                d = float(torch.clamp(ent_features[bi, idx, 0], 0.0, 2.0).item())
                s = float(torch.clamp(ent_features[bi, idx, 1], -1.0, 1.0).item())
                c = float(torch.clamp(ent_features[bi, idx, 2], -1.0, 1.0).item())
                vis_coords.append((idx, d * c, d * s, float(et[bi, idx].item())))

            # Greedy nearest-neighbor matching against ALIVE slots
            pairs: list[tuple[float, int, int, float, float, float]] = []
            for slot_i in range(self.max_slots):
                if slots[bi, slot_i, 2].item() == 1.0:
                    sx_i = float(slots[bi, slot_i, 0].item())
                    sy_i = float(slots[bi, slot_i, 1].item())
                    for e_pos, (idx, vx, vy, etype_v) in enumerate(vis_coords):
                        dist_err = math.hypot(vx - sx_i, vy - sy_i)
                        if dist_err <= MATCH_RADIUS:
                            pairs.append((dist_err, slot_i, e_pos, vx, vy, etype_v))
            pairs.sort(key=lambda x: x[0])

            for _, slot_i, e_pos, vx, vy, etype_v in pairs:
                if slot_i in matched_slots or e_pos in matched_enemies:
                    continue
                matched_slots.add(slot_i)
                matched_enemies.add(e_pos)
                ang = math.atan2(vy, vx)
                was_shot = float(fired[bi].item()) and abs(ang) < math.radians(18.0)
                slots[bi, slot_i, 0] = vx
                slots[bi, slot_i, 1] = vy
                slots[bi, slot_i, 2] = 1.0  # ALIVE
                slots[bi, slot_i, 3] = 1.0  # confidence
                slots[bi, slot_i, 4] = etype_v
                slots[bi, slot_i, 5] = 0.0  # age
                slots[bi, slot_i, 6] = 0.0  # miss_fov_count
                if was_shot:
                    slots[bi, slot_i, 7] = 1.0
                else:
                    slots[bi, slot_i, 7] *= 0.85

            # Allocate unmatched visible enemies into free (or lowest-priority) slots
            for e_pos, (idx, vx, vy, etype_v) in enumerate(vis_coords):
                if e_pos in matched_enemies:
                    continue
                # Prefer empty slot, then oldest killed slot, then lowest-confidence alive slot
                chosen = -1
                for slot_i in range(self.max_slots):
                    if slots[bi, slot_i, 2].item() == 0.0:
                        chosen = slot_i
                        break
                if chosen < 0:
                    best_score = 1e9
                    for slot_i in range(self.max_slots):
                        st_v = slots[bi, slot_i, 2].item()
                        sc = (0.0 if st_v == 2.0 else 10.0) + float(slots[bi, slot_i, 3].item()) - 0.01 * float(slots[bi, slot_i, 5].item())
                        if sc < best_score:
                            best_score = sc
                            chosen = slot_i
                slots[bi, chosen] = torch.tensor(
                    [vx, vy, 1.0, 1.0, etype_v, 0.0, 0.0, 0.0],
                    device=dev, dtype=torch.float32,
                )
                matched_slots.add(chosen)

            # Update unmatched ALIVE slots (distinguish out-of-FOV/occluded vs confirmed killed)
            rays_b = ray_distance[bi]  # (64,)
            for slot_i in range(self.max_slots):
                if slots[bi, slot_i, 2].item() != 1.0 or slot_i in matched_slots:
                    continue
                sx_i = float(slots[bi, slot_i, 0].item())
                sy_i = float(slots[bi, slot_i, 1].item())
                pred_dist = math.hypot(sx_i, sy_i)
                pred_ang = math.atan2(sy_i, sx_i)
                slots[bi, slot_i, 5] += 1.0  # age

                if fired[bi].item() and abs(pred_ang) < math.radians(18.0):
                    slots[bi, slot_i, 7] = 1.0

                # Check if inside central FOV and not occluded by a closer wall ray
                in_center_fov = abs(pred_ang) < CENTER_FOV_RAD and pred_dist < 0.95
                ray_idx = int(round((pred_ang % (2.0 * math.pi)) / (2.0 * math.pi) * S.NUM_RAYS)) % S.NUM_RAYS
                wall_dist = float(rays_b[ray_idx].item())
                occluded = (wall_dist + 0.04) < pred_dist

                if in_center_fov and not occluded:
                    slots[bi, slot_i, 6] += 1.0  # miss_fov_count
                    miss_cnt = float(slots[bi, slot_i, 6].item())
                    shot_recently = float(slots[bi, slot_i, 7].item()) > 0.35
                    if (shot_recently and miss_cnt >= 2.0) or miss_cnt >= 5.0:
                        # Confirmed killed!
                        slots[bi, slot_i, 2] = 2.0
                        slots[bi, slot_i, 3] = 1.0
                        slots[bi, slot_i, 6] = 0.0
                        continue
                else:
                    # Out of FOV or occluded -> STILL BELIEVED ALIVE; decay confidence slowly
                    slots[bi, slot_i, 3] *= 0.994
                    if slots[bi, slot_i, 3].item() < 0.15:
                        slots[bi, slot_i, 2] = 0.0  # stale memory expired

            # Build summary feature vector for policy (31 dims)
            alive_list: list[tuple[float, float, float, float, float, float]] = []
            killed_count = 0
            offscreen_alive = 0.0
            sector_living = [0.0] * self.num_sectors

            for slot_i in range(self.max_slots):
                st_v = slots[bi, slot_i, 2].item()
                if st_v == 1.0:
                    sx_i = float(slots[bi, slot_i, 0].item())
                    sy_i = float(slots[bi, slot_i, 1].item())
                    d = min(1.5, math.hypot(sx_i, sy_i))
                    ang = math.atan2(sy_i, sx_i)
                    conf = float(slots[bi, slot_i, 3].item())
                    is_vis = 1.0 if slot_i in matched_slots else 0.0
                    if is_vis == 0.0:
                        offscreen_alive = 1.0
                    # Priority score: prefer visible & closer & higher confidence
                    prio = d - 0.15 * is_vis - 0.10 * conf
                    alive_list.append((prio, d, math.sin(ang), math.cos(ang), conf, is_vis))
                    sec_i = int(round((ang % (2.0 * math.pi)) / (2.0 * math.pi) * self.num_sectors)) % self.num_sectors
                    sector_living[sec_i] = max(sector_living[sec_i], conf)
                elif st_v == 2.0:
                    killed_count += 1

            alive_list.sort(key=lambda x: x[0])
            if len(alive_list) >= 1:
                _, d, s, c, conf, iv = alive_list[0]
                summary[bi, 0:6] = torch.tensor([d, s, c, conf, iv, 1.0], device=dev)
            if len(alive_list) >= 2:
                _, d, s, c, conf, iv = alive_list[1]
                summary[bi, 6:12] = torch.tensor([d, s, c, conf, iv, 1.0], device=dev)

            sec_live_t = torch.tensor(sector_living, device=dev, dtype=torch.float32)
            sec_cleared_t = sectors[bi] * (1.0 - sec_live_t)
            summary[bi, 12 : 12 + self.num_sectors] = sec_live_t
            summary[bi, 12 + self.num_sectors : 12 + 2 * self.num_sectors] = sec_cleared_t
            summary[bi, 28] = min(1.0, len(alive_list) / 5.0)
            summary[bi, 29] = min(1.0, killed_count / 5.0)
            summary[bi, 30] = offscreen_alive

        next_state = torch.cat([slots.view(b, self.max_slots * SLOT_DIM), sectors], dim=-1)
        return next_state, summary

    def forward(self, summary_feat: torch.Tensor) -> torch.Tensor:
        return self.feat_net(summary_feat)
