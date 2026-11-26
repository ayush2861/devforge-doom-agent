"""Observation-legal Ego-Centric Enemy, Medikit, and Ammo/Weapon Awareness Memory.

Tracks remembered enemies, confirmed kills, cleared angular sectors, medikits,
and weapon/ammo status using ONLY legal agent observations (visible_entities,
rays, vitals) and previous actions.
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
# Medikit memory: 2 slots of [ego_x, ego_y, active(0/1), confidence]
MAX_MED_SLOTS = 2
MED_SLOT_DIM = 4
# Summary feature vector fed to the neural policy (43 dims):
#   nearest alive enemy (6): [dist, sin, cos, conf, is_vis, present]
#   second alive enemy  (6): [dist, sin, cos, conf, is_vis, present]
#   8 sectors * 2      (16): [living_threat(8), cleared_score(8)]
#   global enemy        (3): [alive_norm, killed_norm, any_offscreen_alive]
#   health & medikit    (6): [health_norm, low_health, med_dist, med_sin, med_cos, has_medikit]
#   ammo & weapon       (6): [is_empty_or_melee, low_ammo, has_other_loaded, ammo_dist, ammo_sin, has_ammo_pickup]
MEMORY_FEAT_DIM = 6 + 6 + NUM_SECTORS * 2 + 3 + 6 + 6  # 43

LOW_HEALTH_THRESHOLD = 45.0
LOW_AMMO_THRESHOLD = 5.0

TURN_STEP_RAD = math.radians(14.0625)
CENTER_FOV_RAD = math.radians(42.0)
MATCH_RADIUS = 0.18


class EnemyAwarenessMemory(nn.Module):
    """Stateful ego-centric enemy, medikit, and ammo/cleared-area tracker."""

    def __init__(self, out_dim: int = 128, max_slots: int = MAX_MEM_SLOTS, num_sectors: int = NUM_SECTORS):
        super().__init__()
        self.max_slots = max_slots
        self.num_sectors = num_sectors
        self.state_dim = max_slots * SLOT_DIM + num_sectors + MAX_MED_SLOTS * MED_SLOT_DIM  # 96 + 8 + 8 = 112

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
        """Update ego-centric enemy slots, medikit slots & cleared sectors for one step."""
        b = ent_features.shape[0]
        dev = ent_features.device
        enemy_end = self.max_slots * SLOT_DIM
        sec_end = enemy_end + self.num_sectors
        slots = state[:, :enemy_end].view(b, self.max_slots, SLOT_DIM).clone()
        sectors = state[:, enemy_end:sec_end].clone()
        med_slots = state[:, sec_end:].view(b, MAX_MED_SLOTS, MED_SLOT_DIM).clone()

        # 1. Ego-motion compensation
        if prev_action is not None:
            turn_l = prev_action[:, 4] > 0.5
            turn_r = prev_action[:, 5] > 0.5
            d_theta = torch.where(turn_l & ~turn_r, TURN_STEP_RAD, 0.0)
            d_theta = torch.where(turn_r & ~turn_l, -TURN_STEP_RAD, d_theta)
            fired = prev_action[:, 6] > 0.5
        else:
            d_theta = torch.zeros(b, device=dev)
            fired = torch.zeros(b, dtype=torch.bool, device=dev)

        dx = (vitals[:, 14].float() * S.FRAME_SKIP) / S.RAY_RANGE
        dy = -(vitals[:, 15].float() * S.FRAME_SKIP) / S.RAY_RANGE
        cos_dt = torch.cos(d_theta).unsqueeze(1)
        sin_dt = torch.sin(d_theta).unsqueeze(1)

        # Rotate & translate enemy slots
        non_empty = slots[:, :, 2] > 0.5
        sx = slots[:, :, 0] - dx.unsqueeze(1)
        sy = slots[:, :, 1] - dy.unsqueeze(1)
        slots[:, :, 0] = torch.where(non_empty, sx * cos_dt + sy * sin_dt, 0.0)
        slots[:, :, 1] = torch.where(non_empty, -sx * sin_dt + sy * cos_dt, 0.0)

        # Rotate & translate medikit slots
        med_active = med_slots[:, :, 2] > 0.5
        mx = med_slots[:, :, 0] - dx.unsqueeze(1)
        my = med_slots[:, :, 1] - dy.unsqueeze(1)
        med_slots[:, :, 0] = torch.where(med_active, mx * cos_dt + my * sin_dt, 0.0)
        med_slots[:, :, 1] = torch.where(med_active, -mx * sin_dt + my * cos_dt, 0.0)

        # Rotate 8 sector recency values
        if self.num_sectors == 8:
            frac = d_theta / (2.0 * math.pi / 8.0)
            for bi in range(b):
                f = float(frac[bi].item())
                if abs(f) > 1e-4:
                    src_idx = (torch.arange(8, device=dev, dtype=torch.float32) + f) % 8.0
                    i0 = torch.floor(src_idx).long() % 8
                    i1 = (i0 + 1) % 8
                    w1 = src_idx - torch.floor(src_idx)
                    sectors[bi] = sectors[bi, i0] * (1.0 - w1) + sectors[bi, i1] * w1

        sectors = sectors * 0.994
        sectors[:, 0] = 1.0
        sectors[:, 1] = torch.maximum(sectors[:, 1], torch.full((b,), 0.85, device=dev))
        sectors[:, 7] = torch.maximum(sectors[:, 7], torch.full((b,), 0.85, device=dev))

        summary = torch.zeros(b, MEMORY_FEAT_DIM, device=dev, dtype=torch.float32)
        valid_mask = ent_mask.bool()
        et = ent_type.long()
        is_enemy = valid_mask & ((et == 1) | (et == 2) | (et == 3))
        is_health = valid_mask & (et == 5)
        is_ammo_wpn = valid_mask & ((et == 7) | (et == 8))

        for bi in range(b):
            # ---- A. Enemy tracking ----
            e_indices = torch.nonzero(is_enemy[bi], as_tuple=False).squeeze(-1)
            matched_slots: set[int] = set()
            matched_enemies: set[int] = set()

            vis_coords: list[tuple[int, float, float, float]] = []
            for idx in e_indices.tolist():
                d = float(torch.clamp(ent_features[bi, idx, 0], 0.0, 2.0).item())
                s = float(torch.clamp(ent_features[bi, idx, 1], -1.0, 1.0).item())
                c = float(torch.clamp(ent_features[bi, idx, 2], -1.0, 1.0).item())
                vis_coords.append((idx, d * c, d * s, float(et[bi, idx].item())))

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
                slots[bi, slot_i, 2] = 1.0
                slots[bi, slot_i, 3] = 1.0
                slots[bi, slot_i, 4] = etype_v
                slots[bi, slot_i, 5] = 0.0
                slots[bi, slot_i, 6] = 0.0
                slots[bi, slot_i, 7] = 1.0 if was_shot else float(slots[bi, slot_i, 7].item()) * 0.85

            for e_pos, (idx, vx, vy, etype_v) in enumerate(vis_coords):
                if e_pos in matched_enemies:
                    continue
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

            rays_b = ray_distance[bi]
            for slot_i in range(self.max_slots):
                if slots[bi, slot_i, 2].item() != 1.0 or slot_i in matched_slots:
                    continue
                sx_i = float(slots[bi, slot_i, 0].item())
                sy_i = float(slots[bi, slot_i, 1].item())
                pred_dist = math.hypot(sx_i, sy_i)
                pred_ang = math.atan2(sy_i, sx_i)
                slots[bi, slot_i, 5] += 1.0

                if fired[bi].item() and abs(pred_ang) < math.radians(18.0):
                    slots[bi, slot_i, 7] = 1.0

                in_center_fov = abs(pred_ang) < CENTER_FOV_RAD and pred_dist < 0.95
                ray_idx = int(round((pred_ang % (2.0 * math.pi)) / (2.0 * math.pi) * S.NUM_RAYS)) % S.NUM_RAYS
                # Check 3-ray neighborhood (+-5.6 deg) so narrow corridor side walls don't falsely mask an enemy
                wall_dist = max(
                    float(rays_b[(ray_idx - 1) % S.NUM_RAYS].item()),
                    float(rays_b[ray_idx].item()),
                    float(rays_b[(ray_idx + 1) % S.NUM_RAYS].item()),
                )
                occluded = (wall_dist + 0.05) < pred_dist
                if float(slots[bi, slot_i, 7].item()) > 0.35 and pred_dist < 0.30:
                    occluded = False

                if in_center_fov and not occluded:
                    slots[bi, slot_i, 6] += 1.0
                    miss_cnt = float(slots[bi, slot_i, 6].item())
                    shot_recently = float(slots[bi, slot_i, 7].item()) > 0.35
                    if (shot_recently and miss_cnt >= 1.0) or miss_cnt >= 4.0:
                        slots[bi, slot_i, 2] = 2.0
                        slots[bi, slot_i, 3] = 1.0
                        slots[bi, slot_i, 6] = 0.0
                        continue
                else:
                    age_v = float(slots[bi, slot_i, 5].item())
                    slots[bi, slot_i, 3] *= (0.96 if age_v > 45.0 else 0.994)
                    if slots[bi, slot_i, 3].item() < 0.15:
                        slots[bi, slot_i, 2] = 0.0

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
                    c_ang = math.cos(ang)
                    prio = -(c_ang - 0.5 * d) - 0.25 * is_vis - 0.10 * conf
                    alive_list.append((prio, d, math.sin(ang), c_ang, conf, is_vis))
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

            # ---- B. Medikit tracking (visible + 2-slot memory) ----
            h_indices = torch.nonzero(is_health[bi], as_tuple=False).squeeze(-1)
            matched_med: set[int] = set()
            for idx in h_indices.tolist():
                d = float(torch.clamp(ent_features[bi, idx, 0], 0.0, 2.0).item())
                s = float(torch.clamp(ent_features[bi, idx, 1], -1.0, 1.0).item())
                c = float(torch.clamp(ent_features[bi, idx, 2], -1.0, 1.0).item())
                vx, vy = d * c, d * s
                # Match or insert into med_slots
                best_ms, best_err = -1, 0.15
                for ms in range(MAX_MED_SLOTS):
                    if med_slots[bi, ms, 2].item() > 0.5:
                        err = math.hypot(vx - float(med_slots[bi, ms, 0].item()), vy - float(med_slots[bi, ms, 1].item()))
                        if err < best_err:
                            best_err = err
                            best_ms = ms
                if best_ms < 0:
                    for ms in range(MAX_MED_SLOTS):
                        if med_slots[bi, ms, 2].item() < 0.5:
                            best_ms = ms
                            break
                if best_ms < 0:
                    best_ms = 0
                med_slots[bi, best_ms] = torch.tensor([vx, vy, 1.0, 1.0], device=dev)
                matched_med.add(best_ms)

            for ms in range(MAX_MED_SLOTS):
                if med_slots[bi, ms, 2].item() > 0.5 and ms not in matched_med:
                    mx_i = float(med_slots[bi, ms, 0].item())
                    my_i = float(med_slots[bi, ms, 1].item())
                    md = math.hypot(mx_i, my_i)
                    mang = math.atan2(my_i, mx_i)
                    # If we walked right over it or it's in center FOV unobstructed & gone -> collected!
                    if md < 0.05 or abs(mang) < CENTER_FOV_RAD:
                        med_slots[bi, ms, 2] = 0.0
                    else:
                        med_slots[bi, ms, 3] *= 0.992
                        if med_slots[bi, ms, 3].item() < 0.2:
                            med_slots[bi, ms, 2] = 0.0

            hp = float(vitals[bi, 0].item())
            summary[bi, 31] = max(0.0, min(1.0, hp / 100.0))
            summary[bi, 32] = 1.0 if hp <= LOW_HEALTH_THRESHOLD else 0.0
            best_med_d, best_med_s, best_med_c, has_med = 1.5, 0.0, 1.0, 0.0
            for ms in range(MAX_MED_SLOTS):
                if med_slots[bi, ms, 2].item() > 0.5:
                    mx_i = float(med_slots[bi, ms, 0].item())
                    my_i = float(med_slots[bi, ms, 1].item())
                    md = min(1.5, math.hypot(mx_i, my_i))
                    if md < best_med_d:
                        best_med_d = md
                        ang = math.atan2(my_i, mx_i)
                        best_med_s = math.sin(ang)
                        best_med_c = math.cos(ang)
                        has_med = 1.0
            if has_med > 0.5:
                summary[bi, 33:37] = torch.tensor([best_med_d, best_med_s, best_med_c, 1.0], device=dev)

            # ---- C. Ammo & Weapon status ----
            cur_wpn = int(round(float(vitals[bi, 2].item())))
            cur_ammo = float(vitals[bi, 3].item())
            bul = float(vitals[bi, 4].item())
            shl = float(vitals[bi, 5].item())
            rkt = float(vitals[bi, 6].item())
            cel = float(vitals[bi, 7].item())
            is_empty_or_melee = 1.0 if (cur_wpn <= 1 or (0.0 <= cur_ammo <= 0.5)) else 0.0
            is_low_ammo = 1.0 if (0.0 <= cur_ammo <= LOW_AMMO_THRESHOLD) else 0.0
            other_ammo = (
                (shl > 0 or rkt > 0 or cel > 0) if cur_wpn in (2, 4)
                else ((bul > 0 or rkt > 0 or cel > 0) if cur_wpn == 3
                      else ((bul + shl + rkt + cel) > 0 if cur_wpn <= 1 else (bul > 0 or shl > 0)))
            )
            summary[bi, 37] = is_empty_or_melee
            summary[bi, 38] = is_low_ammo
            summary[bi, 39] = 1.0 if other_ammo else 0.0

            a_indices = torch.nonzero(is_ammo_wpn[bi], as_tuple=False).squeeze(-1)
            if len(a_indices) > 0:
                best_ak = min(a_indices.tolist(), key=lambda i: float(ent_features[bi, i, 0].item()))
                summary[bi, 40] = float(torch.clamp(ent_features[bi, best_ak, 0], 0.0, 1.5).item())
                summary[bi, 41] = float(torch.clamp(ent_features[bi, best_ak, 1], -1.0, 1.0).item())
                summary[bi, 42] = 1.0

        next_state = torch.cat([slots.view(b, enemy_end), sectors, med_slots.view(b, MAX_MED_SLOTS * MED_SLOT_DIM)], dim=-1)
        return next_state, summary

    def forward(self, summary_feat: torch.Tensor) -> torch.Tensor:
        return self.feat_net(summary_feat)
