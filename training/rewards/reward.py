"""Training-only reward shaping with strong combat-centric rewards."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from doom_env import spaces as S

_ENEMY_TYPE_IDS = {i for i, t in enumerate(S.ENTITY_TYPES) if t.startswith("enemy")}


@dataclass
class EpisodeStats:
    reward: float = 0.0
    length: int = 0
    kills: int = 0
    damage_taken: float = 0.0
    items: int = 0
    keys: int = 0
    cells_visited: int = 0
    doors_opened: int = 0
    stuck_steps: int = 0
    ammo_used: int = 0
    wasted_shots: int = 0
    aimed_shots: int = 0
    distance_travelled: float = 0.0
    forward_steps: int = 0
    turn_steps: int = 0
    exit_reached: bool = False
    dead: bool = False


class DoomRewardShaper:
    """Dense reward shaper for training using privileged_state() and obs (never exposed to policy)."""

    def __init__(self, cell_size: float = 64.0, stuck_window: int = 16, stuck_min_dist: float = 12.0):
        self.cell_size = cell_size
        self.stuck_window = stuck_window
        self.stuck_min_dist = stuck_min_dist
        self._prev: dict | None = None
        self._visited: set[tuple[int, int]] = set()
        self._moved_sectors: set[int] = set()
        self._init_ceil: np.ndarray | None = None
        self._pos_hist: deque[tuple[float, float]] = deque(maxlen=stuck_window)
        self._last_action: np.ndarray | None = None
        self._repeat_count: int = 0
        self._prev_enemy_cos: float | None = None
        self._prev_ammo: float | None = None
        self._no_enemy_turn_steps: int = 0
        self._discovered_enemies: set[int] = set()
        self._stats = EpisodeStats()

    def reset(self, priv_state: dict, obs: dict | None = None) -> None:
        self._prev = priv_state
        self._visited.clear()
        self._moved_sectors.clear()
        self._pos_hist.clear()
        self._last_action = None
        self._repeat_count = 0
        self._no_enemy_turn_steps = 0
        self._prev_enemy_cos = None
        self._prev_ammo = float(obs["vitals"][3]) if obs is not None else None
        self._discovered_enemies.clear()

        ceil = priv_state.get("sector_ceiling")
        self._init_ceil = None if ceil is None else ceil.copy()
        cx = int(np.floor(priv_state["x"] / self.cell_size))
        cy = int(np.floor(priv_state["y"] / self.cell_size))
        self._visited.add((cx, cy))
        self._pos_hist.append((priv_state["x"], priv_state["y"]))
        self._stats = EpisodeStats(cells_visited=1)

    def step(
        self,
        priv_state: dict,
        action: np.ndarray | None = None,
        obs: dict | None = None,
        mem_summary: np.ndarray | None = None,
    ) -> tuple[float, EpisodeStats | None]:
        assert self._prev is not None, "call reset() before step()"
        r = -0.003  # slight step penalty for efficiency

        # 1. Exploration coverage (64x64 map-unit grid) — only reward advancing, not backing up blindly
        cx = int(np.floor(priv_state["x"] / self.cell_size))
        cy = int(np.floor(priv_state["y"] / self.cell_size))
        if (cx, cy) not in self._visited:
            self._visited.add((cx, cy))
            if action is None or int(action[1]) == 0:
                r += 0.10

        # 2. COMBAT: kills, aiming, shooting discipline
        d_kills = priv_state["kills"] - self._prev["kills"]
        if d_kills > 0:
            r += 5.0 * d_kills  # Strong kill reward

        d_hp = priv_state["health"] - self._prev["health"]
        if d_hp > 0:
            r += 0.03 * d_hp

        best_cos = -1.0
        best_sin = 0.0
        enemy_visible = False

        if obs is not None:
            cur_ammo = float(obs["vitals"][3])
            ammo_spent = max(0.0, self._prev_ammo - cur_ammo) if self._prev_ammo is not None else 0.0
            self._prev_ammo = cur_ammo
            self._stats.ammo_used += int(ammo_spent)

            ve = obs["visible_entities"]
            mask = ve["mask"].astype(bool)
            etypes = ve["entity_type"]
            enemy_idx = [idx for idx in np.flatnonzero(mask) if int(etypes[idx]) in _ENEMY_TYPE_IDS]

            if enemy_idx:
                enemy_visible = True
                # Find the most-aligned (highest cos) enemy
                best_k = int(np.argmax(ve["features"][enemy_idx, 2]))
                best_i = enemy_idx[best_k]
                best_cos = float(ve["features"][best_i, 2])
                best_sin = float(ve["features"][best_i, 1])
                best_dist = float(ve["features"][best_i, 0])

                # --- ALIGNMENT IMPROVEMENT REWARD (scaled up 2x) ---
                if self._prev_enemy_cos is not None:
                    improvement = best_cos - self._prev_enemy_cos
                    r += 2.0 * improvement

                # --- CENTERED AIM BONUS ---
                if best_cos >= 0.985:
                    r += 0.08  # tight lock
                    # Extra reward for holding steady (not turning when already aimed)
                    if action is not None and int(action[4]) == 0 and int(action[5]) == 0:
                        r += 0.06
                elif best_cos >= 0.95:
                    r += 0.03  # close to centered

                # --- DIRECTIONAL TURNING REWARD (until tightly centered) ---
                if action is not None and best_cos < 0.985:
                    if best_sin > 0.05:  # enemy to the LEFT
                        if int(action[4]) == 1 and int(action[5]) == 0:
                            r += 0.22  # Correct: turn left toward enemy
                        elif int(action[5]) == 1 and int(action[4]) == 0:
                            r -= 0.18  # Wrong: turning away from enemy
                    elif best_sin < -0.05:  # enemy to the RIGHT
                        if int(action[5]) == 1 and int(action[4]) == 0:
                            r += 0.22  # Correct: turn right toward enemy
                        elif int(action[4]) == 1 and int(action[5]) == 0:
                            r -= 0.18  # Wrong: turning away from enemy

                # --- APPROACH REWARD: move toward distant enemies (best_dist is normalized [0,1] by 2048) ---
                if best_cos >= 0.94 and best_dist > 0.15 and action is not None:
                    if int(action[0]) == 1:
                        r += 0.05  # advance toward aimed enemy

                self._prev_enemy_cos = best_cos
                self._no_enemy_turn_steps = 0

                # Discovery bonus (first time seeing an enemy in this position)
                for ei in enemy_idx:
                    ed = float(ve["features"][ei, 0])
                    es = float(ve["features"][ei, 1])
                    ec = float(ve["features"][ei, 2])
                    disc_key = (round(ed * 20.0), round(es, 1), round(ec, 1))
                    hk = hash(disc_key)
                    if hk not in self._discovered_enemies:
                        self._discovered_enemies.add(hk)
                        r += 0.15  # discovery bonus
            else:
                self._prev_enemy_cos = None

            # --- FIRING DISCIPLINE ---
            fired = (ammo_spent > 0) or (action is not None and int(action[6]) == 1)
            if fired:
                if enemy_visible and best_cos >= 0.982:
                    # Good shot: enemy tightly centered and visible (~10 deg)
                    r += 0.35 if best_cos >= 0.994 else 0.18
                    self._stats.aimed_shots += 1
                elif d_kills > 0:
                    pass  # already rewarded via kill
                else:
                    # Bad shot: no enemy visible, or enemy off-center
                    if not enemy_visible:
                        r -= 0.30  # firing at nothing
                    else:
                        r -= 0.25  # enemy visible but not centered
                    self._stats.wasted_shots += 1

        d_items = priv_state["items"] - self._prev["items"]
        if d_items > 0:
            r += 0.3 * d_items

        d_dmg = priv_state["damage_taken"] - self._prev["damage_taken"]
        if d_dmg > 0:
            r -= 0.04 * d_dmg

        # 3. Keycards & door progression
        d_keys = int(priv_state["keys"].sum() - self._prev["keys"].sum())
        if d_keys > 0:
            r += 2.5 * d_keys

        if self._init_ceil is not None and priv_state.get("sector_ceiling") is not None:
            moved = np.flatnonzero(np.abs(priv_state["sector_ceiling"] - self._init_ceil) > 4.0)
            for sid in moved:
                sid_i = int(sid)
                if sid_i not in self._moved_sectors:
                    self._moved_sectors.add(sid_i)
                    r += 1.0

        # 4. Movement and anti-stuck
        px, py = priv_state["x"], priv_state["y"]
        self._pos_hist.append((px, py))
        step_disp = float(np.hypot(px - self._prev["x"], py - self._prev["y"]))
        self._stats.distance_travelled += step_disp

        front_ray = float(obs["rays"]["distance"][0]) if obs is not None else 1.0
        if action is not None:
            if int(action[0]) == 1:
                self._stats.forward_steps += 1
            if int(action[4]) == 1 or int(action[5]) == 1:
                self._stats.turn_steps += 1

            # Penalize contradictory buttons
            if int(action[0]) == 1 and int(action[1]) == 1:
                r -= 0.15
            if int(action[2]) == 1 and int(action[3]) == 1:
                r -= 0.15
            if int(action[4]) == 1 and int(action[5]) == 1:
                r -= 0.15

        # Forward movement when no enemy visible (or turn toward remembered living threat!)
        if not enemy_visible and action is not None:
            has_mem_enemy = mem_summary is not None and float(mem_summary[5]) > 0.5
            if has_mem_enemy:
                mem_sin = float(mem_summary[1])
                if mem_sin > 0.10:
                    if int(action[4]) == 1 and int(action[5]) == 0:
                        r += 0.22  # turn left toward remembered living enemy!
                    elif int(action[5]) == 1 and int(action[4]) == 0:
                        r -= 0.18  # turning away from remembered enemy
                elif mem_sin < -0.10:
                    if int(action[5]) == 1 and int(action[4]) == 0:
                        r += 0.22  # turn right toward remembered living enemy!
                    elif int(action[4]) == 1 and int(action[5]) == 0:
                        r -= 0.18
            else:
                turning = (int(action[4]) == 1) or (int(action[5]) == 1)

                # Look once (<=3 steps), then advance forward when path is clear instead of turning backward
                if turning and (step_disp < 3.0 or front_ray > 0.22):
                    self._no_enemy_turn_steps += 1
                    if self._no_enemy_turn_steps > 3:
                        r -= 0.10  # stop turning, move forward
                else:
                    self._no_enemy_turn_steps = 0

                # Reward forward movement when path is clear and no enemies to engage
                if int(action[0]) == 1 and int(action[1]) == 0 and front_ray > 0.12:
                    r += 0.12 * min(step_disp / 10.0, 1.0)

                # Strongly penalize moving backward when no enemy is visible
                if int(action[1]) == 1:
                    r -= 0.22 if front_ray > 0.12 else 0.08

                # Wall walking penalty
                if int(action[0]) == 1 and front_ray < 0.06 and step_disp < 1.0:
                    r -= 0.04

        # Anti-stuck detection
        is_stuck = False
        if len(self._pos_hist) == self.stuck_window:
            x0, y0 = self._pos_hist[0]
            win_disp = float(np.hypot(px - x0, py - y0))
            if win_disp < self.stuck_min_dist and d_kills == 0:
                r -= 0.04
                is_stuck = True
                if action is not None and (int(action[4]) == 1 or int(action[5]) == 1) and not enemy_visible:
                    r -= 0.04  # extra penalty for spinning with no enemy

        if action is not None:
            if self._last_action is not None and np.array_equal(action, self._last_action):
                self._repeat_count += 1
            else:
                self._repeat_count = 0
                self._last_action = action.copy()
            if self._repeat_count > 12 and step_disp < 1.0:
                r -= 0.02
                is_stuck = True

        if is_stuck:
            self._stats.stuck_steps += 1

        # 5. Terminal events
        if priv_state["exit_reached"]:
            time_bonus = 2.0 * max(0.0, 1.0 - priv_state["tic"] / max(1, priv_state["episode_ticks"]))
            r += 10.0 + time_bonus
        elif priv_state["dead"]:
            r -= 5.0

        self._prev = priv_state
        self._stats.reward += r
        self._stats.length += 1
        self._stats.kills = priv_state["kills"]
        self._stats.damage_taken = priv_state["damage_taken"]
        self._stats.items = priv_state["items"]
        self._stats.keys = int(priv_state["keys"].sum())
        self._stats.cells_visited = len(self._visited)
        self._stats.doors_opened = len(self._moved_sectors)
        self._stats.exit_reached = priv_state["exit_reached"]
        self._stats.dead = priv_state["dead"]

        done = bool(priv_state["finished"])
        return float(r), (self._stats if done else None)
