"""Grandmaster Combat Training: 3-stage curriculum with diagnostic probes.

Stage 1: Aim Training (single enemy, random angles, short episodes)
  - Agent must detect, orient toward, and kill ONE enemy
  - BC warmup with relaxed firing threshold (cos >= 0.92)
  - Heavy directional turning and kill rewards

Stage 2: Multi-Enemy Combat (2-4 enemies, varied positions)
  - Agent must handle multiple threats, prioritize nearest
  - Builds on Stage 1 weights

Stage 3: Full Map (corridors, rooms, enemies, doors, exits)
  - Full gameplay with exploration + combat
  - Builds on Stage 2 weights
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import vizdoom as vd

from doom_env import make_env
from doom_env import spaces as S
from training.combat_scenarios import build_combat_scenarios
from training.maps.progression_wads import compile_udmf_wad
from training.models.doom_model import ESRHAAgentModel
from training.rollout import _batch_obs
from training.algorithms.ppo import PPOConfig, PPOTrainer
from training.rewards.reward import DoomRewardShaper, EpisodeStats
from training.rollout import RolloutBuffer


# ============================================================
# WAD generators for each stage
# ============================================================

def _build_aim_training_wads(out_dir: Path) -> list[dict]:
    """Stage 1: Small arenas with ONE enemy at different angles."""
    specs = []
    # 16 variants: enemy at different angles and distances
    configs = [
        # (enemy_x, enemy_y, label)
        (7, 4, "ahead"),
        (7, 2, "front_right"),
        (7, 6, "front_left"),
        (5, 1, "sharp_right"),
        (5, 7, "sharp_left"),
        (4, 1, "hard_right"),
        (4, 7, "hard_left"),
        (9, 4, "far_ahead"),
        (9, 2, "far_right"),
        (9, 6, "far_left"),
        (3, 4, "close_ahead"),
        (6, 3, "mid_right"),
        (6, 5, "mid_left"),
        (8, 1, "diagonal_right"),
        (8, 7, "diagonal_left"),
        (5, 4, "near_ahead"),
    ]
    for idx, (ex, ey, label) in enumerate(configs):
        w, h = 11, 9
        grid = [["#" for _ in range(w)] for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                grid[y][x] = "."
        grid[4][2] = "P"  # player always at (2, 4)
        grid[ey][ex] = "z"  # zombieman
        # Add health pickup so agent doesn't die quickly
        grid[4][3] = "h"
        wad = compile_udmf_wad("\n".join("".join(r) for r in grid), out_dir / f"aim_{idx:02d}_{label}.wad")
        specs.append({"name": f"aim_{idx:02d}_{label}", "wad": str(wad), "ticks": 350, "kill_target": 1})
    return specs


def _build_multi_enemy_wads(out_dir: Path) -> list[dict]:
    """Stage 2: Medium arenas with 2-4 enemies."""
    specs = []
    enemy_layouts = [
        # List of (x, y, type) for each variant
        [(7, 2, "z"), (7, 6, "z")],  # two enemies, left and right
        [(6, 4, "z"), (9, 4, "z")],  # two enemies, near and far
        [(5, 2, "z"), (5, 6, "z"), (8, 4, "s")],  # three enemies
        [(6, 2, "z"), (6, 6, "z"), (9, 2, "s"), (9, 6, "s")],  # four enemies, corners
        [(4, 4, "z"), (7, 2, "z"), (7, 6, "z")],  # close+flanks
        [(8, 3, "s"), (8, 5, "s")],  # two shotgunners
        [(5, 2, "z"), (8, 4, "i"), (5, 6, "z")],  # mixed types
        [(6, 1, "z"), (6, 7, "z"), (9, 1, "s"), (9, 7, "s")],  # four enemies, spread
    ]
    for idx, enemies in enumerate(enemy_layouts):
        w, h = 12, 9
        grid = [["#" for _ in range(w)] for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                grid[y][x] = "."
        grid[4][2] = "P"
        grid[4][5] = "h"
        grid[4][6] = "a"
        for ex, ey, et in enemies:
            grid[ey][ex] = et
        wad = compile_udmf_wad("\n".join("".join(r) for r in grid), out_dir / f"multi_{idx:02d}.wad")
        specs.append({"name": f"multi_{idx:02d}", "wad": str(wad), "ticks": 700, "kill_target": len(enemies)})
    return specs


def _build_center_room_wads(out_dir: Path) -> list[dict]:
    """Controlled center-room multi-enemy scenarios (Part 10 of spec)."""
    specs = []
    # Player at (3, 4) facing East (+x). gy < 4 is LEFT, gy > 4 is RIGHT.
    layouts = [
        ("lr_sym", 12, 9, (3, 4), [(7, 2, "z"), (7, 6, "z")]),            # Exact case: 1 left, 1 right (both visible)
        ("lr_close_l", 12, 9, (3, 4), [(6, 2, "z"), (8, 6, "z")]),        # Closer left, farther right
        ("lr_close_r", 12, 9, (3, 4), [(8, 2, "z"), (6, 6, "z")]),        # Farther left, closer right
        ("two_l_one_r", 13, 9, (3, 4), [(6, 2, "z"), (8, 1, "z"), (7, 6, "z")]),  # 2 left, 1 right
        ("one_l_two_r", 13, 9, (3, 4), [(7, 2, "z"), (6, 6, "z"), (8, 7, "z")]),  # 1 left, 2 right
        ("front_back", 13, 9, (6, 4), [(10, 4, "z"), (2, 4, "z")]),       # 1 ahead, 1 behind
        ("quad_center", 13, 11, (6, 5), [(9, 3, "z"), (9, 7, "z"), (3, 3, "z"), (3, 7, "z")]),  # 4 quadrants
        ("wide_lr", 13, 9, (2, 4), [(6, 1, "z"), (6, 7, "z"), (10, 4, "s")]),  # wide left + right + far center
    ]
    for name, w, h, (px, py), enemies in layouts:
        grid = [["#" for _ in range(w)] for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                grid[y][x] = "."
        grid[py][px] = "P"
        grid[py][min(w - 2, px + 1)] = "h"
        for ex, ey, et in enemies:
            grid[ey][ex] = et
        wad = compile_udmf_wad("\n".join("".join(r) for r in grid), out_dir / f"center_{name}.wad")
        specs.append({"name": f"center_{name}", "wad": str(wad), "ticks": 600, "kill_target": len(enemies)})
    return specs


def _build_corridor_combat_wads(out_dir: Path) -> list[dict]:
    """Stage 3: Long corridors with enemies and exit."""
    specs = []
    for idx in range(4):
        rng = random.Random(500 + idx)
        w, h = 21, 7
        grid = [["#" for _ in range(w)] for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                grid[y][x] = "."
        grid[3][1] = "P"
        grid[3][19] = "E"
        # Place 3-4 enemies along the corridor
        for cx in (5, 9, 13, 17):
            if rng.random() < 0.8:
                ey = rng.choice([2, 3, 4])
                grid[ey][cx] = rng.choice(["z", "s", "i"])
        grid[3][7] = "h"
        grid[3][11] = "a"
        grid[3][15] = "h"
        wad = compile_udmf_wad("\n".join("".join(r) for r in grid), out_dir / f"corr_combat_{idx:02d}.wad")
        specs.append({"name": f"corr_combat_{idx:02d}", "wad": str(wad), "ticks": 1400, "kill_target": None})
    return specs


# ============================================================
# BC Teacher with relaxed firing threshold
# ============================================================

def _combat_teacher_action(obs: dict, step_idx: int, mem_summary: np.ndarray | None = None) -> np.ndarray:
    """Human-style teacher with memory-guided target switching (training-time only)."""
    a = np.zeros(len(S.BUTTONS), dtype=np.float32)
    ve = obs["visible_entities"]
    mask = ve["mask"].astype(bool)
    etypes = ve["entity_type"]
    enemy_idx = [i for i in np.flatnonzero(mask) if int(etypes[i]) in (1, 2, 3)]

    rays = obs["rays"]["distance"]
    hit_types = obs["rays"]["hit_type"]
    front_ray = float(rays[0])

    if enemy_idx:
        # Target the best-aligned / closest visible enemy (dist is normalized [0, 1] by 1024)
        best_i = max(enemy_idx, key=lambda i: float(ve["features"][i, 2]) - 0.15 * float(ve["features"][i, 0]))
        dist = float(ve["features"][best_i, 0])
        sin_a = float(ve["features"][best_i, 1])
        cos_a = float(ve["features"][best_i, 2])

        if cos_a >= 0.982:  # Tight aim (~10.8 deg) so shots actually hit!
            a[6] = 1.0  # ATTACK
            if dist > 0.10 and front_ray > 0.12:
                a[0] = 1.0  # advance while firing
            # Gentle strafe during close/mid combat
            if dist < 0.25:
                a[2 if (step_idx // 8) % 2 == 0 else 3] = 1.0
        else:
            # Turn toward enemy until centered
            if sin_a > 0.0:
                a[4] = 1.0  # TURN_LEFT
            else:
                a[5] = 1.0  # TURN_RIGHT
            # Also move forward while turning if path is clear and angle is moderate
            if front_ray > 0.20 and abs(sin_a) < 0.4:
                a[0] = 1.0
    elif mem_summary is not None and float(mem_summary[5]) > 0.5:
        # No enemy in FOV, but EnemyAwarenessMemory remembers a living threat! Turn toward it.
        mem_sin = float(mem_summary[1])
        if mem_sin > 0.0:
            a[4] = 1.0  # TURN_LEFT toward remembered enemy
        else:
            a[5] = 1.0  # TURN_RIGHT toward remembered enemy
    else:
        # No enemy visible
        if int(hit_types[0]) == 2 and front_ray < 0.15:
            a[7] = 1.0  # USE door
        if front_ray > 0.20:
            a[0] = 1.0  # run forward
        else:
            left_clear = float(np.mean(rays[1:9]))
            right_clear = float(np.mean(rays[-8:]))
            a[4 if left_clear >= right_clear else 5] = 1.0
            if front_ray > 0.08:
                a[0] = 1.0
    return a


# ============================================================
# BC Warmup Stage
# ============================================================

def bc_warmup(specs: list[dict], save_path: Path, n_steps: int = 12288, seed: int = 700) -> Path:
    """Collect teacher trajectories and pretrain with class-balanced BCE."""
    rng = random.Random(seed)
    device = "cpu"
    model = ESRHAAgentModel(hidden_dim=256, use_visual=False, use_depth=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    seq_len = 32
    obs_seqs: list[list[dict]] = []
    act_seqs: list[list[np.ndarray]] = []
    pa_seqs: list[list[np.ndarray]] = []

    sp = rng.choice(specs)
    env = make_env(sp["wad"], "MAP01", episode_ticks=sp["ticks"], render_buffers=False)
    obs, _ = env.reset(seed=rng.randint(0, 100000))
    prev_a = np.zeros(len(S.BUTTONS), dtype=np.float32)
    em_state = model.enemy_memory.init_state(1, device)
    cur_obs, cur_act, cur_pa = [], [], []

    print(f"[BC] Collecting {n_steps} teacher steps from {len(specs)} maps...", flush=True)
    for t in range(n_steps):
        obs_t = _batch_obs([obs], device)
        pa_t = torch.as_tensor(prev_a, dtype=torch.float32, device=device).unsqueeze(0)
        em_state, mem_sum = model.enemy_memory.step_state(
            em_state, obs_t["ent_features"], obs_t["ent_type"], obs_t["ent_mask"],
            obs_t["ray_distance"], obs_t["vitals"], pa_t,
        )
        act = _combat_teacher_action(obs, t, mem_summary=mem_sum.squeeze(0).cpu().numpy())
        cur_obs.append(obs)
        cur_act.append(act)
        cur_pa.append(prev_a.copy())
        next_obs, _, term, trunc, _ = env.step(act.astype(np.int8))
        prev_a = act
        obs = next_obs

        if len(cur_obs) == seq_len:
            obs_seqs.append(cur_obs)
            act_seqs.append(cur_act)
            pa_seqs.append(cur_pa)
            cur_obs, cur_act, cur_pa = [], [], []

        if term or trunc:
            env.close()
            sp = rng.choice(specs)
            env = make_env(sp["wad"], "MAP01", episode_ticks=sp["ticks"], render_buffers=False)
            obs, _ = env.reset(seed=rng.randint(0, 100000))
            prev_a = np.zeros(len(S.BUTTONS), dtype=np.float32)
            em_state = model.enemy_memory.init_state(1, device)
            cur_obs, cur_act, cur_pa = [], [], []
    env.close()

    n_seq = len(obs_seqs)
    print(f"[BC] Collected {n_seq} sequences ({n_seq * seq_len} steps). Training 20 epochs...", flush=True)

    # Class-balanced weights: FWD gets 2x, TURN gets 4x, ATTACK gets 8x, STRAFE gets 2x
    pos_w = torch.tensor([2.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 2.0, 1.0], device=device)

    model.train()
    indices = list(range(n_seq))
    for epoch in range(1, 21):
        rng.shuffle(indices)
        total_loss = 0.0
        batches = 0
        for i in range(0, n_seq, 16):
            b_idx = indices[i: i + 16]
            bsz = len(b_idx)
            flat_obs = [obs_seqs[j][k] for j in b_idx for k in range(seq_len)]
            b_flat = _batch_obs(flat_obs, device)
            obs_seq = {k: v.reshape(bsz, seq_len, *v.shape[1:]) for k, v in b_flat.items()}
            act_t = torch.from_numpy(np.stack([np.stack(act_seqs[j]) for j in b_idx])).to(device)
            pa_t = torch.from_numpy(np.stack([np.stack(pa_seqs[j]) for j in b_idx])).to(device)

            out = model.forward_sequence(obs_seq, hidden=None, prev_actions=pa_t)
            loss = F.binary_cross_entropy_with_logits(out["action_logits"], act_t, pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
            batches += 1
        if epoch in (1, 5, 10, 15, 20):
            print(f"[BC epoch {epoch:2d}/20] bce_loss={total_loss / max(1, batches):.4f}", flush=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"hidden_dim": 256, "use_visual": False, "use_depth": False},
        },
        save_path,
    )
    print(f"[BC] Saved to {save_path}", flush=True)
    return save_path


# ============================================================
# PPO Training Stage (reusable)
# ============================================================

def ppo_stage(
    name: str,
    specs: list[dict],
    total_steps: int,
    save_dir: Path,
    init_ckpt: Path | None = None,
    lr: float = 3e-4,
    ent_coef: float = 0.005,
    seed: int = 42,
    kill_terminates: bool = False,
) -> Path:
    """Run PPO training stage and return best checkpoint path."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    device = "cpu"
    save_dir.mkdir(parents=True, exist_ok=True)

    num_envs, rollout_steps = 4, 256
    steps_per_upd = num_envs * rollout_steps
    total_updates = max(1, total_steps // steps_per_upd)

    model = ESRHAAgentModel(hidden_dim=256, use_visual=False, use_depth=False).to(device)
    if init_ckpt is not None and init_ckpt.exists():
        ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[{name}] Loaded weights from {init_ckpt}", flush=True)

    trainer = PPOTrainer(
        model,
        PPOConfig(lr=lr, ent_coef=ent_coef, seq_len=32, batch_seqs=8, ppo_epochs=4),
        total_updates,
        device,
    )

    def _spawn():
        sp = rng.choice(specs)
        e = make_env(sp["wad"], "MAP01", episode_ticks=sp["ticks"], render_buffers=False)
        o, _ = e.reset(seed=rng.randint(0, 100000))
        sh = DoomRewardShaper()
        sh.reset(e.privileged_state(), obs=o)
        return e, sh, o, sp

    envs, shapers, obs_list, active_specs = [], [], [], []
    for _ in range(num_envs):
        e, sh, o, sp = _spawn()
        envs.append(e); shapers.append(sh); obs_list.append(o); active_specs.append(sp)

    hidden = model.init_hidden(num_envs, device)
    prev_action = np.zeros((num_envs, len(S.BUTTONS)), dtype=np.float32)
    done_steps = 0
    best_score = -1e9

    try:
        for upd in range(1, total_updates + 1):
            t0 = time.perf_counter()
            model.eval()
            buf = RolloutBuffer(rollout_steps, num_envs, model.state_dim)
            completed: list[EpisodeStats] = []

            for _ in range(rollout_steps):
                obs_t = _batch_obs(obs_list, device=device)
                pa_t = torch.as_tensor(prev_action, dtype=torch.float32, device=device)
                with torch.no_grad():
                    out = model.forward_step(obs_t, hidden=hidden, prev_action=pa_t)
                    action_t = out["tactical_dist"].sample()
                    lp_t = out["tactical_dist"].log_prob(action_t)
                    val_t = out["value"]

                action_np = action_t.cpu().numpy().astype(np.float32)
                action_int = action_np.astype(np.int8)
                hid_in = hidden.squeeze(0).cpu().numpy()
                rewards = np.zeros(num_envs, dtype=np.float32)
                dones = np.zeros(num_envs, dtype=np.float32)
                next_obs_list = []
                next_hidden = out["next_hidden"]
                next_pa = action_np.copy()

                mem_sum_np = out["mem_summary"].cpu().numpy()
                for i in range(num_envs):
                    no, _, term, trunc, _ = envs[i].step(action_int[i])
                    priv = envs[i].privileged_state()
                    r, st = shapers[i].step(priv, action=action_int[i], obs=no, mem_summary=mem_sum_np[i])
                    if kill_terminates and priv["kills"] >= (active_specs[i].get("kill_target") or 1):
                        r += 3.0
                        st = shapers[i]._stats
                        st.reward += 3.0
                        term = True
                    done = bool(term or trunc)
                    rewards[i] = r; dones[i] = float(done)
                    if done:
                        if st is not None:
                            completed.append(st)
                        envs[i].close()
                        envs[i], shapers[i], no, active_specs[i] = _spawn()
                        next_hidden[:, i, :] = 0.0; next_pa[i] = 0.0
                    next_obs_list.append(no)

                buf.add(obs_list, prev_action, action_np, lp_t.cpu().numpy(), val_t.cpu().numpy(), rewards, dones, hid_in)
                obs_list = next_obs_list; hidden = next_hidden; prev_action = next_pa

            with torch.no_grad():
                lo = model.forward_step(
                    _batch_obs(obs_list, device=device), hidden=hidden,
                    prev_action=torch.as_tensor(prev_action, dtype=torch.float32, device=device),
                )
            trainer.update(buf, lo["value"].cpu().numpy(), np.zeros(num_envs, dtype=np.float32))
            done_steps += steps_per_upd

            if completed:
                rew = float(np.mean([e.reward for e in completed]))
                kills = float(np.mean([e.kills for e in completed]))
                aimed = sum(e.aimed_shots for e in completed)
                wasted = sum(e.wasted_shots for e in completed)
                hit_rate = aimed / max(1, aimed + wasted)
                er = float(np.mean([e.exit_reached for e in completed]))
                dist = float(np.mean([e.distance_travelled for e in completed]))
                fwd = sum(e.forward_steps for e in completed) / max(1, sum(e.length for e in completed))
                print(
                    f"[{name} {upd:2d}/{total_updates}] steps={done_steps:5d} sps={steps_per_upd/(time.perf_counter()-t0):5.1f} | "
                    f"rew={rew:+6.2f} kills={kills:.2f} hit={hit_rate:.2f} exit={er:.2f} dist={dist:5.0f} fwd={fwd:.2f}",
                    flush=True,
                )
                score = rew + kills * 5.0 + er * 5.0
                if score > best_score:
                    best_score = score
                    trainer.save_checkpoint(save_dir / "best.pt", upd, done_steps)
            trainer.save_checkpoint(save_dir / "latest.pt", upd, done_steps)
    finally:
        for e in envs:
            e.close()

    best_path = save_dir / "best.pt"
    if not best_path.exists():
        best_path = save_dir / "latest.pt"
    return best_path


# ============================================================
# Diagnostic probe
# ============================================================

def probe_policy(ckpt_path: Path) -> dict:
    """Test if the policy responds correctly to enemy at different angles."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ESRHAAgentModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = "cpu"

    def _probe(angle_deg: float | None, dist: float = 0.15):
        obs_t = {
            "ray_distance": torch.full((1, 64), 0.5, dtype=torch.float32),
            "ray_hit_type": torch.ones((1, 64), dtype=torch.long),
            "vitals": torch.zeros((1, len(S.VITALS_INDEX)), dtype=torch.float32),
            "depth": torch.full((1, 1, 60, 80), 0.5, dtype=torch.float32),
            "rgb": torch.zeros((1, 3, 60, 80), dtype=torch.float32),
            "ent_features": torch.zeros((1, 32, 5), dtype=torch.float32),
            "ent_type": torch.zeros((1, 32), dtype=torch.long),
            "ent_mask": torch.zeros((1, 32), dtype=torch.float32),
        }
        obs_t["vitals"][0, 0] = 100.0  # health
        obs_t["vitals"][0, 3] = 50.0   # ammo

        if angle_deg is not None:
            rad = math.radians(angle_deg)
            obs_t["ent_features"][0, 0] = torch.tensor(
                [dist, math.sin(rad), math.cos(rad), 0.0, 0.0], dtype=torch.float32,
            )
            obs_t["ent_type"][0, 0] = 2  # hitscan enemy
            obs_t["ent_mask"][0, 0] = 1.0

        h = model.init_hidden(1, device)
        pa = torch.zeros((1, len(S.BUTTONS)), dtype=torch.float32)
        with torch.no_grad():
            out = model.forward_step(obs_t, hidden=h, prev_action=pa)
        probs = torch.sigmoid(out["action_logits"]).squeeze(0).cpu().numpy()
        return {
            "FWD": round(float(probs[0]), 3),
            "BACK": round(float(probs[1]), 3),
            "TURN_L": round(float(probs[4]), 3),
            "TURN_R": round(float(probs[5]), 3),
            "ATK": round(float(probs[6]), 3),
        }

    return {
        "no_enemy": _probe(None),
        "enemy_ahead_0": _probe(0.0),
        "enemy_left_15": _probe(15.0),
        "enemy_left_30": _probe(30.0),
        "enemy_left_60": _probe(60.0),
        "enemy_right_-15": _probe(-15.0),
        "enemy_right_-30": _probe(-30.0),
        "enemy_right_-60": _probe(-60.0),
        "enemy_behind_180": _probe(180.0),
    }


# ============================================================
# Main Training Pipeline
# ============================================================

def main() -> None:
    maps_dir = Path("training/maps/grandmaster")
    maps_dir.mkdir(parents=True, exist_ok=True)

    # Build all training maps
    print("=== Building training maps ===", flush=True)
    aim_specs = _build_aim_training_wads(maps_dir)
    center_specs = _build_center_room_wads(maps_dir)
    multi_specs = _build_multi_enemy_wads(maps_dir)
    corridor_specs = _build_corridor_combat_wads(maps_dir)

    # Also include official combat scenarios
    combat = build_combat_scenarios()
    freedoom2 = str(Path(os.path.dirname(vd.__file__)) / "freedoom2.wad")

    run_dir = Path("training/runs/grandmaster_v9")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ============ STAGE 1: BC Warmup on aim + center-room multi-enemy maps ============
    print("\n=== STAGE 1: BC Warmup ===", flush=True)
    bc_specs = center_specs * 4 + aim_specs * 2 + combat["A_basic"][:4] + combat["C_corridor"][:2]
    bc_ckpt = bc_warmup(bc_specs, run_dir / "bc_init.pt", n_steps=12288, seed=920)

    print("\n--- BC Probe ---", flush=True)
    bc_probe = probe_policy(bc_ckpt)
    print(json.dumps(bc_probe, indent=2), flush=True)

    # ============ STAGE 2: PPO Aim Training (single enemy, kill_terminates) ============
    print("\n=== STAGE 2: Aim Training PPO ===", flush=True)
    aim_ckpt = ppo_stage(
        name="S2_Aim",
        specs=aim_specs * 2 + combat["A_basic"][:4],
        total_steps=20480,
        save_dir=run_dir / "s2_aim",
        init_ckpt=bc_ckpt,
        lr=3e-4,
        ent_coef=0.005,
        seed=921,
        kill_terminates=True,
    )

    print("\n--- Aim Training Probe ---", flush=True)
    aim_probe = probe_policy(aim_ckpt)
    print(json.dumps(aim_probe, indent=2), flush=True)

    # ============ STAGE 3: PPO Center-Room & Multi-Enemy Memory Prioritization ============
    print("\n=== STAGE 3: Multi-Enemy Memory PPO ===", flush=True)
    multi_ckpt = ppo_stage(
        name="S3_Multi",
        specs=center_specs * 4 + multi_specs * 2 + combat["B_defend"][:2] + combat["C_corridor"][:2],
        total_steps=24576,
        save_dir=run_dir / "s3_multi",
        init_ckpt=aim_ckpt,
        lr=2e-4,
        ent_coef=0.004,
        seed=922,
        kill_terminates=False,
    )

    print("\n--- Multi-Enemy Probe ---", flush=True)
    multi_probe = probe_policy(multi_ckpt)
    print(json.dumps(multi_probe, indent=2), flush=True)

    # ============ STAGE 4: PPO Full Map (corridors + rooms + exits) ============
    print("\n=== STAGE 4: Full Map PPO ===", flush=True)
    full_specs = (
        corridor_specs * 3
        + center_specs * 2
        + multi_specs[:4]
        + combat["C_corridor"] * 2
        + [{"name": "freedoom2_map01", "wad": freedoom2, "ticks": 1400, "kill_target": None}] * 2
        + [{"name": "two_rooms", "wad": "maps/two_rooms.wad", "ticks": 1400, "kill_target": None}] * 2
    )
    full_ckpt = ppo_stage(
        name="S4_Full",
        specs=full_specs,
        total_steps=28672,
        save_dir=run_dir / "s4_full",
        init_ckpt=multi_ckpt,
        lr=1.5e-4,
        ent_coef=0.003,
        seed=923,
        kill_terminates=False,
    )

    print("\n--- Full Map Probe ---", flush=True)
    full_probe = probe_policy(full_ckpt)
    print(json.dumps(full_probe, indent=2), flush=True)

    # ============ EXPORT ============
    baseline = Path("submission/weights/esrha_best.pt")
    shutil.copy2(full_ckpt, baseline)
    print(f"\nExported {full_ckpt} -> {baseline}", flush=True)

    studio_latest = Path("training/runs/studio_experiments/combat_studio_exp/latest.pt")
    if studio_latest.parent.exists():
        shutil.copy2(full_ckpt, studio_latest)
        print(f"Exported -> {studio_latest}", flush=True)

    # Save summary
    summary = {
        "bc_probe": bc_probe,
        "aim_probe": aim_probe,
        "multi_probe": multi_probe,
        "full_probe": full_probe,
    }
    (run_dir / "grandmaster_v7_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== GRANDMASTER V7 TRAINING COMPLETE ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
