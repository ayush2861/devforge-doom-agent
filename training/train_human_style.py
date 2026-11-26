"""Train ESRHA on human-style DOOM gameplay extracted from the reference video.

Behaviors modeled from the frame-by-frame human gameplay analysis:
1. Purposeful corridor advance & doorway entry (MOVE_FORWARD + strafe, zero wall-grinding).
2. Trigger discipline: hold fire while navigating/turning, fire only when enemy is centered.
3. Multi-enemy courtyard & corridor engagement with strafe-dodging and item/ammo pickups.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
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
from training.run_3_experiments import evaluate_experiment, train_stage


def build_human_courtyard_wads(out_dir: str | Path = "training/maps/combat") -> list[dict]:
    """Build corridor-into-courtyard WADs mirroring the human gameplay video."""
    out = Path(out_dir)
    specs: list[dict] = []

    # Variant 0: Green-corridor doorway opening into a multi-enemy courtyard with ammo/health
    layout0 = "\n".join([
        "###################",
        "#........#........#",
        "#.P......#..z..h..#",
        "#........#........#",
        "#........###.######",
        "#..........s......#",
        "#....h.......a..i.#",
        "#.................#",
        "#..........z....E.#",
        "###################",
    ])
    w0 = compile_udmf_wad(layout0, out / "human_courtyard_00.wad")
    specs.append({"name": "human_courtyard_00", "wad": str(w0), "ticks": 1200, "kill_target": None})

    # Variant 1: Narrow entry hall opening into wide arena with staggered enemies & pickups
    layout1 = "\n".join([
        "###################",
        "######............#",
        "#P...#..z....a....#",
        "#.................#",
        "#....#....s.....h.#",
        "######............#",
        "#.......i....z..E.#",
        "###################",
    ])
    w1 = compile_udmf_wad(layout1, out / "human_courtyard_01.wad")
    specs.append({"name": "human_courtyard_01", "wad": str(w1), "ticks": 1200, "kill_target": None})

    return specs


@torch.no_grad()
def probe_policy_behavior(ckpt_path: Path) -> dict[str, dict[str, float]]:
    """Probe policy action probabilities on synthetic human-video situations."""
    device = "cpu"
    model = ESRHAAgentModel(hidden_dim=256, use_visual=False, use_depth=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    idx_fwd = S.BUTTONS.index("MOVE_FORWARD")
    idx_back = S.BUTTONS.index("MOVE_BACKWARD")
    idx_sl = S.BUTTONS.index("MOVE_LEFT")
    idx_sr = S.BUTTONS.index("MOVE_RIGHT")
    idx_tl = S.BUTTONS.index("TURN_LEFT")
    idx_tr = S.BUTTONS.index("TURN_RIGHT")
    idx_atk = S.BUTTONS.index("ATTACK")

    def _probe(front_ray: float, angle_deg: float | None = None):
        rays = torch.full((1, 64), 0.5, dtype=torch.float32)
        rays[0, 0] = front_ray
        rays[0, 1] = front_ray
        rays[0, -1] = front_ray
        obs_tensors = {
            "ray_distance": rays,
            "ray_hit_type": torch.ones((1, 64), dtype=torch.long),
            "vitals": torch.zeros((1, len(S.VITALS_INDEX)), dtype=torch.float32),
            "depth": torch.full((1, 1, 60, 80), front_ray, dtype=torch.float32),
            "rgb": torch.zeros((1, 3, 60, 80), dtype=torch.float32),
            "ent_features": torch.zeros((1, 32, 5), dtype=torch.float32),
            "ent_type": torch.zeros((1, 32), dtype=torch.long),
            "ent_mask": torch.zeros((1, 32), dtype=torch.float32),
        }
        obs_tensors["vitals"][0, 0] = 100.0
        obs_tensors["vitals"][0, 3] = 20.0
        if angle_deg is not None:
            rad = math.radians(angle_deg)
            obs_tensors["ent_features"][0, 0] = torch.tensor(
                [250.0, math.sin(rad), math.cos(rad), 0.0, 0.0],
                dtype=torch.float32,
            )
            obs_tensors["ent_type"][0, 0] = 2
            obs_tensors["ent_mask"][0, 0] = 1.0
        h = model.memory.init_hidden(1, device)
        pa = torch.zeros((1, len(S.BUTTONS)), dtype=torch.float32)
        out = model.forward_step(obs_tensors, hidden=h, prev_action=pa)
        probs = torch.sigmoid(out["action_logits"]).squeeze(0).cpu().numpy()
        return {
            "FWD": float(probs[idx_fwd]),
            "BACK": float(probs[idx_back]),
            "STRF_L": float(probs[idx_sl]),
            "STRF_R": float(probs[idx_sr]),
            "TURN_L": float(probs[idx_tl]),
            "TURN_R": float(probs[idx_tr]),
            "ATK": float(probs[idx_atk]),
        }

    return {
        "open_corridor": _probe(0.9, None),
        "wall_ahead": _probe(0.05, None),
        "enemy_centered_0": _probe(0.8, 0.0),
        "enemy_off_15": _probe(0.8, 15.0),
        "enemy_left_+30": _probe(0.8, 30.0),
        "enemy_right_-30": _probe(0.8, -30.0),
    }


def _human_teacher_action(obs: dict, step_idx: int) -> np.ndarray:
    """Human-style DOOM teacher for BC warmup (training-time only; never used in agent.py)."""
    a = np.zeros(len(S.BUTTONS), dtype=np.float32)
    ve = obs["visible_entities"]
    mask = ve["mask"].astype(bool)
    etypes = ve["entity_type"]
    enemy_idx = [i for i in np.flatnonzero(mask) if int(etypes[i]) in (1, 2, 3)]

    rays = obs["rays"]["distance"]
    hit_types = obs["rays"]["hit_type"]
    front_ray = float(rays[0])

    if enemy_idx:
        # Target best-aligned / closest visible enemy
        best_i = max(enemy_idx, key=lambda i: float(ve["features"][i, 2]) - 0.0003 * float(ve["features"][i, 0]))
        dist = float(ve["features"][best_i, 0])
        sin_a = float(ve["features"][best_i, 1])
        cos_a = float(ve["features"][best_i, 2])

        if cos_a >= 0.978:
            a[6] = 1.0  # ATTACK when aimed within ~12 deg
            if dist > 180.0 and front_ray > 0.12:
                a[0] = 1.0  # MOVE_FORWARD to close distance
            # Gentle strafe-dodge during close/mid combat
            if dist < 450.0:
                a[2 if (step_idx // 8) % 2 == 0 else 3] = 1.0
        else:
            if sin_a > 0.0:
                a[4] = 1.0  # TURN_LEFT
            else:
                a[5] = 1.0  # TURN_RIGHT
            if front_ray > 0.25:
                a[0] = 1.0  # advance while turning
    else:
        # No enemy visible: open door if facing one, else run forward, else turn toward open side
        if int(hit_types[0]) == 2 and front_ray < 0.15:
            a[7] = 1.0  # USE
        if front_ray > 0.22:
            a[0] = 1.0  # MOVE_FORWARD cleanly
        else:
            left_clear = float(np.mean(rays[1:9]))
            right_clear = float(np.mean(rays[-8:]))
            a[4 if left_clear >= right_clear else 5] = 1.0
            if front_ray > 0.10:
                a[0] = 1.0
    return a


def bc_warmup_stage(specs: list[dict], save_path: Path, n_steps: int = 8192, seed: int = 700) -> Path:
    """Collect human-teacher trajectories and pretrain ESRHAAgentModel with sequence BCE."""
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
    cur_obs, cur_act, cur_pa = [], [], []

    for t in range(n_steps):
        act = _human_teacher_action(obs, t)
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
            cur_obs, cur_act, cur_pa = [], [], []
    env.close()

    # Batch into (N, T, ...) tensors and train 16 epochs with class-balanced pos_weight
    n_seq = len(obs_seqs)
    print(f"[BC_Warmup] Collected {n_seq} sequences ({n_seq * seq_len} steps). Training...", flush=True)
    pos_w = torch.tensor([1.0, 1.0, 1.5, 1.5, 3.0, 3.0, 14.0, 2.0, 1.0], device=device)
    model.train()
    indices = list(range(n_seq))
    for epoch in range(1, 17):
        rng.shuffle(indices)
        total_loss = 0.0
        batches = 0
        for i in range(0, n_seq, 16):
            b_idx = indices[i : i + 16]
            bsz = len(b_idx)
            # Build (B, T, ...)
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
        if epoch in (1, 6, 12, 16):
            print(f"[BC_Warmup epoch {epoch:2d}/16] bce_loss={total_loss / max(1, batches):.4f}", flush=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"hidden_dim": 256, "use_visual": False, "use_depth": False},
        },
        save_path,
    )
    return save_path


def main() -> None:
    combat = build_combat_scenarios()
    courtyard = build_human_courtyard_wads()
    freedoom2_wad = str(Path(os.path.dirname(vd.__file__)) / "freedoom2.wad")

    # Mix human-video style courtyard/campaign maps with corridor, defend, basic, and two_rooms
    train_specs = (
        courtyard * 3
        + [{"name": "freedoom2_map01", "wad": freedoom2_wad, "ticks": 1400, "kill_target": None}] * 2
        + combat["C_corridor"] * 2
        + combat["B_defend"]
        + combat["A_basic"]
        + [{"name": "two_rooms", "wad": "maps/two_rooms.wad", "ticks": 1400, "kill_target": None}] * 2
    )

    baseline_ckpt = Path("submission/weights/esrha_best.pt")
    bc_ckpt = Path("training/runs/grandmaster_v6_bc/bc_init.pt")
    run_dir = Path("training/runs/grandmaster_v6")

    # Stage 1: Class-balanced BC warmup (ensures ATTACK > 0.85 when centered!)
    bc_specs = combat["A_basic"] * 4 + combat["C_corridor"] * 2 + courtyard * 2 + [{"name": "two_rooms", "wad": "maps/two_rooms.wad", "ticks": 1400, "kill_target": None}]
    bc_warmup_stage(bc_specs, bc_ckpt, n_steps=8192, seed=800)

    print("\n=== Probing BC Warmup Policy ===", flush=True)
    bc_probe = probe_policy_behavior(bc_ckpt)
    print(json.dumps(bc_probe, indent=2), flush=True)

    # Stage 2: PPO RL fine-tuning so the agent surpasses the teacher via RL
    best_ckpt = train_stage(
        name="Grandmaster_v6_PPO",
        specs=train_specs,
        total_steps=32768,
        save_dir=run_dir,
        init_ckpt=bc_ckpt,
        kill_terminates=False,
        lr=2e-4,
        ent_coef=0.002,
        seed=801,
    )

    print("\n=== Probing Trained Grandmaster_v6 Policy ===", flush=True)
    new_probe = probe_policy_behavior(best_ckpt)
    print(json.dumps(new_probe, indent=2), flush=True)

    eval_specs = (
        courtyard
        + [{"name": "freedoom2_map01", "wad": freedoom2_wad, "ticks": 1400, "kill_target": None}]
        + combat["C_corridor"][:2]
        + combat["A_basic"][:2]
        + [{"name": "two_rooms", "wad": "maps/two_rooms.wad", "ticks": 1400, "kill_target": None}]
    )

    bc_eval = evaluate_experiment("BC_Warmup", bc_ckpt, eval_specs)
    new_eval = evaluate_experiment(
        "Grandmaster_v6_PPO",
        best_ckpt,
        eval_specs,
        filmstrip_path=run_dir / "filmstrip.png",
    )

    summary = {
        "bc_probe": bc_probe,
        "human_style_probe": new_probe,
        "bc_eval": bc_eval,
        "human_style_eval": new_eval,
    }
    (run_dir / "human_style_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Evaluation Comparison ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    # Export to submission/weights/esrha_best.pt and Studio default latest.pt
    shutil.copy2(best_ckpt, baseline_ckpt)
    studio_latest = Path("training/runs/studio_experiments/combat_studio_exp/latest.pt")
    if studio_latest.parent.exists():
        shutil.copy2(best_ckpt, studio_latest)
        print(f"Exported {best_ckpt} -> {studio_latest}", flush=True)
    print(f"\nExported {best_ckpt} -> {baseline_ckpt}", flush=True)


if __name__ == "__main__":
    main()
