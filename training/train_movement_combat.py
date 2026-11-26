"""5-Stage Movement-First -> Combined Navigation & Combat Curriculum for ESRHA."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from doom_env import make_env
from doom_env import spaces as S
from training.algorithms.ppo import PPOConfig, PPOTrainer
from training.combat_scenarios import build_combat_scenarios
from training.diagnostics.visualize_depth import write_png
from training.maps.progression_wads import compile_udmf_wad, generate_all
from training.models.doom_model import ESRHAAgentModel
from training.rewards.reward import DoomRewardShaper, EpisodeStats
from training.rollout import RolloutBuffer, _batch_obs
from training.train import resolve_device


def build_movement_curriculum(out_dir: str | Path = "training/maps/movement") -> dict[str, list[dict]]:
    out = Path(out_dir)
    combat = build_combat_scenarios()
    prog = generate_all()

    # Stage 1: Straight forward/backward corridors (learn MOVE_FORWARD, stop blind ATTACK)
    s1 = []
    for idx, length in enumerate((9, 11, 13)):
        row = "#" * length
        mid = "#P" + "." * (length - 4) + "E#"
        wad = compile_udmf_wad(f"{row}\n{mid}\n{row}", out / f"s1_corridor_{idx}.wad")
        s1.append({"name": f"s1_corridor_{idx}", "wad": str(wad), "ticks": 420})

    # Stage 2: Turning & strafing (L-turns and S-bends)
    s2_grids = [
        "#######\n#P....#\n#####.#\n#####.#\n####E.#\n#######",
        "#######\n#....E#\n#.#####\n#.#####\n#P....#\n#######",
        "#########\n#P..#...#\n###.#.#.#\n#...#..E#\n#########",
    ]
    s2 = [
        {"name": f"s2_turn_{i}", "wad": str(compile_udmf_wad(g, out / f"s2_turn_{i}.wad")), "ticks": 700}
        for i, g in enumerate(s2_grids)
    ]

    # Stage 3: Corridor navigation & obstacle/wall avoidance
    s3_grids = [
        "###########\n#P..#.....#\n#.#.#.###.#\n#...#....E#\n###########",
        "###########\n#P....#..E#\n###.#.#.#.#\n#...#.....#\n###########",
    ]
    s3 = [
        {"name": f"s3_nav_{i}", "wad": str(compile_udmf_wad(g, out / f"s3_nav_{i}.wad")), "ticks": 900}
        for i, g in enumerate(s3_grids)
    ]

    # Stage 4: Multi-room exploration with doors
    s4 = [
        {"name": "two_rooms", "wad": "maps/two_rooms.wad", "ticks": 1400},
        {"name": "prog_01_door_exit", "wad": str(prog["prog_01_door_exit"]), "ticks": 1050},
    ]

    # Stage 5: Combined navigation & enemy combat
    s5 = [
        {"name": m["name"], "wad": m["wad"], "ticks": m["ticks"]}
        for m in (combat["C_corridor"] + combat["A_basic"][:4] + combat["B_defend"][1:3])
    ]

    return {"S1_fwd": s1, "S2_turn": s2, "S3_corridor": s3, "S4_rooms": s4, "S5_combat": s5}


@torch.no_grad()
def evaluate_navigation_and_combat(
    ckpt_path: Path | None, scenarios: dict[str, list[dict]], device: str = "cpu", filmstrip_out: Path | None = None
) -> dict[str, float]:
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        mcfg = ckpt.get("model_config", {})
        model = ESRHAAgentModel(
            hidden_dim=int(mcfg.get("hidden_dim", 256)),
            use_visual=bool(mcfg.get("use_visual", False)),
            use_depth=bool(mcfg.get("use_depth", False)),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = ESRHAAgentModel(hidden_dim=256, use_visual=False, use_depth=False).to(device)
    model.eval()

    eval_specs = [scenarios["S1_fwd"][1], scenarios["S2_turn"][0], scenarios["S4_rooms"][0], scenarios["S5_combat"][1]]
    stats_all: list[EpisodeStats] = []
    frames: list[np.ndarray] = []

    for idx, spec in enumerate(eval_specs):
        env = make_env(spec["wad"], "MAP01", episode_ticks=spec["ticks"], render_buffers=(filmstrip_out is not None))
        shaper = DoomRewardShaper()
        try:
            obs, _ = env.reset(seed=777 + idx)
            shaper.reset(env.privileged_state(), obs=obs)
            model.reset_state(device=device)
            for step in range(spec["ticks"] // S.FRAME_SKIP):
                if filmstrip_out is not None and idx == 3 and step % 15 == 0 and len(frames) < 6:
                    frames.append(obs["rgb"].copy())
                action = model.act(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                _, st = shaper.step(env.privileged_state(), action=action, obs=obs)
                if term or trunc:
                    if st is not None:
                        stats_all.append(st)
                    break
        finally:
            env.close()

    if filmstrip_out is not None and frames:
        filmstrip_out.parent.mkdir(parents=True, exist_ok=True)
        write_png(filmstrip_out, np.concatenate(frames, axis=1))

    total_len = max(1, sum(s.length for s in stats_all))
    return {
        "ep_reward": float(np.mean([s.reward for s in stats_all])),
        "distance": float(np.mean([s.distance_travelled for s in stats_all])),
        "cells": float(np.mean([s.cells_visited for s in stats_all])),
        "fwd_pct": float(100.0 * sum(s.forward_steps for s in stats_all) / total_len),
        "turn_pct": float(100.0 * sum(s.turn_steps for s in stats_all) / total_len),
        "ammo_used": float(np.mean([s.ammo_used for s in stats_all])),
        "kills": float(np.mean([s.kills for s in stats_all])),
        "exit_rate": float(np.mean([s.exit_reached for s in stats_all])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="5-Stage Movement + Combat Curriculum Training")
    ap.add_argument("--total-steps", type=int, default=65536)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--rollout-steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-dir", default="training/runs/nav_combat_v1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = resolve_device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / "tb"))

    scenarios = build_movement_curriculum()
    stages = ["S1_fwd", "S2_turn", "S3_corridor", "S4_rooms", "S5_combat"]
    stage_idx = 0
    stage_eps: list[EpisodeStats] = []

    steps_per_update = args.num_envs * args.rollout_steps
    total_updates = max(1, args.total_steps // steps_per_update)

    model = ESRHAAgentModel(hidden_dim=256, use_visual=False, use_depth=False).to(device)
    trainer = PPOTrainer(model, PPOConfig(lr=args.lr, ent_coef=0.01, seq_len=32, batch_seqs=8), total_updates, device)

    def _spawn(st_idx: int):
        pick = st_idx if (st_idx == 0 or rng.random() < 0.75) else rng.randint(0, st_idx - 1)
        spec = rng.choice(scenarios[stages[pick]])
        e = make_env(spec["wad"], "MAP01", episode_ticks=spec["ticks"], render_buffers=False)
        o, _ = e.reset(seed=rng.randint(0, 100000))
        sh = DoomRewardShaper()
        sh.reset(e.privileged_state(), obs=o)
        return e, sh, o

    envs, shapers, obs_list = [], [], []
    for _ in range(args.num_envs):
        e, sh, o = _spawn(stage_idx)
        envs.append(e); shapers.append(sh); obs_list.append(o)

    hidden = model.memory.init_hidden(args.num_envs, device)
    prev_action = np.zeros((args.num_envs, len(S.BUTTONS)), dtype=np.float32)
    total_steps = 0
    best_score = -1e9
    early_u, mid_u = max(1, total_updates // 4), max(2, (3 * total_updates) // 5)

    try:
        for upd in range(1, total_updates + 1):
            t0 = time.perf_counter()
            model.eval()
            buf = RolloutBuffer(args.rollout_steps, args.num_envs, model.memory.state_dim)
            completed: list[EpisodeStats] = []
            probs_acc = np.zeros(len(S.BUTTONS), dtype=np.float64)

            for _ in range(args.rollout_steps):
                obs_t = _batch_obs(obs_list, device=device)
                pa_t = torch.as_tensor(prev_action, dtype=torch.float32, device=device)
                with torch.no_grad():
                    out = model.forward_step(obs_t, hidden=hidden, prev_action=pa_t)
                    action_t = out["tactical_dist"].sample()
                    log_prob_t = out["tactical_dist"].log_prob(action_t)
                    value_t = out["value"]
                    probs_acc += out["tactical_dist"].base_dist.probs.mean(dim=0).cpu().numpy()

                action_np = action_t.cpu().numpy().astype(np.float32)
                action_int = action_np.astype(np.int8)
                hid_in = hidden.squeeze(0).cpu().numpy()

                rewards = np.zeros(args.num_envs, dtype=np.float32)
                dones = np.zeros(args.num_envs, dtype=np.float32)
                next_obs_list = []
                next_hidden = out["next_hidden"]
                next_pa = action_np.copy()

                for i in range(args.num_envs):
                    next_obs, _, term, trunc, _ = envs[i].step(action_int[i])
                    r, st = shapers[i].step(envs[i].privileged_state(), action=action_int[i], obs=next_obs)
                    done = bool(term or trunc)
                    rewards[i] = r; dones[i] = float(done)
                    if done:
                        if st is not None:
                            completed.append(st); stage_eps.append(st)
                        envs[i].close()
                        envs[i], shapers[i], next_obs = _spawn(stage_idx)
                        next_hidden[:, i, :] = 0.0; next_pa[i] = 0.0
                    next_obs_list.append(next_obs)

                buf.add(obs_list, prev_action, action_np, log_prob_t.cpu().numpy(), value_t.cpu().numpy(), rewards, dones, hid_in)
                obs_list = next_obs_list; hidden = next_hidden; prev_action = next_pa

            with torch.no_grad():
                lo = model.forward_step(_batch_obs(obs_list, device=device), hidden=hidden, prev_action=torch.as_tensor(prev_action, dtype=torch.float32, device=device))
            t_stats = trainer.update(buf, lo["value"].cpu().numpy(), np.zeros(args.num_envs, dtype=np.float32))
            total_steps += steps_per_update
            dt = time.perf_counter() - t0
            mean_p = probs_acc / args.rollout_steps

            # Performance-gated stage advancement
            if stage_idx < 4 and len(stage_eps) >= 16:
                win = stage_eps[-16:]
                er = np.mean([e.exit_reached for e in win])
                if er >= 0.45:
                    stage_idx += 1
                    stage_eps.clear()
                    print(f"  >>> ADVANCED TO {stages[stage_idx]} (exit_rate={er:.2f})", flush=True)

            for bi, bn in enumerate(S.BUTTONS):
                writer.add_scalar(f"action_probs/{bn}", float(mean_p[bi]), total_steps)
            for k, v in t_stats.items():
                writer.add_scalar(f"train/{k}", v, total_steps)

            if completed:
                rew = float(np.mean([e.reward for e in completed]))
                dist = float(np.mean([e.distance_travelled for e in completed]))
                cells = float(np.mean([e.cells_visited for e in completed]))
                kills = float(np.mean([e.kills for e in completed]))
                er = float(np.mean([e.exit_reached for e in completed]))
                print(
                    f"[{upd:3d}/{total_updates}] {stages[stage_idx]:11s} steps={total_steps:6d} sps={steps_per_update/dt:5.1f} | "
                    f"rew={rew:+5.2f} exit={er:.2f} dist={dist:6.1f} cells={cells:4.1f} kills={kills:.2f} | "
                    f"fwd_p={mean_p[0]:.2f} back_p={mean_p[1]:.2f} turn_p={mean_p[4]+mean_p[5]:.2f} atk_p={mean_p[6]:.2f}",
                    flush=True,
                )
                score = rew + er * 4.0 + kills * 2.0 + (stage_idx * 5.0)
                if score > best_score:
                    best_score = score
                    trainer.save_checkpoint(save_dir / "best.pt", upd, total_steps)
            trainer.save_checkpoint(save_dir / "latest.pt", upd, total_steps)
            if upd == early_u:
                trainer.save_checkpoint(save_dir / "early.pt", upd, total_steps)
            elif upd == mid_u:
                trainer.save_checkpoint(save_dir / "mid.pt", upd, total_steps)
    finally:
        writer.close()
        for e in envs:
            e.close()

    print("\n=== Before vs After Navigation & Combat Comparison (Deterministic) ===")
    comp = {}
    for label, path in (
        ("before_combat_v1", Path("training/runs/combat_v1/best.pt")),
        ("early_nav", save_dir / "early.pt"),
        ("mid_nav", save_dir / "mid.pt"),
        ("after_best", save_dir / "best.pt"),
    ):
        if not path.exists():
            continue
        res = evaluate_navigation_and_combat(path, scenarios, device=device, filmstrip_out=save_dir / "recordings" / f"{label}.png")
        comp[label] = res
        print(
            f"{label:17s} | rew={res['ep_reward']:+6.2f} | exit={res['exit_rate']:.2f} | dist={res['distance']:6.1f} | "
            f"cells={res['cells']:4.1f} | fwd={res['fwd_pct']:5.1f}% | turn={res['turn_pct']:5.1f}% | ammo={res['ammo_used']:4.1f} | kills={res['kills']:.2f}"
        )
    (save_dir / "comparison.json").write_text(json.dumps(comp, indent=2))


if __name__ == "__main__":
    main()
