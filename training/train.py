"""Main PPO training entry point for ESRHA on doom_env."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from doom_env import make_env
from doom_env import spaces as S
from training.algorithms.ppo import PPOConfig, PPOTrainer
from training.curiosity import IntrinsicCuriosityModule
from training.models.doom_model import ESRHAAgentModel
from training.rewards.reward import DoomRewardShaper, EpisodeStats
from training.rollout import collect_rollouts


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train ESRHA with custom Recurrent PPO")
    ap.add_argument("--wad", default="maps/two_rooms.wad")
    ap.add_argument("--map", default="MAP01")
    ap.add_argument("--episode-ticks", type=int, default=2100)
    ap.add_argument("--total-steps", type=int, default=16384)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--rollout-steps", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--batch-seqs", type=int, default=8)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.015)
    ap.add_argument("--use-icm", action="store_true", default=False)
    ap.add_argument("--icm-coef", type=float, default=0.05)
    ap.add_argument("--use-visual", action="store_true", default=False)
    ap.add_argument("--use-depth", action="store_true", default=True)
    ap.add_argument("--no-depth", dest="use_depth", action="store_false")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-dir", default="training/runs/ppo_baseline")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--render", action="store_true", help="Show live pygame window for env 0 during training")
    ap.add_argument("--web", action="store_true", help="Serve live training stream & metrics at http://127.0.0.1:<web-port>")
    ap.add_argument("--web-port", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def summarize_episodes(eps: list[EpisodeStats]) -> dict[str, float]:
    if not eps:
        return {}
    return {
        "ep_reward": float(np.mean([e.reward for e in eps])),
        "ep_len": float(np.mean([e.length for e in eps])),
        "survival_rate": float(np.mean([not e.dead for e in eps])),
        "exit_rate": float(np.mean([e.exit_reached for e in eps])),
        "kills": float(np.mean([e.kills for e in eps])),
        "damage_taken": float(np.mean([e.damage_taken for e in eps])),
        "cells_visited": float(np.mean([e.cells_visited for e in eps])),
        "doors_opened": float(np.mean([e.doors_opened for e in eps])),
        "keys": float(np.mean([e.keys for e in eps])),
        "stuck_steps": float(np.mean([e.stuck_steps for e in eps])),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"

    steps_per_update = args.num_envs * args.rollout_steps
    total_updates = max(1, args.total_steps // steps_per_update)

    render_bufs = args.use_visual or args.use_depth or args.render or args.web
    envs = [
        make_env(args.wad, args.map, episode_ticks=args.episode_ticks, render_buffers=render_bufs)
        for _ in range(args.num_envs)
    ]
    shapers = [DoomRewardShaper() for _ in range(args.num_envs)]
    viewer = None
    if args.web:
        from training.web_viewer import WebLiveViewer

        viewer = WebLiveViewer(port=args.web_port)
        print(f"Live training dashboard: {viewer.url}", flush=True)
    elif args.render:
        from training.viewer import DoomLiveViewer

        viewer = DoomLiveViewer(title=f"ESRHA Training — {args.wad}:{args.map}")

    obs_list = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=args.seed + i)
        shapers[i].reset(env.privileged_state())
        obs_list.append(o)

    model = ESRHAAgentModel(use_visual=args.use_visual, use_depth=args.use_depth).to(device)
    icm = IntrinsicCuriosityModule(obs_dim=model.hidden_dim).to(device) if args.use_icm else None
    cfg = PPOConfig(
        lr=args.lr,
        ent_coef=args.ent_coef,
        icm_coef=args.icm_coef,
        ppo_epochs=args.ppo_epochs,
        seq_len=args.seq_len,
        batch_seqs=args.batch_seqs,
    )
    trainer = PPOTrainer(model, cfg, total_updates=total_updates, device=device, icm=icm)

    start_update = 1
    total_steps = 0
    if args.resume:
        ckpt = PPOTrainer.load_checkpoint(args.resume, model, trainer, device=device)
        start_update = ckpt["update_idx"] + 1
        total_steps = ckpt["total_steps"]
        print(f"Resumed from {args.resume} at update {start_update - 1} ({total_steps} steps)")

    hidden = model.memory.init_hidden(args.num_envs, device)
    prev_action = np.zeros((args.num_envs, len(S.BUTTONS)), dtype=np.float32)

    best_ep_reward = -1e9
    recent_eps: list[EpisodeStats] = []
    print(f"Training ESRHA on {args.wad}:{args.map} | device={device} | updates={total_updates} | steps/upd={steps_per_update}")

    try:
        for upd in range(start_update, start_update + total_updates):
            t0 = time.perf_counter()
            buf, obs_list, hidden, prev_action, last_vals, last_dones, completed = collect_rollouts(
                envs, shapers, model, obs_list, hidden, prev_action, args.rollout_steps, device,
                seed=args.seed + upd * 100, viewer=viewer,
            )
            train_stats = trainer.update(buf, last_vals, last_dones)
            dt = time.perf_counter() - t0
            total_steps += steps_per_update
            recent_eps.extend(completed)
            recent_eps = recent_eps[-40:]

            ep_summary = summarize_episodes(recent_eps)
            sps = steps_per_update / max(1e-6, dt)
            record = {"update": upd, "total_steps": total_steps, "sps": round(sps, 1), **train_stats, **ep_summary}
            if hasattr(viewer, "update_training_metrics"):
                viewer.update_training_metrics(record)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            ep_str = (
                f"rew={ep_summary['ep_reward']:+6.2f} len={ep_summary['ep_len']:5.1f} "
                f"surv={ep_summary['survival_rate']:.2f} exit={ep_summary['exit_rate']:.2f} "
                f"kills={ep_summary['kills']:.2f} dmg={ep_summary['damage_taken']:4.1f} cells={ep_summary['cells_visited']:4.1f}"
                if ep_summary
                else "no completed episodes yet"
            )
            print(
                f"[{upd:3d}/{start_update + total_updates - 1}] steps={total_steps:6d} sps={sps:6.1f} | "
                f"{ep_str} | pg={train_stats['policy_loss']:+.4f} vf={train_stats['value_loss']:.3f} "
                f"ent={train_stats['entropy']:.3f} kl={train_stats['approx_kl']:.4f}"
            )

            trainer.save_checkpoint(save_dir / "latest.pt", upd, total_steps, extra=ep_summary)
            if ep_summary and ep_summary["ep_reward"] > best_ep_reward:
                best_ep_reward = ep_summary["ep_reward"]
                trainer.save_checkpoint(save_dir / "best.pt", upd, total_steps, extra=ep_summary)
    finally:
        if viewer is not None:
            viewer.close()
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
