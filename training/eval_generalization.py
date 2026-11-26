"""Generalization evaluator across train, validation, and unseen test maps (Stages 1..6)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from doom_env import make_env
from training.algorithms.ppo import PPOTrainer
from training.maps.registry import MapRegistry
from training.models.doom_model import ESRHAAgentModel
from training.rewards.reward import DoomRewardShaper


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path | None,
    split: str = "test",
    episodes_per_map: int = 2,
    device: str = "cpu",
    deterministic: bool = False,
) -> dict[int, dict[str, float]]:
    registry = MapRegistry()
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        mcfg = ckpt.get("model_config", {})
        model = ESRHAAgentModel(
            hidden_dim=mcfg.get("hidden_dim", 384),
            use_visual=mcfg.get("use_visual", False),
            use_depth=mcfg.get("use_depth", True),
        ).to(device)
        PPOTrainer.load_checkpoint(checkpoint_path, model, device=device)
    else:
        model = ESRHAAgentModel(use_visual=False, use_depth=True).to(device)
    model.eval()

    results: dict[int, dict[str, float]] = {}
    for stage in range(1, 7):
        specs = registry.get(split, stage)
        ep_stats = []
        for spec in specs:
            env = make_env(spec.wad_path, spec.map_name, episode_ticks=spec.episode_ticks, render_buffers=True)
            shaper = DoomRewardShaper()
            try:
                for ep in range(episodes_per_map):
                    obs, _ = env.reset(seed=spec.seed + ep)
                    shaper.reset(env.privileged_state())
                    model.reset_state(device=device)
                    while True:
                        action = model.act(obs, deterministic=deterministic)
                        obs, _, term, trunc, _ = env.step(action)
                        _, stats = shaper.step(env.privileged_state(), action=action)
                        if term or trunc:
                            if stats is not None:
                                ep_stats.append(stats)
                            break
            finally:
                env.close()

        results[stage] = {
            "exit_rate": float(np.mean([e.exit_reached for e in ep_stats])),
            "cells_visited": float(np.mean([e.cells_visited for e in ep_stats])),
            "keys": float(np.mean([e.keys for e in ep_stats])),
            "doors_opened": float(np.mean([e.doors_opened for e in ep_stats])),
            "survival_rate": float(np.mean([not e.dead for e in ep_stats])),
            "kills": float(np.mean([e.kills for e in ep_stats])),
            "ep_reward": float(np.mean([e.reward for e in ep_stats])),
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate generalization across unseen maps")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--episodes-per-map", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = evaluate_checkpoint(args.checkpoint, split=args.split, episodes_per_map=args.episodes_per_map, device=args.device)
    print(f"\n=== Generalization Evaluation ({args.split.upper()} split) ===")
    print(f"{'Stage':>5} | {'Exit':>6} | {'Cells':>6} | {'Keys':>5} | {'Doors':>5} | {'Surv':>6} | {'Kills':>5} | {'Reward':>7}")
    print("-" * 65)
    for st, m in res.items():
        print(
            f"{st:5d} | {m['exit_rate']:6.2f} | {m['cells_visited']:6.1f} | {m['keys']:5.2f} | "
            f"{m['doors_opened']:5.2f} | {m['survival_rate']:6.2f} | {m['kills']:5.2f} | {m['ep_reward']:+7.2f}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
