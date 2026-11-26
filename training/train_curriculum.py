"""Curriculum PPO training script across Stages 1..6 with unseen map evaluation."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from doom_env import make_env
from doom_env import spaces as S
from training.algorithms.ppo import PPOConfig, PPOTrainer
from training.curiosity import IntrinsicCuriosityModule
from training.curriculum import CurriculumScheduler
from training.maps.registry import MapRegistry
from training.models.doom_model import ESRHAAgentModel
from training.rewards.reward import DoomRewardShaper
from training.rollout import RolloutBuffer, _batch_obs
from training.train import resolve_device, summarize_episodes


def main() -> None:
    ap = argparse.ArgumentParser(description="Curriculum PPO training for ESRHA")
    ap.add_argument("--total-steps", type=int, default=24576)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--rollout-steps", type=int, default=256)
    ap.add_argument("--start-stage", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--use-icm", action="store_true", default=True)
    ap.add_argument("--use-visual", action="store_true", default=False)
    ap.add_argument("--use-depth", action="store_true", default=False)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-dir", default="training/runs/curriculum_v1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = resolve_device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    registry = MapRegistry()
    scheduler = CurriculumScheduler(start_stage=args.start_stage, max_stage=6)

    steps_per_update = args.num_envs * args.rollout_steps
    total_updates = max(1, args.total_steps // steps_per_update)

    model = ESRHAAgentModel(use_visual=args.use_visual, use_depth=args.use_depth).to(device)
    icm = IntrinsicCuriosityModule(obs_dim=model.hidden_dim).to(device) if args.use_icm else None
    trainer = PPOTrainer(model, PPOConfig(lr=args.lr), total_updates=total_updates, device=device, icm=icm)

    def _spawn_env(stage: int):
        spec = registry.sample("train", stage, rng)
        e = make_env(
            spec.wad_path, spec.map_name, episode_ticks=spec.episode_ticks,
            render_buffers=(args.use_visual or args.use_depth),
        )
        o, _ = e.reset(seed=spec.seed)
        sh = DoomRewardShaper()
        sh.reset(e.privileged_state())
        return e, sh, o

    envs, shapers, obs_list = [], [], []
    for _ in range(args.num_envs):
        e, sh, o = _spawn_env(scheduler.stage)
        envs.append(e)
        shapers.append(sh)
        obs_list.append(o)

    hidden = model.memory.init_hidden(args.num_envs, device)
    prev_action = np.zeros((args.num_envs, len(S.BUTTONS)), dtype=np.float32)
    total_steps = 0
    best_score = -1e9

    try:
        for upd in range(1, total_updates + 1):
            t0 = time.perf_counter()
            model.eval()
            buf = RolloutBuffer(args.rollout_steps, args.num_envs, model.memory.state_dim)
            completed = []

            for _ in range(args.rollout_steps):
                obs_t = _batch_obs(obs_list, device=device)
                pa_t = torch.as_tensor(prev_action, dtype=torch.float32, device=device)
                with torch.no_grad():
                    out = model.forward_step(obs_t, hidden=hidden, prev_action=pa_t)
                    action_t = out["tactical_dist"].sample()
                    log_prob_t = out["tactical_dist"].log_prob(action_t)
                    value_t = out["value"]

                action_np = action_t.cpu().numpy().astype(np.float32)
                action_int = action_np.astype(np.int8)
                hid_in_np = hidden.squeeze(0).cpu().numpy()

                rewards = np.zeros(args.num_envs, dtype=np.float32)
                dones = np.zeros(args.num_envs, dtype=np.float32)
                next_obs_list = []
                next_hidden = out["next_hidden"]
                next_prev_action = action_np.copy()

                for i in range(args.num_envs):
                    next_obs, _, term, trunc, _ = envs[i].step(action_int[i])
                    done = bool(term or trunc)
                    r, ep_stats = shapers[i].step(envs[i].privileged_state(), action=action_int[i], obs=next_obs)
                    rewards[i] = r
                    dones[i] = float(done)
                    if done:
                        if ep_stats is not None:
                            completed.append(ep_stats)
                        envs[i].close()
                        envs[i], shapers[i], next_obs = _spawn_env(scheduler.stage)
                        next_hidden[:, i, :] = 0.0
                        next_prev_action[i] = 0.0
                    next_obs_list.append(next_obs)

                buf.add(obs_list, prev_action, action_np, log_prob_t.cpu().numpy(), value_t.cpu().numpy(), rewards, dones, hid_in_np)
                obs_list = next_obs_list
                hidden = next_hidden
                prev_action = next_prev_action

            with torch.no_grad():
                last_out = model.forward_step(_batch_obs(obs_list, device=device), hidden=hidden, prev_action=torch.as_tensor(prev_action, dtype=torch.float32, device=device))
            train_stats = trainer.update(buf, last_out["value"].cpu().numpy(), np.zeros(args.num_envs, dtype=np.float32))
            total_steps += steps_per_update
            advanced = scheduler.record(completed, upd, total_steps)
            ep_sum = summarize_episodes(completed)
            dt = time.perf_counter() - t0

            msg = (
                f"[{upd:3d}/{total_updates}] stage={scheduler.stage} steps={total_steps:6d} sps={steps_per_update/dt:5.1f} | "
                + (
                    f"rew={ep_sum['ep_reward']:+5.2f} exit={ep_sum['exit_rate']:.2f} kills={ep_sum['kills']:.2f} "
                    f"doors={ep_sum['doors_opened']:.2f} cells={ep_sum['cells_visited']:.1f}"
                    if ep_sum
                    else "no eps"
                )
                + f" | kl={train_stats['approx_kl']:.4f}"
            )
            if advanced:
                msg += f"  >>> ADVANCED TO STAGE {scheduler.stage}!"
            print(msg, flush=True)

            trainer.save_checkpoint(save_dir / "latest.pt", upd, total_steps, extra={"stage": scheduler.stage, **ep_sum})
            if ep_sum and ep_sum["ep_reward"] > best_score:
                best_score = ep_sum["ep_reward"]
                trainer.save_checkpoint(save_dir / "best.pt", upd, total_steps, extra={"stage": scheduler.stage, **ep_sum})
    finally:
        for e in envs:
            e.close()
        (save_dir / "curriculum_history.json").write_text(json.dumps(scheduler.history, indent=2))


if __name__ == "__main__":
    main()
