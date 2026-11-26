"""Focused Combat PPO training (Basic -> Defend Center -> Deadly Corridor) with TensorBoard & eval."""

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
from training.models.doom_model import ESRHAAgentModel
from training.rewards.reward import DoomRewardShaper, EpisodeStats
from training.rollout import RolloutBuffer, _batch_obs
from training.train import resolve_device


@torch.no_grad()
def evaluate_combat_checkpoint(
    ckpt_path: Path | None,
    scenarios: dict[str, list[dict]],
    device: str = "cpu",
    deterministic: bool = True,
    filmstrip_out: Path | None = None,
) -> dict[str, float]:
    """Evaluate random/trained checkpoint deterministically on Basic + Defend + Corridor."""
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

    eval_specs = [scenarios["A_basic"][0], scenarios["A_basic"][2], scenarios["B_defend"][1], scenarios["C_corridor"][1]]
    stats_all: list[EpisodeStats] = []
    frames: list[np.ndarray] = []

    for idx, spec in enumerate(eval_specs):
        env = make_env(spec["wad"], "MAP01", episode_ticks=spec["ticks"], render_buffers=(filmstrip_out is not None))
        shaper = DoomRewardShaper()
        try:
            obs, _ = env.reset(seed=999 + idx)
            shaper.reset(env.privileged_state(), obs=obs)
            model.reset_state(device=device)
            for step in range(spec["ticks"] // S.FRAME_SKIP):
                if filmstrip_out is not None and idx == 0 and step % 8 == 0 and len(frames) < 6:
                    frames.append(obs["rgb"].copy())
                action = model.act(obs, deterministic=deterministic)
                obs, _, term, trunc, _ = env.step(action)
                priv = env.privileged_state()
                _, st = shaper.step(priv, action=action, obs=obs)
                if spec["kill_target"] and priv["kills"] >= spec["kill_target"]:
                    st = shaper._stats
                    term = True
                if term or trunc:
                    if st is not None:
                        stats_all.append(st)
                    break
        finally:
            env.close()

    if filmstrip_out is not None and frames:
        strip = np.concatenate(frames, axis=1)
        filmstrip_out.parent.mkdir(parents=True, exist_ok=True)
        write_png(filmstrip_out, strip)

    return {
        "ep_reward": float(np.mean([s.reward for s in stats_all])),
        "ep_len": float(np.mean([s.length for s in stats_all])),
        "kills": float(np.mean([s.kills for s in stats_all])),
        "aimed_shots": float(np.mean([s.aimed_shots for s in stats_all])),
        "wasted_shots": float(np.mean([s.wasted_shots for s in stats_all])),
        "damage_taken": float(np.mean([s.damage_taken for s in stats_all])),
        "survival_rate": float(np.mean([not s.dead for s in stats_all])),
        "combat_success": float(np.mean([s.kills > 0 for s in stats_all])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Combat PPO training with TensorBoard & checkpoint comparison")
    ap.add_argument("--total-steps", type=int, default=65536)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--rollout-steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-dir", default="training/runs/combat_v1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = resolve_device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / "tb"))

    scenarios = build_combat_scenarios()
    stages = ["A_basic", "B_defend", "C_corridor"]
    stage_idx = 0
    stage_eps: list[EpisodeStats] = []

    steps_per_update = args.num_envs * args.rollout_steps
    total_updates = max(1, args.total_steps // steps_per_update)

    model = ESRHAAgentModel(hidden_dim=256, use_visual=False, use_depth=False).to(device)
    cfg = PPOConfig(lr=args.lr, ent_coef=args.ent_coef, seq_len=32, batch_seqs=8, ppo_epochs=4)
    trainer = PPOTrainer(model, cfg, total_updates=total_updates, device=device)

    def _spawn(st_idx: int):
        # 80% current stage, 20% previous stages for retention
        pick = st_idx if (st_idx == 0 or rng.random() < 0.8) else rng.randint(0, st_idx - 1)
        spec = rng.choice(scenarios[stages[pick]])
        e = make_env(spec["wad"], "MAP01", episode_ticks=spec["ticks"], render_buffers=False)
        o, _ = e.reset(seed=rng.randint(0, 100000))
        sh = DoomRewardShaper()
        sh.reset(e.privileged_state(), obs=o)
        return e, sh, o, spec

    envs, shapers, obs_list, specs = [], [], [], []
    for _ in range(args.num_envs):
        e, sh, o, sp = _spawn(stage_idx)
        envs.append(e)
        shapers.append(sh)
        obs_list.append(o)
        specs.append(sp)

    hidden = model.memory.init_hidden(args.num_envs, device)
    prev_action = np.zeros((args.num_envs, len(S.BUTTONS)), dtype=np.float32)
    total_steps = 0
    best_score = -1e9
    early_upd = max(1, total_updates // 4)
    mid_upd = max(2, (3 * total_updates) // 5)

    try:
        for upd in range(1, total_updates + 1):
            t0 = time.perf_counter()
            model.eval()
            buf = RolloutBuffer(args.rollout_steps, args.num_envs, model.memory.state_dim)
            completed: list[EpisodeStats] = []
            action_probs_accum = np.zeros(len(S.BUTTONS), dtype=np.float64)

            for _ in range(args.rollout_steps):
                obs_t = _batch_obs(obs_list, device=device)
                pa_t = torch.as_tensor(prev_action, dtype=torch.float32, device=device)
                with torch.no_grad():
                    out = model.forward_step(obs_t, hidden=hidden, prev_action=pa_t)
                    action_t = out["tactical_dist"].sample()
                    log_prob_t = out["tactical_dist"].log_prob(action_t)
                    value_t = out["value"]
                    action_probs_accum += out["tactical_dist"].base_dist.probs.mean(dim=0).cpu().numpy()

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
                    priv = envs[i].privileged_state()
                    r, ep_stats = shapers[i].step(priv, action=action_int[i], obs=next_obs)
                    if specs[i]["kill_target"] and priv["kills"] >= specs[i]["kill_target"]:
                        r += 2.0
                        shapers[i]._stats.reward += 2.0
                        ep_stats = shapers[i]._stats
                        term = True

                    done = bool(term or trunc)
                    rewards[i] = r
                    dones[i] = float(done)
                    if done:
                        if ep_stats is not None:
                            completed.append(ep_stats)
                            stage_eps.append(ep_stats)
                        envs[i].close()
                        envs[i], shapers[i], next_obs, specs[i] = _spawn(stage_idx)
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
            dt = time.perf_counter() - t0

            # Curriculum check
            if stage_idx == 0 and len(stage_eps) >= 24:
                kr = np.mean([e.kills > 0 for e in stage_eps[-24:]])
                if kr >= 0.60:
                    stage_idx = 1
                    stage_eps.clear()
                    print(f"  >>> ADVANCED TO {stages[stage_idx]} (kill_rate={kr:.2f})", flush=True)
            elif stage_idx == 1 and len(stage_eps) >= 20:
                kr = np.mean([e.kills for e in stage_eps[-20:]])
                sr = np.mean([not e.dead for e in stage_eps[-20:]])
                if kr >= 1.2 and sr >= 0.50:
                    stage_idx = 2
                    stage_eps.clear()
                    print(f"  >>> ADVANCED TO {stages[stage_idx]} (kills={kr:.2f}, surv={sr:.2f})", flush=True)

            # TensorBoard logging
            mean_probs = action_probs_accum / args.rollout_steps
            for b_i, b_name in enumerate(S.BUTTONS):
                writer.add_scalar(f"action_probs/{b_name}", float(mean_probs[b_i]), total_steps)
            for k, v in train_stats.items():
                writer.add_scalar(f"train/{k}", v, total_steps)

            if completed:
                ep_rew = float(np.mean([e.reward for e in completed]))
                ep_len = float(np.mean([e.length for e in completed]))
                kills = float(np.mean([e.kills for e in completed]))
                dmg = float(np.mean([e.damage_taken for e in completed]))
                ammo = float(np.mean([e.ammo_used for e in completed]))
                aimed = float(np.mean([e.aimed_shots for e in completed]))
                wasted = float(np.mean([e.wasted_shots for e in completed]))
                c_succ = float(np.mean([e.kills > 0 for e in completed]))
                surv = float(np.mean([not e.dead for e in completed]))

                for k, v in (
                    ("ep_reward", ep_rew), ("ep_len", ep_len), ("kills", kills), ("damage_taken", dmg),
                    ("ammo_used", ammo), ("aimed_shots", aimed), ("wasted_shots", wasted),
                    ("combat_success_rate", c_succ), ("survival_rate", surv),
                ):
                    writer.add_scalar(f"episode/{k}", v, total_steps)

                print(
                    f"[{upd:3d}/{total_updates}] {stages[stage_idx]:10s} steps={total_steps:6d} sps={steps_per_update/dt:5.1f} | "
                    f"rew={ep_rew:+5.2f} kills={kills:.2f} succ={c_succ:.2f} aimed={aimed:.1f} wasted={wasted:.1f} dmg={dmg:4.1f} | "
                    f"atk_p={mean_probs[6]:.2f} ent={train_stats['entropy']:.2f} kl={train_stats['approx_kl']:.4f}",
                    flush=True,
                )
                score = kills * 3.0 + ep_rew * 0.2 - wasted * 0.05
                if score > best_score:
                    best_score = score
                    trainer.save_checkpoint(save_dir / "best.pt", upd, total_steps)

            trainer.save_checkpoint(save_dir / "latest.pt", upd, total_steps)
            if upd == early_upd:
                trainer.save_checkpoint(save_dir / "early.pt", upd, total_steps)
            elif upd == mid_upd:
                trainer.save_checkpoint(save_dir / "mid.pt", upd, total_steps)
    finally:
        writer.close()
        for e in envs:
            e.close()

    # Evaluate progression: Random -> Early -> Mid -> Best
    print("\n=== Deterministic Checkpoint Comparison (Basic + Defend + Corridor) ===")
    comp = {}
    for label, path in (
        ("random_init", None),
        ("early", save_dir / "early.pt"),
        ("mid", save_dir / "mid.pt"),
        ("best", save_dir / "best.pt"),
    ):
        if path is not None and not path.exists():
            continue
        res = evaluate_combat_checkpoint(
            path, scenarios, device=device, deterministic=True,
            filmstrip_out=save_dir / "recordings" / f"{label}_filmstrip.png",
        )
        comp[label] = res
        print(
            f"{label:12s} | rew={res['ep_reward']:+6.2f} | kills={res['kills']:.2f} | succ={res['combat_success']:.2f} | "
            f"aimed={res['aimed_shots']:4.1f} | wasted={res['wasted_shots']:4.1f} | dmg={res['damage_taken']:4.1f} | surv={res['survival_rate']:.2f}"
        )
    (save_dir / "comparison.json").write_text(json.dumps(comp, indent=2))


if __name__ == "__main__":
    main()
