"""Phases 2-8: Independent Movement (A), Single-Enemy Combat (B), and Combined (C) PPO Experiments."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from doom_env import make_env
from doom_env import spaces as S
from training.algorithms.ppo import PPOConfig, PPOTrainer
from training.combat_scenarios import build_combat_scenarios
from training.diagnostics.visualize_depth import write_png
from training.models.doom_model import ESRHAAgentModel
from training.train_movement_combat import build_movement_curriculum
from training.rewards.reward import DoomRewardShaper, EpisodeStats
from training.rollout import RolloutBuffer, _batch_obs


def train_stage(
    name: str,
    specs: list[dict],
    total_steps: int,
    save_dir: Path,
    init_ckpt: Path | None = None,
    kill_terminates: bool = False,
    lr: float = 5e-4,
    ent_coef: float = 0.01,
    seed: int = 42,
) -> Path:
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
        print(f"[{name}] Initialized weights from {init_ckpt}", flush=True)

    trainer = PPOTrainer(model, PPOConfig(lr=lr, ent_coef=ent_coef, seq_len=32, batch_seqs=8), total_updates, device)

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

    hidden = model.memory.init_hidden(num_envs, device)
    prev_action = np.zeros((num_envs, len(S.BUTTONS)), dtype=np.float32)
    done_steps = 0
    best_score = -1e9

    try:
        for upd in range(1, total_updates + 1):
            t0 = time.perf_counter()
            model.eval()
            buf = RolloutBuffer(rollout_steps, num_envs, model.memory.state_dim)
            completed: list[EpisodeStats] = []
            probs_acc = np.zeros(len(S.BUTTONS), dtype=np.float64)

            for _ in range(rollout_steps):
                obs_t = _batch_obs(obs_list, device=device)
                pa_t = torch.as_tensor(prev_action, dtype=torch.float32, device=device)
                with torch.no_grad():
                    out = model.forward_step(obs_t, hidden=hidden, prev_action=pa_t)
                    action_t = out["tactical_dist"].sample()
                    lp_t = out["tactical_dist"].log_prob(action_t)
                    val_t = out["value"]
                    probs_acc += out["tactical_dist"].base_dist.probs.mean(dim=0).cpu().numpy()

                action_np = action_t.cpu().numpy().astype(np.float32)
                action_int = action_np.astype(np.int8)
                hid_in = hidden.squeeze(0).cpu().numpy()
                rewards = np.zeros(num_envs, dtype=np.float32)
                dones = np.zeros(num_envs, dtype=np.float32)
                next_obs_list = []
                next_hidden = out["next_hidden"]
                next_pa = action_np.copy()

                for i in range(num_envs):
                    no, _, term, trunc, _ = envs[i].step(action_int[i])
                    priv = envs[i].privileged_state()
                    r, st = shapers[i].step(priv, action=action_int[i], obs=no)
                    if kill_terminates and priv["kills"] >= 1:
                        r += 2.0
                        shapers[i]._stats.reward += 2.0
                        st = shapers[i]._stats
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
                lo = model.forward_step(_batch_obs(obs_list, device=device), hidden=hidden, prev_action=torch.as_tensor(prev_action, dtype=torch.float32, device=device))
            trainer.update(buf, lo["value"].cpu().numpy(), np.zeros(num_envs, dtype=np.float32))
            done_steps += steps_per_upd
            mp = probs_acc / rollout_steps

            if completed:
                rew = float(np.mean([e.reward for e in completed]))
                er = float(np.mean([e.exit_reached for e in completed]))
                kr = float(np.mean([e.kills for e in completed]))
                dist = float(np.mean([e.distance_travelled for e in completed]))
                print(
                    f"[{name} {upd:2d}/{total_updates}] steps={done_steps:5d} sps={steps_per_upd/(time.perf_counter()-t0):5.1f} | "
                    f"rew={rew:+5.2f} exit={er:.2f} kills={kr:.2f} dist={dist:5.1f} | "
                    f"fwd={mp[0]:.2f} back={mp[1]:.2f} turn={mp[4]+mp[5]:.2f} atk={mp[6]:.2f}",
                    flush=True,
                )
                score = rew + er * 5.0 + kr * 4.0
                if score > best_score:
                    best_score = score
                    trainer.save_checkpoint(save_dir / "best.pt", upd, done_steps)
            trainer.save_checkpoint(save_dir / "latest.pt", upd, done_steps)
    finally:
        for e in envs:
            e.close()
    return save_dir / "best.pt"


@torch.no_grad()
def evaluate_experiment(
    label: str, ckpt_path: Path, specs: list[dict], kill_terminates: bool = False, filmstrip_path: Path | None = None
) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ESRHAAgentModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    stats_all: list[EpisodeStats] = []
    wall_hits = 0
    stuck_eps = 0
    frames: list[np.ndarray] = []

    for idx, sp in enumerate(specs):
        e = make_env(sp["wad"], "MAP01", episode_ticks=sp["ticks"], render_buffers=(filmstrip_path is not None and idx == 0))
        sh = DoomRewardShaper()
        obs, _ = e.reset(seed=500 + idx)
        sh.reset(e.privileged_state(), obs=obs)
        model.reset_state()
        for step in range(sp["ticks"] // S.FRAME_SKIP):
            if filmstrip_path is not None and idx == 0 and step % 8 == 0 and len(frames) < 6:
                frames.append(obs["rgb"].copy())
            act = model.act(obs, deterministic=True)
            obs, _, term, trunc, _ = e.step(act)
            priv = e.privileged_state()
            _, st = sh.step(priv, action=act, obs=obs)
            if float(obs["rays"]["distance"][0]) < 0.05 and int(act[0]) == 1:
                wall_hits += 1
            if kill_terminates and priv["kills"] >= 1:
                st = sh._stats
                term = True
            if term or trunc:
                if st is not None:
                    stats_all.append(st)
                    if st.stuck_steps > 10:
                        stuck_eps += 1
                break
        e.close()

    if filmstrip_path is not None and frames:
        filmstrip_path.parent.mkdir(parents=True, exist_ok=True)
        write_png(filmstrip_path, np.concatenate(frames, axis=1))

    tot_len = max(1, sum(s.length for s in stats_all))
    aimed = sum(s.aimed_shots for s in stats_all)
    wasted = sum(s.wasted_shots for s in stats_all)
    return {
        "experiment": label,
        "episodes": len(stats_all),
        "avg_distance": round(float(np.mean([s.distance_travelled for s in stats_all])), 1),
        "completion_rate": round(float(np.mean([s.exit_reached for s in stats_all])), 2),
        "turn_pct": round(100.0 * sum(s.turn_steps for s in stats_all) / tot_len, 1),
        "fwd_pct": round(100.0 * sum(s.forward_steps for s in stats_all) / tot_len, 1),
        "wall_collisions": wall_hits,
        "stuck_episodes": stuck_eps,
        "kills": round(float(np.mean([s.kills for s in stats_all])), 2),
        "hit_rate": round(aimed / max(1, aimed + wasted), 2),
        "ammo_used": round(float(np.mean([s.ammo_used for s in stats_all])), 1),
        "damage_received": round(float(np.mean([s.damage_taken for s in stats_all])), 1),
        "survival_rate": round(float(np.mean([not s.dead for s in stats_all])), 2),
        "avg_ep_len": round(float(np.mean([s.length for s in stats_all])), 1),
        "ep_reward": round(float(np.mean([s.reward for s in stats_all])), 2),
    }


def main() -> None:
    mov = build_movement_curriculum()
    comb = build_combat_scenarios()

    # Experiment A: Movement-Only
    exp_a_specs = mov["S1_fwd"] + mov["S2_turn"] + mov["S3_corridor"]
    ckpt_a = train_stage("ExpA_Move", exp_a_specs, total_steps=24576, save_dir=Path("training/runs/exp_A_movement"), lr=5e-4)

    # Experiment B: Single-Enemy Combat (randomized enemy spawns)
    exp_b_specs = comb["A_basic"]
    ckpt_b = train_stage("ExpB_Combat", exp_b_specs, total_steps=32768, save_dir=Path("training/runs/exp_B_combat"), kill_terminates=True, lr=5e-4)

    # Experiment C: Combined Movement + Combat (warm-started from Exp A)
    exp_c_specs = comb["C_corridor"] + mov["S2_turn"][:2] + comb["A_basic"][:4]
    ckpt_c = train_stage("ExpC_Combined", exp_c_specs, total_steps=32768, save_dir=Path("training/runs/exp_C_combined"), init_ckpt=ckpt_a, lr=4e-4)

    # Evaluate all 3
    res_a = evaluate_experiment("ExpA_Movement", ckpt_a, mov["S1_fwd"] + mov["S2_turn"], filmstrip_path=Path("training/runs/exp_A_movement/filmstrip.png"))
    res_b = evaluate_experiment("ExpB_Combat", ckpt_b, comb["A_basic"], kill_terminates=True, filmstrip_path=Path("training/runs/exp_B_combat/filmstrip.png"))
    res_c = evaluate_experiment("ExpC_Combined", ckpt_c, comb["C_corridor"] + [{"name": "two_rooms", "wad": "maps/two_rooms.wad", "ticks": 1400}], filmstrip_path=Path("training/runs/exp_C_combined/filmstrip.png"))

    report = {"A_movement": res_a, "B_combat": res_b, "C_combined": res_c}
    Path("training/runs/phase7_8_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== PHASE 8 PERFORMANCE REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
