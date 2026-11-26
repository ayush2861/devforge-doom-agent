"""Human-Feedback Diagnosis, Continuous Online PPO Learning, and Safe Policy Rollback."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from doom_env import make_env
from doom_env import spaces as S
from training.algorithms.ppo import PPOTrainer
from training.models.doom_model import ESRHAAgentModel
from training.rewards.reward import DoomRewardShaper
from training.rollout import RolloutBuffer, _batch_obs

PROBLEM_CATEGORIES = {
    "movement": [
        ("not_moving", "AI is not moving"),
        ("moving_too_slowly", "AI is moving too slowly"),
        ("moving_too_fast", "AI is moving too fast"),
        ("rotating_in_circles", "AI keeps rotating in circles"),
        ("not_moving_forward", "AI is not moving forward"),
        ("returning_same_location", "AI keeps returning to the same location"),
        ("stuck_against_wall", "AI is stuck against a wall"),
        ("not_exploring", "AI is not exploring the map"),
    ],
    "combat": [
        ("not_detecting_enemies", "AI is not detecting enemies"),
        ("detects_no_kill", "AI detects enemies but cannot kill them"),
        ("firing_randomly", "AI is firing randomly"),
        ("firing_at_empty", "AI is firing at empty locations"),
        ("not_aiming_accurately", "AI is not aiming accurately"),
        ("shooting_after_kill", "AI keeps shooting after killing an enemy"),
        ("ignores_nearby_enemies", "AI ignores nearby enemies"),
        ("not_finding_other_enemies", "AI is not finding enemies in other areas"),
    ],
    "survival": [
        ("not_collecting_medikits", "AI is not collecting medikits"),
        ("ignores_low_health", "AI ignores low health"),
        ("wastes_ammo", "AI wastes ammunition"),
        ("not_switching_weapons", "AI is not switching weapons"),
        ("not_reloading", "AI is not reloading when reload is supported"),
    ],
    "other": [
        ("other_issue", "Something else is wrong"),
    ],
}

VALID_STATUSES = (
    "Feedback received",
    "Capturing gameplay context",
    "Analyzing behaviour",
    "Problem identified",
    "Generating learning signal",
    "Online learning update in progress",
    "Candidate policy ready",
    "Evaluating updated policy",
    "Applying updated policy",
    "Improvement verified",
    "More learning required",
    "Correction unsuccessful",
)


@dataclass
class FeedbackShapingModifiers:
    """Dynamic reward multipliers/bonuses adjusted by human feedback (training-time only)."""

    fwd_explore_mult: float = 1.0
    spin_penalty_mult: float = 1.0
    blind_fire_penalty_mult: float = 1.0
    aim_turn_bonus_mult: float = 1.0
    aimed_shot_bonus_mult: float = 1.0
    medikit_bonus_mult: float = 1.0
    wpn_switch_bonus_mult: float = 1.0
    combat_slowdown_mult: float = 1.0


def diagnose_feedback(
    categories: list[str],
    description: str,
    history: list[dict],
    current_ctx: dict,
    modifiers: FeedbackShapingModifiers,
) -> tuple[dict, FeedbackShapingModifiers]:
    """Inspect rolling gameplay telemetry to diagnose root cause and produce reward adjustments."""
    recent = history[-60:] if history else []
    n = max(1, len(recent))

    fwd_rate = sum(1 for h in recent if h["action"][0] == 1) / n
    turn_rate = sum(1 for h in recent if h["action"][4] == 1 or h["action"][5] == 1) / n
    atk_rate = sum(1 for h in recent if h["action"][6] == 1) / n
    no_tgt_atk = sum(1 for h in recent if h["action"][6] == 1 and h["visible_enemies"] == 0) / n
    avg_disp = float(np.mean([h["disp"] for h in recent])) if recent else 0.0
    avg_ray = float(np.mean([h["front_ray"] for h in recent])) if recent else 0.5
    vis_rate = sum(1 for h in recent if h["visible_enemies"] > 0) / n
    cells_delta = (recent[-1]["cells_visited"] - recent[0]["cells_visited"]) if len(recent) >= 2 else 0

    cats = set(categories)
    new_mod = copy.deepcopy(modifiers)
    causes: list[str] = []
    origins: list[str] = []
    actions_taken: list[str] = []

    if cats & {"not_moving", "not_moving_forward", "rotating_in_circles", "not_exploring", "returning_same_location", "stuck_against_wall", "moving_too_slowly"}:
        if turn_rate > 0.45 and fwd_rate < 0.35:
            causes.append(
                f"Policy selects turning on {turn_rate*100:.0f}% of steps vs forward on {fwd_rate*100:.0f}% "
                f"(exploration gained {cells_delta} cells, avg front clearance {avg_ray:.2f})."
            )
            origins.append("Movement Policy / Exploration Strategy")
        elif avg_disp < 1.5 and avg_ray < 0.12:
            causes.append(f"Player stalled near obstacle (avg displacement {avg_disp:.1f} u/step, front ray {avg_ray:.2f}).")
            origins.append("Movement Policy")
        else:
            causes.append(f"Low exploration progress ({cells_delta} new cells over last {n} steps, fwd={fwd_rate*100:.0f}%).")
            origins.append("Exploration Strategy")
        new_mod.fwd_explore_mult = min(2.5, new_mod.fwd_explore_mult * 1.45)
        new_mod.spin_penalty_mult = min(3.0, new_mod.spin_penalty_mult * 1.5)
        actions_taken.append("Boosted forward exploration reward & spin/oscillation penalty in online PPO")

    if cats & {"moving_too_fast"}:
        causes.append(f"High forward/strafe rate during engagement (fwd={fwd_rate*100:.0f}%, avg_disp={avg_disp:.1f} u/step).")
        origins.append("Movement Policy")
        new_mod.combat_slowdown_mult = min(2.5, new_mod.combat_slowdown_mult * 1.5)
        actions_taken.append("Increased close-combat slowdown penalty to stabilize aim")

    if cats & {"firing_randomly", "firing_at_empty", "not_aiming_accurately", "shooting_after_kill", "detects_no_kill", "ignores_nearby_enemies", "not_detecting_enemies", "not_finding_other_enemies"}:
        if no_tgt_atk > 0.05:
            causes.append(f"Unjustified firing without visible target on {no_tgt_atk*100:.1f}% of recent steps (atk_rate={atk_rate*100:.0f}%).")
            origins.append("Aiming & Firing")
        elif vis_rate > 0.2 and atk_rate < 0.1:
            causes.append(f"Enemies visible on {vis_rate*100:.0f}% of steps but attack rate is only {atk_rate*100:.0f}%.")
            origins.append("Target Selection / Aiming")
        else:
            causes.append(f"Suboptimal target alignment during combat (vis_rate={vis_rate*100:.0f}%, atk_rate={atk_rate*100:.0f}%).")
            origins.append("Aiming & Firing / Target Selection")
        new_mod.blind_fire_penalty_mult = min(3.0, new_mod.blind_fire_penalty_mult * 1.5)
        new_mod.aim_turn_bonus_mult = min(2.5, new_mod.aim_turn_bonus_mult * 1.35)
        new_mod.aimed_shot_bonus_mult = min(2.5, new_mod.aimed_shot_bonus_mult * 1.35)
        actions_taken.append("Increased blind-fire penalty, target-turn bonus & aimed-shot reward in online PPO")

    if cats & {"not_collecting_medikits", "ignores_low_health"}:
        causes.append(f"Health at {current_ctx.get('health', 100):.0f}% with insufficient medikit recovery priority.")
        origins.append("Survival / Health Management")
        new_mod.medikit_bonus_mult = min(3.0, new_mod.medikit_bonus_mult * 1.6)
        actions_taken.append("Boosted low-health medikit approach & collection reward")

    if cats & {"wastes_ammo", "not_switching_weapons", "not_reloading"}:
        note = " (Note: DOOM has no reload action; uses SELECT_NEXT_WEAPON.)" if "not_reloading" in cats else ""
        causes.append(f"Ammo efficiency / weapon selection issue (weapon={current_ctx.get('weapon_name')}, ammo={current_ctx.get('ammo')}){note}")
        origins.append("Ammo & Weapon Management")
        new_mod.wpn_switch_bonus_mult = min(3.0, new_mod.wpn_switch_bonus_mult * 1.5)
        new_mod.blind_fire_penalty_mult = min(3.0, new_mod.blind_fire_penalty_mult * 1.3)
        actions_taken.append("Boosted empty-weapon switch reward & wasted-ammo penalty")

    if not causes:
        causes.append(f"General policy issue reported ({description or 'user observation'}); fwd={fwd_rate*100:.0f}%, turn={turn_rate*100:.0f}%, atk={atk_rate*100:.0f}%.")
        origins.append("Reward Design / Policy")
        actions_taken.append("Applied balanced online PPO refinement step")

    diagnosis = {
        "origin": " + ".join(dict.fromkeys(origins)),
        "detected_cause": " ".join(causes),
        "learning_action": "; ".join(actions_taken),
        "telemetry_snapshot": {
            "window_steps": len(recent),
            "fwd_pct": round(fwd_rate * 100, 1),
            "turn_pct": round(turn_rate * 100, 1),
            "atk_pct": round(atk_rate * 100, 1),
            "no_tgt_atk_pct": round(no_tgt_atk * 100, 1),
            "avg_disp": round(avg_disp, 2),
            "cells_delta": int(cells_delta),
        },
    }
    return diagnosis, new_mod


@torch.no_grad()
def evaluate_policy_episodes(
    model: ESRHAAgentModel,
    map_path: str,
    ticks: int = 700,
    n_episodes: int = 2,
    seed_base: int = 700,
    device: str = "cpu",
    deterministic: bool = True,
) -> dict:
    """Evaluate a policy on real ViZDoom episodes (target map + regression check)."""
    model.eval()
    kills_l, cells_l, disp_l, fwd_l, turn_l, no_tgt_l, aimed_l, wasted_l, rew_l = [[] for _ in range(9)]

    for ep in range(n_episodes):
        env = make_env(map_path, "MAP01", episode_ticks=ticks, render_buffers=False)
        sh = DoomRewardShaper()
        obs, _ = env.reset(seed=seed_base + ep)
        sh.reset(env.privileged_state(), obs=obs)
        model.reset_state(batch_size=1, device=device)

        ep_no_tgt, ep_disp = 0, 0.0
        max_steps = min(200, ticks // S.FRAME_SKIP)
        for _ in range(max_steps):
            ve = obs["visible_entities"]
            has_e = any(int(ve["entity_type"][i]) in (1, 2, 3) for i in np.flatnonzero(ve["mask"].astype(bool)))
            act = model.act(obs, deterministic=deterministic)
            if act[6] and not has_e:
                ep_no_tgt += 1
            obs, _, term, trunc, _ = env.step(act)
            p = env.privileged_state()
            sh.step(p, action=act, obs=obs)
            if term or trunc:
                break
        st = sh._stats
        kills_l.append(p["kills"])
        cells_l.append(st.cells_visited)
        disp_l.append(st.distance_travelled)
        fwd_l.append(100.0 * st.forward_steps / max(1, st.length))
        turn_l.append(100.0 * st.turn_steps / max(1, st.length))
        no_tgt_l.append(ep_no_tgt)
        aimed_l.append(st.aimed_shots)
        wasted_l.append(st.wasted_shots)
        rew_l.append(st.reward)
        env.close()

    tot_shots = sum(aimed_l) + sum(wasted_l)
    return {
        "avg_kills": round(float(np.mean(kills_l)), 2),
        "avg_cells": round(float(np.mean(cells_l)), 1),
        "avg_disp": round(float(np.mean(disp_l)), 1),
        "fwd_pct": round(float(np.mean(fwd_l)), 1),
        "turn_pct": round(float(np.mean(turn_l)), 1),
        "no_tgt_shots": round(float(np.mean(no_tgt_l)), 2),
        "hit_rate": round(sum(aimed_l) / max(1, tot_shots), 2),
        "avg_reward": round(float(np.mean(rew_l)), 2),
    }


def check_candidate_improvement(
    categories: list[str],
    before: dict,
    after: dict,
) -> tuple[bool, str]:
    """Verify candidate improved (or maintained) target behavior without catastrophic regression."""
    cats = set(categories)
    # Regression guard: kills must not collapse
    if before["avg_kills"] >= 0.5 and after["avg_kills"] < before["avg_kills"] - 0.6:
        return False, f"Combat regression: kills dropped {before['avg_kills']} -> {after['avg_kills']}."

    if cats & {"not_moving", "not_moving_forward", "rotating_in_circles", "not_exploring", "returning_same_location", "stuck_against_wall", "moving_too_slowly"}:
        improved = (after["fwd_pct"] >= before["fwd_pct"] - 2.0 and after["avg_disp"] >= before["avg_disp"] * 0.9) or (
            after["avg_cells"] > before["avg_cells"]
        )
        if not improved:
            return False, f"Navigation did not improve (fwd {before['fwd_pct']}%->{after['fwd_pct']}%, disp {before['avg_disp']}->{after['avg_disp']})."

    if cats & {"firing_randomly", "firing_at_empty", "not_aiming_accurately", "shooting_after_kill", "wastes_ammo"}:
        # Account for longer survival / more kills (e.g. 6 kills vs 3 kills)
        kills_gained = after["avg_kills"] - before["avg_kills"]
        if after["no_tgt_shots"] > before["no_tgt_shots"] + 2.0 and after["hit_rate"] < before["hit_rate"] - 0.05 and kills_gained < 0.5:
            return False, f"Firing did not improve (no_tgt {before['no_tgt_shots']}->{after['no_tgt_shots']}, hit {before['hit_rate']}->{after['hit_rate']})."

    if after["avg_reward"] < before["avg_reward"] - 8.0 and after["avg_kills"] < before["avg_kills"]:
        return False, f"Overall reward & kills regressed (rew {before['avg_reward']}->{after['avg_reward']})."

    return True, (
        f"Verified: kills {before['avg_kills']}->{after['avg_kills']}, "
        f"fwd {before['fwd_pct']}%->{after['fwd_pct']}%, "
        f"disp {before['avg_disp']}->{after['avg_disp']}, "
        f"no_tgt {before['no_tgt_shots']}->{after['no_tgt_shots']}."
    )


def apply_modifiers_to_shaper(shaper: DoomRewardShaper, raw_reward: float, action: np.ndarray, obs: dict, mod: FeedbackShapingModifiers) -> float:
    """Apply feedback-aware shaping deltas on top of base DoomRewardShaper reward."""
    r = raw_reward
    ve = obs["visible_entities"]
    mask = ve["mask"].astype(bool)
    enemy_vis = any(int(ve["entity_type"][i]) in (1, 2, 3) for i in np.flatnonzero(mask))
    front_ray = float(obs["rays"]["distance"][0])

    if not enemy_vis:
        if front_ray < 0.15:
            # Stuck against wall: discourage pushing into wall, reward turning toward open side
            if int(action[0]) == 1:
                r -= 0.12 * mod.fwd_explore_mult
            rays = obs["rays"]["distance"]
            left_clear = float(np.mean(rays[1:9]))
            right_clear = float(np.mean(rays[-8:]))
            if left_clear >= right_clear and int(action[4]) == 1:
                r += 0.18 * mod.fwd_explore_mult
            elif right_clear > left_clear and int(action[5]) == 1:
                r += 0.18 * mod.fwd_explore_mult
        elif int(action[0]) == 1 and int(action[1]) == 0 and front_ray > 0.18:
            r += 0.08 * (mod.fwd_explore_mult - 1.0)
        if (int(action[4]) == 1 or int(action[5]) == 1) and front_ray > 0.22:
            r -= 0.08 * (mod.spin_penalty_mult - 1.0)
        if int(action[6]) == 1:
            r -= 0.15 * (mod.blind_fire_penalty_mult - 1.0)
    else:
        if int(action[0]) == 1 and mod.combat_slowdown_mult > 1.0:
            r -= 0.06 * (mod.combat_slowdown_mult - 1.0)
    return r
