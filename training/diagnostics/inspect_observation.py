"""Capture and inspect all sensory observation channels across key gameplay situations."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import vizdoom as vd

from doom_env import make_env
from doom_env import spaces as S
from tests.fixtures import fixture_wad
from training.diagnostics.visualize_depth import save_rgb_and_depth
from training.diagnostics.visualize_entities import format_entities_table, save_entities_plot
from training.diagnostics.visualize_rays import format_rays_summary, save_rays_plot
from training.diagnostics.visualize_vitals import decode_vitals, format_vitals

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"

FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
BACK = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], np.int8)
TURN_L = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], np.int8)
TURN_R = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0], np.int8)
USE = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], np.int8)
NOOP = np.zeros(9, np.int8)


def _capture(tag: str, env, obs: dict) -> dict:
    depth_stats = save_rgb_and_depth(obs, OUT_DIR, tag)
    ray_stats = save_rays_plot(obs["rays"], OUT_DIR, tag)
    ent_list = save_entities_plot(obs["visible_entities"], OUT_DIR, tag)
    vitals_dict = decode_vitals(obs["vitals"])

    print(f"\n=== Scenario: {tag} ===")
    print(format_vitals(obs["vitals"]))
    print(format_rays_summary(obs["rays"]))
    print(format_entities_table(obs["visible_entities"]))
    print(
        f"Depth: min={depth_stats['min']}, max={depth_stats['max']}, "
        f"p10={depth_stats['p10']:.1f}, p50={depth_stats['p50']:.1f}, p90={depth_stats['p90']:.1f}, "
        f"near(<20)={depth_stats['near_frac_lt_20']:.2%}, far(>100)={depth_stats['far_frac_gt_100']:.2%}"
    )
    return {
        "scenario": tag,
        "vitals": vitals_dict,
        "depth": depth_stats,
        "rays": ray_stats,
        "visible_entities": ent_list,
    }


def _step_n(env, action: np.ndarray, n: int) -> dict:
    obs = None
    for _ in range(n):
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    return obs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []

    # 1. Empty corridor (fixture: "#####\n#P..#\n#####")
    env = make_env(fixture_wad("#####\n#P..#\n#####"), episode_ticks=400)
    obs, _ = env.reset(seed=0)
    summaries.append(_capture("1_empty_corridor", env, obs))
    env.close()

    # 2. Wall in front (step forward toward wall in two_rooms or turn toward wall)
    env = make_env(fixture_wad("#####\n#P..#\n#####"), episode_ticks=400)
    env.reset(seed=0)
    obs = _step_n(env, TURN_L, 4)  # turn 90 deg to face north wall closely
    summaries.append(_capture("2_wall_in_front", env, obs))
    env.close()

    # 3. Door in front (fixture: "##########\n#P..D...E#\n##########")
    env = make_env(fixture_wad("##########\n#P..D...E#\n##########"), episode_ticks=400)
    obs, _ = env.reset(seed=0)
    summaries.append(_capture("3_door_in_front", env, obs))
    env.close()

    # 4. Locked door (fixture: "#######\n#P..B.E#\n#######")
    env = make_env(fixture_wad("#######\n#P..B.E#\n#######"), episode_ticks=400)
    obs, _ = env.reset(seed=0)
    summaries.append(_capture("4_locked_door", env, obs))
    env.close()

    # 5. Item visible + Enemy visible (fixture: "#########\n#P..h..z#\n#########")
    env = make_env(fixture_wad("#########\n#P..h..z#\n#########"), episode_ticks=400)
    obs, _ = env.reset(seed=0)
    summaries.append(_capture("5_item_and_enemy_visible", env, obs))
    env.close()

    # 6. Enemy visible alone (fixture: "##########\n#P......i#\n##########")
    env = make_env(fixture_wad("##########\n#P......i#\n##########"), episode_ticks=400)
    obs, _ = env.reset(seed=0)
    obs = _step_n(env, NOOP, 20)  # let imp wake up and move/fire
    summaries.append(_capture("6_enemy_visible", env, obs))
    env.close()

    # 7. Multiple enemies (Freedoom2 MAP01: advance and scan until >= 2 enemies are visible)
    freedoom = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
    env = make_env(freedoom, "MAP01", episode_ticks=800)
    obs, _ = env.reset(seed=0)
    best_obs = obs
    best_enemies = 0
    for step_idx in range(60):
        act = FWD if step_idx < 18 else TURN_L
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break
        m = obs["visible_entities"]["mask"].astype(bool)
        n_en = sum(1 for t in obs["visible_entities"]["entity_type"][m] if S.ENTITY_TYPES[int(t)].startswith("enemy"))
        if n_en > best_enemies:
            best_enemies = n_en
            best_obs = obs
        if n_en >= 2:
            break
    summaries.append(_capture("7_multiple_enemies", env, best_obs))
    env.close()

    (OUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nSaved {len(summaries)} scenario diagnostics and PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
