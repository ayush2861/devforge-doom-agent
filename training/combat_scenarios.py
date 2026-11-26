"""Focused combat scenarios (Basic Combat, Defend Center, Deadly Corridor) for doom_env."""

from __future__ import annotations

import os
import random
from pathlib import Path

import vizdoom as vd

from training.maps.progression_wads import compile_udmf_wad


def build_combat_scenarios(out_dir: str | Path = "training/maps/combat") -> dict[str, list[dict]]:
    """Build Stage A (Basic Combat), Stage B (Defend Center), and Stage C (Deadly Corridor)."""
    out = Path(out_dir)
    sdir = Path(os.path.dirname(vd.__file__)) / "scenarios"

    # Stage A: Randomized single-enemy arenas (left, center, right, near, far) — no exit distractor
    basic_maps = []
    # Spawn enemy at different offsets relative to player at (2, 4) facing East
    spawns = [
        (8, 4),   # straight ahead
        (8, 2),   # front-right
        (8, 6),   # front-left
        (6, 1),   # sharp right
        (6, 7),   # sharp left
        (10, 3),  # far right
        (10, 5),  # far left
        (5, 4),   # close ahead
    ]
    for idx, (ex, ey) in enumerate(spawns):
        w, h = 13, 9
        grid = [["#" for _ in range(w)] for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                grid[y][x] = "."
        grid[4][2] = "P"
        grid[ey][ex] = "z"
        wad = compile_udmf_wad("\n".join("".join(r) for r in grid), out / f"basic_{idx:02d}.wad")
        basic_maps.append({"name": f"basic_{idx:02d}", "wad": str(wad), "ticks": 420, "kill_target": 1})

    # Also include official ViZDoom basic.wad
    basic_maps.append({"name": "vizdoom_basic", "wad": str(sdir / "basic.wad"), "ticks": 300, "kill_target": 1})

    # Stage B: Defend Center (official defend_the_center.wad + randomized 4-way arenas)
    defend_maps = [
        {"name": "vizdoom_defend", "wad": str(sdir / "defend_the_center.wad"), "ticks": 1050, "kill_target": None}
    ]
    for idx in range(4):
        rng = random.Random(200 + idx)
        w, h = 13, 13
        grid = [["#" for _ in range(w)] for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                grid[y][x] = "."
        grid[6][6] = "P"
        # Enemies in 4 quadrants
        for qx, qy in [(2, 2), (10, 2), (2, 10), (10, 10)]:
            grid[qy + rng.randint(-1, 1)][qx + rng.randint(-1, 1)] = rng.choice(["z", "s", "i"])
        grid[6][3] = "h"
        wad = compile_udmf_wad("\n".join("".join(r) for r in grid), out / f"defend_{idx:02d}.wad")
        defend_maps.append({"name": f"defend_{idx:02d}", "wad": str(wad), "ticks": 1050, "kill_target": None})

    # Stage C: Deadly Corridor (official deadly_corridor.wad + multi-enemy corridor with exit)
    corridor_maps = [
        {"name": "vizdoom_corridor", "wad": str(sdir / "deadly_corridor.wad"), "ticks": 1400, "kill_target": None}
    ]
    for idx in range(3):
        rng = random.Random(300 + idx)
        w, h = 19, 7
        grid = [["#" for _ in range(w)] for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                grid[y][x] = "."
        grid[3][1] = "P"
        grid[3][17] = "E"
        for cx in (5, 9, 13):
            grid[rng.choice([1, 2, 4, 5])][cx] = rng.choice(["z", "s", "i"])
        grid[3][7] = "h"
        grid[3][11] = "a"
        wad = compile_udmf_wad("\n".join("".join(r) for r in grid), out / f"corridor_{idx:02d}.wad")
        corridor_maps.append({"name": f"corridor_{idx:02d}", "wad": str(wad), "ticks": 1400, "kill_target": None})

    return {"A_basic": basic_maps, "B_defend": defend_maps, "C_corridor": corridor_maps}
