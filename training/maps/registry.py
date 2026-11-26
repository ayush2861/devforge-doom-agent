"""Reproducible training, validation, and unseen test map registry across 6 curriculum stages."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from training.maps.progression_wads import compile_udmf_wad


@dataclass(frozen=True)
class MapSpec:
    name: str
    stage: int
    split: str  # "train" | "val" | "test"
    seed: int
    wad_path: str
    map_name: str = "MAP01"
    episode_ticks: int = 1400


def _generate_stage_grid(stage: int, seed: int) -> str:
    """Deterministically generate a 2D ASCII DOOM layout for stages 1..6."""
    rng = random.Random(seed)

    if stage == 1:
        # Stage 1: Basic movement, shooting, survival (corridor/room + weak enemy + medikit + exit)
        w = rng.randint(9, 13)
        h = rng.randint(5, 7)
        grid = [["#" for _ in range(w)] for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                grid[y][x] = "."
        py = rng.randint(1, h - 2)
        ey = rng.randint(1, h - 2)
        grid[py][1] = "P"
        grid[ey][w - 2] = "E"
        grid[rng.randint(1, h - 2)][w // 2] = rng.choice(["z", "h", "."])
        return "\n".join("".join(r) for r in grid)

    if stage == 2:
        # Stage 2: Enemy encounters and combat (multiple enemies + cover pillars + health/ammo)
        w = rng.randint(11, 15)
        h = rng.randint(7, 9)
        grid = [["#" for _ in range(w)] for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                grid[y][x] = "."
        grid[h // 2][1] = "P"
        grid[rng.randint(1, h - 2)][w - 2] = "E"
        # Cover pillars
        for _ in range(rng.randint(1, 3)):
            px, py = rng.randint(3, w - 4), rng.randint(2, h - 3)
            grid[py][px] = "#"
        # Enemies & supplies
        for _ in range(rng.randint(2, 4)):
            ex, ey = rng.randint(4, w - 3), rng.randint(1, h - 2)
            if grid[ey][ex] == ".":
                grid[ey][ex] = rng.choice(["z", "s", "i"])
        grid[rng.randint(1, h - 2)][2] = "h"
        grid[rng.randint(1, h - 2)][3] = "a"
        return "\n".join("".join(r) for r in grid)

    if stage == 3:
        # Stage 3: Multi-room navigation (2-3 connected rooms with normal doors, optional dead-end)
        w = rng.randint(13, 17)
        h = rng.randint(7, 9)
        grid = [["#" for _ in range(w)] for _ in range(h)]
        c1 = w // 3
        c2 = (2 * w) // 3
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if x not in (c1, c2):
                    grid[y][x] = "."
        d1y = rng.randint(1, h - 2)
        d2y = rng.randint(1, h - 2)
        grid[d1y][c1] = "D"
        grid[d2y][c2] = "D"
        grid[rng.randint(1, h - 2)][1] = "P"
        grid[rng.randint(1, h - 2)][w - 2] = "E"
        if rng.random() < 0.7:
            grid[rng.randint(1, h - 2)][c1 + 2] = rng.choice(["z", "h"])
        return "\n".join("".join(r) for r in grid)

    if stage == 4:
        # Stage 4: Item and keycard collection across rooms (unblocked exit or normal door)
        w, h = 15, 9
        grid = [["#" for _ in range(w)] for _ in range(h)]
        mid_x, mid_y = 7, 4
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if x != mid_x and y != mid_y:
                    grid[y][x] = "."
        grid[rng.randint(1, 3)][mid_x] = "D"
        grid[mid_y][rng.randint(1, 6)] = "D"
        grid[mid_y][rng.randint(8, 13)] = "D"
        grid[2][2] = "P"
        grid[7][2] = rng.choice(["b", "r", "y"])
        grid[2][12] = "h"
        grid[7][12] = "E"
        grid[6][10] = "z"
        return "\n".join("".join(r) for r in grid)

    if stage == 5:
        # Stage 5: Locked-door progression and backtracking
        key_char, door_char = rng.choice([("b", "B"), ("r", "R"), ("y", "Y")])
        w, h = 15, 9
        grid = [["#" for _ in range(w)] for _ in range(h)]
        mid_x, mid_y = 7, 4
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if x != mid_x and y != mid_y:
                    grid[y][x] = "."
        # Top-left: Player; Bottom-left: side room with keycard via normal door
        # Top-right: locked door leading to Exit room
        grid[mid_y][rng.randint(2, 5)] = "D"
        grid[rng.randint(1, 3)][mid_x] = door_char
        grid[2][2] = "P"
        grid[7][rng.randint(1, 5)] = key_char
        grid[rng.randint(1, 3)][w - 2] = "E"
        if rng.random() < 0.6:
            grid[6][5] = "h"
        return "\n".join("".join(r) for r in grid)

    # Stage 6: Full combination (combat + exploration + multi-room + locked door + resources + exit)
    key1, door1 = rng.choice([("b", "B"), ("r", "R")])
    key2, door2 = ("y", "Y")
    w, h = 17, 11
    grid = [["#" for _ in range(w)] for _ in range(h)]
    c1, c2, r1 = 5, 11, 5
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if x not in (c1, c2) and y != r1:
                grid[y][x] = "."
    grid[r1][2] = "D"
    grid[2][c1] = door1
    grid[r1][8] = "D"
    grid[7][c2] = door2
    grid[2][2] = "P"
    grid[8][2] = key1
    grid[8][3] = "z"
    grid[2][8] = key2
    grid[3][9] = "i"
    grid[8][8] = "h"
    grid[7][9] = "a"
    grid[8][14] = "E"
    grid[7][13] = "s"
    return "\n".join("".join(r) for r in grid)


class MapRegistry:
    """Builds and manages disjoint train, val, and unseen test maps for Stages 1..6."""

    def __init__(
        self,
        root_dir: str | Path = "training/maps/curriculum",
        train_per_stage: int = 4,
        val_per_stage: int = 2,
        test_per_stage: int = 2,
        base_seed: int = 1000,
    ):
        self.root_dir = Path(root_dir)
        self.maps: list[MapSpec] = []
        self._by_split_stage: dict[tuple[str, int], list[MapSpec]] = {}

        split_counts = [("train", train_per_stage), ("val", val_per_stage), ("test", test_per_stage)]
        for stage in range(1, 7):
            ticks = 1050 if stage <= 2 else (1400 if stage <= 4 else 2100)
            for split, count in split_counts:
                for idx in range(count):
                    # Disjoint deterministic seed per (split, stage, idx)
                    h = int(hashlib.sha1(f"{base_seed}:{split}:{stage}:{idx}".encode()).hexdigest()[:8], 16)
                    name = f"s{stage}_{split}_{idx:02d}"
                    wad_path = self.root_dir / split / f"{name}.wad"
                    if not wad_path.exists():
                        grid = _generate_stage_grid(stage, h)
                        compile_udmf_wad(grid, wad_path)
                    spec = MapSpec(
                        name=name,
                        stage=stage,
                        split=split,
                        seed=h,
                        wad_path=str(wad_path),
                        episode_ticks=ticks,
                    )
                    self.maps.append(spec)
                    self._by_split_stage.setdefault((split, stage), []).append(spec)

    def get(self, split: str, stage: int) -> list[MapSpec]:
        return list(self._by_split_stage.get((split, stage), []))

    def sample(self, split: str, max_stage: int, rng: random.Random, current_stage_prob: float = 0.8) -> MapSpec:
        if max_stage <= 1 or rng.random() < current_stage_prob:
            st = max_stage
        else:
            st = rng.randint(1, max_stage - 1)
        pool = self._by_split_stage[(split, st)]
        return rng.choice(pool)

    def manifest(self) -> list[dict]:
        return [asdict(m) for m in self.maps]
