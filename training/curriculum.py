"""Performance-based curriculum scheduler across Stages 1..6."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from training.rewards.reward import EpisodeStats


@dataclass
class StageCriteria:
    min_episodes: int
    min_exit_rate: float = 0.0
    min_survival: float = 0.0
    min_kills: float = 0.0
    min_doors: float = 0.0
    min_keys: float = 0.0
    min_cells: float = 0.0


DEFAULT_CRITERIA: dict[int, StageCriteria] = {
    1: StageCriteria(min_episodes=16, min_exit_rate=0.40, min_survival=0.75, min_cells=10.0),
    2: StageCriteria(min_episodes=20, min_survival=0.65, min_kills=0.50, min_exit_rate=0.25),
    3: StageCriteria(min_episodes=20, min_exit_rate=0.30, min_doors=0.50),
    4: StageCriteria(min_episodes=24, min_exit_rate=0.25, min_keys=0.45),
    5: StageCriteria(min_episodes=24, min_exit_rate=0.20, min_keys=0.50, min_doors=0.60),
    6: StageCriteria(min_episodes=999999),  # final stage
}


class CurriculumScheduler:
    """Advances curriculum stage only when actual rolling performance meets stage thresholds."""

    def __init__(self, start_stage: int = 1, max_stage: int = 6, window: int = 32):
        self.stage = start_stage
        self.max_stage = max_stage
        self.window = window
        self.episodes_in_stage = 0
        self._buf: deque[EpisodeStats] = deque(maxlen=window)
        self.history: list[dict] = []

    def record(self, stats: list[EpisodeStats], update_idx: int, total_steps: int) -> bool:
        """Record completed episodes; return True if stage advanced."""
        for s in stats:
            self._buf.append(s)
            self.episodes_in_stage += 1

        if self.stage >= self.max_stage:
            return False

        crit = DEFAULT_CRITERIA[self.stage]
        if self.episodes_in_stage < crit.min_episodes or len(self._buf) < min(crit.min_episodes, self.window):
            return False

        n = len(self._buf)
        exit_rate = sum(1 for e in self._buf if e.exit_reached) / n
        survival = sum(1 for e in self._buf if not e.dead) / n
        kills = sum(e.kills for e in self._buf) / n
        doors = sum(e.doors_opened for e in self._buf) / n
        keys = sum(e.keys for e in self._buf) / n
        cells = sum(e.cells_visited for e in self._buf) / n

        passed = (
            exit_rate >= crit.min_exit_rate
            and survival >= crit.min_survival
            and kills >= crit.min_kills
            and doors >= crit.min_doors
            and keys >= crit.min_keys
            and cells >= crit.min_cells
        )
        if not passed:
            return False

        self.history.append(
            {
                "from_stage": self.stage,
                "to_stage": self.stage + 1,
                "update": update_idx,
                "total_steps": total_steps,
                "episodes_in_stage": self.episodes_in_stage,
                "metrics": {
                    "exit_rate": round(exit_rate, 3),
                    "survival": round(survival, 3),
                    "kills": round(kills, 3),
                    "doors": round(doors, 3),
                    "keys": round(keys, 3),
                    "cells": round(cells, 2),
                },
            }
        )
        self.stage += 1
        self.episodes_in_stage = 0
        self._buf.clear()
        return True
