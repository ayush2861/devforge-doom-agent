"""Dynamic WAD map discovery for DOOM AI Training Studio (real files only)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import vizdoom as vd

from training.combat_scenarios import build_combat_scenarios
from training.maps.progression_wads import generate_all

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MapEntry:
    id: str
    name: str
    filename: str
    path: str
    category: str
    difficulty: str
    description: str
    episode_ticks: int
    available: bool


def discover_maps() -> list[dict]:
    """Ensure procedural training WADs exist and return all verified available WAD maps."""
    build_combat_scenarios()
    generate_all()

    sdir = Path(os.path.dirname(vd.__file__)) / "scenarios"
    candidates: list[tuple[str, str, Path, str, str, str, int]] = [
        ("two_rooms", "Two Rooms (Starter)", ROOT / "maps" / "two_rooms.wad", "Multi-Room Exploration", "Easy", "Starter 2-room map with one door, one enemy, and exit.", 2100),
        ("vizdoom_basic", "ViZDoom Basic", sdir / "basic.wad", "Basic Combat", "Easy", "Official ViZDoom single-room target shooting scenario.", 300),
        ("vizdoom_defend", "Defend The Center", sdir / "defend_the_center.wad", "Defend Center", "Medium", "Official ViZDoom 360-degree arena survival against approaching monsters.", 1050),
        ("vizdoom_corridor", "Deadly Corridor", sdir / "deadly_corridor.wad", "Deadly Corridor", "Hard", "Official ViZDoom corridor combat with multiple enemies and exit.", 1400),
        ("prog_01_door_exit", "Door & Exit", ROOT / "training" / "maps" / "generated" / "prog_01_door_exit.wad", "Multi-Room Exploration", "Easy", "Corridor with a closed door leading to the exit.", 1050),
        ("prog_02_key_backtrack", "Blue Key Backtrack", ROOT / "training" / "maps" / "generated" / "prog_02_key_backtrack.wad", "Keycard and Locked Door", "Medium", "Discover locked blue door, explore side room for blue key, backtrack, unlock, exit.", 1400),
        ("prog_03_ambush_key", "Ambush & Red Key", ROOT / "training" / "maps" / "generated" / "prog_03_ambush_key.wad", "Keycard and Locked Door", "Hard", "Multi-room ambush guarding the red keycard and locked red exit door.", 1400),
        ("prog_04_two_keys", "Two-Key Gauntlet", ROOT / "training" / "maps" / "generated" / "prog_04_two_keys.wad", "Full Gauntlet", "Hard", "Sequential blue and red keycards, locked doors, enemies, and exit.", 2100),
    ]

    # Add randomized Basic Combat, Defend, and Corridor WADs on disk
    combat_dir = ROOT / "training" / "maps" / "combat"
    for p in sorted(combat_dir.glob("basic_*.wad")):
        candidates.append((p.stem, f"Basic Combat ({p.stem})", p, "Basic Combat", "Easy", "Randomized single-enemy spawn for aiming & trigger discipline.", 420))
    for p in sorted(combat_dir.glob("defend_*.wad")):
        candidates.append((p.stem, f"Defend Arena ({p.stem})", p, "Defend Center", "Medium", "4-quadrant enemy arena with health pickup.", 1050))
    for p in sorted(combat_dir.glob("corridor_*.wad")):
        candidates.append((p.stem, f"Combat Corridor ({p.stem})", p, "Deadly Corridor", "Hard", "Multi-enemy corridor with supplies and exit.", 1400))

    entries: list[dict] = []
    for mid, name, path, cat, diff, desc, ticks in candidates:
        resolved = path.resolve()
        avail = resolved.is_file()
        entries.append(
            asdict(
                MapEntry(
                    id=mid,
                    name=name,
                    filename=resolved.name,
                    path=str(resolved),
                    category=cat,
                    difficulty=diff,
                    description=desc,
                    episode_ticks=ticks,
                    available=avail,
                )
            )
        )
    return entries
