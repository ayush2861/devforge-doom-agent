"""Procedural UDMF WAD generator for long-horizon DOOM progression scenarios."""

from __future__ import annotations

from pathlib import Path

from doom_env.wad import Lump, write_wad

TILE = 64

# ZDoom DoomEdNums
THINGS = {
    "P": 1,     # Player 1 start
    "z": 3004,  # Zombieman
    "s": 9,     # ShotgunGuy
    "i": 3001,  # Imp
    "h": 2012,  # Medikit
    "a": 2007,  # Clip
    "b": 5,     # BlueCard
    "y": 6,     # YellowCard
    "r": 13,    # RedCard
}

DOOR_LOCK = {
    "D": 0,  # Unlocked door
    "R": 1,  # RedCard / RedSkull
    "B": 2,  # BlueCard / BlueSkull
    "Y": 3,  # YellowCard / YellowSkull
}


def compile_udmf_wad(grid: str, out_path: str | Path) -> Path:
    """Compile a 2D ASCII grid into a valid UDMF MAP01 PWAD."""
    rows = [r for r in grid.strip("\n").splitlines() if r]
    h = len(rows)
    w = max(len(r) for r in rows)
    rows = [r.ljust(w, "#") for r in rows]

    # Assign sector id per non-wall cell
    sec_id: dict[tuple[int, int], int] = {}
    sectors_txt: list[str] = []
    things_txt: list[str] = []

    for gy, row in enumerate(rows):
        y_world = (h - 1 - gy) * TILE
        for gx, ch in enumerate(row):
            if ch == "#":
                continue
            sid = len(sectors_txt)
            sec_id[(gx, gy)] = sid
            is_door = ch in DOOR_LOCK
            ceil = 0 if is_door else 128
            ftex = "FLOOR0_1"
            ctex = "CEIL1_1"
            sectors_txt.append(
                f'sector {{ heightfloor = 0; heightceiling = {ceil}; '
                f'texturefloor = "{ftex}"; textureceiling = "{ctex}"; lightlevel = 192; }}'
            )
            cx = gx * TILE + TILE // 2
            cy = y_world + TILE // 2
            if ch in THINGS:
                tid = THINGS[ch]
                things_txt.append(
                    f'thing {{ x = {float(cx)}; y = {float(cy)}; angle = 0; type = {tid}; '
                    f'skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; '
                    f'single = true; }}'
                )

    # Vertices on (w+1) x (h+1) lattice
    verts_txt: list[str] = []
    v_idx: dict[tuple[int, int], int] = {}
    for vy in range(h + 1):
        for vx in range(w + 1):
            v_idx[(vx, vy)] = len(verts_txt)
            verts_txt.append(f"vertex {{ x = {float(vx * TILE)}; y = {float(vy * TILE)}; }}")

    sidedefs_txt: list[str] = []
    linedefs_txt: list[str] = []

    def add_side(sid: int, tex: str = "STARTAN2") -> int:
        idx = len(sidedefs_txt)
        sidedefs_txt.append(f'sidedef {{ sector = {sid}; texturemiddle = "{tex}"; texturetop = "{tex}"; texturebottom = "{tex}"; }}')
        return idx

    def add_line(v1: int, v2: int, f_sec: int, b_sec: int | None, special: int = 0, arg0: int = 0, arg3: int = 0, exit_line: bool = False) -> None:
        tex = "EXITDOOR" if exit_line else ("BIGDOOR2" if (f_sec in door_secs or (b_sec in door_secs if b_sec is not None else False)) else "STARTAN2")
        sf = add_side(f_sec, "-" if b_sec is not None else tex)
        sb = add_side(b_sec, "-") if b_sec is not None else -1
        props = [f"v1 = {v1};", f"v2 = {v2};", f"sidefront = {sf};"]
        if sb >= 0:
            props.append(f"sideback = {sb};")
            props.append("twosided = true;")
        else:
            props.append("blocking = true;")
        if special:
            props.append(f"special = {special};")
            props.append(f"arg0 = {arg0}; arg1 = 16; arg2 = 150; arg3 = {arg3};")
            if special == 243:
                props.append("playeruse = true;")
            else:
                props.append("playeruse = true; repeatspecial = true;")
        linedefs_txt.append("linedef { " + " ".join(props) + " }")

    door_secs = {sec_id[(gx, gy)] for gy, row in enumerate(rows) for gx, ch in enumerate(row) if ch in DOOR_LOCK}

    for gy, row in enumerate(rows):
        wy = h - 1 - gy
        for gx, ch in enumerate(row):
            if ch == "#":
                continue
            s = sec_id[(gx, gy)]
            # Four edges of cell (gx, wy): N, S, W, E
            edges = [
                ((gx, wy + 1), (gx + 1, wy + 1), (gx, gy - 1)),  # North
                ((gx + 1, wy), (gx, wy), (gx, gy + 1)),          # South
                ((gx, wy), (gx, wy + 1), (gx - 1, gy)),          # West
                ((gx + 1, wy + 1), (gx + 1, wy), (gx + 1, gy)),  # East
            ]
            for (vx1, vy1), (vx2, vy2), (nx, ny) in edges:
                nb = sec_id.get((nx, ny))
                if nb is not None and (nx, ny) < (gx, gy):
                    continue  # already emitted from neighbor
                if nb is None:
                    # Wall; if this cell is 'E', make one wall an exit switch (special 243 = Exit_Normal)
                    is_exit_wall = ch == "E" and (nx > gx or ny == gy)
                    add_line(v_idx[(vx1, vy1)], v_idx[(vx2, vy2)], s, None, special=243 if is_exit_wall else 0, exit_line=is_exit_wall)
                else:
                    # Shared boundary: check if either side is a door
                    n_ch = rows[ny][nx]
                    lock = DOOR_LOCK.get(ch, DOOR_LOCK.get(n_ch, None))
                    if lock is not None:
                        d_sec = s if ch in DOOR_LOCK else nb
                        spec = 13 if lock > 0 else 12  # Door_LockedRaise=13, Door_Raise=12
                        # Orient front toward non-door side if possible
                        if ch in DOOR_LOCK:
                            add_line(v_idx[(vx2, vy2)], v_idx[(vx1, vy1)], nb, s, special=spec, arg0=0, arg3=lock)
                        else:
                            add_line(v_idx[(vx1, vy1)], v_idx[(vx2, vy2)], s, nb, special=spec, arg0=0, arg3=lock)
                    else:
                        add_line(v_idx[(vx1, vy1)], v_idx[(vx2, vy2)], s, nb)

    # Also add a walk-over exit trigger inside 'E' cell so stepping onto E completes the level
    for gy, row in enumerate(rows):
        wy = h - 1 - gy
        for gx, ch in enumerate(row):
            if ch == "E":
                s = sec_id[(gx, gy)]
                sf = add_side(s, "-")
                sb = add_side(s, "-")
                linedefs_txt.append(
                    f"linedef {{ v1 = {v_idx[(gx, wy)]}; v2 = {v_idx[(gx + 1, wy + 1)]}; "
                    f"sidefront = {sf}; sideback = {sb}; twosided = true; special = 243; playercross = true; }}"
                )

    textmap = "\n".join(
        ['namespace = "zdoom";', *things_txt, *verts_txt, *linedefs_txt, *sidedefs_txt, *sectors_txt]
    ).encode("utf-8")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wad(out, [Lump("MAP01", b""), Lump("TEXTMAP", textmap), Lump("ENDMAP", b"")])
    return out


SCENARIOS = {
    # 1. Discover door -> open -> exit
    "prog_01_door_exit": """
###########
#P..D....E#
###########
""",
    # 2. Discover locked Blue door ahead, explore side room for Blue key, backtrack, unlock, exit
    "prog_02_key_backtrack": """
#############
#....#......#
#.b..#..B..E#
#....D......#
#..P.#......#
#############
""",
    # 3. Multi-room with enemy guarding keycard + locked Red door + exit
    "prog_03_ambush_key": """
#############
#P....D....E#
###.###R#####
#..z..#..h..#
#.r...D.....#
#############
""",
    # 4. Two-key progression (Blue key -> Blue door -> Red key -> Red door -> Exit)
    "prog_04_two_keys": """
###############
#P..#.....#...#
#.b.D..r..B.R.E#
#...#..z..#...#
###############
""",
}


def generate_all(out_dir: str | Path = "training/maps/generated") -> dict[str, Path]:
    out = Path(out_dir)
    return {name: compile_udmf_wad(grid, out / f"{name}.wad") for name, grid in SCENARIOS.items()}


if __name__ == "__main__":
    paths = generate_all()
    for name, p in paths.items():
        print(f"Generated {name}: {p}")
