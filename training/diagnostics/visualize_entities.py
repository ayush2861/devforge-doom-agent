"""Visible entity decoding and ego-centric top-down visualization."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from doom_env import spaces as S
from training.diagnostics.visualize_depth import write_png
from training.diagnostics.visualize_rays import draw_circle, draw_line

ENTITY_COLORS: dict[str, tuple[int, int, int]] = {
    "enemy_melee": (255, 60, 60),
    "enemy_hitscan": (255, 110, 40),
    "enemy_projectile": (255, 40, 160),
    "projectile": (255, 230, 50),
    "health": (60, 230, 100),
    "armor": (70, 170, 255),
    "ammo": (220, 180, 90),
    "weapon": (180, 120, 255),
    "key": (255, 255, 80),
    "barrel": (180, 100, 60),
    "other": (180, 180, 180),
}


def decode_entities(visible_entities: dict) -> list[dict]:
    """Decode active visible entity slots using the official binary mask."""
    feats = np.asarray(visible_entities["features"], dtype=np.float32)
    etypes = np.asarray(visible_entities["entity_type"], dtype=np.int64)
    mask = np.asarray(visible_entities["mask"], dtype=np.int8)

    decoded = []
    for i in range(S.MAX_ENTITIES):
        if mask[i] == 0:
            continue
        dist_norm, sin_a, cos_a, vf, vr = feats[i]
        angle_rad = math.atan2(float(sin_a), float(cos_a))
        decoded.append(
            {
                "slot": i,
                "type": S.ENTITY_TYPES[int(etypes[i])],
                "type_id": int(etypes[i]),
                "distance_norm": float(dist_norm),
                "distance_units": float(dist_norm * S.RAY_RANGE),
                "sin_rel_angle": float(sin_a),
                "cos_rel_angle": float(cos_a),
                "rel_angle_deg": math.degrees(angle_rad),
                "rel_vel_forward": float(vf),
                "rel_vel_right": float(vr),
            }
        )
    return decoded


def render_entities_topdown(visible_entities: dict, size: int = 360, max_range: float = 1024.0) -> np.ndarray:
    """Render ego-centric top-down FOV plot of masked visible entities (player at bottom-center)."""
    canvas = np.full((size, size, 3), 18, dtype=np.uint8)
    px, py = size // 2, size - 24
    scale = (size - 48) / max_range

    # 90-degree FOV boundary lines
    fov_len = int(max_range * scale)
    for sign in (-1, 1):
        ex = int(round(px + sign * math.sin(math.radians(45)) * fov_len))
        ey = int(round(py - math.cos(math.radians(45)) * fov_len))
        draw_line(canvas, px, py, ex, ey, (65, 65, 90))

    # Range arcs (256, 512, 768, 1024)
    for r_units in (256, 512, 768, 1024):
        draw_circle(canvas, px, py, int(r_units * scale), (40, 44, 52), fill=False)

    for ent in decode_entities(visible_entities):
        # rel_angle > 0 is to the player's left
        d_px = min(ent["distance_units"], max_range) * scale
        ex = int(round(px - ent["sin_rel_angle"] * d_px))
        ey = int(round(py - ent["cos_rel_angle"] * d_px))
        color = ENTITY_COLORS.get(ent["type"], (220, 220, 220))
        draw_circle(canvas, ex, ey, 5, color, fill=True)
        # Velocity vector (scaled for visibility)
        vx = int(round(ex + ent["rel_vel_right"] * 2.0))
        vy = int(round(ey - ent["rel_vel_forward"] * 2.0))
        draw_line(canvas, ex, ey, vx, vy, (255, 255, 255))

    draw_circle(canvas, px, py, 5, (255, 255, 255), fill=True)
    return canvas


def format_entities_table(visible_entities: dict) -> str:
    ents = decode_entities(visible_entities)
    if not ents:
        return f"Visible entities: 0 / {S.MAX_ENTITIES}"
    lines = [f"Visible entities: {len(ents)} / {S.MAX_ENTITIES}"]
    for e in ents:
        lines.append(
            f"  [{e['slot']:2d}] {e['type']:16s} dist={e['distance_units']:6.1f}u "
            f"ang={e['rel_angle_deg']:+6.1f}deg v_fwd={e['rel_vel_forward']:+5.2f} v_r={e['rel_vel_right']:+5.2f}"
        )
    return "\n".join(lines)


def save_entities_plot(visible_entities: dict, out_dir: Path, tag: str) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = render_entities_topdown(visible_entities)
    write_png(out_dir / f"{tag}_entities.png", img)
    return decode_entities(visible_entities)
