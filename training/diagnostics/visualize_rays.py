"""64-ray ego-centric raycast decoding and polar visualization."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from doom_env import spaces as S
from training.diagnostics.visualize_depth import write_png

# RGB colors per RAY_HIT_TYPES index:
# 0: none, 1: wall, 2: door, 3: locked_door, 4: enemy, 5: item
HIT_COLORS: dict[int, tuple[int, int, int]] = {
    0: (70, 70, 80),       # none
    1: (220, 220, 230),    # wall
    2: (0, 210, 255),      # door (cyan)
    3: (255, 110, 0),      # locked_door (orange)
    4: (255, 45, 45),      # enemy (red)
    5: (50, 235, 90),      # item (green)
}


def draw_line(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    """Bresenham line drawing on an (H, W, 3) uint8 canvas."""
    h, w, _ = canvas.shape
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            canvas[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_circle(canvas: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int], fill: bool = False) -> None:
    """Draw a circle outline or filled disc on an (H, W, 3) uint8 canvas."""
    h, w, _ = canvas.shape
    r2 = radius * radius
    y_min = max(0, cy - radius)
    y_max = min(h - 1, cy + radius)
    x_min = max(0, cx - radius)
    x_max = min(w - 1, cx + radius)
    for y in range(y_min, y_max + 1):
        dy = y - cy
        for x in range(x_min, x_max + 1):
            dx = x - cx
            d2 = dx * dx + dy * dy
            if fill:
                if d2 <= r2:
                    canvas[y, x] = color
            else:
                if abs(math.sqrt(d2) - radius) <= 0.7:
                    canvas[y, x] = color


def decode_rays(rays: dict) -> dict:
    """Decode normalized ray distances and hit types into summary metrics."""
    dist_norm = np.asarray(rays["distance"], dtype=np.float32)
    hit_type = np.asarray(rays["hit_type"], dtype=np.int64)
    dist_units = dist_norm * S.RAY_RANGE
    counts = {name: int((hit_type == i).sum()) for i, name in enumerate(S.RAY_HIT_TYPES)}

    # Forward 90-degree cone: indices [0..8] and [56..63]
    fwd_idx = np.r_[np.arange(0, 9), np.arange(56, 64)]
    fwd_min_idx = int(fwd_idx[np.argmin(dist_norm[fwd_idx])])
    overall_min_idx = int(np.argmin(dist_norm))

    return {
        "counts": counts,
        "min_dist_units": float(dist_units[overall_min_idx]),
        "min_hit_type": S.RAY_HIT_TYPES[int(hit_type[overall_min_idx])],
        "fwd_min_dist_units": float(dist_units[fwd_min_idx]),
        "fwd_min_hit_type": S.RAY_HIT_TYPES[int(hit_type[fwd_min_idx])],
        "front_ray_dist_units": float(dist_units[0]),
        "front_ray_hit_type": S.RAY_HIT_TYPES[int(hit_type[0])],
    }


def render_rays_polar(rays: dict, size: int = 360) -> np.ndarray:
    """Render an ego-centric 360° polar plot (player at center, facing UP)."""
    canvas = np.full((size, size, 3), 18, dtype=np.uint8)
    cx = cy = size // 2
    max_r = int(size * 0.45)

    # Range rings (256, 512, 768, 1024 map units)
    for frac in (0.25, 0.5, 0.75, 1.0):
        draw_circle(canvas, cx, cy, int(max_r * frac), (45, 48, 58), fill=False)

    # 90-degree FOV guide lines (+/- 45 deg from UP)
    for ang in (math.radians(45), math.radians(-45)):
        ex = int(round(cx - math.sin(ang) * max_r))
        ey = int(round(cy - math.cos(ang) * max_r))
        draw_line(canvas, cx, cy, ex, ey, (65, 65, 90))

    dist = np.asarray(rays["distance"], dtype=np.float32)
    hits = np.asarray(rays["hit_type"], dtype=np.int64)
    for i in range(S.NUM_RAYS):
        # Contract: index 0 = straight ahead (UP), counter-clockwise (index 16 = LEFT)
        ang = i * (2.0 * math.pi / S.NUM_RAYS)
        r = float(dist[i]) * max_r
        ex = int(round(cx - math.sin(ang) * r))
        ey = int(round(cy - math.cos(ang) * r))
        color = HIT_COLORS.get(int(hits[i]), (255, 255, 255))
        draw_line(canvas, cx, cy, ex, ey, tuple(max(20, c // 3) for c in color))
        draw_circle(canvas, ex, ey, 3, color, fill=True)

    # Player marker at center
    draw_circle(canvas, cx, cy, 5, (255, 255, 255), fill=True)
    return canvas


def format_rays_summary(rays: dict) -> str:
    info = decode_rays(rays)
    counts_str = ", ".join(f"{k}={v}" for k, v in info["counts"].items() if v > 0)
    return (
        f"Rays(64): hits=[{counts_str}] | "
        f"front={info['front_ray_hit_type']}@{info['front_ray_dist_units']:.1f}u | "
        f"fwd90_min={info['fwd_min_hit_type']}@{info['fwd_min_dist_units']:.1f}u | "
        f"all_min={info['min_hit_type']}@{info['min_dist_units']:.1f}u"
    )


def save_rays_plot(rays: dict, out_dir: Path, tag: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = render_rays_polar(rays)
    write_png(out_dir / f"{tag}_rays.png", img)
    return decode_rays(rays)
