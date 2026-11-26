"""Depth buffer and RGB observation visualization utilities."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: str | Path, rgb: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 RGB array to a PNG file using only Python stdlib."""
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, c = arr.shape
    if c != 3:
        raise ValueError(f"Expected (H, W, 3) RGB array, got {arr.shape}")

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, level=6)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Map (H, W) uint8 depth buffer to a high-contrast RGB colormap (near=warm/bright, far=cool/dark)."""
    d = depth.astype(np.float32) / 255.0
    # Invert so closer surfaces are bright/warm (yellow-red) and distant surfaces are deep blue
    inv = 1.0 - d
    r = np.clip(1.5 - np.abs(4.0 * inv - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * inv - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * inv - 1.0), 0.0, 1.0)
    # Near-field boost for very close walls
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def analyze_depth(depth: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for the (120, 160) uint8 depth buffer."""
    return {
        "min": int(depth.min()),
        "max": int(depth.max()),
        "mean": float(depth.mean()),
        "p10": float(np.percentile(depth, 10)),
        "p50": float(np.percentile(depth, 50)),
        "p90": float(np.percentile(depth, 90)),
        "near_frac_lt_20": float((depth < 20).mean()),
        "far_frac_gt_100": float((depth > 100).mean()),
    }


def save_rgb_and_depth(obs: dict, out_dir: Path, tag: str) -> dict[str, float]:
    """Save raw 120x160 RGB, 120x160 depth colormap, and side-by-side composite."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = obs["rgb"]
    depth = obs["depth"]
    depth_rgb = depth_to_rgb(depth)

    write_png(out_dir / f"{tag}_rgb.png", rgb)
    write_png(out_dir / f"{tag}_depth.png", depth_rgb)
    # Side-by-side preserving exact 120px height and 160px per panel
    sep = np.full((rgb.shape[0], 4, 3), 40, dtype=np.uint8)
    composite = np.concatenate([rgb, sep, depth_rgb], axis=1)
    write_png(out_dir / f"{tag}_rgb_depth.png", composite)
    return analyze_depth(depth)
