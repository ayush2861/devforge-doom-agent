"""Player vitals decoding and formatting using VITALS_INDEX."""

from __future__ import annotations

import numpy as np

from doom_env import spaces as S


def decode_vitals(vitals: np.ndarray) -> dict[str, float]:
    """Decode all 17 vitals entries using the official VITALS_INDEX mapping."""
    v = np.asarray(vitals, dtype=np.float32)
    if v.shape != (len(S.VITALS_FIELDS),):
        raise ValueError(f"Expected vitals shape {(len(S.VITALS_FIELDS),)}, got {v.shape}")
    return {name: float(v[idx]) for name, idx in S.VITALS_INDEX.items()}


def format_vitals(vitals: np.ndarray) -> str:
    """Human-readable summary of health, armor, weapon/ammo, keys, and ego velocity."""
    d = decode_vitals(vitals)
    held_keys = [
        k.replace("key_", "")
        for k in ("key_blue_card", "key_yellow_card", "key_red_card", "key_blue_skull", "key_yellow_skull", "key_red_skull")
        if d[k] > 0.5
    ]
    return (
        f"Vitals(17): HP={d['health']:.0f} Armor={d['armor']:.0f} | "
        f"Weapon={int(d['selected_weapon'])} (ammo={d['selected_weapon_ammo']:.0f}) | "
        f"Ammo[bul={d['ammo_bullets']:.0f}, shl={d['ammo_shells']:.0f}, rkt={d['ammo_rockets']:.0f}, cel={d['ammo_cells']:.0f}] | "
        f"Keys={held_keys or 'none'} | "
        f"Vel(fwd={d['vel_forward']:+.2f}, right={d['vel_right']:+.2f}, up={d['vel_up']:+.2f})"
    )
