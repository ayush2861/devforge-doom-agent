"""Modular lightweight CNN encoder for (120, 160, 3) RGB observations."""

from __future__ import annotations

import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """Optional lightweight RGB CNN. Can be disabled (enabled=False) for pure telemetry training."""

    def __init__(self, out_dim: int = 128, enabled: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.enabled = enabled
        if enabled:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=5, stride=4, padding=2),
                nn.SiLU(),
                nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((3, 4)),
            )
            self.proj = nn.Sequential(
                nn.Linear(64 * 3 * 4, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU(),
            )

    def forward(self, rgb: torch.Tensor | None) -> torch.Tensor:
        if not self.enabled:
            if rgb is None:
                raise ValueError("rgb or batch reference required")
            return torch.zeros(rgb.shape[0], self.out_dim, device=rgb.device, dtype=torch.float32)

        x = rgb.float()
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # (B, H, W, 3) -> (B, 3, H, W)
        if x.max() > 1.0:
            x = x / 255.0
        return self.proj(self.conv(x).flatten(1))
