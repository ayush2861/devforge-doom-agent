"""Official evaluation entry point for the ESRHA DOOM agent.

Contract: docs/CONTRACT.md §6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from training.models.doom_model import ESRHAAgentModel

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SUBMISSION_DIR / "weights" / "esrha_best.pt"


class DoomAgent:
    """Ego-Spatial Recurrent Hierarchical Agent (ESRHA) for zero-shot DOOM evaluation."""

    def __init__(self, observation_space, action_space, device: str):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)

        if self.device.type == "cpu":
            torch.set_num_threads(1)

        ckpt = torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=False)
        mcfg = ckpt.get("model_config", {})
        self.model = ESRHAAgentModel(
            hidden_dim=int(mcfg.get("hidden_dim", 384)),
            use_visual=bool(mcfg.get("use_visual", False)),
            use_depth=bool(mcfg.get("use_depth", True)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.reset()

    def reset(self) -> None:
        """Clear GRU + episodic slot memory and previous-action state between maps."""
        self.model.reset_state(batch_size=1, device=self.device)

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        """Run ESRHA inference and return an int8 (9,) binary action."""
        action = self.model.act(observation, deterministic=False)
        return np.asarray(action, dtype=np.int8).reshape(self.action_space.shape)
