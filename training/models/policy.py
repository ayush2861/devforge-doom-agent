"""Hierarchical Actor (Strategic + 9-Independent-Bernoulli MultiBinary(9)) and Critic."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Independent

from doom_env import spaces as S

STRATEGIES = (
    "exploration",
    "combat",
    "item_collection",
    "door_progression",
    "exit_seeking",
)
NUM_STRATEGIES = len(STRATEGIES)
NUM_ACTIONS = len(S.BUTTONS)  # 9


class HierarchicalPolicy(nn.Module):
    """Two-level policy: Strategic mode selector -> 9 independent Bernoulli Tactical head."""

    def __init__(self, hidden_dim: int = 256, obs_skip_dim: int = 0):
        super().__init__()
        self.obs_skip_dim = obs_skip_dim
        in_dim = hidden_dim + obs_skip_dim
        self.strategic_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, NUM_STRATEGIES),
        )
        self.tactical_head = nn.Sequential(
            nn.Linear(in_dim + NUM_STRATEGIES, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, NUM_ACTIONS),
        )
        # Low initial bias on ATTACK(6), USE(7), SELECT_NEXT_WEAPON(8) to avoid blind spam at step 0
        with torch.no_grad():
            self.tactical_head[-1].bias[6] = -1.8
            self.tactical_head[-1].bias[7] = -2.0
            self.tactical_head[-1].bias[8] = -3.0

    def forward(
        self, memory_out: torch.Tensor, obs_skip: torch.Tensor | None = None
    ) -> tuple[Independent, Categorical, torch.Tensor, torch.Tensor]:
        """Return (tactical_dist, strategic_dist, action_logits, strategy_probs).

        tactical_dist is Independent(Bernoulli(logits), 1): joint log_prob is the exact
        sum of the 9 independent Bernoulli log probabilities.
        """
        feat = torch.cat([memory_out, obs_skip], dim=-1) if (self.obs_skip_dim > 0 and obs_skip is not None) else memory_out
        strategy_logits = self.strategic_head(feat)
        strategy_probs = F.softmax(strategy_logits, dim=-1)
        strategic_dist = Categorical(logits=strategy_logits)

        tactical_in = torch.cat([feat, strategy_probs], dim=-1)
        action_logits = self.tactical_head(tactical_in)
        tactical_dist = Independent(Bernoulli(logits=action_logits), 1)
        return tactical_dist, strategic_dist, action_logits, strategy_probs


class ValueNetwork(nn.Module):
    """Critic estimating expected discounted return V(s_t)."""

    def __init__(self, hidden_dim: int = 256, obs_skip_dim: int = 0):
        super().__init__()
        self.obs_skip_dim = obs_skip_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + obs_skip_dim + NUM_STRATEGIES, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self, memory_out: torch.Tensor, strategy_probs: torch.Tensor, obs_skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        parts = [memory_out, obs_skip, strategy_probs] if (self.obs_skip_dim > 0 and obs_skip is not None) else [memory_out, strategy_probs]
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)
