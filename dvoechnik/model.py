from __future__ import annotations

import torch
from torch import nn


class PolicyValueNet(nn.Module):
    """Small policy/value network for the first self-play experiments."""

    def __init__(self, observation_size: int = 769, action_size: int = 4096) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy = nn.Linear(256, action_size)
        self.value = nn.Sequential(nn.Linear(256, 1), nn.Tanh())

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(observation.float())
        return self.policy(hidden), self.value(hidden).squeeze(-1)
