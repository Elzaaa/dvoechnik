from __future__ import annotations

import torch

from .environment import ChessEnvironment
from .model import PolicyValueNet


class NeuralAgent:
    def __init__(self, model: PolicyValueNet, temperature: float = 1.0) -> None:
        self.model = model
        self.temperature = temperature

    def select_action(self, env: ChessEnvironment, explore: bool = True) -> int:
        observation = torch.tensor(env.observe(), dtype=torch.float32).unsqueeze(0)
        logits, _ = self.model(observation)
        legal = env.legal_actions()
        mask = torch.full_like(logits, float("-inf"))
        mask[0, legal] = logits[0, legal]
        if explore:
            probabilities = torch.softmax(mask / self.temperature, dim=-1)
            return int(torch.multinomial(probabilities[0], 1).item())
        return int(torch.argmax(mask, dim=-1).item())
