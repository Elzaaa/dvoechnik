from __future__ import annotations

import torch
from torch import optim

from .agent import NeuralAgent
from .environment import ChessEnvironment
from .model import PolicyValueNet


def train_episode(model: PolicyValueNet, max_plies: int = 200) -> float:
    """Run one self-play episode and update the shared model."""
    env = ChessEnvironment()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    observations: list[tuple[int, ...]] = []
    actions: list[int] = []
    rewards: list[float] = []
    agent = NeuralAgent(model)

    for _ in range(max_plies):
        observations.append(env.observe())
        action = agent.select_action(env)
        actions.append(action)
        _, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    states = torch.tensor(observations, dtype=torch.float32)
    action_tensor = torch.tensor(actions)
    target = torch.tensor(rewards, dtype=torch.float32)
    logits, values = model(states)
    log_prob = torch.log_softmax(logits, dim=-1).gather(1, action_tensor[:, None]).squeeze(1)
    loss = -(log_prob * target).mean() + (values - target).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


if __name__ == "__main__":
    network = PolicyValueNet()
    loss = train_episode(network)
    print(f"Completed one self-play training episode. Loss: {loss:.4f}")
