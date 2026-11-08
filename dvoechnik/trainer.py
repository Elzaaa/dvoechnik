from __future__ import annotations

import torch
from torch import optim

from .agent import NeuralAgent
from .environment import ChessEnvironment
from .model import PolicyValueNet


def train_episode(model: PolicyValueNet, optimizer: optim.Optimizer, max_plies: int = 200) -> tuple[float, str]:
    """Run one self-play game and train from its final outcome.

    Both colors use the same network. Rewards are assigned from each player's
    perspective after the game ends: win=+1, loss=-1, draw=0.
    """
    env = ChessEnvironment()
    observations: list[tuple[int, ...]] = []
    actions: list[int] = []
    players: list[bool] = []
    agent = NeuralAgent(model)

    for _ in range(max_plies):
        observations.append(env.observe())
        players.append(env.board.turn)
        action = agent.select_action(env)
        actions.append(action)
        _, _, done, _ = env.step(action)
        if done:
            break

    if env.board.is_checkmate():
        winner = not env.board.turn
        rewards = [1.0 if player == winner else -1.0 for player in players]
        result = "white" if winner else "black"
    else:
        rewards = [0.0] * len(players)
        result = "draw"

    states = torch.tensor(observations, dtype=torch.float32)
    action_tensor = torch.tensor(actions)
    target = torch.tensor(rewards, dtype=torch.float32)
    logits, values = model(states)
    log_prob = torch.log_softmax(logits, dim=-1).gather(1, action_tensor[:, None]).squeeze(1)
    loss = -(log_prob * target).mean() + (values - target).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item()), result


if __name__ == "__main__":
    network = PolicyValueNet()
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    loss, result = train_episode(network, optimizer)
    print(f"Completed one self-play training episode. Result: {result}. Loss: {loss:.4f}")
