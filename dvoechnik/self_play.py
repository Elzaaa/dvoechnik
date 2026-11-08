from __future__ import annotations

from .environment import ChessEnvironment, Transition


def play_random_game(seed: int | None = None, max_plies: int = 200) -> list[Transition]:
    """Produce a baseline game; this validates the environment before training."""
    env = ChessEnvironment(seed=seed)
    observation = env.reset()
    game: list[Transition] = []
    for _ in range(max_plies):
        action = env.random_action()
        next_observation, reward, done, _ = env.step(action)
        game.append(Transition(observation, action, reward, next_observation, done))
        observation = next_observation
        if done:
            break
    return game


if __name__ == "__main__":
    game = play_random_game(seed=0)
    print(f"Generated {len(game)} plies.")
