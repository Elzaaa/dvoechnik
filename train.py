"""Minimal self-play scaffold. Stockfish and precomputed evaluations are intentionally excluded."""
from dataclasses import dataclass

import chess


@dataclass
class Step:
    fen: str
    move_uci: str
    reward: float
    done: bool


def random_self_play(max_plies: int = 200) -> list[Step]:
    """Verify that the environment can produce games; this is not training yet."""
    board = chess.Board()
    history: list[Step] = []
    for _ in range(max_plies):
        if board.is_game_over():
            break
        move = next(iter(board.legal_moves))
        before = board.fen()
        board.push(move)
        done = board.is_game_over()
        reward = 0.0
        if done and board.is_checkmate():
            reward = 1.0
        history.append(Step(before, move.uci(), reward, done))
    return history


if __name__ == "__main__":
    game = random_self_play()
    print(f"Plies played: {len(game)}")
    print("Next step: replace the first legal move with a policy network and self-play.")
