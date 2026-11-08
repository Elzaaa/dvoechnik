"""Минимальный каркас self-play. Stockfish и готовые оценки намеренно не используются."""
from dataclasses import dataclass

import chess


@dataclass
class Step:
    fen: str
    move_uci: str
    reward: float
    done: bool


def random_self_play(max_plies: int = 200) -> list[Step]:
    """Проверяет, что среда умеет выдавать партии; это ещё не обучение."""
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
    print(f"Сыграно полуходов: {len(game)}")
    print("Следующий шаг: заменить первый легальный ход policy-сетью и self-play.")
