from __future__ import annotations

import random
from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class Transition:
    observation: tuple[int, ...]
    action: int
    reward: float
    next_observation: tuple[int, ...]
    done: bool


class ChessEnvironment:
    """Small Gym-like chess environment with no engine evaluation."""

    action_size = 64 * 64

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.board = chess.Board()

    def reset(self) -> tuple[int, ...]:
        self.board.reset()
        return self.observe()

    def observe(self) -> tuple[int, ...]:
        """Encode 12 piece planes plus side-to-move into a flat tuple."""
        planes = [0] * (12 * 64 + 1)
        for square, piece in self.board.piece_map().items():
            piece_plane = (piece.piece_type - 1) + (0 if piece.color else 6)
            planes[piece_plane * 64 + square] = 1
        planes[-1] = int(self.board.turn)
        return tuple(planes)

    @staticmethod
    def encode_action(move: chess.Move) -> int:
        return move.from_square * 64 + move.to_square

    def legal_actions(self) -> list[int]:
        return [self.encode_action(move) for move in self.board.legal_moves]

    def step(self, action: int) -> tuple[tuple[int, ...], float, bool, dict]:
        from_square, to_square = divmod(action, 64)
        move = chess.Move(from_square, to_square)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal action: {action}")
        self.board.push(move)
        done = self.board.is_game_over()
        reward = 0.0
        if done:
            if self.board.is_checkmate():
                reward = 1.0
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                reward = 0.0
        return self.observe(), reward, done, {"fen": self.board.fen()}

    def random_action(self) -> int:
        return self.rng.choice(self.legal_actions())
