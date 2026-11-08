# Dvoechnik

A neural network that learns chess from scratch — with the confidence of a failing student.

## Concept

Dvoechnik does not receive Stockfish games, an opening book, or precomputed chess evaluations.
It learns by interacting with an environment:

1. observes the board state;
2. chooses an action;
3. receives a new state and a reward;
4. plays thousands of games against itself;
5. slowly learns to distinguish a good move from a spectacular blunder.

At the first stage, chess rules are provided by the environment, not by the neural network. This is not cheating: without a correct environment, the agent cannot understand when a game ends, which moves are illegal, or why its queen suddenly disappeared.

## Learning roadmap

- **Stage 0 — board:** position visualization and manual moves.
- **Stage 1 — environment:** legal moves, promotion, check, checkmate, stalemate, and draws.
- **Stage 2 — baseline agent:** random play and a verified self-play loop.
- **Stage 3 — training:** a policy/value network, illegal-move masking, and reinforcement learning.
- **Stage 4 — search:** MCTS, inspired by AlphaZero, without a ready-made engine.
- **Stage 5 — evaluation:** matches against random agents, old checkpoints, and humans.

## Running the board preview

Open `board/index.html` in a browser. It is only a visual preview; the training system is Python-first.

## Status

The project is at Stage 0. No ready-made chess bot is used for training.

