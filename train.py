from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch import optim

from dvoechnik.model import PolicyValueNet
from dvoechnik.trainer import train_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dvoechnik through self-play.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = PolicyValueNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    results: Counter[str] = Counter()
    for episode in range(1, args.episodes + 1):
        loss, result = train_episode(model, optimizer, max_plies=args.max_plies)
        results[result] += 1
        if episode == 1 or episode % 10 == 0:
            print(f"episode={episode} loss={loss:.4f} result={result} stats={dict(results)}")
        if episode % args.checkpoint_every == 0 or episode == args.episodes:
            path = args.out / f"dvoechnik_{episode:06d}.pt"
            torch.save({"episode": episode, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "results": dict(results)}, path)
            print(f"saved={path}")


if __name__ == "__main__":
    main()
