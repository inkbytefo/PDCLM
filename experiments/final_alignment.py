## Developer: inkbytefo
## Modified: 2025-11-17

import sys
import os
import argparse
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pdclm import PDCLM, full_train
from src.hcl import task_generator


def evaluate(model: PDCLM, samples: int = 50) -> float:
    rewards = []
    for _ in range(samples):
        task = task_generator()
        _, reward = model.full_forward(task)
        rewards.append(reward)
    return sum(rewards) / max(len(rewards), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()

    model = PDCLM(embed_dim=args.embed_dim)
    if torch.cuda.is_available():
        model = model.cuda()

    full_train(model, num_epochs=args.epochs, steps_per_epoch=args.steps)
    avg_reward = evaluate(model, samples=50)
    print(f"Final Avg Reward: {avg_reward:.4f}")
    if avg_reward > 0.85:
        print("PD-CLM Prototip TAMAM, Deployment Hazır.")
    else:
        print("Epoch artır, dim=512 yap.")


if __name__ == '__main__':
    main()