## Developer: inkbytefo
## Modified: 2025-11-17

import sys
import os
import argparse
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pdclm import PDCLM
from src.hcl import task_generator
from src.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--ckpt', type=str, default='checkpoints/pdclm_final.pt')
    parser.add_argument('--samples', type=int, default=20)
    args = parser.parse_args()

    model = PDCLM(embed_dim=args.embed_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if os.path.exists(args.ckpt):
        load_checkpoint(model, optimizer, args.ckpt)
    if torch.cuda.is_available():
        model = model.cuda()

    rewards = []
    for _ in range(args.samples):
        task = task_generator()
        _, reward = model.full_forward(task)
        rewards.append(reward)
    avg = sum(rewards) / max(len(rewards), 1)
    print(f"Inference Avg Reward: {avg:.4f}")


if __name__ == '__main__':
    main()