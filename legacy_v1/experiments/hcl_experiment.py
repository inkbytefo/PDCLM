## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import PDCLMBase
from src.hcl import CognitiveControlModule, HCLAgent, hcl_train_step


def main():
    model = PDCLMBase(embed_dim=256, window_size=32)
    ccm = CognitiveControlModule()
    proposer = HCLAgent(model, ccm)
    critic = HCLAgent(model, ccm)
    rewards: list[float] = []
    for _ in range(50):
        reward = hcl_train_step(proposer, critic, num_tasks=10)
        rewards.append(reward)
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.title("Faz-2 Self-Play Reward")
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.tight_layout()
    plt.savefig("experiments/hcl_reward.png")
    avg = sum(rewards) / len(rewards)
    if avg > 0.75:
        print("Faz-2 çalışıyor, Reflection'a geç.")
    elif avg > 0.7:
        print("Faz-2 çalışıyor, Reflection'a geç.")
    else:
        print("Task çeşitlendir, CoT decode ekle.")


if __name__ == "__main__":
    main()