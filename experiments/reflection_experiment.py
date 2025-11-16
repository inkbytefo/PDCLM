## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.reflection import ReflectivePDCLM, reflective_train_step
from src.hcl import HCLAgent, CognitiveControlModule


def main():
    model = ReflectivePDCLM(embed_dim=512)
    if torch.cuda.is_available():
        model = model.cuda()
    ccm = CognitiveControlModule()
    agent = HCLAgent(model, ccm)
    losses: list[float] = []
    refls: list[float] = []
    task = "What is 37 * 24?"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(200):
        loss, refl = reflective_train_step(model, agent, task, max_steps=6, optimizer=optimizer)
        losses.append(loss)
        refls.append(refl)
    fig, ax1 = plt.subplots()
    ax1.plot(losses, 'b-')
    ax2 = ax1.twinx()
    ax2.plot(refls, 'r-')
    plt.title("Faz-3 Reflective Loss (blue) & Reflection Error (red)")
    plt.savefig("experiments/reflection_loss.png")
    final_refl = refls[-1]
    if final_refl < 0.25:
        print("Faz-3 TAMAM, Final Alignment.")
    else:
        print("Reflection dim artır (512), veya CoT uzunluğu 6 yap.")


if __name__ == "__main__":
    main()