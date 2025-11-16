## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.reflection import ReflectivePDCLM, reflective_train_step
from src.hcl import HCLAgent, CognitiveControlModule, task_generator


def main():
    model = ReflectivePDCLM(embed_dim=512)
    if torch.cuda.is_available():
        model = model.cuda()
    ccm = CognitiveControlModule()
    agent = HCLAgent(model, ccm)
    losses: list[float] = []
    pred_errors: list[float] = []
    target_errors: list[float] = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50)
    for _ in range(500):
        t = task_generator()
        loss, refl, mean_logit, mean_target = reflective_train_step(model, agent, t, max_steps=8, optimizer=optimizer)
        losses.append(loss)
        pred_errors.append(torch.sigmoid(torch.tensor(mean_logit)).item())
        target_errors.append(mean_target)
        scheduler.step(mean_target)
    fig, ax1 = plt.subplots()
    ax1.plot(losses, 'b-')
    ax2 = ax1.twinx()
    ax2.plot(pred_errors, 'r-')
    ax2.plot(target_errors, 'g-')
    plt.title("Faz-3 Reflective Loss (blue), Pred Error (red), Target Error (green)")
    plt.savefig("experiments/reflection_loss_improved.png")
    last50 = target_errors[-50:]
    avg_last50 = sum(last50) / len(last50)
    if avg_last50 < 0.20:
        print("Faz-3 TAMAM, PD-CLM Prototip Hazır.")
    else:
        print("CoT çeşitliliği artır, dim=512 yap.")


if __name__ == "__main__":
    main()