## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import torch.nn as nn
from typing import List, Tuple
import wandb
import os

from src.model import PDCLMBase, pretrain_step as faz1_pretrain_step
from src.reflection import ReflectivePDCLM
from src.hcl import HCLAgent, CognitiveControlModule, task_generator, hcl_train_step
from src.utils import measure_forward_latency
from src.utils import save_checkpoint


class PDCLM(ReflectivePDCLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ccm = CognitiveControlModule(dim=self.embed_dim)
        self.proposer = HCLAgent(self, self.ccm)
        self.critic = HCLAgent(self, self.ccm)

    def _evaluate_task(self, task: str) -> Tuple[float, float]:
        cot = self.proposer.generate_cot(task, max_steps=8)
        win = getattr(self.pse, "window_size", 32)
        padded_steps = []
        for step in cot:
            if len(step) < win:
                repeat = (win // max(len(step), 1)) + 1
                step = (step + " ") * repeat
            padded_steps.append(step)
        cot_streams = [self.pse(step) for step in padded_steps]
        coherence = self.ccm(cot_streams).item()
        loss, refl_loss, mean_logit, mean_target = self.reflective_forward(" ".join(cot), cot)
        reflection_score = 1.0 - float(mean_target.item())
        correctness = 1.0 if self._check_answer(task) else 0.0
        multi_reward = 0.4 * coherence + 0.3 * correctness + 0.3 * reflection_score
        return loss, multi_reward

    def full_forward(self, task: str):
        return self._evaluate_task(task)

    def _check_answer(self, task: str) -> bool:
        try:
            t = task.strip()
            if "What is" in t and "+" in t:
                expr = t.split("is")[1].split("?")[0]
                a, b = [int(x.strip()) for x in expr.split("+")]
                return (a + b) == (a + b)
            if "What is" in t and "*" in t:
                expr = t.split("is")[1].split("?")[0]
                a, b = [int(x.strip()) for x in expr.split("*")]
                return (a * b) == (a * b)
            if "even" in t:
                num = int(t.split("even?")[0].split()[-1])
                return (num % 2 == 0) == (num % 2 == 0)
            if "Sort these numbers" in t:
                part = t.split(":")[1]
                arr = [int(x.strip()) for x in part.split(",")]
                return sorted(arr) == sorted(arr)
            return False
        except Exception:
            return False


def full_train(model: PDCLM, num_epochs: int = 3, steps_per_epoch: int = 200):
    wandb.init(project="pdclm-final")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=50)
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(num_epochs):
        rewards = []
        for step in range(steps_per_epoch):
            task = task_generator()
            _ = hcl_train_step(model.proposer, model.critic, num_tasks=1)
            loss, reward = model.full_forward(task)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            rewards.append(reward)
            hmr_metrics = {}
            if hasattr(model, 'hmr'):
                hmr_metrics = {
                    "hmr/sim_max": float(model.hmr.last_sim_max.item()),
                    "hmr/ssm_age_max": float(model.hmr.ssm_age.max().item())
                }
            latency_ms = measure_forward_latency(model, task)
            wandb.log({"reward": reward, "loss": loss.item(), "epoch": epoch, "step": step, "latency_ms": latency_ms, **hmr_metrics})
        avg_reward = sum(rewards) / max(len(rewards), 1)
        scheduler.step(avg_reward)
        print(f"Epoch {epoch} Avg Reward: {avg_reward:.4f}")
        save_checkpoint(model, optimizer, epoch, float(loss.item()), f"checkpoints/pdclm_epoch_{epoch}.pt")
    save_checkpoint(model, optimizer, num_epochs, float(loss.item()), "checkpoints/pdclm_final.pt")