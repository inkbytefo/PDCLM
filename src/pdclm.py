## Developer: inkbytefo
## Modified: 2025-11-18

import torch
import torch.nn as nn
from typing import List, Tuple
import wandb
import torch.distributed as dist
import os

from src.model import PDCLMBase, pretrain_step as faz1_pretrain_step
from src.reflection import ReflectivePDCLM
from src.hcl import HCLAgent, CognitiveControlModule, task_generator, hcl_train_step
from src.utils import measure_forward_latency
from src.utils import compute_hallucination_penalty
from src.utils import save_checkpoint
from src.utils import build_sft_dataset
from src.utils import build_tool_augmented_sft_dataset
from src.utils import compute_hallucination_penalty
from src.tools import ToolExecutor, CalculatorTool, SearchTool


class PDCLM(ReflectivePDCLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ccm = CognitiveControlModule(dim=self.embed_dim)
        self.proposer = HCLAgent(self, self.ccm)
        self.critic = HCLAgent(self, self.ccm)

    def _evaluate_task(self, task: str) -> Tuple[float, float]:
        cot = self.proposer.generate_rule_based_cot(task, max_steps=8)
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
        final = cot[-1]
        ans = None
        if "Final answer:" in final:
            ans = final.split("Final answer:")[1].strip()
        correctness = 1.0 if self._check_answer(task, ans) else 0.0
        hallucination_penalty = compute_hallucination_penalty(task, ans)
        multi_reward = 0.25 * coherence + 0.5 * correctness + 0.25 * reflection_score - 0.05 * hallucination_penalty
        return loss, multi_reward

    def evaluate_with_answer(self, task: str) -> Tuple[float, float, str | None]:
        cot = self.proposer.generate_rule_based_cot(task, max_steps=8)
        loss, reward = self._evaluate_task(task)
        final = cot[-1]
        ans = None
        if "Final answer:" in final:
            ans = final.split("Final answer:")[1].strip()
        return loss, reward, ans

    def full_forward(self, task: str):
        return self._evaluate_task(task)

    def _check_answer(self, task: str, answer: str | None) -> bool:
        try:
            t = task.strip()
            if answer is None:
                return False
            if "What is" in t and "+" in t:
                expr = t.split("is")[1].split("?")[0]
                a, b = [int(x.strip()) for x in expr.split("+")]
                return int(answer) == (a + b)
            if "What is" in t and "*" in t:
                expr = t.split("is")[1].split("?")[0]
                a, b = [int(x.strip()) for x in expr.split("*")]
                return int(answer) == (a * b)
            if "even" in t:
                num = int(t.split("even?")[0].split()[-1])
                return (answer.lower() == "true") == (num % 2 == 0)
            if "Sort these numbers" in t:
                part = t.split(":")[1]
                arr = [int(x.strip()) for x in part.split(",")]
                parsed = [int(x.strip()) for x in answer.strip('[]').split(',')]
                return parsed == sorted(arr)
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
            loss, base_reward = model.full_forward(task)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            latency_ms = measure_forward_latency(model, task)
            lat_norm = min(latency_ms / 50.0, 1.0)
            reward = max(0.0, base_reward - 0.05 * lat_norm)
            rewards.append(reward)
            hmr_metrics = {}
            if hasattr(model, 'hmr'):
                hmr_metrics = {
                    "hmr/sim_max": float(model.hmr.last_sim_max.item()),
                    "hmr/ssm_age_max": float(model.hmr.ssm_age.max().item())
                }
            wandb.log({"reward": reward, "reward_base": base_reward, "loss": loss.item(), "epoch": epoch, "step": step, "latency_ms": latency_ms, **hmr_metrics})
        avg_reward = sum(rewards) / max(len(rewards), 1)
        scheduler.step(avg_reward)
        print(f"Epoch {epoch} Avg Reward: {avg_reward:.4f}")
        save_checkpoint(model, optimizer, epoch, float(loss.item()), f"checkpoints/pdclm_epoch_{epoch}.pt")
    save_checkpoint(model, optimizer, num_epochs, float(loss.item()), "checkpoints/pdclm_final.pt")


def instruction_sft_train(model: PDCLM, num_samples: int = 1000, epochs: int = 1, lr: float = 3e-4, use_tools: bool = False):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world > 1
    if ddp and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    if rank == 0:
        wandb.init(project="pdclm-sft")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    os.makedirs("checkpoints", exist_ok=True)
    ccm = model.ccm
    agent = HCLAgent(model, ccm)
    if use_tools:
        tools = setup_default_tools()
        samples = build_tool_augmented_sft_dataset(num_samples, agent, tools)
    else:
        samples = build_sft_dataset(num_samples, agent)
    if ddp:
        model_wrapped = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model_wrapped = model
    for epoch in range(epochs):
        total_loss = 0.0
        for i, s in enumerate(samples[rank::world] if ddp else samples):
            text = s["input"] + "\n" + s["target"]
            loss = model_wrapped(text)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            if rank == 0:
                wandb.log({"sft/loss": float(loss.item()), "epoch": epoch, "step": i})
        avg = total_loss / max(len(samples[rank::world]) if ddp else len(samples), 1)
        if rank == 0:
            print(f"SFT Epoch {epoch} Avg Loss: {avg:.4f}")
            save_checkpoint(model, optimizer, epoch, avg, f"checkpoints/pdclm_sft_epoch_{epoch}.pt")
    if rank == 0:
        save_checkpoint(model, optimizer, epochs, avg, "checkpoints/pdclm_sft_final.pt")


def evaluate_generated_text(model: PDCLM, task: str, text: str) -> float:
    win = getattr(model.pse, "window_size", 32)
    steps = []
    parts = text.split("\nStep ")
    for i, p in enumerate(parts):
        s = p if i == 0 else ("Step " + p)
        if len(s) < win:
            repeat = (win // max(len(s), 1)) + 1
            s = (s + " ") * repeat
        steps.append(s)
    cot_streams = [model.pse(step) for step in steps]
    coherence = model.ccm(cot_streams).item()
    loss, refl_loss, mean_logit, mean_target = model.reflective_forward(" ".join(steps), steps)
    reflection_score = 1.0 - float(mean_target.item())
    ans = None
    if "Final answer:" in text:
        ans = text.split("Final answer:")[-1].strip()
    correctness = 1.0 if model._check_answer(task, ans) else 0.0
    if "capital of Turkey" in task:
        if ans is not None and ("ankara" in ans.lower()):
            correctness = min(1.0, correctness + 0.5)
    hallucination_penalty = compute_hallucination_penalty(task, ans)
    tool_used = ("[TOOL_RESULT:" in text)
    tool_bonus = 0.05 if tool_used else 0.0
    if ("*" in task or "+" in task) and not tool_used:
        tool_bonus -= 0.03
    if "capital of Turkey" in task and not tool_used:
        tool_bonus -= 0.05
    reward = 0.25 * coherence + 0.5 * correctness + 0.25 * reflection_score - 0.05 * hallucination_penalty + tool_bonus
    return reward


def ppo_train(model: PDCLM, epochs: int = 1, tasks_per_epoch: int = 10, lr: float = 3e-4, clip: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.0, use_tools: bool = True, gen_max_bytes: int = 32, cot_max_steps: int = 3):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world > 1
    if ddp and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    if rank == 0:
        wandb.init(project="pdclm-ppo")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    os.makedirs("checkpoints", exist_ok=True)
    tools = setup_default_tools() if use_tools else None
    if ddp:
        model_wrapped = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model_wrapped = model
    for epoch in range(epochs):
        avg_loss = 0.0
        avg_reward = 0.0
        tool_calls = 0
        tool_errors = 0
        for _ in range(tasks_per_epoch):
            task = task_generator()
            if use_tools:
                agent = HCLAgent(model, model.ccm)
                cot = agent.generate_cot_with_tools(task, tools, max_steps=cot_max_steps)
                text = cot[-1]
                tool_calls += int("[TOOL_CALL:" in "\n".join(cot))
                tool_errors += 0
                prompt = task + "\nStep 1: "
                trace_text, trace = model_wrapped.module.generate_text(prompt, max_new_bytes=gen_max_bytes, return_traces=True) if ddp else model.generate_text(prompt, max_new_bytes=gen_max_bytes, return_traces=True)
            else:
                prompt = task + "\nStep 1: "
                text, trace = model_wrapped.module.generate_text(prompt, max_new_bytes=gen_max_bytes, return_traces=True) if ddp else model.generate_text(prompt, max_new_bytes=gen_max_bytes, return_traces=True)
            reward = evaluate_generated_text(model, task, text)
            new_lp, new_vals = (model_wrapped.module if ddp else model).evaluate_sequence_logprobs(prompt, trace["actions"])
            old_lp = torch.tensor(trace["logprobs"], dtype=torch.float32)
            old_vals = torch.tensor(trace["values"], dtype=torch.float32)
            R = torch.full_like(new_vals, float(reward))
            adv = R.detach() - old_vals
            ratios = torch.exp(new_lp - old_lp)
            clipped = torch.clamp(ratios, 1.0 - clip, 1.0 + clip)
            policy_loss = -(torch.min(ratios * adv, clipped * adv)).mean()
            value_loss = torch.nn.functional.mse_loss(new_vals, R)
            loss = policy_loss + vf_coef * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            avg_loss += float(loss.item())
            avg_reward += float(reward)
            if rank == 0:
                wandb.log({"ppo/loss": float(loss.item()), "ppo/reward": float(reward), "tools/calls": tool_calls, "tools/errors": tool_errors})
        avg_loss /= max(tasks_per_epoch, 1)
        avg_reward /= max(tasks_per_epoch, 1)
        if rank == 0:
            print(f"PPO Epoch {epoch} loss={avg_loss:.4f} reward={avg_reward:.4f}")
            save_checkpoint(model, optimizer, epoch, avg_loss, f"checkpoints/pdclm_ppo_epoch_{epoch}.pt")
    if rank == 0:
        save_checkpoint(model, optimizer, epochs, avg_loss, "checkpoints/pdclm_ppo_final.pt")


def setup_default_tools() -> ToolExecutor:
    execu = ToolExecutor()
    execu.register("calculator", CalculatorTool())
    index_dir = "data/index"
    if os.path.exists(os.path.join(index_dir, "faiss.index")):
        execu.register("search", SearchTool(kb_text="", index_dir=index_dir))
        return execu
    kb = ""
    try:
        with open("data/raw/wikitext_sample.txt", "r", encoding="utf-8") as f:
            kb = f.read()
    except Exception:
        kb = "Ankara is the capital of Turkey. Istanbul is the largest city in Turkey."
    execu.register("search", SearchTool(kb_text=kb))
    return execu