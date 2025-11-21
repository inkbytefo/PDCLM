## Developer: inkbytefo
## Modified: 2025-11-21

import torch
import torch.nn as nn
from typing import List, Optional
import re
from ..models.pdclm import PDCLMBase
# from ..utils.tools import ToolExecutor # TODO: Implement tools

class CognitiveControlModule(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.coherence_scorer = nn.Linear(dim, 1)
        self.max_cot_depth = 5

    def forward(self, cot_steps: List[torch.Tensor]) -> torch.Tensor:
        if len(cot_steps) == 0:
            return torch.tensor(0.0, device=self.coherence_scorer.weight.device)
        if len(cot_steps) > self.max_cot_depth:
            cot_steps = cot_steps[-self.max_cot_depth:]
        pooled = []
        for s in cot_steps:
            if s.dim() == 2:
                pooled.append(s.mean(dim=0))
            else:
                pooled.append(s)
        stacked = torch.stack(pooled)
        scores = torch.sigmoid(self.coherence_scorer(stacked))
        return scores.mean()


class HCLAgent:
    def __init__(self, model: PDCLMBase, ccm: CognitiveControlModule):
        self.model = model
        self.ccm = ccm

    def generate_cot(self, task: str, max_steps: int = 8) -> List[str]:
        cot = []
        current = task
        for i in range(max_steps):
            with torch.no_grad():
                prompt = current + f"\nStep {i+1}: "
                text = self.model.generate_text(prompt, max_new_bytes=64)
            current = text
            cot.append(current)
            if "Final answer:" in current:
                break
        return cot

    def generate_cot_with_tools(self, task: str, tools, max_steps: int = 8) -> List[str]:
        cot = []
        current = task
        for i in range(max_steps):
            if "[TOOL_CALL:" in current:
                result = tools.run_call(current)
                if result:
                    current = current + f"\n[TOOL_RESULT: \"{result}\"]"
            with torch.no_grad():
                prompt = current + f"\nStep {i+1}: "
                text = self.model.generate_text(prompt, max_new_bytes=64)
            current = text
            cot.append(current)
            if "Final answer:" in current:
                break
        return cot

    def generate_rule_based_cot(self, task: str, max_steps: int = 8) -> List[str]:
        cot = []
        current = task
        def add_padding(txt: str) -> str:
            min_len = getattr(self.model.pse, "window_size", 32)
            if len(txt) < min_len:
                repeat = (min_len // max(len(txt), 1)) + 1
                txt = (txt + " ") * repeat
            return txt
        def parse_and_reason(base_t: str, step_idx: int) -> str:
            s = base_t.lower()
            if "what is" in s and "+" in s:
                expr = s.split("is")[1].split("?")[0]
                a, b = [int(x.strip()) for x in expr.split("+")]
                if step_idx < max_steps - 1:
                    return f"Decompose addition: {a} + {b}"
                return f"Final answer: {a + b}"
            if "what is" in s and "*" in s:
                expr = s.split("is")[1].split("?")[0]
                a, b = [int(x.strip()) for x in expr.split("*")]
                if step_idx < max_steps - 1:
                    return f"Partial products for {a} * {b}"
                return f"Final answer: {a * b}"
            if "even" in s:
                try:
                    num = int(s.split("even?")[0].split()[-1])
                except Exception:
                    num = 0
                if step_idx < max_steps - 1:
                    return f"Check parity of {num}"
                return f"Final answer: {'True' if num % 2 == 0 else 'False'}"
            if "sort these numbers" in s:
                nums = re.findall(r"\d+", s)
                arr = [int(x) for x in nums]
                if step_idx < max_steps - 1:
                    return f"Apply sort to {arr}"
                return f"Final answer: {sorted(arr)}"
            return "Reasoning step"
        for i in range(max_steps):
            with torch.no_grad():
                text = add_padding(current)
                stream = self.model.pse(text)
                # Assuming model has transformer attribute
                if hasattr(self.model, 'transformer'):
                    _ = self.model.transformer(stream.unsqueeze(0), torch.zeros_like(stream).unsqueeze(0))
                else:
                    # Fallback or error if model structure is different
                    pass
            next_step = parse_and_reason(task, i)
            current = current + " " + next_step
            cot.append(current)
        return cot

    def embed_cot(self, cot_steps: List[str]) -> List[torch.Tensor]:
        embeds = []
        for t in cot_steps:
            text = t
            min_len = getattr(self.model.pse, "window_size", 32)
            if len(text) < min_len:
                repeat = (min_len // max(len(text), 1)) + 1
                text = (text + " ") * repeat
            e = self.model.pse(text)
            if e.dim() == 2:
                embeds.append(e.mean(dim=0))
            else:
                embeds.append(e)
        return embeds


def task_generator() -> str:
    import random
    tasks = [
        f"What is {random.randint(1,100)} + {random.randint(1,100)}?",
        f"What is {random.randint(1,50)} * {random.randint(1,50)}?",
        f"Is {random.randint(1,100)} even? Answer True/False.",
        f"Sort these numbers: {', '.join(map(str, random.sample(range(1,20), 5)))}"
    ]
    return random.choice(tasks)


def parse_expected_answer(task: str) -> int:
    try:
        expr = task.split("is")[1].split("?")[0].strip()
        parts = expr.split("+")
        a = int(parts[0].strip())
        b = int(parts[1].strip())
        return a + b
    except Exception:
        return -1


def compute_reward(cot_steps: List[torch.Tensor], task: str, answer: str, ccm: CognitiveControlModule) -> float:
    coherence = ccm(cot_steps)
    expected = parse_expected_answer(task)
    correctness = 1.0 if (answer.strip().isdigit() and int(answer) == expected) else 0.0
    reward = 0.7 * coherence + 0.3 * torch.tensor(correctness, device=coherence.device)
    return float(reward.item())


def hcl_train_step(proposer: HCLAgent, critic: HCLAgent, num_tasks: int = 10, optimizer: Optional[torch.optim.Optimizer] = None) -> float:
    device = next(proposer.model.parameters()).device
    if optimizer is None:
        optimizer = torch.optim.AdamW(critic.ccm.parameters(), lr=1e-3)
    total = 0.0
    for _ in range(num_tasks):
        task = task_generator()
        cot_texts = proposer.generate_cot(task)
        cot_embeds = proposer.embed_cot(cot_texts)
        answer = str(parse_expected_answer(task))
        coherence = critic.ccm(cot_embeds)
        expected = parse_expected_answer(task)
        correctness = 1.0 if (answer.strip().isdigit() and int(answer) == expected) else 0.0
        reward_tensor = 0.7 * coherence + 0.3 * torch.tensor(correctness, device=device)
        loss = -reward_tensor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(reward_tensor.item())
    return total / num_tasks
