## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import torch.nn as nn
from typing import List
from src.model import PDCLMBase
from src.hcl import HCLAgent, CognitiveControlModule


class ReflectionNetwork(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.reflector = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        for m in self.reflector.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_stream: torch.Tensor, pred_stream: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([input_stream.mean(dim=0), pred_stream.mean(dim=0)])
        return torch.sigmoid(self.reflector(concat.unsqueeze(0)))


class ReflectivePDCLM(PDCLMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reflection = ReflectionNetwork(dim=self.embed_dim)
        self.reflection_loss_weight = 1.0

    def _pad_text(self, text: str) -> str:
        min_len = getattr(self.pse, "window_size", 32)
        if len(text) < min_len:
            repeat = (min_len // max(len(text), 1)) + 1
            text = (text + " ") * repeat
        return text

    def reflective_forward(self, raw_text: str, cot_steps: List[str]):
        streams = []
        for s in cot_steps[:-1]:
            s_padded = self._pad_text(s)
            streams.append(self.pse(s_padded))
        pred_stream = self.pse(self._pad_text(cot_steps[-1]))
        reflection_losses = []
        for inp, pred in zip(streams, streams[1:] + [pred_stream]):
            refl_score = self.reflection(inp, pred)
            reflection_losses.append(refl_score)
        refl_loss = torch.stack(reflection_losses).mean()
        task_loss = super().forward(self._pad_text(raw_text))
        return task_loss + self.reflection_loss_weight * refl_loss, refl_loss


def reflective_train_step(model: ReflectivePDCLM, agent: HCLAgent, task: str, max_steps: int = 6, optimizer: torch.optim.Optimizer | None = None):
    model.train()
    cot = agent.generate_cot(task, max_steps=max_steps)
    loss, refl_loss = model.reflective_forward(" ".join(cot), cot)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item(), refl_loss.item()