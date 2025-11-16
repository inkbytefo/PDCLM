## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
from src.model import PDCLMBase
from src.hcl import HCLAgent, CognitiveControlModule


class ReflectionNetwork(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.reflector = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 1)
        )
        for m in self.reflector:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.constant_(m.bias, -1.0)

    def forward(self, input_stream: torch.Tensor, pred_stream: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([input_stream.mean(dim=0), pred_stream.mean(dim=0)])
        return self.reflector(concat.unsqueeze(0))


class ReflectivePDCLM(PDCLMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reflection = ReflectionNetwork(dim=self.embed_dim)
        self.reflection_loss_weight = 1.5
        self.bce_loss = nn.BCEWithLogitsLoss()

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
        refl_logits = []
        targets = []
        for inp, pred in zip(streams, streams[1:] + [pred_stream]):
            logit = self.reflection(inp, pred)
            target_err = torch.clamp(1.0 - F.cosine_similarity(inp.mean(dim=0), pred.mean(dim=0), dim=0), 0.0, 1.0)
            refl_logits.append(logit)
            targets.append(target_err)
        refl_logits = torch.cat(refl_logits)
        targets = torch.stack(targets).unsqueeze(1)
        refl_loss = self.bce_loss(refl_logits, targets)
        task_loss = super().forward(self._pad_text(raw_text))
        total = task_loss + self.reflection_loss_weight * refl_loss
        return total, refl_loss


def reflective_train_step(model: ReflectivePDCLM, agent: HCLAgent, task: str, max_steps: int = 8, optimizer: torch.optim.Optimizer | None = None):
    model.train()
    cot = agent.generate_cot(task, max_steps=max_steps)
    loss, refl_loss = model.reflective_forward(" ".join(cot), cot)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item(), refl_loss.item()