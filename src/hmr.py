## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import torch.nn as nn


class HierarchicalMemoryRouter(nn.Module):
    def __init__(self, embed_dim: int = 256, num_ssm: int = 4, num_msm: int = 2, decay: float = 0.9):
        super().__init__()
        self.embed_dim = embed_dim
        self.decay = decay
        self.ssm_slots = nn.Parameter(torch.randn(num_ssm, embed_dim))
        self.msm_slots = nn.Parameter(torch.randn(num_msm, embed_dim))
        self.router = nn.Linear(embed_dim, num_ssm + num_msm)

    def forward(self, input_stream: torch.Tensor) -> torch.Tensor:
        recent = input_stream[-min(4, input_stream.size(0)):, :].mean(dim=0)
        logits = self.router(recent)
        weights = torch.softmax(logits, dim=-1)
        slots = torch.cat([self.ssm_slots, self.msm_slots], dim=0)
        weighted = slots * weights.unsqueeze(1)
        with torch.no_grad():
            target = recent.unsqueeze(0).expand_as(self.ssm_slots)
            self.ssm_slots.mul_(self.decay).add_((1.0 - self.decay) * target)
        return weighted