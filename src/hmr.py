## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import torch.nn as nn


class HierarchicalMemoryRouter(nn.Module):
    def __init__(self, embed_dim: int = 256, num_ssm: int = 4, num_msm: int = 2, decay: float = 0.9, age_limit: int = 50):
        super().__init__()
        self.embed_dim = embed_dim
        self.decay = decay
        self.age_limit = age_limit
        self.ssm_slots = nn.Parameter(torch.randn(num_ssm, embed_dim))
        self.msm_slots = nn.Parameter(torch.randn(num_msm, embed_dim))
        self.router = nn.Linear(embed_dim, num_ssm + num_msm)
        self.compress = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("ssm_age", torch.zeros(num_ssm))

    def forward(self, input_stream: torch.Tensor) -> torch.Tensor:
        recent = input_stream[-min(8, input_stream.size(0)):, :]
        recent_mean = recent.mean(dim=0)
        pos_logits = self.router(input_stream)
        pos_weights = torch.softmax(pos_logits, dim=-1)
        avg_weights = pos_weights.mean(dim=0)
        slots = torch.cat([self.ssm_slots, self.msm_slots], dim=0)
        weighted = slots * avg_weights.unsqueeze(1)
        with torch.no_grad():
            target_vec = self.compress(recent_mean).unsqueeze(0).expand_as(self.ssm_slots)
            self.ssm_slots.mul_(self.decay).add_((1.0 - self.decay) * target_vec)
            self.ssm_age += 1
            if int(self.ssm_age.max().item()) > self.age_limit:
                num_ssm = self.ssm_slots.size(0)
                ssm_weights = avg_weights[:num_ssm]
                idx = int(torch.argmin(ssm_weights).item())
                self.ssm_slots[idx] = self.ssm_slots[idx] * self.decay + (1.0 - self.decay) * target_vec[0]
                self.ssm_age[idx] = 0
        return weighted