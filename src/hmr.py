## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import torch.nn as nn


class HierarchicalMemoryRouter(nn.Module):
    def __init__(self, embed_dim: int = 256, num_ssm: int = 4, num_msm: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.ssm_slots = nn.Parameter(torch.randn(num_ssm, embed_dim))
        self.msm_slots = nn.Parameter(torch.randn(num_msm, embed_dim))
        self.router = nn.Linear(embed_dim, num_ssm + num_msm)

    def forward(self, input_stream: torch.Tensor) -> torch.Tensor:
        combined_memory = torch.cat([self.ssm_slots, self.msm_slots], dim=0)
        return combined_memory