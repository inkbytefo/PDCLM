## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalMemoryRouter(nn.Module):
    def __init__(self, embed_dim: int = 256, num_ssm: int = 4, num_msm: int = 2, decay: float = 0.9, age_limit: int = 50, num_lsm: int = 8, lsm_threshold: float = 0.8):
        super().__init__()
        self.embed_dim = embed_dim
        self.decay = decay
        self.age_limit = age_limit
        self.lsm_threshold = lsm_threshold
        self.ssm_slots = nn.Parameter(torch.randn(num_ssm, embed_dim))
        self.msm_slots = nn.Parameter(torch.randn(num_msm, embed_dim))
        self.router = nn.Linear(embed_dim, num_ssm + num_msm)
        self.compress = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("ssm_age", torch.zeros(num_ssm))
        self.register_buffer("lsm_keys", F.normalize(torch.randn(num_lsm, embed_dim), dim=1))
        self.register_buffer("lsm_values", torch.randn(num_lsm, embed_dim))
        self.register_buffer("lsm_age", torch.zeros(num_lsm))
        self.register_buffer("last_sim_max", torch.tensor(0.0))

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
            sims = F.cosine_similarity(recent_mean.unsqueeze(0), self.lsm_keys, dim=1)
            sim_max, sim_idx = sims.max(dim=0)
            self.last_sim_max = sim_max.detach()
            self.lsm_age += 1
            if float(sim_max.item()) >= self.lsm_threshold:
                num_ssm = self.ssm_slots.size(0)
                ssm_weights = avg_weights[:num_ssm]
                victim = int(torch.argmin(ssm_weights).item())
                self.ssm_slots[victim] = self.ssm_slots[victim] * self.decay + (1.0 - self.decay) * self.lsm_values[sim_idx]
                self.ssm_age[victim] = 0
            else:
                victim = int(torch.argmax(self.lsm_age).item())
                self.lsm_keys[victim] = F.normalize(recent_mean, dim=0)
                self.lsm_values[victim] = self.compress(recent_mean)
                self.lsm_age[victim] = 0
        return weighted