## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.hmr import HierarchicalMemoryRouter


def test_hmr_initialization_and_forward():
    embed_dim = 256
    hmr = HierarchicalMemoryRouter(embed_dim=embed_dim)
    input_stream = torch.randn(10, embed_dim)
    memory_output, weights = hmr(input_stream)
    assert memory_output.dim() == 2
    assert memory_output.size(1) == embed_dim
    assert memory_output.size(0) > 0
    assert weights.dim() == 1


def test_hmr_integration_with_model():
    from src.model import PDCLMBase
    model = PDCLMBase(embed_dim=256, num_layers=2, heads=2, window_size=128)
    assert hasattr(model, 'hmr')
    assert isinstance(model.hmr, HierarchicalMemoryRouter)
    test_text = "Bu bir test metnidir." * 10
    loss = model(test_text)
    assert not torch.isnan(loss)
    assert loss.item() >= 0


def test_hmr_slot_update():
    embed_dim = 256
    hmr = HierarchicalMemoryRouter(embed_dim=embed_dim, decay=0.8)
    input_stream = torch.randn(8, embed_dim)
    before = hmr.ssm_slots.clone().detach()
    _ = hmr(input_stream)
    after = hmr.ssm_slots.clone().detach()
    diff = (after - before).abs().sum().item()
    assert diff > 0


def test_hmr_eviction_policy():
    embed_dim = 256
    hmr = HierarchicalMemoryRouter(embed_dim=embed_dim, decay=0.9, age_limit=1)
    x1 = torch.randn(8, embed_dim)
    _ = hmr(x1)
    before = hmr.ssm_slots.clone().detach()
    x2 = torch.randn(8, embed_dim)
    _ = hmr(x2)
    after = hmr.ssm_slots.clone().detach()
    change = (after - before).abs().sum().item()
    assert change > 0


def test_hmr_lsm_retrieval_and_insert():
    embed_dim = 64
    hmr = HierarchicalMemoryRouter(embed_dim=embed_dim, num_lsm=4, lsm_threshold=0.5)
    x = torch.randn(8, embed_dim)
    _ = hmr(x)
    sim1 = float(hmr.last_sim_max.item())
    _ = hmr(x + 0.01)
    sim2 = float(hmr.last_sim_max.item())
    assert sim2 >= sim1