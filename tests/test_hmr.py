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
    memory_output = hmr(input_stream)
    assert memory_output.dim() == 2
    assert memory_output.size(1) == embed_dim
    assert memory_output.size(0) > 0


def test_hmr_integration_with_model():
    from src.model import PDCLMBase
    model = PDCLMBase(embed_dim=256, num_layers=2, heads=2, window_size=128)
    assert hasattr(model, 'hmr')
    assert isinstance(model.hmr, HierarchicalMemoryRouter)
    test_text = "Bu bir test metnidir." * 10
    loss = model(test_text)
    assert not torch.isnan(loss)
    assert loss.item() >= 0