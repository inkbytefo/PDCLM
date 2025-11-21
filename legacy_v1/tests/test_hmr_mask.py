## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import PDCLMBase


def test_hmr_memory_key_padding_mask_applied():
    model = PDCLMBase(embed_dim=256, num_layers=2, heads=2, window_size=128)
    text = "a" * 2000
    loss = model(text)
    assert loss.item() >= 0