## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import sys
sys.path.append('..')
from src.reflection import ReflectivePDCLM, ReflectionNetwork


def test_reflection_score():
    model = ReflectivePDCLM(embed_dim=256)
    stream1 = torch.randn(10, 256)
    stream2 = stream1.clone()
    score = model.reflection(stream1, stream2)
    assert 0.4 < score.item() < 0.6


def test_reflective_loss():
    model = ReflectivePDCLM()
    cot = ["a" * 1000] * 5
    loss, refl = model.reflective_forward("a" * 5000, cot)
    assert loss.item() > refl.item()