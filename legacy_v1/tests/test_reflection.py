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
    logit = model.reflection(stream1, stream2)
    prob = torch.sigmoid(logit).item()
    assert 0.2 < prob < 0.8


def test_reflective_loss():
    model = ReflectivePDCLM()
    cot = ["a" * 1000] * 5
    loss, refl, mean_logit, mean_target = model.reflective_forward("a" * 5000, cot)
    assert loss.item() > 0
    assert 0.0 <= mean_target.item() <= 1.0