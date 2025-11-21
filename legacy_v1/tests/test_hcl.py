## Developer: inkbytefo
## Modified: 2025-11-17

import torch
import sys
import os

sys.path.append('..')

from src.hcl import CognitiveControlModule


def test_ccm_coherence():
    ccm = CognitiveControlModule()
    steps = [torch.randn(10, 256) for _ in range(3)]
    score = ccm(steps)
    assert 0 <= score.item() <= 1