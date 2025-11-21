## Developer: inkbytefo
## Modified: 2025-11-17

import sys
sys.path.append('..')

from src.pdclm import PDCLM


def test_full_reward():
    model = PDCLM(embed_dim=256)
    task = "What is 2 + 2?"
    _, reward = model.full_forward(task)
    assert reward > 0.5