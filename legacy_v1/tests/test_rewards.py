## Developer: inkbytefo
## Modified: 2025-11-17

import sys
sys.path.append('..')

from src.utils import compute_hallucination_penalty


def test_hallucination_penalty_add():
    t = "What is 2 + 3?"
    assert compute_hallucination_penalty(t, "5") == 0.0
    assert compute_hallucination_penalty(t, "6") > 0.0


def test_hallucination_penalty_sort():
    t = "Sort these numbers: 3, 1, 2"
    assert compute_hallucination_penalty(t, "[1, 2, 3]") == 0.0
    assert compute_hallucination_penalty(t, "[3, 2, 1]") > 0.0