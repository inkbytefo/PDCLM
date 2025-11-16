## Developer: inkbytefo
## Modified: 2025-11-17

import sys
import os
sys.path.append('..')

from src.utils import load_data, preprocess_text, build_reasoning_tasks, measure_forward_latency
from src.model import PDCLMBase


def test_build_reasoning_tasks():
    tasks = build_reasoning_tasks(8)
    assert isinstance(tasks, list)
    assert len(tasks) == 8
    assert all(isinstance(t, str) for t in tasks)


def test_measure_forward_latency():
    model = PDCLMBase()
    text = "a" * 1000
    ms = measure_forward_latency(model, text)
    assert ms >= 0.0


def test_load_data_basic():
    data = load_data(os.path.join('data', 'raw', 'wikitext_sample.txt'))
    assert 'text' in data and 'length' in data