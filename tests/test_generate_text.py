## Developer: inkbytefo
## Modified: 2025-11-17

import pytest
import torch
from src.model import PDCLMBase


def test_generate_text_produces_output():
    model = PDCLMBase()
    out = model.generate_text("Task: What is 2 + 3? \n")
    assert isinstance(out, str)
    assert len(out) >= len("Task: What is 2 + 3? \n")


if __name__ == "__main__":
    pytest.main([__file__])