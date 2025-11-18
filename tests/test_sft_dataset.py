## Developer: inkbytefo
## Modified: 2025-11-17

import pytest
from src.model import PDCLMBase
from src.hcl import HCLAgent, CognitiveControlModule
from src.utils import build_sft_dataset


def test_build_sft_dataset_basic():
    model = PDCLMBase()
    ccm = CognitiveControlModule()
    agent = HCLAgent(model, ccm)
    samples = build_sft_dataset(5, agent)
    assert isinstance(samples, list)
    assert len(samples) == 5
    assert all("input" in s and "target" in s for s in samples)


if __name__ == "__main__":
    pytest.main([__file__])