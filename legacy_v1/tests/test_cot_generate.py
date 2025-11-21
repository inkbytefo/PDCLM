## Developer: inkbytefo
## Modified: 2025-11-17

import pytest
from src.model import PDCLMBase
from src.hcl import HCLAgent, CognitiveControlModule


def test_generate_cot_returns_steps():
    model = PDCLMBase()
    ccm = CognitiveControlModule()
    agent = HCLAgent(model, ccm)
    steps = agent.generate_cot("What is 2 + 3?")
    assert isinstance(steps, list)
    assert len(steps) >= 1


if __name__ == "__main__":
    pytest.main([__file__])