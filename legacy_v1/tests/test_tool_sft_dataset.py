## Developer: inkbytefo
## Modified: 2025-11-18

import pytest
from src.model import PDCLMBase
from src.hcl import HCLAgent, CognitiveControlModule
from src.pdclm import setup_default_tools
from src.utils import build_tool_augmented_sft_dataset


def test_tool_augmented_sft_dataset_has_calls_and_results():
    model = PDCLMBase()
    agent = HCLAgent(model, CognitiveControlModule())
    tools = setup_default_tools()
    samples = build_tool_augmented_sft_dataset(3, agent, tools)
    assert all("[TOOL_CALL:" in s["target"] and "[TOOL_RESULT:" in s["target"] for s in samples)


if __name__ == "__main__":
    pytest.main([__file__])