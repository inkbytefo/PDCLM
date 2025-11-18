## Developer: inkbytefo
## Modified: 2025-11-18

import pytest
from src.model import PDCLMBase
from src.hcl import HCLAgent, CognitiveControlModule
from src.tools import ToolExecutor, CalculatorTool


def test_generate_cot_with_tools_injects_result():
    model = PDCLMBase()
    ccm = CognitiveControlModule()
    agent = HCLAgent(model, ccm)
    tools = ToolExecutor()
    tools.register("calculator", CalculatorTool())
    task = 'Solve: [TOOL_CALL: calculator("2 * 3")] then provide Final answer: 6'
    steps = agent.generate_cot_with_tools(task, tools, max_steps=2)
    assert any("[TOOL_RESULT:" in s for s in steps)


if __name__ == "__main__":
    pytest.main([__file__])