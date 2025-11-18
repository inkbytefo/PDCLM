## Developer: inkbytefo
## Modified: 2025-11-18

import pytest
from src.tools import ToolExecutor, CalculatorTool, SearchTool


def test_calculator_tool_exec():
    t = ToolExecutor()
    t.register("calculator", CalculatorTool())
    r = t.run_call('[TOOL_CALL: calculator("2 * 3")]')
    assert r == "6"


def test_search_tool_exec():
    kb = "Ankara is the capital of Turkey. Istanbul is the largest city."
    t = ToolExecutor()
    t.register("search", SearchTool(kb_text=kb))
    r = t.run_call('[TOOL_CALL: search("capital of Turkey")]')
    assert isinstance(r, str)
    assert len(r) > 0


if __name__ == "__main__":
    pytest.main([__file__])