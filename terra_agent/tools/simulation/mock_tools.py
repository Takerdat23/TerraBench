"""Small deterministic mock tools for smoke tests."""

from __future__ import annotations

from terra_agent.tools.base import BaseTool, ToolResult
from terra_agent.tools.registry import register_tool


@register_tool
class MockNumericAnswerTool(BaseTool):
    name = "mock_numeric_answer"
    group = "simulation"
    description = "Deterministic mock numeric answer tool for smoke tests."
    requires_credentials = False
    deterministic = True

    def run(self, inputs: dict, context: dict) -> ToolResult:
        return ToolResult(
            status="success",
            summary="Mock TerraAgent numeric answer generated.",
            data={"value": 1.0, "mock": bool(context.get("mock", True))},
            metadata={"item_id": context.get("item_id"), "mode": "mock"},
        )
