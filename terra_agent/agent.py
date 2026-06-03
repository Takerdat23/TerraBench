"""Minimal TerraAgent runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from terrabench.data.loaders import BenchmarkItem, load_item
from terrabench.utils.json_utils import write_json
from terra_agent.tools.registry import ToolRegistry


class TerraAgent:
    """Simple public runner with mock and dry-run modes."""

    def __init__(self, *, registry: ToolRegistry | None = None, mock_tools: bool = False, dry_run: bool = False):
        self.registry = registry or ToolRegistry.default()
        self.mock_tools = mock_tools
        self.dry_run = dry_run

    def run(self, item: BenchmarkItem | str | Path, *, out_dir: str | Path) -> dict[str, Any]:
        loaded = load_item(item) if isinstance(item, (str, Path)) else item
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tool = self.registry.get("mock_numeric_answer")
        result = tool.run(
            {"question": loaded.question.question, "answer_keys": loaded.question.answer_keys},
            {"mock": self.mock_tools, "dry_run": self.dry_run, "item_id": loaded.question.item_id},
        )
        answer_key = loaded.question.answer_keys[0] if loaded.question.answer_keys else "answer"
        value = result.data.get("value", 1.0)
        prediction = {
            "answers": [{"key": answer_key, "value": value}],
            "trace": [
                {
                    "type": "tool",
                    "name": tool.name,
                    "arguments": {"answer_key": answer_key},
                    "status": result.status,
                    "content": result.summary,
                }
            ],
            "metadata": {"mode": "mock" if self.mock_tools else "dry_run" if self.dry_run else "local"},
        }
        write_json(out / "prediction.json", prediction)
        return prediction
