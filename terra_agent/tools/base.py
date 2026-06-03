"""Common TerraAgent tool interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolArtifact:
    path: str
    type: str
    description: str | None = None


@dataclass(frozen=True)
class ToolResult:
    status: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ToolArtifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool:
    name: str = "base_tool"
    group: str = "general"
    description: str = ""
    requires_credentials: bool = False
    deterministic: bool = True

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        return None

    def run(self, inputs: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        raise NotImplementedError
