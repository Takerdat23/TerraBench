"""Plugin-like TerraAgent tool registry."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from terra_agent.tools.base import BaseTool, ToolResult

_REGISTERED: dict[str, type[BaseTool]] = {}


def register_tool(cls: type[BaseTool]) -> type[BaseTool]:
    if not getattr(cls, "name", None):
        raise ValueError("Registered tool classes must define a name.")
    _REGISTERED[cls.name] = cls
    return cls


class ToolRegistry:
    """Lookup and execute tools by name or group."""

    def __init__(self, tools: dict[str, BaseTool] | None = None):
        self._tools = tools or {name: cls() for name, cls in _REGISTERED.items()}

    @classmethod
    def default(cls) -> "ToolRegistry":
        from terra_agent.tools.execution_server.code_agent import CodeAgentTool  # noqa: F401
        from terra_agent.tools.satellite.tools import (  # noqa: F401
            CompositeSatelliteEnvironmentalImageTool,
            GeeFetchSentinel2IndicesTool,
            GeeFetchTruecolorThumbnailTool,
            GeeIndexAreaStatsTool,
        )
        from terra_agent.tools.simulation.mock_tools import MockNumericAnswerTool  # noqa: F401
        from terra_agent.tools.web_search.tools import (  # noqa: F401
            SummarizeWebPageTool,
            WebSearchSerperTool,
        )
        import terra_agent.tools.full_climateagent  # noqa: F401

        return cls()

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool {name!r}. Available tools: {self.list_tools()}")
        return self._tools[name]

    def list_tools(self, group: str | None = None) -> list[str]:
        names = []
        for name, tool in self._tools.items():
            if group is None or tool.group == group:
                names.append(name)
        return sorted(names)

    def list_by_group(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for name, tool in self._tools.items():
            groups[tool.group].append(name)
        return {group: sorted(names) for group, names in groups.items()}

    def validate_tool_arguments(self, name: str, arguments: dict[str, Any]) -> None:
        self.get(name).validate_inputs(arguments)

    def credential_errors(self, name: str) -> list[str]:
        tool = self.get(name)
        if not tool.requires_credentials:
            return []
        missing = []
        explicit_env_names = tuple(getattr(tool, "credential_env_vars", ()) or ())
        any_env_names = tuple(getattr(tool, "credential_env_any", ()) or ())

        for env_name in explicit_env_names:
            if not os.getenv(env_name):
                missing.append(f"Missing credential environment variable: {env_name}")

        if any_env_names and not any(os.getenv(env_name) for env_name in any_env_names):
            missing.append(
                "Missing one of credential environment variables: "
                + ", ".join(any_env_names)
            )

        if missing:
            return missing
        if explicit_env_names or any_env_names:
            return []

        env_name = f"{tool.name.upper()}_API_KEY"
        return [] if os.getenv(env_name) else [f"Missing credential environment variable: {env_name}"]

    def run(self, name: str, inputs: dict[str, Any], context: dict[str, Any] | None = None) -> ToolResult:
        context = dict(context or {})
        if context.get("dry_run"):
            return ToolResult(status="success", summary=f"Dry run for {name}.", metadata={"dry_run": True})
        errors = self.credential_errors(name)
        if errors:
            return ToolResult(status="error", summary="; ".join(errors), metadata={"credential_errors": errors})
        tool = self.get(name)
        tool.validate_inputs(inputs)
        return tool.run(inputs, context)
