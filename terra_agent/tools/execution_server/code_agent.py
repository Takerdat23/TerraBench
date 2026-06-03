"""TerraAgent wrapper around the InterCode math/code-agent API."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from terra_agent.tools.base import BaseTool, ToolResult
from terra_agent.tools.registry import register_tool

DEFAULT_CODE_AGENT_URL = "http://127.0.0.1:8000/code-agent"


@register_tool
class CodeAgentTool(BaseTool):
    name = "code_agent"
    group = "execution_server"
    description = "Execute math and Python reasoning tasks through the InterCode service."
    requires_credentials = False
    deterministic = False

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        query = inputs.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("code_agent requires a non-empty string input named 'query'.")

    def run(self, inputs: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        self.validate_inputs(inputs)
        endpoint = (
            context.get("code_agent_url")
            or inputs.get("code_agent_url")
            or os.getenv("CODE_AGENT_URL")
            or context.get("math_agent_url")
            or inputs.get("math_agent_url")
            or os.getenv("MATH_AGENT_URL")
            or DEFAULT_CODE_AGENT_URL
        )
        timeout = float(inputs.get("request_timeout_seconds") or context.get("request_timeout_seconds") or 120)
        payload = _build_payload(inputs)
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            str(endpoint),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw) if raw else {}
                status_code = response.status
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return ToolResult(
                status="error",
                summary=f"code_agent HTTP {exc.code}: {detail}",
                metadata={"endpoint": endpoint, "status_code": exc.code},
            )
        except (error.URLError, TimeoutError, OSError) as exc:
            return ToolResult(
                status="error",
                summary=f"code_agent request failed: {exc}",
                metadata={"endpoint": endpoint},
            )

        summary = data.get("summary") or {}
        turns = summary.get("turns_taken")
        reward = summary.get("max_reward")
        message = "code_agent completed"
        if turns is not None or reward is not None:
            message = f"code_agent completed with reward={reward}, turns={turns}"
        return ToolResult(
            status="success",
            summary=message,
            data=data,
            metadata={"endpoint": endpoint, "status_code": status_code},
        )


def _build_payload(inputs: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "query",
        "context",
        "variables",
        "model",
        "template",
        "dialogue_limit",
        "max_turns",
        "execution_timeout_seconds",
        "metadata",
    }
    return {key: value for key, value in inputs.items() if key in allowed and value is not None}
