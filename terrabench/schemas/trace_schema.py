"""Schemas for TerraBench traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceStep:
    """One normalized trace step."""

    type: str
    name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str | None = None
    content: Any = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "TraceStep":
        step_type = str(payload.get("type") or payload.get("role") or "unknown")
        name = payload.get("name") or payload.get("tool") or payload.get("tool_name")
        raw_args = payload.get("arguments") or payload.get("args") or payload.get("input") or {}
        arguments = raw_args if isinstance(raw_args, dict) else {"value": raw_args}
        status = payload.get("status")
        content = payload.get("content", payload.get("observation"))
        return cls(
            type=step_type,
            name=str(name) if name is not None else None,
            arguments=arguments,
            status=str(status) if status is not None else None,
            content=content,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "arguments": dict(self.arguments),
            "status": self.status,
            "content": self.content,
        }


@dataclass(frozen=True)
class Trace:
    """A normalized benchmark trace."""

    steps: list[TraceStep]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "Trace":
        if isinstance(payload, dict):
            raw_steps = payload.get("steps") or payload.get("trace") or payload.get("messages") or []
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        elif isinstance(payload, list):
            raw_steps = payload
            metadata = {}
        else:
            raise ValueError("Trace payload must be a JSON object or list.")
        if not isinstance(raw_steps, list):
            raise ValueError("Trace steps must be a list.")
        steps = [TraceStep.from_mapping(step) for step in raw_steps if isinstance(step, dict)]
        return cls(steps=steps, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps], "metadata": dict(self.metadata)}
