"""Trace canonicalization helpers."""

from __future__ import annotations

from terrabench.schemas.trace_schema import Trace


def canonical_tool_sequence(trace: Trace) -> list[str]:
    return [step.name for step in trace.steps if step.name and step.type in {"tool", "tools", "tool_call"}]
