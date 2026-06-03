"""Tool-use metrics for TerraBench traces."""

from __future__ import annotations

from collections import Counter
from typing import Any

from terrabench.schemas.trace_schema import Trace
from terrabench.traces.alignment import order_score
from terrabench.traces.canonicalization import canonical_tool_sequence

DEFAULT_TOOLUSE_WEIGHTS = {
    "ToolAcc": 0.30,
    "InstAcc": 0.15,
    "ArgAcc": 0.20,
    "CategoryF1": 0.15,
    "OrderScore": 0.15,
    "ToolCallSuccessRate": 0.05,
}


def _counter_f1(predicted: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0 if not predicted else 0.0
    pred_counts = Counter(predicted)
    exp_counts = Counter(expected)
    true_pos = sum(min(pred_counts[key], exp_counts[key]) for key in exp_counts)
    precision = true_pos / max(1, sum(pred_counts.values()))
    recall = true_pos / max(1, sum(exp_counts.values()))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _success_rate(trace: Trace) -> float:
    tool_steps = [step for step in trace.steps if step.name and step.type in {"tool", "tools", "tool_call"}]
    if not tool_steps:
        return 1.0
    successes = sum(1 for step in tool_steps if str(step.status or "success").lower() in {"success", "ok", "completed"})
    return successes / len(tool_steps)


def compute_tool_metrics(predicted_trace: Trace | None, expected_trace: Trace | None, weights: dict[str, float] | None = None) -> dict[str, Any]:
    weights = dict(weights or DEFAULT_TOOLUSE_WEIGHTS)
    predicted = canonical_tool_sequence(predicted_trace) if predicted_trace else []
    expected = canonical_tool_sequence(expected_trace) if expected_trace else []
    tool_acc = _counter_f1(predicted, expected)
    metrics = {
        "InstAcc": 1.0 if predicted or not expected else 0.0,
        "ToolCallSuccessRate": _success_rate(predicted_trace) if predicted_trace else 0.0,
        "ToolAcc": tool_acc,
        "CategoryF1": tool_acc,
        "ArgAcc": tool_acc,
        "OrderScore": order_score(predicted, expected),
    }
    total_weight = sum(weights.values()) or 1.0
    metrics["ToolUseScore"] = sum(metrics.get(name, 0.0) * weight for name, weight in weights.items()) / total_weight
    return {"metrics": metrics, "details": {"predicted_tools": predicted, "expected_tools": expected}}
