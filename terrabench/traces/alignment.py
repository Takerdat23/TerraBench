"""Trace alignment utilities."""

from __future__ import annotations


def order_score(predicted: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0 if not predicted else 0.0
    matches = sum(1 for left, right in zip(predicted, expected) if left == right)
    return matches / len(expected)
