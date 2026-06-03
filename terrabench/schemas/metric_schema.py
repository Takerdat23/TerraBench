"""Schemas for evaluation metric reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricReport:
    """Structured metric report returned by evaluators."""

    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": dict(self.metrics), "details": dict(self.details)}
