"""Schemas for answer fields and numeric ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnswerField:
    """A prediction or answer field."""

    key: str
    value: Any

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "AnswerField":
        if "key" not in payload:
            raise ValueError("Answer field must include key.")
        if "value" not in payload:
            raise ValueError("Answer field must include value.")
        return cls(key=str(payload["key"]), value=payload["value"])


@dataclass(frozen=True)
class NumericGroundTruth:
    """Ground-truth field with explicit numeric tolerance metadata."""

    key: str
    value: Any
    abs_tol: float
    rel_tol: float
    floor_scale: float

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "NumericGroundTruth":
        missing = [name for name in ("key", "value", "abs_tol", "rel_tol", "floor_scale") if name not in payload]
        if missing:
            raise ValueError(f"Ground-truth field is missing required keys: {missing}.")
        return cls(
            key=str(payload["key"]),
            value=payload["value"],
            abs_tol=float(payload["abs_tol"]),
            rel_tol=float(payload["rel_tol"]),
            floor_scale=float(payload["floor_scale"]),
        )
