"""Answer-format metrics."""

from __future__ import annotations

from typing import Any

from terrabench.evaluation.numeric_metrics import normalize_answer_fields


def answer_format_valid(prediction: Any) -> float:
    return 1.0 if normalize_answer_fields(prediction) else 0.0
