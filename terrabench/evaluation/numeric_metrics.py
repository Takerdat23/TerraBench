"""Numeric and exact-answer metrics."""

from __future__ import annotations

from typing import Any

from terrabench.schemas.answer_schema import NumericGroundTruth


def normalize_answer_fields(payload: Any) -> dict[str, Any]:
    """Normalize prediction payloads to `{key: value}`."""

    if isinstance(payload, dict) and "answers" in payload:
        payload = payload["answers"]
    if isinstance(payload, dict) and "final_json" in payload:
        payload = payload["final_json"]
    if isinstance(payload, list):
        result = {}
        for item in payload:
            if isinstance(item, dict) and "key" in item and "value" in item:
                result[str(item["key"])] = item["value"]
        return result
    if isinstance(payload, dict):
        if "key" in payload and "value" in payload:
            return {str(payload["key"]): payload["value"]}
        return {str(key): value for key, value in payload.items() if key not in {"trace", "metadata"}}
    return {}


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def field_hit(predicted: Any, truth: NumericGroundTruth) -> tuple[float, str]:
    """Score one field using exact or tolerance matching."""

    if predicted is None:
        return 0.0, "missing"
    expected = truth.value
    if _is_number(expected):
        if not _is_number(predicted):
            return 0.0, "unparseable"
        expected_float = float(expected)
        predicted_float = float(predicted)
        tolerance = max(truth.abs_tol, truth.rel_tol * max(abs(expected_float), truth.floor_scale))
        return (1.0, "within_tolerance") if abs(predicted_float - expected_float) <= tolerance else (0.0, "outside_tolerance")
    if isinstance(expected, bool):
        return (1.0, "exact") if bool(predicted) is expected else (0.0, "mismatch")
    return (1.0, "exact") if predicted == expected else (0.0, "mismatch")


def compute_numscore(prediction: Any, ground_truth: list[NumericGroundTruth]) -> dict[str, Any]:
    predictions = normalize_answer_fields(prediction)
    details = []
    hits = []
    for truth in ground_truth:
        score, reason = field_hit(predictions.get(truth.key), truth)
        hits.append(score)
        details.append({"key": truth.key, "score": score, "reason": reason})
    score = sum(hits) / len(hits) if hits else 0.0
    return {"Hit@tol": score, "NumScore": score, "field_scores": details}
