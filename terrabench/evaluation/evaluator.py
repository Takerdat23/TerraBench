"""Top-level evaluation entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from terrabench.data.loaders import load_ground_truth
from terrabench.evaluation.answer_metrics import answer_format_valid
from terrabench.evaluation.numeric_metrics import compute_numscore
from terrabench.evaluation.tool_metrics import DEFAULT_TOOLUSE_WEIGHTS, compute_tool_metrics
from terrabench.schemas.metric_schema import MetricReport
from terrabench.traces.io import load_trace
from terrabench.utils.json_utils import read_json, write_json


def _weights_from_config(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return dict(DEFAULT_TOOLUSE_WEIGHTS)
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    weights = payload.get("tooluse_weights", payload)
    return {str(key): float(value) for key, value in weights.items()}


def _trace_from_prediction(prediction: Any):
    if isinstance(prediction, dict) and "trace" in prediction:
        from terrabench.schemas.trace_schema import Trace

        return Trace.from_payload(prediction["trace"])
    return None


def evaluate_prediction(
    prediction_path: str | Path,
    ground_truth_path: str | Path,
    *,
    trace_path: str | Path | None = None,
    weights_path: str | Path | None = None,
) -> MetricReport:
    """Evaluate one prediction file against one ground-truth file."""

    prediction = read_json(prediction_path)
    ground_truth = load_ground_truth(ground_truth_path)
    numeric = compute_numscore(prediction, ground_truth)
    expected_trace = load_trace(trace_path) if trace_path else None
    predicted_trace = _trace_from_prediction(prediction)
    tool = compute_tool_metrics(predicted_trace, expected_trace, _weights_from_config(weights_path))
    metrics = {
        "AnswerFormatValid": answer_format_valid(prediction),
        "Hit@tol": float(numeric["Hit@tol"]),
        "NumScore": float(numeric["NumScore"]),
        **{key: float(value) for key, value in tool["metrics"].items()},
    }
    return MetricReport(metrics=metrics, details={"numeric": numeric["field_scores"], "tool": tool["details"]})


def evaluate_folder(
    prediction_dir: str | Path,
    ground_truth_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    weights_path: str | Path | None = None,
) -> MetricReport:
    """Evaluate a simple folder containing prediction and ground-truth JSON files."""

    pred_root = Path(prediction_dir)
    gt_root = Path(ground_truth_dir)
    prediction_path = pred_root if pred_root.is_file() else pred_root / "prediction_example.json"
    ground_truth_path = gt_root if gt_root.is_file() else gt_root / "number_ground_truth.json"
    trace_path = None if gt_root.is_file() else gt_root / "Main_trace.json"
    if trace_path is not None and not trace_path.exists():
        trace_path = None
    report = evaluate_prediction(prediction_path, ground_truth_path, trace_path=trace_path, weights_path=weights_path)
    if out_dir is not None:
        write_json(Path(out_dir) / "metrics.json", report.to_dict())
    return report
