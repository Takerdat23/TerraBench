"""Portable TerraBench item loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terrabench.schemas.answer_schema import NumericGroundTruth
from terrabench.schemas.item_schema import BenchmarkItemSchema
from terrabench.schemas.trace_schema import Trace
from terrabench.utils.json_utils import read_json


@dataclass(frozen=True)
class BenchmarkItem:
    """Loaded benchmark item folder."""

    root: Path
    question: BenchmarkItemSchema
    trace: Trace
    ground_truth: list[NumericGroundTruth]
    raw: dict[str, Any]


def load_ground_truth(path: str | Path) -> list[NumericGroundTruth]:
    payload = read_json(path)
    if isinstance(payload, dict) and "answers" in payload:
        payload = payload["answers"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("number_ground_truth.json must be a list or object.")
    return [NumericGroundTruth.from_mapping(item) for item in payload if isinstance(item, dict)]


def load_item(item_dir: str | Path) -> BenchmarkItem:
    """Load a TerraBench item from a folder."""

    root = Path(item_dir).expanduser().resolve()
    question_path = root / "question_input.json"
    trace_path = root / "Main_trace.json"
    truth_path = root / "number_ground_truth.json"
    question_payload = read_json(question_path)
    trace_payload = read_json(trace_path)
    truth = load_ground_truth(truth_path)
    question = BenchmarkItemSchema.from_mapping(question_payload)
    trace = Trace.from_payload(trace_payload)
    return BenchmarkItem(
        root=root,
        question=question,
        trace=trace,
        ground_truth=truth,
        raw={
            "question_input": question_payload,
            "Main_trace": trace_payload,
            "number_ground_truth": [item.__dict__ for item in truth],
        },
    )
