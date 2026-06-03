#!/usr/bin/env python
"""Run the public TerraBench smoke test."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from terrabench.data import load_item
from terrabench.evaluation import evaluate_prediction
from terrabench.utils.json_utils import write_json
from terra_agent.runner import run_agent


def main() -> None:
    root = ROOT
    item_dir = root / "examples" / "minimal_item"
    out_dir = root / "outputs" / "smoke_test"
    item = load_item(item_dir)
    prediction = run_agent(item_dir, out_dir, mock_tools=True)
    prediction_path = out_dir / "prediction.json"
    report = evaluate_prediction(
        prediction_path,
        item_dir / "number_ground_truth.json",
        trace_path=item_dir / "Main_trace.json",
    )
    write_json(out_dir / "smoke_metrics.json", report.to_dict())
    print(
        "Smoke test passed: "
        f"item={item.question.item_id} "
        f"Hit@tol={report.metrics['Hit@tol']:.3f} "
        f"ToolUseScore={report.metrics['ToolUseScore']:.3f}"
    )


if __name__ == "__main__":
    main()
