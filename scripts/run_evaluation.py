#!/usr/bin/env python
"""Evaluate TerraBench predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from terrabench.evaluation import evaluate_prediction
from terrabench.utils.json_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one prediction against ground truth.")
    parser.add_argument("--pred", required=True, help="Prediction JSON file.")
    parser.add_argument("--gt", required=True, help="Ground-truth JSON file.")
    parser.add_argument("--out", required=True, help="Directory for metrics.json.")
    parser.add_argument("--trace", help="Optional reference Main_trace.json.")
    parser.add_argument("--weights", help="Optional evaluation YAML with tooluse_weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_prediction(args.pred, args.gt, trace_path=args.trace, weights_path=args.weights)
    out = Path(args.out)
    write_json(out / "metrics.json", report.to_dict())
    print(f"Wrote metrics to {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
