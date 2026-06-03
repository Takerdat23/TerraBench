#!/usr/bin/env python
"""Generate a summary report from saved metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from terrabench.evaluation.reports import generate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate metric summary reports.")
    parser.add_argument("--metrics", required=True, help="Directory containing metric JSON/CSV files.")
    parser.add_argument("--out", required=True, help="Output report directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_report(args.metrics, args.out)
    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
