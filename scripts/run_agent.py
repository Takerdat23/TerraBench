#!/usr/bin/env python
"""Run TerraAgent on one benchmark item."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from terra_agent.runner import run_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TerraAgent on a TerraBench item.")
    parser.add_argument("--item", required=True, help="Path to item directory.")
    parser.add_argument("--out", required=True, help="Output directory for prediction.json.")
    parser.add_argument("--mock-tools", action="store_true", help="Use deterministic mock tools.")
    parser.add_argument("--dry-run", action="store_true", help="Validate execution without calling tools.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_agent(args.item, args.out, mock_tools=args.mock_tools, dry_run=args.dry_run)
    print(f"Wrote prediction to {args.out}/prediction.json")


if __name__ == "__main__":
    main()
