"""Validation helpers for benchmark folders."""

from __future__ import annotations

from pathlib import Path

REQUIRED_ITEM_FILES = ("question_input.json", "Main_trace.json", "number_ground_truth.json")


def validate_item_folder(path: str | Path) -> list[str]:
    root = Path(path)
    errors = []
    for filename in REQUIRED_ITEM_FILES:
        if not (root / filename).exists():
            errors.append(f"Missing required file: {root / filename}")
    return errors
