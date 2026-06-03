"""Dataset manifest utilities."""

from __future__ import annotations

from pathlib import Path

from terrabench.utils.json_utils import read_json


def load_manifest(path: str | Path) -> dict:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object.")
    return payload
