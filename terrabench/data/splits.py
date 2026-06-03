"""Split-file utilities."""

from __future__ import annotations

from pathlib import Path

from terrabench.utils.json_utils import read_json


def load_split(path: str | Path) -> list[str]:
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = payload.get("items", [])
    if not isinstance(payload, list):
        raise ValueError("Split file must be a list or object with an items list.")
    return [str(item) for item in payload]
