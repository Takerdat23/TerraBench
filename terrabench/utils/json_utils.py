"""JSON helpers with clear error messages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    json_path = Path(path)
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file not found: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {json_path}: {exc}") from exc


def write_json(path: str | Path, payload: Any) -> Path:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return json_path
