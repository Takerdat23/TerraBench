"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from terrabench.utils.paths import project_root, resolve_path


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    config_path = resolve_path(path, root=project_root())
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return payload
