"""Trace I/O helpers."""

from __future__ import annotations

from pathlib import Path

from terrabench.schemas.trace_schema import Trace
from terrabench.utils.json_utils import read_json, write_json


def load_trace(path: str | Path) -> Trace:
    return Trace.from_payload(read_json(path))


def write_trace(path: str | Path, trace: Trace) -> Path:
    return write_json(path, trace.to_dict())
