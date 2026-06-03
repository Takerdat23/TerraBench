#!/usr/bin/env python
"""Run the packaged TerraBench full-agent CLI."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    runpy.run_module("terra_agent.full_agent.main", run_name="__main__")
