"""Convenience runner helpers."""

from __future__ import annotations

from pathlib import Path

from terra_agent.agent import TerraAgent


def run_agent(item_dir: str | Path, out_dir: str | Path, *, mock_tools: bool = False, dry_run: bool = False) -> dict:
    agent = TerraAgent(mock_tools=mock_tools, dry_run=dry_run)
    return agent.run(item_dir, out_dir=out_dir)
