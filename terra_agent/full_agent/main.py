"""Compatibility entrypoint for the packaged full TerraBench agent."""

from __future__ import annotations

from terra_agent.full_agent.cli import cli_main, parse_args
from terra_agent.full_agent.config import (
    BASE_DIR,
    PROMPT_DIR,
    _env_bool,
    _env_value,
    _parse_bool_arg,
)
from terra_agent.full_agent.prompts import _resolve_path
from terra_agent.full_agent.runtime import DEFAULT_USER_REQUEST, main

__all__ = [
    "BASE_DIR",
    "PROMPT_DIR",
    "DEFAULT_USER_REQUEST",
    "_env_bool",
    "_env_value",
    "_parse_bool_arg",
    "_resolve_path",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    cli_main()
