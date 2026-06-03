"""Configuration, environment helpers, and shared logging for the full agent."""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import Any

from agentscope.agent import ReActAgent
from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parents[1]
PROMPT_DIR = PACKAGE_DIR.parent / "prompts" / "full_agent"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "tool_errors.log"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger("terra_agent.full_agent")

_REACT_COMPRESSION_CONFIG_CLS = getattr(ReActAgent, "CompressionConfig", None)
_REACT_INIT_SUPPORTS_COMPRESSION_CONFIG = False
try:
    _REACT_INIT_SUPPORTS_COMPRESSION_CONFIG = (
        "compression_config" in inspect.signature(ReActAgent.__init__).parameters
    )
except (TypeError, ValueError):
    _REACT_INIT_SUPPORTS_COMPRESSION_CONFIG = False

_REACT_HAS_COMPRESS_HOOK = callable(getattr(ReActAgent, "_compress_memory_if_needed", None))
_REACT_SUPPORTS_HISTORY_COMPRESSION = (
    _REACT_COMPRESSION_CONFIG_CLS is not None
    and _REACT_INIT_SUPPORTS_COMPRESSION_CONFIG
    and _REACT_HAS_COMPRESS_HOOK
)

def _coerce_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Expected a boolean value, got {value!r}. Use true/false, yes/no, or 1/0.",
    )


def _parse_bool_arg(value: str | None) -> bool:
    try:
        return _coerce_bool(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _env_value(name: str, default: str | None = None) -> str | None:
    terra_name = name.replace("CLIMATE_AGENT_", "TERRABENCH_", 1)
    return os.getenv(terra_name) or os.getenv(name) or default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_value(name)
    if raw is None:
        return default
    return _coerce_bool(raw)


def _env_int(name: str, default: int) -> int:
    raw = _env_value(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


def _load_env_json_object(name: str) -> dict[str, Any]:
    raw = _env_value(name)
    if raw is None or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Environment variable {name} must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Environment variable {name} must decode to a JSON object.")
    return dict(value)
