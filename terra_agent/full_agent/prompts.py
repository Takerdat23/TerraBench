"""Prompt, path, and trace-loading helpers for full-agent runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from terra_agent.full_agent.config import BASE_DIR, PROMPT_DIR, _env_value
from terra_agent.full_agent.utils import _extract_trace, _load_json_file, _prettify_json

def load_system_prompt() -> tuple[Path, str]:
    """Read the agent system prompt from disk so users can tweak it freely."""
    prompt_env = _env_value("CLIMATE_AGENT_PROMPT_PATH")
    if prompt_env:
        prompt_path = Path(prompt_env).expanduser()
        if not prompt_path.is_absolute():
            prompt_path = (BASE_DIR / prompt_path).resolve()
    else:
        prompt_path = PROMPT_DIR / "Annotating_prompt.yaml"

    if not prompt_path.is_file():
        raise FileNotFoundError(f"System prompt file not found at {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8")
    return prompt_path, prompt_text


def load_history_compression_prompt() -> tuple[Path, str]:
    """Read the history compression prompt from disk so it can be tuned locally."""
    prompt_env = _env_value("CLIMATE_AGENT_COMPRESS_HISTORY_PROMPT_PATH")
    if prompt_env:
        prompt_path = Path(prompt_env).expanduser()
        if not prompt_path.is_absolute():
            prompt_path = (BASE_DIR / prompt_path).resolve()
    else:
        prompt_path = PROMPT_DIR / "history_compression_prompt.md"

    if not prompt_path.is_file():
        raise FileNotFoundError(f"History compression prompt file not found at {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    return prompt_path, prompt_text


def _resolve_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (BASE_DIR / resolved).resolve()
    return resolved


def _load_trace_for_prompt(
    trace_path: Path | None,
    *,
    prefer_reasoning: bool,
) -> Any | None:
    if not trace_path:
        return None
    payload = _load_json_file(trace_path)
    return _extract_trace(payload, prefer_reasoning=prefer_reasoning)


def _fill_prompt_template(
    template: str,
    *,
    current_trace: Any | None,
    extra_message: str | None,
) -> str:
    current_trace_text = (
        current_trace
        if isinstance(current_trace, str)
        else _prettify_json(current_trace, []) if current_trace is not None else ""
    )
    replacements = {
        "<CURRENT_TRACE_JSON>": current_trace_text,
        "<EXTRA_GUIDANCE>": (extra_message or "").strip(),
        "<DUMMY_TRACE_JSON>": "",
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template
