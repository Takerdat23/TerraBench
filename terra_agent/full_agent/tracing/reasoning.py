"""Reasoning-trace extraction and writing helpers."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from agentscope.message import Msg

from terra_agent.full_agent.tracing.writer import TraceWriter

TOOLS_PATTERN = re.compile(r"<tools>(.*?)</tools>", re.IGNORECASE | re.DOTALL)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)

def _content_to_text(msg: Msg) -> str:
    """Best-effort conversion of a Msg content payload to plain text."""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        parts: list[str] = []
        for block in msg.content:
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return ""


def _parse_json_objects(raw: str) -> List[Dict[str, Any]]:
    """
    Parse one or more JSON objects contained in a <tools> block.

    Handles:
    - A single JSON object.
    - Multiple sibling JSON objects separated by whitespace or newlines.
    - A JSON array containing multiple objects.
    """
    cleaned = raw.strip()
    parsed: List[Dict[str, Any]] = []
    if not cleaned:
        return parsed

    # 1) Direct parse (object or array)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            parsed.extend([obj for obj in data if isinstance(obj, dict)])
        elif isinstance(data, dict):
            parsed.append(data)
        return parsed
    except Exception:
        pass

    # 2) Split sibling objects like "} {"
    pieces = re.split(r"}\s*{", cleaned)
    if len(pieces) > 1:
        stitched = "},{".join(pieces)
        try:
            data = json.loads(f"[{stitched}]")
            if isinstance(data, list):
                parsed.extend([obj for obj in data if isinstance(obj, dict)])
                return parsed
        except Exception:
            pass

    # 3) Fallback: extract balanced braces at the top level
    stack = 0
    start = None
    for idx, ch in enumerate(cleaned):
        if ch == "{":
            if stack == 0:
                start = idx
            stack += 1
        elif ch == "}":
            stack = max(0, stack - 1)
            if stack == 0 and start is not None:
                candidate = cleaned[start : idx + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        parsed.append(obj)
                except Exception:
                    parsed.append({"raw": candidate, "parse_error": True})
                start = None
    return parsed


def _prune_generate_response(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop tool_call objects that only invoke generate_response to keep traces compact."""
    pruned: List[Dict[str, Any]] = []
    for obj in objs:
        calls = obj.get("tool_calls")
        if isinstance(calls, list):
            filtered_calls = []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                name = call.get("function", {}).get("name")
                if name == "generate_response":
                    continue
                filtered_calls.append(call)
            if not filtered_calls:
                continue
            obj = dict(obj)
            obj["tool_calls"] = filtered_calls
        pruned.append(obj)
    return pruned


def extract_tools_blocks(messages: Sequence[Msg]) -> List[dict]:
    """Extract raw and parsed <tools> blocks from assistant messages."""
    results: List[dict] = []
    for idx, msg in enumerate(messages):
        if msg.role != "assistant":
            continue
        text = _content_to_text(msg)
        if not text:
            continue
        think_match = THINK_PATTERN.search(text)
        think_item = {"plan": think_match.group(1).strip()} if think_match else None
        for match in TOOLS_PATTERN.finditer(text):
            raw_block = match.group(1).strip()
            parsed_objs = _prune_generate_response(_parse_json_objects(raw_block))
            if think_item:
                parsed_objs = [think_item] + parsed_objs
            if not parsed_objs:
                continue
            results.append(
                {
                    "message_index": idx,
                    "timestamp": msg.timestamp,
                    # "raw": raw_block,
                    "parsed": parsed_objs,
                }
            )
    return results


def flatten_tool_calls(blocks: Iterable[dict]) -> List[Dict[str, Any]]:
    """Flatten all ``tool_calls`` arrays inside parsed blocks into a single list."""
    flat: List[Dict[str, Any]] = []
    for block in blocks:
        for obj in block.get("parsed", []) or []:
            calls = obj.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if isinstance(call, dict):
                        flat.append(call)
    return flat


class ReasoningTraceWriter(TraceWriter):
    """Extend TraceWriter by allowing an explicit reasoning_trace field."""

    def __init__(self, output_dir: str | Path = "./logs/trajectories", file_prefix: str = "single_agent") -> None:
        super().__init__(output_dir=output_dir, file_prefix=file_prefix)
        self._reasoning_trace: Any = None

    def set_reasoning_trace(self, trace: Any) -> None:
        """Attach structured reasoning content to embed in the final JSON."""
        self._reasoning_trace = trace

    def finalize(self) -> Path:
        """Flush the collected records to disk and return the file path (with reasoning_trace)."""
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload = {
            "run_id": self.run_id,
            "metadata": {**self.metadata, "finished_at": finished_at},
            # "trajectory": self._records,
        }
        if self._reasoning_trace:
            payload["reasoning_trace"] = self._reasoning_trace
        filename = f"{self.file_prefix}_{finished_at.replace(':', '').replace('-', '')}_{self.run_id[:8]}.json"
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path
