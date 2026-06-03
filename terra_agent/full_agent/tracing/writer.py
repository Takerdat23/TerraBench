"""
Utilities for capturing single-agent trajectories and storing them as JSON.

The TraceWriter consumes AgentScope `Msg` objects and normalises them into a
lightweight record format that mirrors the high-level conversation stages:

<plan>         – planner-style kickoff messages
<think>        – internal reasoning emitted by the agent
<tools>        – tool invocation requests (name + arguments)
<observation>  – tool call results
<reflection>   – mid-run critiques, synonyms with <verify>
<answer>       – final user-facing reply

Plain user or assistant messages without tags are still preserved so runs can
be replayed later.  Each run emits a single JSON file inside the configured
output directory.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from agentscope.message import Msg

TaggedRecord = Dict[str, Any]


class TraceWriter:
    """Collect messages from a run and persist them as structured JSON."""

    _TAG_PATTERN = re.compile(
        r"<(plan|think|tools|observation|reflection|verify|answer)>(.*?)</\1>",
        re.IGNORECASE | re.DOTALL,
    )
    _FINAL_JSON_PATTERN = re.compile(
        r"<final_json>\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*</final_json>",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(
        self,
        output_dir: str | Path = "./logs/trajectories",
        file_prefix: str = "single_agent",
        number_json_filename: str | None = "number_ground_truth.json",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_prefix = file_prefix
        self.number_json_filename = number_json_filename
        self.run_id = str(uuid.uuid4())
        self.metadata: Dict[str, Any] = {}
        self._records: List[TaggedRecord] = []

    # ------------------------------------------------------------------#
    # Lifecycle helpers
    # ------------------------------------------------------------------#
    def start_run(
        self,
        *,
        prompt_path: str | None = None,
        user_request: str | None = None,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Initialise per-run metadata."""
        self.metadata = {
            "prompt_path": prompt_path,
            "user_request": user_request,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra_metadata:
            self.metadata.update(extra_metadata)

    def add_messages(self, messages: Sequence[Msg]) -> None:
        """Append a batch of AgentScope messages to the trace."""
        total = len(messages)
        for idx, msg in enumerate(messages):
            is_last = idx == total - 1
            self._records.extend(self._serialise_msg(msg, is_last=is_last))

    def finalize(self) -> Path:
        """Flush the collected records to disk and return the file path."""
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload = {
            "run_id": self.run_id,
            "metadata": {**self.metadata, "finished_at": finished_at},
            "trajectory": self._records,
        }
        filename = f"{self.file_prefix}_{finished_at.replace(':', '').replace('-', '')}_{self.run_id[:8]}.json"
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._write_number_ground_truth(payload)
        return path

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _serialise_msg(self, msg: Msg, *, is_last: bool) -> List[TaggedRecord]:
        records: List[TaggedRecord] = []

        if msg.role == "user":
            records.append(
                {
                    "type": "user",
                    "sender": msg.name,
                    "timestamp": msg.timestamp,
                    "content": self._normalise_text(msg.content),
                }
            )
            return records

        if isinstance(msg.content, str):
            records.extend(
                self._serialise_text_content(
                    msg.content, sender=msg.name, role=msg.role, timestamp=msg.timestamp
                )
            )
        else:
            records.extend(
                self._serialise_blocks(
                    blocks=msg.content,
                    sender=msg.name,
                    role=msg.role,
                    timestamp=msg.timestamp,
                )
            )

        if is_last:
            for rec in reversed(records):
                if rec.get("type") in {"assistant", "text"}:
                    rec["type"] = "answer"
                    break
                if rec.get("type") == "answer":
                    break

        return records

    def _serialise_text_content(
        self,
        content: str,
        *,
        sender: str,
        role: str,
        timestamp: str,
    ) -> List[TaggedRecord]:
        matches = list(self._TAG_PATTERN.finditer(content))
        if not matches:
            return [
                {
                    "type": "assistant" if role == "assistant" else role,
                    "sender": sender,
                    "timestamp": timestamp,
                    "content": content.strip(),
                }
            ]

        records: List[TaggedRecord] = []
        cursor = 0
        for match in matches:
            start, end = match.span()
            if start > cursor:
                leading = content[cursor:start].strip()
                if leading:
                    records.append(
                        {
                            "type": role if role != "assistant" else "assistant",
                            "sender": sender,
                            "timestamp": timestamp,
                            "content": leading,
                        }
                    )
            tag = match.group(1).lower()
            rec_type = "reflection" if tag in {"reflection", "verify"} else tag
            records.append(
                {
                    "type": rec_type,
                    "sender": sender,
                    "timestamp": timestamp,
                    "content": match.group(2).strip(),
                }
            )
            cursor = end

        if cursor < len(content):
            trailing = content[cursor:].strip()
            if trailing:
                records.append(
                    {
                        "type": role if role != "assistant" else "assistant",
                        "sender": sender,
                        "timestamp": timestamp,
                        "content": trailing,
                    }
                )
        return records

    def _serialise_blocks(
        self,
        *,
        blocks: Iterable[Dict[str, Any]],
        sender: str,
        role: str,
        timestamp: str,
    ) -> List[TaggedRecord]:
        records: List[TaggedRecord] = []
        for block in blocks:
            # print("=============================")
            # print("Block", block)
            # print("=============================")
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text", ""))
                if text:
                    records.extend(
                        self._serialise_text_content(
                            text,
                            sender=sender,
                            role=role,
                            timestamp=timestamp,
                        )
                    )
            elif block_type == "tool_use":
                records.append(
                    {
                        "type": "tools",
                        "sender": sender,
                        "timestamp": timestamp,
                        "name": block.get("name"),
                        "args": self._safe_json(block.get("input")),
                    }
                )
            elif block_type == "tool_result":
                records.append(
                    {
                        "type": "observation",
                        "sender": sender,
                        "timestamp": timestamp,
                        "name": block.get("name"),
                        "output": self._safe_json(block.get("output")),
                    }
                )
            else:
                records.append(
                    {
                        "type": block_type or "content",
                        "sender": sender,
                        "timestamp": timestamp,
                        "raw": self._safe_json(block),
                    }
                )
        if not records and role == "assistant":
            records.append(
                {
                    "type": "assistant",
                    "sender": sender,
                    "timestamp": timestamp,
                    "content": "",
                }
            )
        return records

    def _write_number_ground_truth(self, payload: Dict[str, Any]) -> None:
        if not self.number_json_filename:
            return
        final_json = self._extract_final_json_payload(payload.get("trajectory", []))
        if final_json is None:
            return
        ground_truth_path = self.output_dir / self.number_json_filename
        with ground_truth_path.open("w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)

    def _extract_final_json_payload(self, trajectory: Sequence[TaggedRecord]) -> Any | None:
        for record in reversed(list(trajectory)):
            content = record.get("content")
            if not isinstance(content, str):
                continue
            match = self._FINAL_JSON_PATTERN.search(content)
            if not match:
                continue
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(data, (dict, list)):
                return data
        return None

    @staticmethod
    def _normalise_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return repr(content)

    @staticmethod
    def _safe_json(obj: Any) -> Any:
        try:
            json.dumps(obj, ensure_ascii=False)
            return obj
        except TypeError:
            try:
                return json.loads(json.dumps(obj, default=str))
            except Exception:
                return str(obj)
