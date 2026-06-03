"""Lightweight trace logger capturing agent requests and environment actions."""

import datetime
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from rpyc.core.netref import BaseNetref
from rpyc.utils.classic import obtain


class TraceLogger:
    """Record a sequence of trace events to a JSONL file for later inspection."""

    def __init__(self, path: Optional[str] = None) -> None:
        trace_path = Path(path or os.path.join("logs", "code_agent_trace.jsonl"))
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = trace_path
        self._lock = threading.Lock()
        self._file = trace_path.open("a", encoding="utf-8")
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def path(self) -> str:
        return str(self._path)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def begin_task(self, task_id: int, payload: Dict[str, Any]) -> str:
        """Record a new task request and return its trace identifier."""
        trace_id = uuid.uuid4().hex
        self._write(
            {
                "type": "agent_request",
                "task_id": task_id,
                "trace_id": trace_id,
                "payload": payload,
            }
        )
        return trace_id

    def log_environment_step(
        self,
        task_id: int,
        trace_id: str,
        turn: int,
        action: str,
        observation: Any,
        reward: Optional[float],
        valid_action: Optional[bool],
        info: Dict[str, Any],
        debug: Dict[str, Any],
    ) -> None:
        """Append a single environment action + observation pair to the trace."""
        self._write(
            {
                "type": "env_step",
                "task_id": task_id,
                "trace_id": trace_id,
                "turn": turn,
                "action": action,
                "observation": observation,
                "reward": reward,
                "valid_action": valid_action,
                "info": info,
                "debug": debug,
            }
        )

    def log_agent_response(
        self,
        task_id: int,
        trace_id: str,
        summary: Dict[str, Any],
        final_observation: Optional[str],
        turn_history: Dict[str, Any],
    ) -> None:
        self._write(
            {
                "type": "agent_response",
                "task_id": task_id,
                "trace_id": trace_id,
                "summary": summary,
                "final_observation": final_observation,
                "turn_history": turn_history,
            }
        )

    def _write(self, event: Dict[str, Any]) -> None:
        event.setdefault("timestamp", datetime.datetime.utcnow().isoformat() + "Z")
        safe_event = self._normalize(event)
        with self._lock:
            try:
                json.dump(safe_event, self._file, default=self._stringify)
                self._file.write("\n")
                self._file.flush()
            except Exception as err:
                self._logger.exception("Failed to write trace event: %s", err)

    def _stringify(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, BaseNetref):
            try:
                return self._stringify(obtain(obj))
            except Exception:
                return repr(obj)

        if isinstance(obj, dict):
            return {self._stringify(k): self._stringify(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            converted = [self._stringify(v) for v in obj]
            return tuple(converted) if isinstance(obj, tuple) else converted

        try:
            return obj.__dict__
        except Exception:
            return repr(obj)

    def _normalize(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, BaseNetref):
            try:
                return self._normalize(obtain(value))
            except Exception:
                return repr(value)

        if isinstance(value, dict):
            normalized = {}
            for key, val in value.items():
                normalized[self._normalize(key)] = self._normalize(val)
            return normalized

        if isinstance(value, (list, tuple, set)):
            normalized = [self._normalize(item) for item in value]
            return tuple(normalized) if isinstance(value, tuple) else normalized

        try:
            return self._normalize(value.__dict__)
        except Exception:
            return repr(value)
