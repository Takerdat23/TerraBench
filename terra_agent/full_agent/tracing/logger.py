# A lightweight JSONL trace logger for agent research and production debugging.
# - Captures: run_started, plan, model_call (summary), tool_call/result, state_snapshot,
#             decision, final, run_finished, error
# - Avoids:   storing raw, free-form chain-of-thought (only short structured rationales)
# - Extras:   secret/PII scrubbing, span IDs, simple helper decorator for tools

from __future__ import annotations
import json, time, uuid, hashlib, re
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------
# Basic scrubbing utilities
# --------------------------
_SECRET_PATTERNS = [
    re.compile(r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}['\"]?", re.I),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # generic OpenAI-style pattern
]
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def _now_iso() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((t%1)*1000):03d}Z"

def _hash(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:16]

def scrub(obj: Any) -> Any:
    """Scrub secrets/PII from strings; pass through other types."""
    if isinstance(obj, str):
        t = _EMAIL_PATTERN.sub("[EMAIL]", obj)
        for pat in _SECRET_PATTERNS:
            t = pat.sub("[SECRET]", t)
        return t
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj

# --------------------------
# TraceLogger
# --------------------------
class TraceLogger:
    """
    JSONL logger for agent traces. Each line is a JSON record with fields:
      event_type, ts, run_id, agent, parent_span_id, span_id, payload

    Design choices:
      * Keep *summaries* of model I/O, not free-form CoT.
      * Strongly typed events for easy post-processing.
      * Span IDs so you can stitch call/result pairs and export to OTEL if desired.
    """
    def __init__(self, path: str | Path, run_id: Optional[str] = None, agent: str = "agent"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # newline-separated JSON; safe for large runs and streaming readers
        self._f = self.path.open("a", encoding="utf-8")
        self.run_id = run_id or str(uuid.uuid4())
        self.agent = agent
        self._dialog_buffer: list[dict[str, Any]] = []

    # ---- core emitter ----
    def _emit(self, event_type: str, payload: Dict[str, Any],
              parent_span_id: str | None = None, span_id: str | None = None):
        rec = {
            "event_type": event_type,
            "ts": _now_iso(),
            "run_id": self.run_id,
            "agent": self.agent,
            "parent_span_id": parent_span_id,
            "span_id": span_id,
            "payload": payload,
        }
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()

    # ---- lifecycle ----
    def run_started(self, task: str, model: str, tools: list[str] | None = None,
                    model_provider: str = "", data_provenance: Dict[str, Any] | None = None,
                    params: Dict[str, Any] | None = None):
        meta = {
            "task": scrub(task),
            "prompt_hash": _hash(task),
            "model": model,
            "model_provider": model_provider,
            "data_provenance": data_provenance or {},
            "params": params or {},
        }
        if tools:
            meta["tools"] = tools
        self._emit("run_started", {"meta": meta})

    def run_finished(self):
        self._emit("run_finished", {"meta": {}})

    # ---- planning & state ----
    def plan(self, steps: list[dict]):
        # expected step keys: id, goal, why_short (optional)
        clean = []
        for s in steps:
            s = dict(s)
            if "why_short" in s:
                s["why_short"] = scrub(s["why_short"])
            clean.append(s)
        self._emit("plan", {"plan": {"steps": clean}})

    def state_snapshot(self, memory_keys: list[str], state: Dict[str, Any]):
        # store only keys/metadata you’re comfortable keeping
        self._emit("state_snapshot", {"state_snapshot": {
            "memory_keys": memory_keys, "state": scrub(state)
        }})

    # ---- model I/O (summary only) ----
    def model_call(self, model: str, system: str, user: str,
                   assistant: str, function_call: Dict[str, Any] | None = None,
                   prompt_tokens: int | None = None, completion_tokens: int | None = None,
                   latency_ms: float | None = None, cost: float | None = None):
        self._emit("model_call", {
            "model_call": {
                "model": model,
                "input": {"system": scrub(system), "user": scrub(user), "messages": [], "tools": []},
                "output": {"assistant": scrub(assistant), "function_call": scrub(function_call or {})},
                "metrics": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                            "latency_ms": latency_ms, "cost": cost}
            }
        })

    # ---- tool calls/results ----
    def tool_call(self, tool_name: str, args: Dict[str, Any],
                  approval: str = "auto", parent_span_id: str | None = None) -> str:
        span_id = str(uuid.uuid4())
        self._emit("tool_call", {"tool_call": {
            "tool_name": tool_name,
            "args": scrub(args),
            "approval": approval
        }}, parent_span_id, span_id)
        return span_id

    def tool_result(self, tool_name: str, ok: bool, result: Any = None,
                    elapsed_ms: float | None = None, error: str | None = None,
                    parent_span_id: str | None = None, span_id: str | None = None):
        self._emit("tool_result", {"tool_result": {
            "tool_name": tool_name, "ok": ok, "result": scrub(result),
            "elapsed_ms": elapsed_ms, "error": scrub(error)
        }}, parent_span_id, span_id)

    # ---- decisions & final outputs ----
    def decision(self, action: str, reason_short: str = "", confidence: float | None = None):
        self._emit("decision", {"decision": {
            "action": action, "reason_short": scrub(reason_short), "confidence": confidence
        }})

    def final(self, answer: str, artifacts: list[str] | None = None, metrics: Dict[str, Any] | None = None):
        self._emit("final", {"final": {
            "answer": scrub(answer), "artifacts": artifacts or [], "metrics": metrics or {}
        }})

    # ---- GTA-style dialog capture ----
    def start_dialog(self) -> None:
        """Reset the in-memory buffer used for GTA-style dialog serialization."""
        self._dialog_buffer = []

    def log_dialog_turn(
        self,
        *,
        role: str,
        content: Any | None = None,
        thought: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        name: str | None = None,
    ) -> None:
        """
        Append a single dialog turn so we can later emit a GTA-style record.

        Args:
            role: Typically ``user``, ``assistant``, or ``tool``.
            content: Free-form response (string or dict). Scrubbed before storage.
            thought: Optional chain-of-thought summary to attach to assistant turns.
            tool_calls: Optional list of tool call descriptors that mirrors the
                OpenAI function-call schema.
            name: When ``role == "tool"`` provides the tool function name.
        """
        if not isinstance(role, str):
            raise TypeError("role must be a string")
        if not self._dialog_buffer:
            self._dialog_buffer = []
        entry: dict[str, Any] = {"role": role}
        if name and role == "tool":
            entry["name"] = name
        if content is not None:
            entry["content"] = scrub(content)
        if thought:
            entry["thought"] = scrub(thought)
        if tool_calls:
            entry["tool_calls"] = scrub(tool_calls)
        self._dialog_buffer.append(entry)

    def commit_dialog(
        self,
        *,
        whitelist: list[list[str]] | None = None,
        blacklist: list[list[str]] | None = None,
        extra_answer_fields: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit the buffered dialog in GTA-style format and clear the buffer.

        Args:
            whitelist: Nested list of acceptable answers (mirrors GTA spec).
            blacklist: Nested list of disallowed answers.
            extra_answer_fields: Optional additional metadata merged into the
                ``gt_answer`` object.
        """
        if not self._dialog_buffer:
            return
        gt_answer: dict[str, Any] = {
            "whitelist": whitelist,
            "blacklist": blacklist,
        }
        if extra_answer_fields:
            gt_answer.update(extra_answer_fields)
        payload = {
            "dialogs": list(self._dialog_buffer),
            "gt_answer": gt_answer,
        }
        self._emit("dialog_sample", payload)
        self._dialog_buffer = []

    # ---- errors & cleanup ----
    def error(self, message: str, stack: str | None = None):
        self._emit("error", {"error": {"message": scrub(message), "stack": scrub(stack or "")}})

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

# --------------------------
# Helper decorator for tools
# --------------------------
def trace_tool(logger: TraceLogger, tool_name: str):
    """
    Decorate a tool function so calls/results are logged automatically.
    Works for sync or async functions.
    """
    def decorator(func):
        if _is_coro(func):
            async def awrap(*args, **kwargs):
                import time as _t
                start = _t.time()
                span = logger.tool_call(tool_name, {"args": args, "kwargs": kwargs}, approval="auto")
                try:
                    out = await func(*args, **kwargs)
                    logger.tool_result(tool_name, ok=True, result=_shorten(out),
                                       elapsed_ms=(_t.time()-start)*1000, span_id=span)
                    return out
                except Exception as e:
                    logger.tool_result(tool_name, ok=False, error=str(e),
                                       elapsed_ms=(_t.time()-start)*1000, span_id=span)
                    raise
            return awrap
        else:
            def wrap(*args, **kwargs):
                import time as _t
                start = _t.time()
                span = logger.tool_call(tool_name, {"args": args, "kwargs": kwargs}, approval="auto")
                try:
                    out = func(*args, **kwargs)
                    logger.tool_result(tool_name, ok=True, result=_shorten(out),
                                       elapsed_ms=(_t.time()-start)*1000, span_id=span)
                    return out
                except Exception as e:
                    logger.tool_result(tool_name, ok=False, error=str(e),
                                       elapsed_ms=(_t.time()-start)*1000, span_id=span)
                    raise
            return wrap
    return decorator

def _is_coro(fn) -> bool:
    import inspect
    return inspect.iscoroutinefunction(fn)

def _shorten(obj: Any, max_len: int = 5000) -> Any:
    """Limit very large results; emit strings or small summaries."""
    try:
        s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if len(s) > max_len:
        return s[:max_len] + f"... <truncated {len(s)-max_len} chars>"
    return s

# --------------------------
# Example usage (comment out if importing)
# --------------------------
if __name__ == "__main__":
    # Minimal demo
    logger = TraceLogger("traces.jsonl", agent="DemoAgent")
    logger.run_started(task="demo task", model="local-hf", tools=["calc"])

    @trace_tool(logger, "calc")
    def calc(x: int, y: int) -> dict:
        return {"sum": x + y, "prod": x * y}

    logger.plan([{"id": "s1", "goal": "compute", "why_short": "need baseline"}])
    span = logger.tool_call("echo", {"msg": "hello"})
    logger.tool_result("echo", True, {"ok": True}, elapsed_ms=1.2, span_id=span)
    _ = calc(2, 3)
    logger.decision("report", "have numbers", 0.9)
    logger.final("done", artifacts=["/path/out.json"], metrics={"rmse": 2.1})
    logger.run_finished()
    logger.close()
