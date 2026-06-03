"""Agent subclasses and stop-state helpers for full-agent execution."""

from __future__ import annotations

from typing import Any

from agentscope.agent import ReActAgent

class MathAgentObservationStopError(RuntimeError):
    """Abort the current run after a fatal math-agent observation is recorded."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _MathAgentStopState:
    """Track whether a math-agent observation should terminate the current run."""

    def __init__(self, stop_status_codes: set[int] | None = None) -> None:
        self.stop_status_codes = set(stop_status_codes or set())
        self.exception: MathAgentObservationStopError | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.stop_status_codes)

    def capture(self, response: Any) -> None:
        if not self.enabled or self.exception is not None:
            return

        metadata = getattr(response, "metadata", None)
        if not isinstance(metadata, dict) or not metadata.get("error"):
            return

        raw_status = metadata.get("status_code")
        try:
            status_code = int(raw_status)
        except (TypeError, ValueError):
            return
        if status_code not in self.stop_status_codes:
            return

        message = str(metadata.get("message") or f"Math agent returned status {status_code}")
        self.exception = MathAgentObservationStopError(message, status_code=status_code)


class _StopAwareReActAgent(ReActAgent):
    """Raise after a fatal tool observation has been written into memory."""

    def __init__(
        self,
        *args: Any,
        math_agent_stop_state: _MathAgentStopState | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._math_agent_stop_state = math_agent_stop_state

    def _raise_if_math_agent_stop_requested(self) -> None:
        stop_state = self._math_agent_stop_state
        if stop_state and stop_state.exception is not None:
            raise stop_state.exception

    async def _acting(self, tool_call: Any) -> dict | None:
        structured_output = await super()._acting(tool_call)
        self._raise_if_math_agent_stop_requested()
        return structured_output


class _ToolStepCompressionReActAgent(_StopAwareReActAgent):
    """Run the built-in compression check immediately after each tool step."""

    async def _acting(self, tool_call: Any) -> dict | None:
        structured_output = await super()._acting(tool_call)
        await self._compress_memory_if_needed()
        return structured_output
