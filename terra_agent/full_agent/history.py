"""History compression and current-trace summarization helpers."""

from __future__ import annotations

import os
from typing import Any

from agentscope.message import Msg
from agentscope.model import ChatModelBase
from agentscope.token import TokenCounterBase
from pydantic import BaseModel, Field

from terra_agent.full_agent.config import (
    LOGGER,
    _REACT_COMPRESSION_CONFIG_CLS,
    _REACT_SUPPORTS_HISTORY_COMPRESSION,
    _env_int,
    _env_value,
    _load_env_json_object,
)
from terra_agent.full_agent.prompts import load_history_compression_prompt
from terra_agent.full_agent.utils import _prettify_json

class _AnthropicHistoryTokenCounter(TokenCounterBase):
    """Anthropic token counter compatible with the current SDK count_tokens shape."""

    def __init__(self, model_name: str, api_key: str, **kwargs: Any) -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key, **kwargs)
        self.model_name = model_name

    async def count(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> int:
        payload_messages = list(messages)
        system_blocks: Any = None

        if payload_messages and payload_messages[0].get("role") == "system":
            system_message = payload_messages.pop(0)
            system_blocks = system_message.get("content")

        extra_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": payload_messages,
            **kwargs,
        }
        if tools:
            extra_kwargs["tools"] = tools
        if system_blocks is not None:
            extra_kwargs["system"] = system_blocks

        res = await self.client.messages.count_tokens(**extra_kwargs)
        return res.input_tokens


class _GapAnalysisCompressionSummary(BaseModel):
    """Structured summary schema for conversation compression."""

    task_overview: str = Field(
        default="",
        max_length=500,
        description=(
            "The user's task, scope, constraints, and what counts as success. "
            "Mention key date windows, AOIs, thresholds, and required deliverables when relevant."
        ),
    )
    gap_analysis: str = Field(
        default="",
        max_length=1400,
        description=(
            "Gap analysis against the visible task or CSV instruction fields. "
            "Write concise bullet lines starting with DONE:, OUTSTANDING:, BLOCKED:, or CHECK:. "
            "Include grounded numeric results already established and clearly identify missing deliverables."
        ),
    )
    next_required_step: str = Field(
        default="",
        max_length=700,
        description=(
            "The single most important next step to continue the task. "
            "Name the tool(s), artifact(s), and concrete action needed next."
        ),
    )
    artifact_paths: str = Field(
        default="",
        max_length=2500,
        description=(
            "List every concrete artifact reference from the compressed history, one per line. "
            "Include local filesystem paths, upload paths, container paths, file_id values, output files, "
            "artifact names, and other exact execution references. Deduplicate exact duplicates, but do not "
            "omit distinct artifact references."
        ),
    )
    important_discoveries: str = Field(
        default="",
        max_length=1200,
        description=(
            "Critical findings, errors, blockers, interpretations, and resolved issues discovered so far. "
            "Include why failed approaches failed and what technical constraints matter for continuation."
        ),
    )
    context_to_preserve: str = Field(
        default="",
        max_length=1000,
        description=(
            "Any user preferences, formatting requirements, output schema requirements, benchmark constraints, "
            "or operational rules that the future continuation must preserve."
        ),
    )


_GAP_ANALYSIS_COMPRESSION_TEMPLATE = (
    "<system-info>Here is a summary of your previous work.\n"
    "# Task Overview\n"
    "{task_overview}\n\n"
    "# Gap Analysis Against Task Requirements\n"
    "{gap_analysis}\n\n"
    "# Next Required Step\n"
    "{next_required_step}\n\n"
    "# Artifact Paths\n"
    "{artifact_paths}\n\n"
    "# Important Discoveries\n"
    "{important_discoveries}\n\n"
    "# Context To Preserve\n"
    "{context_to_preserve}\n"
    "</system-info>"
)


def _default_gap_analysis_metadata() -> dict[str, str]:
    return _GapAnalysisCompressionSummary().model_dump()


def _normalize_gap_analysis_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    normalized = _default_gap_analysis_metadata()
    if not isinstance(metadata, dict):
        return normalized
    for key in normalized:
        value = metadata.get(key, normalized[key])
        normalized[key] = "" if value is None else str(value)
    return normalized


class _GapAnalysisCompressionChatModel(ChatModelBase):
    """Ensure structured compression metadata always contains all template keys."""

    def __init__(self, model: Any) -> None:
        super().__init__(
            model_name=getattr(model, "model_name", "unknown"),
            stream=bool(getattr(model, "stream", False)),
        )
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def _normalize_response_metadata(self, response: Any) -> Any:
        metadata = getattr(response, "metadata", None)
        setattr(response, "metadata", _normalize_gap_analysis_metadata(metadata))
        return response

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        response = await self._model(*args, **kwargs)
        if self.stream:
            async def _stream():
                async for chunk in response:
                    yield self._normalize_response_metadata(chunk)

            return _stream()
        return self._normalize_response_metadata(response)


def _build_history_compression_config(
    *,
    enabled: bool,
    provider: str,
    model: Any,
    formatter: Any,
) -> Any | None:
    if not enabled:
        return None

    if not _REACT_SUPPORTS_HISTORY_COMPRESSION:
        LOGGER.warning(
            "History compression was requested, but this AgentScope runtime does not expose "
            "ReActAgent compression hooks. Skipping --compress-history.",
        )
        return None

    normalized_provider = (provider or "").strip().lower()
    if normalized_provider not in {"anthropic", "gemini", "google", "google-gemini"}:
        LOGGER.warning(
            "History compression was requested, but it is only configured for the Anthropic and Gemini providers. "
            "Skipping compression for provider=%s.",
            provider,
        )
        return None

    if normalized_provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("--compress-history requires ANTHROPIC_API_KEY when using --model-provider anthropic.")

        client_kwargs = _load_env_json_object("ANTHROPIC_CLIENT_ARGS")
        base_url = os.getenv("ANTHROPIC_BASE_URL") or _env_value("CLIMATE_AGENT_ANTHROPIC_BASE_URL")
        if base_url:
            client_kwargs.setdefault("base_url", base_url.strip())

        token_counter = _AnthropicHistoryTokenCounter(
            model_name=model.model_name,
            api_key=api_key,
            **client_kwargs,
        )
        provider_label = "Anthropic"

    else:
        from agentscope.token import GeminiTokenCounter

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "--compress-history requires GEMINI_API_KEY or GOOGLE_API_KEY "
                "when using --model-provider gemini.",
            )

        client_kwargs = _load_env_json_object("GEMINI_CLIENT_ARGS")
        token_counter = GeminiTokenCounter(
            model_name=model.model_name,
            api_key=api_key,
            **client_kwargs,
        )
        provider_label = "Gemini"

    trigger_threshold = _env_int("CLIMATE_AGENT_COMPRESS_HISTORY_TRIGGER_TOKENS", 12000)
    keep_recent = _env_int("CLIMATE_AGENT_COMPRESS_HISTORY_KEEP_RECENT", 6)
    if trigger_threshold <= 0:
        raise ValueError("CLIMATE_AGENT_COMPRESS_HISTORY_TRIGGER_TOKENS must be greater than 0.")
    if keep_recent <= 0:
        raise ValueError("CLIMATE_AGENT_COMPRESS_HISTORY_KEEP_RECENT must be greater than 0.")

    LOGGER.info(
        "%s history compression enabled: trigger_threshold=%d keep_recent=%d model=%s",
        provider_label,
        trigger_threshold,
        keep_recent,
        getattr(model, "model_name", "unknown"),
    )

    compression_prompt_path, compression_prompt = load_history_compression_prompt()
    LOGGER.info("%s history compression prompt path: %s", provider_label, compression_prompt_path)

    compression_config_cls = _REACT_COMPRESSION_CONFIG_CLS
    if compression_config_cls is None:
        return None

    return compression_config_cls(
        enable=True,
        agent_token_counter=token_counter,
        trigger_threshold=trigger_threshold,
        keep_recent=keep_recent,
        compression_prompt=compression_prompt,
        summary_template=_GAP_ANALYSIS_COMPRESSION_TEMPLATE,
        summary_schema=_GapAnalysisCompressionSummary,
        compression_model=_GapAnalysisCompressionChatModel(model),
        compression_formatter=formatter,
    )


async def _collect_structured_metadata(response: Any, *, stream: bool) -> dict[str, Any] | None:
    """Collect the final structured metadata from a model response."""
    last_chunk = None
    if stream:
        async for chunk in response:
            last_chunk = chunk
    else:
        last_chunk = response

    metadata = getattr(last_chunk, "metadata", None) if last_chunk is not None else None
    return dict(metadata) if isinstance(metadata, dict) else None


async def _summarize_current_trace_for_prompt(
    *,
    current_trace: Any,
    user_request: str,
    model: Any,
    formatter: Any,
) -> str | None:
    """Compress the bootstrap current trace into a gap-analysis summary for prompt injection."""
    compression_prompt_path, compression_prompt = load_history_compression_prompt()
    LOGGER.info("Compressing current trace for prompt injection using %s", compression_prompt_path)

    trace_json = _prettify_json(current_trace, [])
    compression_request = await formatter.format(
        [
            Msg(
                "system",
                (
                    "You are preparing a compact continuation summary for an unfinished agent trace. "
                    "The summary will replace the raw current trace in the main agent prompt."
                ),
                "system",
            ),
            Msg("user", f"Task:\n{user_request}", "user"),
            Msg("user", f"Current trace JSON:\n{trace_json}", "user"),
            Msg("user", compression_prompt, "user"),
        ],
    )

    try:
        response = await model(
            compression_request,
            structured_model=_GapAnalysisCompressionSummary,
        )
        metadata = await _collect_structured_metadata(
            response,
            stream=bool(getattr(model, "stream", False)),
        )
    except Exception:
        LOGGER.exception("Failed to compress current trace for prompt injection")
        return None

    if not metadata:
        LOGGER.warning("Current trace compression returned no structured metadata; falling back to raw trace.")
        return None

    metadata = _normalize_gap_analysis_metadata(metadata)

    return (
        "<COMPRESSED_CURRENT_TRACE>\n"
        + _GAP_ANALYSIS_COMPRESSION_TEMPLATE.format(**metadata)
        + "\n</COMPRESSED_CURRENT_TRACE>"
    )
