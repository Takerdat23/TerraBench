"""
Shared utilities for TerraBench entrypoints and tooling wrappers.
"""
from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import re
import ast
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable
from collections import Counter

from agentscope.formatter import AnthropicChatFormatter, DeepSeekChatFormatter, OpenAIChatFormatter
from agentscope.message import Msg, TextBlock, ToolUseBlock
from agentscope.model import OpenAIChatModel, ChatResponse
from agentscope.tool import ToolResponse, Toolkit

from terra_agent.full_agent.tracing.logger import TraceLogger

LOGGER = logging.getLogger("climate_agent.utils")
TOOL_LOGGER = logging.getLogger("climate_agent.tools")

_TAG_PATTERN = re.compile(
    r"<(plan|think|tools|observation|reflection|verify|answer)>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_OPENAI_REASONING_EFFORT_ENV = "OPENAI_REASONING_EFFORT"
_SUPPORTED_OPENAI_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)


def _looks_like_deepseek(provider: str, base_url: str | None, model_name: str | None) -> bool:
    if provider == "deepseek":
        return True
    if base_url and "deepseek" in base_url.lower():
        return True
    if model_name and "deepseek" in model_name.lower():
        return True
    return False


def _looks_like_mistral_family(model_name: str | None) -> bool:
    normalized = (model_name or "").strip().lower()
    return any(tag in normalized for tag in ("mistral", "ministral", "mixtral"))


def _looks_like_llama_family(model_name: str | None) -> bool:
    normalized = (model_name or "").strip().lower()
    return "llama" in normalized


def _supports_openai_reasoning_effort(model_name: str | None) -> bool:
    normalized = (model_name or "").strip().lower()
    if not normalized:
        return False
    if normalized.startswith("gpt-5") and "-chat" not in normalized:
        return True
    if normalized.startswith("gpt-oss-"):
        return True
    return bool(re.match(r"^o\d(?:$|[-_])", normalized))


def _resolve_openai_reasoning_effort(model_name: str | None) -> str | None:
    if not _supports_openai_reasoning_effort(model_name):
        return None

    raw_effort = os.getenv(_OPENAI_REASONING_EFFORT_ENV, "high").strip().lower()
    if not raw_effort or raw_effort in {"off", "disable", "disabled"}:
        return None
    if raw_effort not in _SUPPORTED_OPENAI_REASONING_EFFORTS:
        LOGGER.warning(
            "Ignoring unsupported %s=%r for model=%s",
            _OPENAI_REASONING_EFFORT_ENV,
            raw_effort,
            model_name or "unknown",
        )
        return None
    return raw_effort


def _is_unsupported_openai_reasoning_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "reasoning_effort" not in message and "reasoning effort" not in message:
        return False
    return any(
        marker in message
        for marker in (
            "unsupported",
            "unexpected keyword argument",
            "unknown parameter",
            "unrecognized request argument",
            "extra inputs are not permitted",
            "did you mean",
            "not allowed",
            "not permitted",
            "invalid",
        )
    )


class _DeepSeekReasoningFormatter(DeepSeekChatFormatter):
    async def _format(self, msgs: list[Msg]) -> list[dict[str, Any]]:
        messages = await super()._format(msgs)
        for msg in messages:
            if msg.get("role") == "assistant" and "reasoning_content" not in msg:
                # DeepSeek requires reasoning_content on assistant messages, even if empty.
                msg["reasoning_content"] = ""
        return messages


class _MistralCompatFormatter(OpenAIChatFormatter):
    async def _format(self, msgs: list[Msg]) -> list[dict[str, Any]]:
        messages = await super()._format(msgs)
        for msg in messages:
            if msg.get("role") != "tool":
                msg.pop("name", None)
        return messages


class _ToolForcedOpenAIChatModel(OpenAIChatModel):
    @staticmethod
    def _strip_code_fences(text: str) -> list[str]:
        stripped = text.strip()
        candidates = [stripped]
        fenced = re.findall(r"```(?:json|python)?\s*(.*?)```", stripped, re.DOTALL)
        candidates.extend(block.strip() for block in fenced if block.strip())
        return candidates

    @staticmethod
    def _extract_balanced_objects(text: str) -> list[str]:
        snippets: list[str] = []
        start = None
        depth = 0
        in_string = False
        escaped = False

        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and start is not None:
                    snippets.append(text[start : index + 1])
                    start = None

        return snippets

    @staticmethod
    def _parse_tool_payload(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return None

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @classmethod
    def _normalize_text_tool_call(
        cls,
        obj: Any,
    ) -> ToolUseBlock | None:
        if not isinstance(obj, dict):
            return None

        name = obj.get("name")
        params = obj.get("parameters", obj.get("arguments", {}))
        params_dict = cls._parse_tool_payload(params)

        if not isinstance(name, str) or not name.strip() or params_dict is None:
            return None

        return ToolUseBlock(
            type="tool_use",
            id=f"llama_call_{uuid.uuid4().hex[:12]}",
            name=name.strip(),
            input=params_dict,
        )

    @classmethod
    def _parse_text_tool_calls(cls, text: str) -> list[ToolUseBlock]:
        for candidate in cls._strip_code_fences(text):
            tool_block = cls._normalize_text_tool_call(cls._parse_tool_payload(candidate))
            if tool_block is not None:
                return [tool_block]

            objects = [
                cls._normalize_text_tool_call(cls._parse_tool_payload(snippet))
                for snippet in cls._extract_balanced_objects(candidate)
            ]
            parsed = [obj for obj in objects if obj is not None]
            if parsed:
                return parsed

        return []

    @classmethod
    def _rewrite_llama_text_tool_calls(
        cls,
        response: ChatResponse,
    ) -> ChatResponse:
        if any(block.get("type") == "tool_use" for block in response.content):
            return response

        text_parts = [
            block.get("text", "")
            for block in response.content
            if block.get("type") == "text"
        ]
        if not text_parts:
            return response

        tool_calls = cls._parse_text_tool_calls("\n".join(text_parts))
        if not tool_calls:
            return response

        LOGGER.warning(
            "Recovered %d plain-text Llama tool call(s) into structured "
            "tool_use blocks.",
            len(tool_calls),
        )
        return ChatResponse(
            content=tool_calls,
            id=response.id,
            created_at=response.created_at,
            type=response.type,
            usage=response.usage,
            metadata=response.metadata,
        )

    @staticmethod
    def _is_retryable_mistral_tool_error(exc: Exception) -> bool:
        message = str(exc).lower()
        markers = (
            "invalid json: eof while parsing a list",
            "invalid json",
            "json_invalid",
        )
        return any(marker in message for marker in markers)

    @staticmethod
    def _build_retry_messages(messages: list[dict]) -> list[dict]:
        retried = deepcopy(messages)
        repair_instruction = (
            "Your previous response produced malformed function-call JSON. "
            "Retry now. Respond with exactly one valid tool call or a valid "
            "sequence of tool calls only. Do not output prose, markdown, code "
            "fences, or explanations. Ensure every JSON object and array is "
            "fully closed and syntactically valid."
        )

        if retried and retried[0].get("role") == "system":
            content = retried[0].get("content")
            if isinstance(content, str):
                retried[0]["content"] = f"{content}\n\n{repair_instruction}"
            elif isinstance(content, list):
                content.append({"type": "text", "text": f"\n\n{repair_instruction}"})
            else:
                retried[0]["content"] = repair_instruction
        else:
            retried.insert(0, {"role": "system", "content": repair_instruction})

        return retried

    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        structured_model: Any = None,
        **kwargs: Any,
    ) -> Any:
        if (
            tools
            and tool_choice is None
            and structured_model is None
            and _looks_like_mistral_family(self.model_name)
        ):
            tool_choice = "required"

        call_kwargs = dict(kwargs)
        reasoning_effort = _resolve_openai_reasoning_effort(self.model_name)
        inserted_reasoning_effort = False
        if reasoning_effort and "reasoning_effort" not in call_kwargs:
            call_kwargs["reasoning_effort"] = reasoning_effort
            inserted_reasoning_effort = True

        super_call = super(_ToolForcedOpenAIChatModel, self).__call__

        async def _invoke(call_messages: list[dict], invoke_kwargs: dict[str, Any]) -> Any:
            response = await super_call(
                messages=call_messages,
                tools=tools,
                tool_choice=tool_choice,
                structured_model=structured_model,
                **invoke_kwargs,
            )
            if (
                tools
                and structured_model is None
                and isinstance(response, ChatResponse)
                and _looks_like_llama_family(self.model_name)
            ):
                return self._rewrite_llama_text_tool_calls(response)
            return response

        try:
            return await _invoke(messages, call_kwargs)
        except Exception as exc:
            if inserted_reasoning_effort and _is_unsupported_openai_reasoning_error(exc):
                LOGGER.warning(
                    "Retrying model=%s without reasoning_effort after unsupported "
                    "parameter error.",
                    self.model_name,
                )
                retry_kwargs = dict(call_kwargs)
                retry_kwargs.pop("reasoning_effort", None)
                return await _invoke(messages, retry_kwargs)

            if not (
                tools
                and structured_model is None
                and _looks_like_mistral_family(self.model_name)
                and self._is_retryable_mistral_tool_error(exc)
            ):
                raise

            LOGGER.warning(
                "Retrying Mistral-family tool call after malformed JSON "
                "response from the model server."
            )
            retry_messages = self._build_retry_messages(messages)
            return await _invoke(retry_messages, call_kwargs)


class TokenUsageReporter:
    """Accumulate and log per-call token usage from model responses."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or LOGGER
        self._calls = 0
        self._total_input = 0
        self._total_output = 0
        self._total_time = 0.0

    def record(self, usage: Any, *, model_name: str) -> None:
        self._calls += 1
        if usage is None:
            self._logger.info(
                "Token usage call %d (model=%s): unavailable",
                self._calls,
                model_name,
            )
            return

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        time_sec = getattr(usage, "time", None)
        if isinstance(time_sec, (int, float)):
            self._total_time += float(time_sec)

        self._total_input += input_tokens
        self._total_output += output_tokens

        self._logger.info(
            "Token usage call %d (model=%s): input=%d output=%d time=%s",
            self._calls,
            model_name,
            input_tokens,
            output_tokens,
            f"{time_sec:.2f}s" if isinstance(time_sec, (int, float)) else "n/a",
        )

    def log_summary(self) -> None:
        if not self._calls:
            self._logger.info("Token usage summary: no calls recorded")
            return
        self._logger.info(
            "Token usage summary: calls=%d input=%d output=%d time=%s",
            self._calls,
            self._total_input,
            self._total_output,
            f"{self._total_time:.2f}s" if self._total_time else "n/a",
        )


class ReportingChatModel:
    """Wrap a chat model to report token usage from responses."""

    def __init__(
        self,
        model: Any,
        reporter: TokenUsageReporter,
        *,
        report_detail: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self._model = model
        self._reporter = reporter
        self._report_detail = report_detail
        self._logger = logger or LOGGER

    @property
    def model_name(self) -> str:
        return getattr(self._model, "model_name", "unknown")

    @property
    def stream(self) -> bool:
        return bool(getattr(self._model, "stream", False))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._report_detail:
            self._log_payload_detail(args, kwargs)
        res = await self._model(*args, **kwargs)
        if self.stream:
            async def _stream():
                last_usage = None
                async for chunk in res:
                    if getattr(chunk, "usage", None):
                        last_usage = chunk.usage
                    yield chunk
                self._reporter.record(last_usage, model_name=self.model_name)

            return _stream()

        self._reporter.record(getattr(res, "usage", None), model_name=self.model_name)
        return res

    def _log_payload_detail(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        messages = args[0] if args else kwargs.get("messages")
        tools = kwargs.get("tools")
        if not isinstance(messages, list):
            self._logger.info(
                "Prompt detail (model=%s): messages unavailable (type=%s)",
                self.model_name,
                type(messages).__name__,
            )
            return

        role_counts = Counter()
        total_text_chars = 0
        system_text_chars = 0
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or "unknown"
            role_counts[role] += 1
            content = msg.get("content")
            # print(content)
            char_count = self._count_text_chars(content)
            total_text_chars += char_count
            if idx == 0 and msg.get("role") == "system":
                system_text_chars = char_count

        tools_count = len(tools) if isinstance(tools, list) else 0
        prompt_json_chars = self._safe_json_len(messages)
        tools_json_chars = self._safe_json_len(tools) if tools is not None else 0

        self._logger.info(
            "Prompt detail (model=%s): messages=%d roles=%s system_text_chars=%d "
            "text_chars=%d payload_chars=%d tools=%d tools_chars=%d",
            self.model_name,
            len(messages),
            dict(role_counts),
            system_text_chars,
            total_text_chars,
            prompt_json_chars,
            tools_count,
            tools_json_chars,
        )

    @staticmethod
    def _count_text_chars(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value)
        if isinstance(value, list):
            return sum(ReportingChatModel._count_text_chars(item) for item in value)
        if isinstance(value, dict):
            return sum(
                ReportingChatModel._count_text_chars(item)
                for item in value.values()
            )
        return 0

    @staticmethod
    def _safe_json_len(value: Any) -> int:
        try:
            return len(json.dumps(value, ensure_ascii=False))
        except TypeError:
            return len(str(value))


def _extract_text_from_blocks(blocks: Iterable[dict[str, Any]]) -> str:
    texts: list[str] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            texts.append(str(block.get("text", "")))
    return "\n".join(part for part in texts if part).strip()


def _strip_known_tags(text: str) -> str:
    return _TAG_PATTERN.sub("", text).strip()


def _extract_thought(text: str | None) -> str | None:
    if not text:
        return None
    match = _THINK_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def _extract_tool_calls(msg: Msg) -> list[dict[str, Any]]:
    if not isinstance(msg.content, list):
        return []
    calls: list[dict[str, Any]] = []
    for block in msg.content:
        if block.get("type") != "tool_use":
            continue
        calls.append(
            {
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": block.get("input", {}),
                },
            }
        )
    return calls


def _format_tool_result_output(output: Any) -> Any:
    if isinstance(output, list):
        texts = []
        for item in output:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
        if texts:
            return {"type": "text", "content": "\n".join(texts).strip()}
    return output


def _record_dialog(logger: TraceLogger, messages: list[Msg]) -> str | None:
    """Convert AgentScope memory messages into GTA-style dialog entries."""
    logger.start_dialog()
    final_answer: str | None = None
    for msg in messages:
        if msg.role == "user":
            text = (
                msg.content.strip()
                if isinstance(msg.content, str)
                else _extract_text_from_blocks(msg.content or [])
            )
            text = _strip_known_tags(text) if text else ""
            if text:
                logger.log_dialog_turn(role="user", content=text)
        elif msg.role == "assistant":
            raw_text = (
                msg.content.strip()
                if isinstance(msg.content, str)
                else _extract_text_from_blocks(msg.content or [])
            )
            thought = _extract_thought(raw_text) if raw_text else None
            reply = _strip_known_tags(raw_text) if raw_text else ""
            tool_calls = _extract_tool_calls(msg)
            payload: dict[str, Any] = {"role": "assistant"}
            if tool_calls:
                payload["tool_calls"] = tool_calls
            if thought:
                payload["thought"] = thought
            if reply:
                payload["content"] = reply
                final_answer = reply
            if payload.keys() - {"role"}:
                logger.log_dialog_turn(**payload)
        elif msg.role == "system" and isinstance(msg.content, list):
            for block in msg.content:
                if block.get("type") != "tool_result":
                    continue
                logger.log_dialog_turn(
                    role="tool",
                    name=block.get("name"),
                    content=_format_tool_result_output(block.get("output")),
                )
    whitelist = [[final_answer.strip()]] if final_answer else None
    logger.commit_dialog(whitelist=whitelist)
    return final_answer


def log_tool_errors(tool_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a tool so failures record full stack traces before re-raising."""

    @functools.wraps(tool_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return tool_fn(*args, **kwargs)
        except Exception:  # pragma: no cover - safety logging
            TOOL_LOGGER.exception("Tool %s failed", tool_fn.__name__)
            raise

    return wrapper


def _tool_response_error(message: str) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {message}")],
        metadata={"error": True, "message": message},
    )


def _env_value(var_name: str, default: str | None = None) -> str | None:
    terra_name = var_name.replace("CLIMATE_AGENT_", "TERRABENCH_", 1)
    return os.getenv(terra_name) or os.getenv(var_name) or default


def _load_env_json(var_name: str) -> dict[str, Any] | None:
    """Parse a JSON object stored in an environment variable."""
    raw = _env_value(var_name)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Ignoring %s because it is not valid JSON (%s)", var_name, exc)
        return None
    if not isinstance(value, dict):
        LOGGER.warning("Ignoring %s because it must decode to a JSON object", var_name)
        return None
    return value


def _load_env_float(var_name: str) -> float | None:
    """Parse a float environment variable, returning None when unset/invalid."""
    raw = _env_value(var_name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        LOGGER.warning("Ignoring %s because %r is not a float", var_name, raw)
        return None


def _load_env_int(var_name: str) -> int | None:
    """Parse an integer environment variable, returning None when unset/invalid."""
    raw = _env_value(var_name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("Ignoring %s because %r is not an integer", var_name, raw)
        return None


def _env_flag(var_name: str, default: bool) -> bool:
    """Parse boolean-like flags such as '1', 'true', 'no', etc."""
    raw = _env_value(var_name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    LOGGER.warning("Ignoring %s because %r is ambiguous; using default %s", var_name, raw, default)
    return default


def _load_json_file(path: str | Path | None) -> Any:
    """Load a JSON file when the path is provided."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _prettify_json(data: Any, fallback: Any) -> str:
    """Stringify JSON content with indentation."""
    try:
        return json.dumps(data if data is not None else fallback, ensure_ascii=False, indent=2)
    except TypeError:
        return json.dumps(fallback, ensure_ascii=False, indent=2)


def _extract_question_from_payload(payload: Any) -> str | None:
    """Best-effort question extraction from a trace payload."""
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    candidates: list[str] = []
    if isinstance(metadata, dict):
        for key in ("user_request", "question_text", "question"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    for key in ("question", "question_text", "user_request"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _extract_trace(payload: Any, *, prefer_reasoning: bool = False) -> Any:
    """
    Extract a trace-like structure from a JSON payload.

    prefer_reasoning:
        When True, try reasoning_trace/tool_blocks before trajectory.
    """
    if not isinstance(payload, dict):
        return payload
    if prefer_reasoning:
        reasoning = payload.get("reasoning_trace")
        if isinstance(reasoning, dict):
            if "tool_blocks" in reasoning:
                return reasoning.get("tool_blocks")
            return reasoning
        if reasoning is not None:
            return reasoning
    trajectory = payload.get("trajectory")
    if trajectory is not None:
        return trajectory
    return payload


def _build_tool_catalog(toolkit: Toolkit, allowed: set[str] | None = None) -> list[dict[str, Any]]:
    """Serialize the toolkit's JSON schemas into GTA-style tool metadata."""
    catalog: list[dict[str, Any]] = []
    try:
        schemas = toolkit.get_json_schemas()
    except Exception:  # pragma: no cover - guard against unexpected toolkit failures
        LOGGER.exception("Unable to build tool catalog from toolkit schemas")
        return catalog

    registered_groups = {name: entry.group for name, entry in getattr(toolkit, "tools", {}).items()}

    for schema in schemas:
        func = schema.get("function") if isinstance(schema, dict) else None
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not name:
            continue
        if allowed and name not in allowed:
            continue
        params = func.get("parameters") if isinstance(func.get("parameters"), dict) else {}
        properties = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        required = set(params.get("required") or [])
        inputs: list[dict[str, Any]] = []
        for arg_name, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            inputs.append(
                {
                    "type": spec.get("type", "json"),
                    "name": arg_name,
                    "description": spec.get("description"),
                    "optional": arg_name not in required,
                    "default": spec.get("default"),
                    "enum": spec.get("enum"),
                    }
                )
        group_name = registered_groups.get(name, "basic")
        resource_estimate = {
            "cpu_cores": properties.get("cpu_cores", {}).get("default") if isinstance(properties.get("cpu_cores"), dict) else None,
            "memory_gb": properties.get("memory_gb", {}).get("default") if isinstance(properties.get("memory_gb"), dict) else None,
            "timeout_seconds": properties.get("timeout_seconds", {}).get("default") if isinstance(properties.get("timeout_seconds"), dict) else None,
        }
        if all(value is None for value in resource_estimate.values()):
            resource_estimate = {}
        determinism = {
            "has_seed": "seed" in properties,
            "seed_default": properties.get("seed", {}).get("default") if isinstance(properties.get("seed"), dict) else None,
            "model_version": properties.get("model_version", {}).get("default")
            if isinstance(properties.get("model_version"), dict)
            else None,
            "container_image": properties.get("container_image", {}).get("default")
            if isinstance(properties.get("container_image"), dict)
            else None,
            "threads": properties.get("threads", {}).get("default") if isinstance(properties.get("threads"), dict) else None,
        }
        if not any(value is not None and value is not False for value in determinism.values()):
            determinism = {}
        example_args: dict[str, Any] = {}
        for arg_name, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            if "default" in spec:
                example_args[arg_name] = spec.get("default")
                continue
            if arg_name in required:
                arg_type = spec.get("type")
                if arg_type == "string":
                    example_args[arg_name] = f"<{arg_name}>"
                elif arg_type == "integer":
                    example_args[arg_name] = 0
                elif arg_type == "number":
                    example_args[arg_name] = 0.0
                elif arg_type == "boolean":
                    example_args[arg_name] = False
                elif arg_type == "array":
                    example_args[arg_name] = []
                elif arg_type == "object":
                    example_args[arg_name] = {}
                else:
                    example_args[arg_name] = None

        returns_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "metrics": {"type": "object"},
                "derived_metrics": {"type": "object"},
                "outputs_artifacts": {"type": "array"},
                "provenance": {"type": "object"},
            },
            "required": ["status", "metrics", "derived_metrics", "outputs_artifacts", "provenance"],
        }
        if name.startswith("impact_") or group_name == "simulators":
            returns_schema["properties"]["run_context"] = {"type": "object"}
            returns_schema["properties"]["agentic_summary"] = {"type": "object"}
            returns_schema["required"].append("run_context")

        catalog.append(
            {
                "name": name,
                "group": group_name,
                "description": func.get("description"),
                "args_schema": params,
                "returns_schema": returns_schema,
                "determinism": determinism,
                "resource_estimate": resource_estimate,
                "example_call": {"name": name, "arguments": example_args},
                "example_output": {
                    "status": "success",
                    "metrics": {"metric_name": {"value": 0.0, "unit": "unit"}},
                    "derived_metrics": {},
                    "outputs_artifacts": [
                        {"path": "/path/to/artifact", "sha256": "<sha256>", "content_type": "text/csv", "description": "artifact"}
                    ],
                    "provenance": {"command": f"inprocess:{name}", "runtime_seconds": 0.0, "warnings": [], "errors": []},
                },
                "inputs": inputs,
                "outputs": [
                    {
                        "type": "json",
                        "description": "ToolResponse payload (content + metadata)",
                        "optional": False,
                        "default": None,
                    }
                ],
            }
        )

    catalog.sort(key=lambda entry: entry["name"])
    return catalog


def _build_chat_backend(provider: str) -> tuple[Any, Any]:
    """Return the (model, formatter) pair for the requested provider."""
    normalized = (provider or "openai").strip().lower()

    if normalized in {"openai", "openai_compat", "openai-compatible", "deepseek"}:
        client_args = _load_env_json("OPENAI_CLIENT_ARGS")
        base_url = os.getenv("OPENAI_BASE_URL") or _env_value("CLIMATE_AGENT_OPENAI_BASE_URL")
        if base_url:
            client_args = client_args or {}
            client_args.setdefault("base_url", base_url.strip())

        http_timeout = _load_env_float("OPENAI_HTTP_TIMEOUT")
        if http_timeout is not None:
            client_args = client_args or {}
            client_args["timeout"] = http_timeout

        generate_kwargs = _load_env_json("OPENAI_GENERATE_KWARGS")
        env_model = os.getenv("OPENAI_MODEL")
        use_deepseek = _looks_like_deepseek(normalized, base_url, env_model)
        use_mistral_compat = _looks_like_mistral_family(env_model)
        model_name = env_model or ("deepseek-reasoner" if use_deepseek else "gpt-4o-mini")

        model = _ToolForcedOpenAIChatModel(
            model_name=model_name,
            api_key=os.environ.get("OPENAI_API_KEY"),
            stream=_env_flag("OPENAI_STREAM", True),
            client_args=client_args,
            generate_kwargs=generate_kwargs,
        )
        formatter = (
            _DeepSeekReasoningFormatter()
            if use_deepseek
            else _MistralCompatFormatter()
            if use_mistral_compat
            else OpenAIChatFormatter()
        )
        return model, formatter

    if normalized == "anthropic":
        from agentscope.model import AnthropicChatModel  # local import to avoid unused dependency

        client_args = _load_env_json("ANTHROPIC_CLIENT_ARGS")
        generate_kwargs = _load_env_json("ANTHROPIC_GENERATE_KWARGS")
        thinking_cfg = _load_env_json("ANTHROPIC_THINKING")
        max_tokens = _load_env_int("ANTHROPIC_MAX_TOKENS") or 4096
        base_url = os.getenv("ANTHROPIC_BASE_URL") or _env_value("CLIMATE_AGENT_ANTHROPIC_BASE_URL")
        if base_url:
            client_args = client_args or {}
            client_args.setdefault("base_url", base_url.strip())

        model = AnthropicChatModel(
            model_name=os.getenv(
                "ANTHROPIC_MODEL",
                os.getenv("OPENAI_MODEL", "claude-3-5-sonnet-20241022"),
            ),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            stream=_env_flag("ANTHROPIC_STREAM", True),
            max_tokens=max_tokens,
            thinking=thinking_cfg,
            client_args=client_args,
            generate_kwargs=generate_kwargs,
        )
        formatter = AnthropicChatFormatter()
        return model, formatter

    if normalized in {"gemini", "google", "google-gemini"}:
        from agentscope.formatter import GeminiChatFormatter
        from agentscope.model import GeminiChatModel

        class _ToolAutoGeminiChatModel(GeminiChatModel):
            @classmethod
            def _strip_unsupported_schema_fields(cls, value: Any) -> Any:
                if isinstance(value, list):
                    return [
                        cls._strip_unsupported_schema_fields(item)
                        for item in value
                    ]
                if not isinstance(value, dict):
                    return value

                unsupported = {
                    "additionalProperties",
                    "additional_properties",
                }
                return {
                    key: cls._strip_unsupported_schema_fields(item)
                    for key, item in value.items()
                    if key not in unsupported
                }

            def _format_tools_json_schemas(
                self,
                schemas: list[dict[str, Any]],
            ) -> list[dict[str, Any]]:
                sanitized = self._strip_unsupported_schema_fields(
                    deepcopy(schemas),
                )
                return super()._format_tools_json_schemas(sanitized)

            async def __call__(
                self,
                messages: list[dict],
                tools: list[dict] | None = None,
                tool_choice: str | None = None,
                structured_model: Any = None,
                **kwargs: Any,
            ) -> Any:
                if tools and tool_choice is None and structured_model is None:
                    tool_choice = "auto"
                return await super().__call__(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    structured_model=structured_model,
                    **kwargs,
                )

        client_args = _load_env_json("GEMINI_CLIENT_ARGS")
        generate_kwargs = _load_env_json("GEMINI_GENERATE_KWARGS")
        thinking_config = _load_env_json("GEMINI_THINKING_CONFIG") or _load_env_json(
            "GEMINI_THINKING",
        )

        model = _ToolAutoGeminiChatModel(
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            api_key=(
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            ),
            stream=_env_flag("GEMINI_STREAM", True),
            thinking_config=thinking_config,
            client_args=client_args,
            generate_kwargs=generate_kwargs,
        )
        formatter = GeminiChatFormatter()
        return model, formatter

    raise ValueError(
        "Unsupported model provider "
        f"'{provider}'. Supported values: 'openai', 'anthropic', 'deepseek', 'gemini'."
    )


def _normalize_messages(payload: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    raw_metadata = payload.get("metadata") or {}
    if isinstance(raw_metadata, dict):
        metadata = dict(raw_metadata)
    else:
        metadata = {"raw_metadata": raw_metadata}
    messages = payload.get("messages") or []
    if isinstance(messages, str):
        messages = [messages]
    if not messages:
        if metadata:
            messages = [json.dumps(metadata, ensure_ascii=False)]
        else:
            messages = ["No message returned by remote tool."]
    return [str(msg) for msg in messages], dict(metadata)


def _build_textblocks(messages: Iterable[str]) -> list[TextBlock]:
    return [TextBlock(type="text", text=str(message)) for message in messages]


def make_http_tool(
    original_fn: Callable[..., ToolResponse],
    *,
    tool_name: str,
    base_url: str,
    timeout: float = 120.0,
) -> Callable[..., ToolResponse]:
    """Create an HTTP proxy around a local tool implementation."""

    endpoint = f"{base_url.rstrip('/')}/{tool_name}"
    signature = inspect.signature(original_fn)

    @functools.wraps(original_fn)
    def remote_tool(*args: Any, **kwargs: Any) -> ToolResponse:
        try:
            import httpx  # type: ignore[import]
        except ImportError:
            return _tool_response_error(
                "Remote tool mode requires the 'httpx' package. Install httpx and retry."
            )

        try:
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            payload = dict(bound.arguments)
        except TypeError as exc:
            return _tool_response_error(f"Invalid arguments for remote tool '{tool_name}': {exc}")

        try:
            response = httpx.post(endpoint, json=payload, timeout=timeout)
        except Exception as exc:  # pragma: no cover - runtime dependencies
            return _tool_response_error(f"Failed to reach remote tool '{tool_name}': {exc}")

        if response.is_error:
            detail = response.text.strip() or response.reason_phrase
            return _tool_response_error(
                f"HTTP {response.status_code} error from remote tool '{tool_name}': {detail}"
            )

        try:
            data = response.json()
        except ValueError:
            return _tool_response_error(
                f"Remote tool '{tool_name}' returned an invalid JSON response."
            )

        messages, metadata = _normalize_messages(data)
        if "ok" in data and "ok" not in metadata:
            metadata["ok"] = data["ok"]
        blocks = _build_textblocks(messages)
        if not blocks:
            blocks = [TextBlock(type="text", text=json.dumps(metadata))]
        return ToolResponse(content=blocks, metadata=metadata)

    remote_tool.__doc__ = (
        f"HTTP proxy for {original_fn.__name__} using endpoint {endpoint}."
    )
    return remote_tool


__all__ = [
    "_build_chat_backend",
    "_build_textblocks",
    "_build_tool_catalog",
    "_extract_question_from_payload",
    "_extract_text_from_blocks",
    "_extract_thought",
    "_extract_tool_calls",
    "_extract_trace",
    "_format_tool_result_output",
    "_load_env_float",
    "_load_env_int",
    "_load_env_json",
    "_load_json_file",
    "_normalize_messages",
    "_prettify_json",
    "_record_dialog",
    "_strip_known_tags",
    "_tool_response_error",
    "_env_flag",
    "log_tool_errors",
    "make_http_tool",
    "ReportingChatModel",
    "TokenUsageReporter",
    "TOOL_LOGGER",
    "LOGGER",
]
