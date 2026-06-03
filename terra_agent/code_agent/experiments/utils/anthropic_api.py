import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .text_utils import clean_completion_text
from .key_config import load_key_config

try:
    from anthropic import Anthropic, AI_PROMPT, HUMAN_PROMPT
except ImportError:  # pragma: no cover
    Anthropic = None
    AI_PROMPT = None
    HUMAN_PROMPT = None

logger = logging.getLogger(__name__)

# Set Anthropic key if provided
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
if anthropic_api_key is None or anthropic_api_key == "":
    cfg = load_key_config()
    if cfg:
        anthropic_api_key = cfg.get("ANTHROPIC_API_KEY")

anthropic_client = None
if Anthropic and anthropic_api_key:
    anthropic_client = Anthropic(api_key=anthropic_api_key)
elif anthropic_api_key and not Anthropic:
    logger.warning("ANTHROPIC_API_KEY provided but the 'anthropic' package is unavailable.")


def is_anthropic_model(model_name: str) -> bool:
    if not model_name:
        return False
    normalized = model_name.strip().lower()
    return "claude" in normalized or normalized.startswith("anthropic")


def _anthropic_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    human_prompt = HUMAN_PROMPT or "\n\nHuman: "
    ai_prompt = AI_PROMPT or "\n\nAssistant: "
    assembled = ""
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")
        if role == "assistant":
            assembled += f"{ai_prompt}{content}"
        else:
            assembled += f"{human_prompt}{content}"
    assembled += ai_prompt
    return assembled


def _anthropic_supports_chat_completions() -> bool:
    if not anthropic_client:
        return False
    chat_api = getattr(anthropic_client, "chat", None)
    return bool(chat_api and hasattr(chat_api, "completions"))


def _anthropic_supports_messages_api() -> bool:
    if not anthropic_client:
        return False
    messages_api = getattr(anthropic_client, "messages", None)
    return bool(messages_api and hasattr(messages_api, "create"))


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def anthropic_messages_with_backoff(**kwargs):
    if not anthropic_client:
        raise RuntimeError(
            "Anthropic client is not configured; set ANTHROPIC_API_KEY and install the `anthropic` package."
        )
    if not _anthropic_supports_messages_api():
        raise RuntimeError("Anthropic client does not expose the messages API.")
    return anthropic_client.messages.create(**kwargs)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def anthropic_chat_completion_with_backoff(**kwargs):
    if not anthropic_client:
        raise RuntimeError(
            "Anthropic client is not configured; set ANTHROPIC_API_KEY and install the `anthropic` package."
        )
    if _anthropic_supports_chat_completions():
        return anthropic_client.chat.completions.create(**kwargs)
    return anthropic_client.completions.create(**kwargs)


def _anthropic_extract_completion(response: Any) -> str:
    if hasattr(response, "completion"):
        return getattr(response, "completion", "")

    content_attr = getattr(response, "content", None)
    if isinstance(content_attr, (list, tuple)) and content_attr:
        first = content_attr[0]
        if isinstance(first, dict):
            return first.get("text", first.get("content", ""))
        return getattr(first, "text", getattr(first, "content", ""))

    if isinstance(response, dict):
        completion = response.get("completion")
        if isinstance(completion, str):
            return completion
        choices = response.get("choices") or []
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return message.get("content", "")
                return first.get("text", "") or first.get("completion", "")
            return getattr(first, "message", getattr(first, "text", ""))

    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message:
            return getattr(message, "content", "")
        return getattr(first, "text", getattr(first, "completion", ""))
    return ""


def _normalize_anthropic_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    normalized = []
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _split_system_prompt(messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    system_prompt = None
    filtered: List[Dict[str, str]] = []
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")
        if not content:
            continue
        if role == "system":
            if system_prompt is None:
                system_prompt = content
            continue
        filtered.append({"role": role, "content": content})
    return system_prompt, filtered


def AnthropicChat(
    messages: List[Dict[str, str]],
    model: str = "claude-3.5",
    num_samples: int = 1,
    max_tokens: int = 2048,
):
    candidates = []
    normalized_messages = _normalize_anthropic_messages(messages)
    system_prompt, messages_for_api = _split_system_prompt(normalized_messages)
    stop_sequences = [HUMAN_PROMPT or "\n\nHuman: "]
    use_messages_api = _anthropic_supports_messages_api()
    use_chat_api = _anthropic_supports_chat_completions()
    prompt_text = _anthropic_prompt_from_messages(messages)
    for _ in range(num_samples):
        if use_messages_api:
            response = anthropic_messages_with_backoff(
                model=model,
                messages=messages_for_api,
                system=system_prompt,
                max_tokens=max_tokens,
                top_p=1,
                stop_sequences=stop_sequences,
            )
        elif use_chat_api:
            response = anthropic_chat_completion_with_backoff(
                model=model,
                messages=messages_for_api,
                max_tokens_to_sample=max_tokens,
                temperature=0,
                top_p=1,
                stop_sequences=stop_sequences,
            )
        else:
            response = anthropic_chat_completion_with_backoff(
                model=model,
                prompt=prompt_text,
                max_tokens_to_sample=max_tokens,
                temperature=0,
                top_p=1,
                stop_sequences=stop_sequences,
            )
        completion = _anthropic_extract_completion(response)
        candidates.append(clean_completion_text(completion))
    return candidates
