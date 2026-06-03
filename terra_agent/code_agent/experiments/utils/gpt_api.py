import logging
import os
from time import sleep
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential

import openai

from .anthropic_api import AnthropicChat
from .key_config import load_key_config
from .text_utils import clean_completion_text

logger = logging.getLogger(__name__)

# Set OpenAI key from environment or config file
cfg = None
api_key = os.environ.get("OPENAI_API_KEY")
if api_key is None or api_key == "":
    cfg = load_key_config()
    if cfg:
        api_key = cfg.get("OPENAI_API_KEY")
openai.api_key = api_key

# Optional OpenAI base URL override (env or config file)
api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")
if api_base is None or api_base == "":
    if cfg is None:
        cfg = load_key_config()
    if cfg:
        api_base = (
            cfg.get("OPENAI_API_BASE")
            or cfg.get("OPENAI_BASE_URL")
            or cfg.get("BASE_URL")
        )
if api_base:
    openai.api_base = api_base

PROVIDER_ENV_VARS = ("MODEL_PROVIDER", "LLM_PROVIDER", "API_PROVIDER")


def _normalize_provider(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in ("anthropic", "claude"):
        return "anthropic"
    if normalized in ("openai", "open-ai"):
        return "openai"
    return None


def _load_provider_override(config: Optional[object]) -> Optional[str]:
    for key in PROVIDER_ENV_VARS:
        value = os.environ.get(key)
        if value:
            return value
    if not config:
        return None
    for key in PROVIDER_ENV_VARS:
        value = config.get(key)
        if value:
            return value
    return None


def _resolve_provider() -> str:
    config = cfg or load_key_config()
    override = _load_provider_override(config)
    provider = _normalize_provider(override)
    if provider:
        return provider
    if override:
        logger.warning(
            "Unrecognized MODEL_PROVIDER value '%s'; falling back to API key detection.",
            override,
        )

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY") or (config and config.get("ANTHROPIC_API_KEY")))
    has_openai = bool(os.environ.get("OPENAI_API_KEY") or (config and config.get("OPENAI_API_KEY")))
    if has_anthropic and not has_openai:
        return "anthropic"
    if has_openai and not has_anthropic:
        return "openai"
    if has_anthropic and has_openai:
        logger.warning(
            "Both OPENAI_API_KEY and ANTHROPIC_API_KEY are set; defaulting to OpenAI. "
            "Set MODEL_PROVIDER to override."
        )
    return "openai"


def _short_sleep(seconds: int = 2):
    sleep(seconds)

@retry(wait=wait_random_exponential(min=20, max=100), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def CompletionGPT(
    phrase: str,
    model: str = "text-davinci-003",
    num_samples: int = 1,
    max_tokens: int = 2048,
):
    if _resolve_provider() != "openai":
        raise RuntimeError("CompletionGPT only supports the OpenAI-compatible provider.")
    _short_sleep()
    response = completion_with_backoff(
        model=model,
        prompt=phrase.strip(),
        temperature=0,
        top_p=1,
        max_tokens=max_tokens,
        n=num_samples,
    )
    candidates = []
    for candidate in response.choices:
        text = getattr(candidate, "text", "")
        candidates.append(clean_completion_text(text))
    return candidates

def ChatGPT(
    messages: list[dict[str, str]],
    model: str = "gpt-3.5-turbo",
    num_samples: int = 1,
    max_tokens: int = 2048,
):
    if _resolve_provider() == "anthropic":
        return AnthropicChat(messages, model=model, num_samples=num_samples, max_tokens=max_tokens)

    response = chat_with_backoff(
        model=model,
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=max_tokens,
        n=num_samples,
    )
    candidates = []
    for candidate in response.choices:
        content = candidate.message.content
        candidates.append(clean_completion_text(content))
    return candidates


if __name__ == "__main__":
    pass
