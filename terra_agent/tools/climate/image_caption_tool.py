"""
Image description tool built from the test.py prototype.

Supports switching between 4B and 8B LLaVA-OneVision checkpoints, with an override
for arbitrary model ids so other VLM families can be added later without code changes.
"""

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor

__all__ = ["describe_image_vlm"]

MODEL_ALIASES: Dict[str, Dict[str, str]] = {
    # Update these aliases when adding new families or sizes.
    "llava-onevision": {
        "4b": "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        "8b": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
    },
}


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def _ok(payload: Dict[str, Any]) -> ToolResponse:
    cleaned = _sanitize(payload)
    return ToolResponse(
        content=[TextBlock(type="text", text=json.dumps(cleaned))],
        metadata=cleaned,
    )


def _error(msg: str) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {msg}")],
        metadata={"error": True, "message": msg},
    )


def _resolve_model_id(model_family: str, model_size: str, override: Optional[str]) -> str:
    if override:
        return override
    family = MODEL_ALIASES.get(model_family.lower())
    if not family:
        raise ValueError(f"Unknown model_family '{model_family}'. Provide model_id to override.")
    model_id = family.get(model_size.lower())
    if not model_id:
        raise ValueError(f"Unknown model_size '{model_size}' for family '{model_family}'.")
    return model_id


def describe_image_vlm(
    image: str | Sequence[str],
    *,
    prompt: str = "Describe this image.",
    model_family: str = "llava-onevision",
    model_size: str = "4b",
    model_id: Optional[str] = None,
    device: str = "cuda:0",
    max_new_tokens: int = 256,
) -> ToolResponse:
    """
    Run a vision-language model to describe an image.

    Args:
        image: Image URL/path or a list of images for multi-image captioning.
        prompt: Text prompt paired with the image(s).
        model_family: Alias group (default llava-onevision).
        model_size: Size key within the family ("4b" or "8b").
        model_id: Optional explicit model id to bypass aliases (for future VLMs).
        device: Torch device string.
        max_new_tokens: Cap on generated tokens.
    """
    try:
        resolved_model = _resolve_model_id(model_family, model_size, model_id)

        model = AutoModelForCausalLM.from_pretrained(
            resolved_model,
            trust_remote_code=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(resolved_model, trust_remote_code=True)

        image_list = [image] if isinstance(image, str) else list(image)
        if not image_list:
            raise ValueError("At least one image must be provided.")

        content_blocks = [{"type": "image", "image": img} for img in image_list]
        content_blocks.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content_blocks}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        caption = output_text[0] if output_text else ""

        return _ok(
            {
                "model_id": resolved_model,
                "model_family": model_family,
                "model_size": model_size,
                "device": device,
                "prompt": prompt,
                "images": image_list,
                "caption": caption,
                "max_new_tokens": max_new_tokens,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _error(f"describe_image_vlm failed: {exc}")
