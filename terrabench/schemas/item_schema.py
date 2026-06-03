"""Schemas for benchmark item metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_TRACKS = {
    "Fundamentals",
    "Simulator-Grounded",
    "Document-Grounded Verification",
}


@dataclass(frozen=True)
class BenchmarkItemSchema:
    """Minimal public schema for `question_input.json`."""

    item_id: str
    question: str
    track: str
    answer_keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "BenchmarkItemSchema":
        item_id = str(payload.get("item_id") or payload.get("id") or "").strip()
        question = str(payload.get("question") or payload.get("query") or "").strip()
        track = str(payload.get("track") or "Fundamentals").strip()
        if not item_id:
            raise ValueError("question_input.json must define item_id or id.")
        if not question:
            raise ValueError("question_input.json must define question.")
        if track not in VALID_TRACKS:
            raise ValueError(f"Unsupported track {track!r}; expected one of {sorted(VALID_TRACKS)}.")
        keys = payload.get("answer_keys") or payload.get("expected_answer_keys") or []
        if isinstance(keys, str):
            answer_keys = [keys]
        elif isinstance(keys, list):
            answer_keys = [str(value) for value in keys]
        else:
            raise ValueError("answer_keys must be a string or list of strings.")
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        return cls(item_id=item_id, question=question, track=track, answer_keys=answer_keys, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "track": self.track,
            "answer_keys": list(self.answer_keys),
            "metadata": dict(self.metadata),
        }
