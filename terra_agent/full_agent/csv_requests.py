"""CSV row selection and request-building helpers."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

def _build_request_from_row(row: dict[str, Any]) -> str:
    normalized_row = {_normalize_header(k): v for k, v in row.items() if isinstance(k, str)}
    parts: list[str] = []
    emitted_keys: set[str] = set()

    def append_field(label: str, *aliases: str) -> None:
        for alias in aliases:
            key = _normalize_header(alias)
            if key not in normalized_row:
                continue
            emitted_keys.add(key)
            value = str(normalized_row[key]).strip() if normalized_row[key] is not None else ""
            if value:
                parts.append(f"{label}: {value}")
                return

    append_field("Context", "Context")
    append_field("Use case", "Use case", "Use case ")
    append_field("Subtask", "Subtask")
    append_field("Level", "Level")
    append_field("Question", "Question", "Question ")
    append_field("Reference (authoritative)", "Reference (authoritative)", "Reference")
    append_field("Ground truth", "Ground truth", "Ground truth")
    append_field(
        "Reference numeric-evidence requirement",
        "FAO/WMO/IPCC/peer-reviewed, must contain explicit numeric value(s)",
    )
    append_field("Answer (ground-truth objective)", "Answer (ground-truth objective)", "Answer")
    append_field(
        "Answer numeric requirement",
        "numeric value(s) + units + snippet containing the number",
    )
    append_field(
        "Instruction prompt (annotation-only)",
        "Instruction prompt (annotation-only)",
        "Instruction prompt",
    )
    append_field(
        "Instruction procedure requirement",
        "document-grounded step-by-step procedure to reproduce/approximate the Answer using tools",
    )
    append_field(
        "Instruction checks/tolerance requirement",
        "includes checks/tolerances so results are audit-friendly",
    )

    for raw_key, value in row.items():
        if not isinstance(raw_key, str):
            continue
        key = _normalize_header(raw_key)
        if key in emitted_keys:
            continue
        text = str(value).strip() if value is not None else ""
        if text:
            parts.append(f"{raw_key.strip()}: {text}")
    return "\n".join(parts)


def _get_row_value(row: dict[str, Any], *aliases: str) -> str:
    normalized_row = {_normalize_header(k): v for k, v in row.items() if isinstance(k, str)}
    for alias in aliases:
        key = _normalize_header(alias)
        if key not in normalized_row:
            continue
        value = normalized_row[key]
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return ""


def _extract_output_template_text(row: dict[str, Any]) -> str:
    answer_text = _get_row_value(row, "Answer", "Answer (ground-truth objective)")
    if not answer_text:
        return ""

    for marker in ("OUTPUT TEMPLATE", "OUTPUT SPEC"):
        index = answer_text.find(marker)
        if index >= 0:
            return answer_text[index:].strip()

    for pattern in (
        r"(?is)<final_json_template>.*?</final_json_template>",
        r"(?is)<final_json>.*?</final_json>",
    ):
        match = re.search(pattern, answer_text)
        if match:
            return match.group(0).strip()
    return answer_text.strip()


def _build_inference_request_from_row(row: dict[str, Any]) -> str:
    parts: list[str] = []

    context = _get_row_value(row, "Context")
    if context:
        parts.append(f"Context: {context}")

    question = _get_row_value(row, "Question", "Question ")
    if question:
        parts.append(f"Question: {question}")

    output_template = _extract_output_template_text(row)
    if output_template:
        parts.append(output_template)

    return "\n\n".join(parts)


def _build_request_for_mode(row: dict[str, Any], *, request_mode: str) -> str:
    if request_mode == "inference":
        return _build_inference_request_from_row(row)
    if request_mode == "full":
        return _build_request_from_row(row)
    raise ValueError(f"Unsupported CSV request mode: {request_mode}")


def _load_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _normalize_header(header: str) -> str:
    return " ".join(header.strip().lower().split())


def _resolve_selected_rows(
    total_rows: int,
    *,
    index: int | None,
    start_index: int | None,
    end_index: int | None,
) -> list[int]:
    if total_rows <= 0:
        return []
    if index is not None and (start_index is not None or end_index is not None):
        raise ValueError("Use either --agent-index or --agent-start-index/--agent-end-index, not both.")
    if index is not None:
        start = end = index
    else:
        start = start_index or 1
        end = end_index or total_rows
    if start < 1 or end < 1:
        raise ValueError("CSV row indices are 1-based and must be >= 1.")
    if start > end:
        raise ValueError(f"Invalid CSV row range: start index {start} is greater than end index {end}.")
    if end > total_rows:
        raise ValueError(f"CSV row range {start}-{end} is out of bounds; file has {total_rows} rows.")
    return list(range(start, end + 1))


def _build_row_destination(base_dest: Path, *, row_index: int) -> Path:
    return base_dest / f"Q{row_index}"


def _find_latest_trace(dest_dir: Path) -> Path | None:
    candidates = sorted(dest_dir.glob("single_agent_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _write_question_input(dest_dir: Path, *, row_index: int, row: dict[str, Any], user_request: str) -> None:
    payload = {"row_index": row_index, "row": row, "user_request": user_request}
    out_path = dest_dir / "question_input.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
