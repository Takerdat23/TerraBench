"""
Tool that forwards math or coding queries to a locally hosted Intercode agent.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

import requests
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse


DEFAULT_BASE_URL = (
    os.environ.get("CODE_AGENT_BASE_URL")
    or os.environ.get("MATH_AGENT_BASE_URL")
    or "http://127.0.0.1:8000"
)
DEFAULT_ROUTE = os.environ.get("CODE_AGENT_ROUTE") or os.environ.get("MATH_AGENT_ROUTE") or "/code-agent"
_MAX_TURNS_PATTERN = re.compile(
    r"<parameter\s+name=[\"']max_turns[\"']>\s*([0-9]+)\s*",
    re.IGNORECASE,
)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return value
    return str(value)


def _error(message: str, *, status_code: Optional[int] = None) -> ToolResponse:
    metadata: Dict[str, Any] = {"error": True, "message": message}
    if status_code is not None:
        metadata["status_code"] = status_code
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {message}")],
        metadata=metadata,
    )


def _summarize(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("result", "answer", "output", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        try:
            return json.dumps(payload, indent=2)
        except TypeError:
            return str(payload)
    if isinstance(payload, (list, tuple)):
        try:
            return json.dumps(payload, indent=2)
        except TypeError:
            return ", ".join(str(item) for item in payload)
    return str(payload)


def _download_artifact(
    artifact: Dict[str, Any],
    dest_root: Path,
    *,
    max_bytes: int = 50 * 1024 * 1024,
) -> Dict[str, Any]:
    """
    Download a single artifact, preferring download_url.
    Guards against oversized payloads via a simple max_bytes cap.
    """
    download_url = artifact.get("download_url")
    filename = artifact.get("filename") or "artifact"
    file_id = artifact.get("file_id") or "artifact"

    if not download_url:
        return {"file_id": file_id, "filename": filename, "error": "Missing download_url."}

    dest_dir = dest_root / str(file_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    try:
        with requests.get(download_url, stream=True, timeout=30) as resp:
            if resp.status_code >= 400:
                return {
                    "file_id": file_id,
                    "filename": filename,
                    "error": f"HTTP {resp.status_code} when fetching artifact.",
                }
            total = 0
            with dest_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        return {
                            "file_id": file_id,
                            "filename": filename,
                            "error": f"Artifact exceeds max_bytes limit ({max_bytes}).",
                        }
                    fh.write(chunk)
    except requests.RequestException as exc:
        return {"file_id": file_id, "filename": filename, "error": f"Download failed: {exc}"}

    return {
        "file_id": file_id,
        "filename": filename,
        "saved_path": str(dest_path),
        "size_bytes": dest_path.stat().st_size if dest_path.exists() else None,
        "sha256": artifact.get("sha256"),
        "source_url": download_url,
    }


def _strip_embedded_max_turns(raw_query: str) -> tuple[str, Optional[int]]:
    """Remove `<parameter name="max_turns">X` fragments from query text and return value."""
    if not raw_query:
        return "", None

    matches = list(_MAX_TURNS_PATTERN.finditer(raw_query))
    embedded_turns: Optional[int] = None
    if matches:
        # Use the last occurrence if multiple are present
        try:
            embedded_turns = int(matches[-1].group(1))
        except ValueError:
            embedded_turns = None

    cleaned = _MAX_TURNS_PATTERN.sub("", raw_query)
    return cleaned, embedded_turns


def run_math_agent(
    query: str,
    *,
    variables: Optional[Dict[str, Any]] = None,
    max_turns: Optional[int] = None,
    base_url: Optional[str] = None,
    route: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    artifact_dir: str = "./data/code_agent_artifacts",
    max_artifact_bytes: int = 1000 * 1024 * 1024,
) -> ToolResponse:
    """Send a query to the Intercode code agent service and auto-download artifacts."""

    cleaned_query, embedded_turns = _strip_embedded_max_turns(query or "")
    clean_query = cleaned_query.strip()
    if not clean_query:
        return _error("Query must be a non-empty string.")

    resolved_base = (
        base_url
        or os.environ.get("CODE_AGENT_BASE_URL")
        or os.environ.get("MATH_AGENT_BASE_URL")
        or DEFAULT_BASE_URL
    ).strip()
    if not resolved_base:
        return _error("Base URL for code agent service is empty.")
    resolved_route = (
        route
        or os.environ.get("CODE_AGENT_ROUTE")
        or os.environ.get("MATH_AGENT_ROUTE")
        or DEFAULT_ROUTE
    ).strip() or "/code-agent"
    if not resolved_route.startswith("/"):
        resolved_route = f"/{resolved_route}"

    url = f"{resolved_base.rstrip('/')}{resolved_route}"

    payload: Dict[str, Any] = {"query": clean_query}
    payload_variables: Dict[str, Any] = {}
    if variables is not None:
        if not isinstance(variables, dict):
            return _error("Variables payload must be a JSON object.")
        payload_variables.update(variables)
    if payload_variables:
        payload["variables"] = payload_variables

    turns_value: Optional[int] = max_turns if max_turns is not None else embedded_turns
    if turns_value is None:
        turns_value = 1  # default if nothing provided
    try:
        turns = int(turns_value)
    except (TypeError, ValueError):
        return _error("max_turns must be a positive integer.")
    if turns <= 0:
        return _error("max_turns must be a positive integer.")
    payload["max_turns"] = turns

    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    try:
        request_body_json = json.dumps(payload)
        # print("Json Payload:", request_body_json)
        # print("=============================")
    except (TypeError, ValueError) as exc:
        return _error(f"Unable to serialize payload as JSON: {exc}")

    try:
        response = requests.post(
            url,
            headers=request_headers,
            data=request_body_json,
        )
    except requests.RequestException as exc:
        return _error(f"Request to code agent failed: {exc}")

    if response.status_code >= 400:
        detail = response.text[:800] if response.text else "no response body"
        return _error(
            f"Code agent returned status {response.status_code}: {detail}",
            status_code=response.status_code,
        )

    try:
        response_payload = response.json()
    except ValueError:
        response_payload = response.text
    # print("Response Payload:", response_payload)
    artifacts: List[Dict[str, Any]] = []
    downloaded: List[Dict[str, Any]] = []
    if isinstance(response_payload, dict):
        raw_artifacts = response_payload.get("artifacts")
        if isinstance(raw_artifacts, list):
            artifacts = [_sanitize(item) for item in raw_artifacts]  # type: ignore[arg-type]
            downloadable = [
                entry
                for entry in raw_artifacts
                if isinstance(entry, dict) and entry.get("download_url")
            ]
            if downloadable:
                dest_root = Path(artifact_dir).expanduser()
                for entry in downloadable:
                    downloaded.append(_download_artifact(entry, dest_root, max_bytes=max_artifact_bytes))
        elif isinstance(response_payload.get("download_url"), str):
            artifact_entry = {
                "download_url": response_payload.get("download_url"),
                "file_id": response_payload.get("file_id"),
                "filename": response_payload.get("filename") or response_payload.get("name"),
            }
            artifacts = [_sanitize(artifact_entry)]
            dest_root = Path(artifact_dir).expanduser()
            downloaded.append(_download_artifact(artifact_entry, dest_root, max_bytes=max_artifact_bytes))

    summary_text = _summarize(response_payload)
    if downloaded:
        saved = [d for d in downloaded if "saved_path" in d and "error" not in d]
        if saved:
            saved_paths = ", ".join(item["saved_path"] for item in saved)
            summary_text = f"{summary_text}\nArtifacts saved: {saved_paths}"

    metadata = {
        "request": {
            "url": url,
            "payload": _sanitize(payload),
            "headers": _sanitize(request_headers),
        },
        "response": {
            "status_code": response.status_code,
            "payload": _sanitize(response_payload),
        },
        "error": False,
    }
    if artifacts:
        metadata["artifacts"] = artifacts
    if downloaded:
        metadata["downloaded_artifacts"] = downloaded

    return ToolResponse(
        content=[TextBlock(type="text", text=summary_text)],
        metadata=metadata,
    )
