"""Upload local data files (NetCDF, CSV, text) to the code agent staging endpoint."""

import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
from dotenv import load_dotenv

load_dotenv()

DEFAULT_UPLOAD_URL_ENV = "CODE_AGENT_UPLOAD_URL"
LEGACY_UPLOAD_URL_ENV = "MATH_AGENT_UPLOAD_URL"
DEFAULT_BASE_ENV = "CODE_AGENT_BASE_URL"
LEGACY_BASE_ENV = "MATH_AGENT_BASE_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_UPLOAD_ROUTE = "/code-agent/upload"


def _error(message: str, *, status_code: Optional[int] = None) -> ToolResponse:
    metadata: Dict[str, Any] = {"error": True, "message": message}
    if status_code is not None:
        metadata["status_code"] = status_code
    return ToolResponse(content=[TextBlock(type="text", text=f"Error: {message}")], metadata=metadata)


def _resolve_upload_url(upload_url: Optional[str]) -> str:
    if upload_url:
        return upload_url
    env_url = os.getenv(DEFAULT_UPLOAD_URL_ENV) or os.getenv(LEGACY_UPLOAD_URL_ENV)
    if env_url:
        return env_url
    base_url = (os.getenv(DEFAULT_BASE_ENV) or os.getenv(LEGACY_BASE_ENV) or DEFAULT_BASE_URL).rstrip("/")
    return f"{base_url}{DEFAULT_UPLOAD_ROUTE}"


def stream_data_to_math_agent(
    file_path: str,
    *,
    upload_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> ToolResponse:
    """Stream a local file to the code-agent file ingestion endpoint."""

    if not file_path:
        return _error("file_path must be provided")

    path = Path(file_path).expanduser()
    if not path.is_file():
        return _error(f"File not found: {path}")

    if metadata is not None and not isinstance(metadata, dict):
        return _error("metadata must be a JSON object when provided")

    resolved_url = _resolve_upload_url(upload_url)
    file_size = path.stat().st_size
    file_name = path.name
    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

    form_fields: Dict[str, str] = {}
    if metadata:
        try:
            form_fields["metadata"] = json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            return _error(f"metadata must be JSON-serializable: {exc}")

    request_headers = dict(headers or {})

    try:
        with path.open("rb") as fh:
            response = requests.post(
                resolved_url,
                data=form_fields,
                files={"file": (file_name, fh, mime_type)},
                headers=request_headers,
            )
    except requests.RequestException as exc:
        return _error(f"Failed to upload file: {exc}")

    if response.status_code >= 400:
        detail = response.text[:800] if response.text else "no response body"
        return _error(
            f"Upload endpoint returned status {response.status_code}: {detail}",
            status_code=response.status_code,
        )

    try:
        response_payload: Any = response.json()
    except ValueError:
        response_payload = {"raw_text": response.text}

    payload = {
        "file_name": file_name,
        "file_size_bytes": file_size,
        "upload_url": resolved_url,
        "response": response_payload,
    }

    metadata_payload: Dict[str, Any] = {
        "file": {
            "path": str(path),
            "size_bytes": file_size,
            "mime_type": mime_type,
        },
        "request": {
            "url": resolved_url,
            "headers": request_headers,
            "metadata": metadata or {},
        },
        "response": response_payload,
        "error": False,
    }

    return ToolResponse(
        content=[TextBlock(type="text", text=json.dumps(payload))],
        metadata=metadata_payload,
    )
