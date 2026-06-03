import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from terra_agent.code_agent.service import CodeAgentServicePool, CodeAgentSlotRecycledError
from terra_agent.code_agent.upload_manager import UploadManager


class CodeAgentRequest(BaseModel):
    query: str
    context: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    template: Optional[str] = None
    dialogue_limit: Optional[int] = None
    max_turns: Optional[int] = None
    execution_timeout_seconds: Optional[float] = Field(default=None, gt=0)
    metadata: Optional[Dict[str, Any]] = None


MathAgentRequest = CodeAgentRequest

app = FastAPI(title="TerraBench Code Agent API")
DEFAULT_SERVICE_MODEL = "gpt-4"
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 30.0
DEFAULT_WORKSPACE_HOST_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE_CONTAINER_DIR = "/workspace"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    code_agent_name = name.replace("MATH_AGENT_", "CODE_AGENT_", 1)
    return os.getenv(code_agent_name) or os.getenv(name) or default


DEPLOYMENT_MODEL = _env("MATH_AGENT_MODEL", DEFAULT_SERVICE_MODEL)
VERBOSE = _env("MATH_AGENT_VERBOSE", "false")
POOL_SIZE = _env("MATH_AGENT_POOL_SIZE")
POOL_PORT_BASE = _env("MATH_AGENT_POOL_PORT_BASE")
CONTAINER_NANO_CPUS = _env("MATH_AGENT_CONTAINER_NANO_CPUS")
CONTAINER_MEM_LIMIT = _env("MATH_AGENT_CONTAINER_MEM_LIMIT")
CONTAINER_PIDS_LIMIT = _env("MATH_AGENT_CONTAINER_PIDS_LIMIT")
LOAD_BALANCING_STRATEGY = _env("MATH_AGENT_LOAD_BALANCING_STRATEGY")
MAX_CONTAINER_CPU_PERCENT = _env("MATH_AGENT_MAX_CONTAINER_CPU_PERCENT")
DISABLE_TIMEOUT = _env("MATH_AGENT_DISABLE_TIMEOUT", "false")
UPLOAD_DIR = Path(_env("MATH_AGENT_UPLOAD_DIR", "uploads") or "uploads")
UPLOAD_MOUNT = _env("MATH_AGENT_UPLOAD_MOUNT", "/uploads")
UPLOAD_TTL = _env("MATH_AGENT_UPLOAD_TTL_SECONDS")
UPLOAD_MAX_BYTES = _env("MATH_AGENT_UPLOAD_MAX_BYTES")
UPLOAD_DELETE_AFTER_USE = _env("MATH_AGENT_UPLOAD_DELETE_AFTER_USE", "false")
UPLOAD_TOKEN = _env("MATH_AGENT_UPLOAD_TOKEN")
MAX_RESPONSE_TOKENS = _env("MATH_AGENT_MAX_RESPONSE_TOKENS")
EXECUTION_TIMEOUT_SECONDS = _env("MATH_AGENT_EXECUTION_TIMEOUT_SECONDS")
WORKSPACE_HOST_DIR = _env("MATH_AGENT_WORKSPACE_HOST_DIR")
WORKSPACE_CONTAINER_DIR = _env("MATH_AGENT_WORKSPACE_CONTAINER_DIR")
WORKSPACE_READ_ONLY = _env("MATH_AGENT_WORKSPACE_READ_ONLY", "true")


def _parse_positive_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logging.getLogger(__name__).warning("Invalid integer '%s'; falling back to %s", value, default)
        return default
    if parsed <= 0:
        return None
    return parsed


def _parse_positive_float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        logging.getLogger(__name__).warning("Invalid float '%s'; falling back to %s", value, default)
        return default
    if parsed <= 0:
        return None
    return parsed


def _to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_non_empty_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_workspace_host_dir(value: Optional[str]) -> Optional[str]:
    if value is None:
        return str(DEFAULT_WORKSPACE_HOST_DIR)
    stripped = value.strip()
    if not stripped or stripped.lower() in {"0", "false", "no", "none", "off"}:
        return None
    return str(Path(stripped).expanduser().resolve())


TIMEOUT_DISABLED = _to_bool(DISABLE_TIMEOUT)


def _effective_timeout(request_timeout: Optional[float]) -> Optional[float]:
    if TIMEOUT_DISABLED:
        return None
    return request_timeout


upload_manager = UploadManager(
    storage_dir=UPLOAD_DIR,
    container_mount=UPLOAD_MOUNT or "/uploads",
    ttl_seconds=_parse_positive_int(UPLOAD_TTL, default=86_400),
    max_file_size_bytes=_parse_positive_int(UPLOAD_MAX_BYTES, default=512 * 1024 * 1024) or 0,
    delete_after_use=_to_bool(UPLOAD_DELETE_AFTER_USE),
)
service_pool = CodeAgentServicePool(
    size=_parse_positive_int(POOL_SIZE, default=1) or 1,
    host_port_base=_parse_positive_int(POOL_PORT_BASE, default=3006),
    container_nano_cpus=_parse_positive_int(CONTAINER_NANO_CPUS),
    container_mem_limit=_parse_non_empty_str(CONTAINER_MEM_LIMIT),
    container_pids_limit=_parse_positive_int(CONTAINER_PIDS_LIMIT),
    load_balancing_strategy=_parse_non_empty_str(LOAD_BALANCING_STRATEGY),
    max_container_cpu_percent=_parse_positive_float(MAX_CONTAINER_CPU_PERCENT),
    auto_submit=True,
    verbose=_to_bool(VERBOSE),
    model=DEPLOYMENT_MODEL,
    max_response_tokens=_parse_positive_int(MAX_RESPONSE_TOKENS, default=2048),
    workspace_host_dir=_parse_workspace_host_dir(WORKSPACE_HOST_DIR),
    workspace_container_dir=_parse_non_empty_str(WORKSPACE_CONTAINER_DIR) or DEFAULT_WORKSPACE_CONTAINER_DIR,
    workspace_read_only=_to_bool(WORKSPACE_READ_ONLY),
    execution_timeout_seconds=(
        None
        if TIMEOUT_DISABLED
        else _parse_positive_float(
            EXECUTION_TIMEOUT_SECONDS,
            default=DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        )
    ),
    upload_manager=upload_manager,
)
logger = logging.getLogger(__name__)


def _preview(text: Optional[str], limit: int = 80) -> str:
    if not text:
        return ""
    single_line = " ".join(text.strip().splitlines())
    return single_line[:limit] + ("..." if len(single_line) > limit else "")


@app.post("/math-agent")
@app.post("/code-agent")
async def invoke_code_agent(request: Request, payload: CodeAgentRequest) -> Dict[str, Any]:
    try:
        request_timeout = _effective_timeout(payload.execution_timeout_seconds)
        logger.info(
            "Received /code-agent request | query='%s' | model=%s | template=%s | max_turns=%s | execution_timeout=%s",
            _preview(payload.query),
            payload.model or service_pool.default_model,
            payload.template or service_pool.default_template,
            payload.max_turns or service_pool.default_max_turns,
            request_timeout or service_pool.default_execution_timeout_seconds,
        )
        if payload.metadata:
            logger.debug("Request metadata: %s", payload.metadata)
        result = await run_in_threadpool(
            service_pool.run_task,
            query=payload.query,
            context=payload.context,
            variables=payload.variables,
            model=payload.model,
            template=payload.template,
            dialogue_limit=payload.dialogue_limit,
            max_turns=payload.max_turns,
            execution_timeout_seconds=request_timeout,
            metadata=payload.metadata,
        )
        # Attach download URLs for any artifacts the task produced
        for artifact in result.get("artifacts", []) or []:
            artifact["download_url"] = _build_download_url(request, artifact["file_id"])
        # Mirror download URLs into summary copy if present
        summary_artifacts = (result.get("summary") or {}).get("artifacts") or []
        for artifact in summary_artifacts:
            if "download_url" not in artifact:
                artifact["download_url"] = _build_download_url(request, artifact["file_id"])
        summary = result.get("summary", {})
        logger.info(
            "Completed /code-agent request | reward=%s | turns=%s/%s",
            summary.get("max_reward"),
            summary.get("turns_taken"),
            summary.get("turns_max"),
        )
    except CodeAgentSlotRecycledError as err:
        logger.warning("Execution timeout triggered slot recycle: %s", err)
        raise HTTPException(status_code=504, detail=str(err))
    except ValueError as err:
        logger.exception("Bad code-agent request: %s", err)
        raise HTTPException(status_code=400, detail=str(err))
    return result


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/pool-status")
async def pool_status() -> Dict[str, Any]:
    return service_pool.get_status()


def _parse_metadata(metadata_raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if metadata_raw is None:
        return None
    if not metadata_raw.strip():
        return {}
    try:
        parsed = json.loads(metadata_raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Metadata must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="Metadata must decode to an object/dict.")
    return parsed


def _check_upload_token(header_token: Optional[str]) -> None:
    if UPLOAD_TOKEN and header_token != UPLOAD_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing upload token.")


def _build_download_url(request: Request, file_id: str) -> str:
    base = str(request.base_url)
    return urljoin(base, f"code-agent/upload/{file_id}")


@app.post("/math-agent/upload")
@app.post("/code-agent/upload")
async def upload_binary(
    request: Request,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    upload_token: Optional[str] = Header(None, alias="X-Upload-Token"),
) -> Dict[str, Any]:
    _check_upload_token(upload_token)
    meta_dict = _parse_metadata(metadata)
    upload_manager.purge_expired()
    try:
        record = upload_manager.save_stream(
            file.file,
            filename=file.filename,
            content_type=file.content_type,
            metadata=meta_dict,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    finally:
        await file.close()

    response_payload = {
        "file_id": record["file_id"],
        "filename": record["filename"],
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
        "content_type": record.get("content_type"),
        "created_at": record.get("created_at"),
        "expires_at": record.get("expires_at"),
        "metadata": record.get("metadata"),
        # Absolute path inside the execution container; use this instead of downloading via HTTP
        "container_path": record.get("container_path"),
        "download_url": _build_download_url(request, record["file_id"]),
    }
    return response_payload


@app.get("/math-agent/upload/{file_id}")
@app.get("/code-agent/upload/{file_id}")
async def download_binary(
    file_id: str,
    upload_token: Optional[str] = Header(None, alias="X-Upload-Token"),
) -> FileResponse:
    _check_upload_token(upload_token)
    try:
        record = upload_manager.get_record(file_id)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return FileResponse(
        record["host_path"],
        media_type=record.get("content_type") or "application/octet-stream",
        filename=record["filename"],
    )


@app.on_event("shutdown")
def shutdown_service() -> None:
    logger.info("Shutting down code-agent service")
    service_pool.close()
