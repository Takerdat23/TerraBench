#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

workspace_host_dir="${CODE_AGENT_WORKSPACE_HOST_DIR:-${MATH_AGENT_WORKSPACE_HOST_DIR:-}}"
if [[ -z "${workspace_host_dir}" ]]; then
  workspace_host_dir="${REPO_ROOT}"
elif [[ "${workspace_host_dir}" != /* ]] && [[ "${workspace_host_dir}" != "none" ]]; then
  workspace_host_dir="${REPO_ROOT}/${workspace_host_dir}"
fi
export CODE_AGENT_WORKSPACE_HOST_DIR="${workspace_host_dir}"
export MATH_AGENT_WORKSPACE_HOST_DIR="${workspace_host_dir}"

upload_dir="${CODE_AGENT_UPLOAD_DIR:-${MATH_AGENT_UPLOAD_DIR:-}}"
if [[ -z "${upload_dir}" ]]; then
  upload_dir="${REPO_ROOT}/outputs/code_agent_uploads"
elif [[ "${upload_dir}" != /* ]]; then
  upload_dir="${REPO_ROOT}/${upload_dir}"
fi
export CODE_AGENT_UPLOAD_DIR="${upload_dir}"
export MATH_AGENT_UPLOAD_DIR="${upload_dir}"

export CODE_AGENT_WORKSPACE_CONTAINER_DIR="${CODE_AGENT_WORKSPACE_CONTAINER_DIR:-${MATH_AGENT_WORKSPACE_CONTAINER_DIR:-/workspace}}"
export MATH_AGENT_WORKSPACE_CONTAINER_DIR="${CODE_AGENT_WORKSPACE_CONTAINER_DIR}"
export CODE_AGENT_WORKSPACE_READ_ONLY="${CODE_AGENT_WORKSPACE_READ_ONLY:-${MATH_AGENT_WORKSPACE_READ_ONLY:-true}}"
export MATH_AGENT_WORKSPACE_READ_ONLY="${CODE_AGENT_WORKSPACE_READ_ONLY}"
export CODE_AGENT_UPLOAD_MOUNT="${CODE_AGENT_UPLOAD_MOUNT:-${MATH_AGENT_UPLOAD_MOUNT:-/uploads}}"
export MATH_AGENT_UPLOAD_MOUNT="${CODE_AGENT_UPLOAD_MOUNT}"
export CODE_AGENT_MODEL="${CODE_AGENT_MODEL:-${MATH_AGENT_MODEL:-gpt-4o-mini}}"
export MATH_AGENT_MODEL="${CODE_AGENT_MODEL}"

host="${CODE_AGENT_HOST:-${MATH_AGENT_HOST:-127.0.0.1}}"
port="${CODE_AGENT_PORT:-${MATH_AGENT_PORT:-8000}}"

exec uvicorn terra_agent.code_agent.api:app --host "${host}" --port "${port}"
