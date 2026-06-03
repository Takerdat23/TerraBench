import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from terra_agent.code_agent.intercode.utils import TraceLogger
from terra_agent.code_agent.runner import CodeAgentRunner, DEFAULT_IMAGE
from terra_agent.code_agent.upload_manager import UploadManager

DEFAULT_WORKSPACE_CONTAINER_DIR = "/workspace"
PYTHON_SERVER_WORKDIR_ENV = "PYTHON_SERVER_WORKDIR"


class CodeAgentService:
    """
    Thin wrapper that turns CodeAgentRunner into a reusable service.
    Designed for API deployments where external agents submit ad-hoc code queries.
    """

    def __init__(
        self,
        image_name: str = DEFAULT_IMAGE,
        container_name: Optional[str] = None,
        host_port: Optional[int] = None,
        slot_index: Optional[int] = None,
        container_nano_cpus: Optional[int] = None,
        container_mem_limit: Optional[str] = None,
        container_pids_limit: Optional[int] = None,
        model: str = "gpt-4",
        template: str = "claude_prompt",
        dialogue_limit: Optional[int] = None,
        max_turns: int = 7,
        auto_submit: bool = False,
        verbose: bool = False,
        trace_log_path: Optional[str] = None,
        max_action_chars: Optional[int] = 8000,
        max_output_chars: Optional[int] = 8000,
        max_response_tokens: Optional[int] = 2048,
        execution_timeout_seconds: Optional[float] = None,
        upload_manager: Optional[UploadManager] = None,
        workspace_host_dir: Optional[str] = None,
        workspace_container_dir: Optional[str] = DEFAULT_WORKSPACE_CONTAINER_DIR,
        workspace_read_only: bool = True,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        self.image_name = image_name
        self.container_name = container_name or f"{image_name}_ic_ctr"
        self.host_port = host_port
        self.slot_index = slot_index
        self.container_nano_cpus = container_nano_cpus
        self.container_mem_limit = container_mem_limit
        self.container_pids_limit = container_pids_limit
        self.model = model
        self.template = template
        self.dialogue_limit = dialogue_limit
        self.max_turns = max_turns
        self.auto_submit = auto_submit
        self.trace_log_path = trace_log_path
        self.max_action_chars = max_action_chars
        self.max_output_chars = max_output_chars
        self.max_response_tokens = max_response_tokens
        self.execution_timeout_seconds = execution_timeout_seconds
        self.upload_manager = upload_manager
        self.workspace_host_dir = _normalize_workspace_host_dir(workspace_host_dir)
        self.workspace_container_dir = _normalize_workspace_container_dir(workspace_container_dir)
        self.workspace_read_only = bool(workspace_read_only)
        self.verbose = verbose
        self.trace_logger = TraceLogger(trace_log_path)
        self.runner = None
        self._initialize_runner()
        self.lock = threading.Lock()
        self.task_counter = 0

    def run_task(
        self,
        query: str,
        context: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        template: Optional[str] = None,
        dialogue_limit: Optional[int] = None,
        max_turns: Optional[int] = None,
        execution_timeout_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a math task and return the turn-by-turn trace.
        """
        prompt = query.strip()
        extra_sections = []
        resolved_variables, used_uploads = self._resolve_variables(variables)
        if self.workspace_host_dir:
            extra_sections.append(
                "Filesystem:\n"
                f"The TerraBench project root is mounted at {self.workspace_container_dir} "
                "inside the Python execution container, and that is the current working directory. "
                "Use relative paths from there for repository files. Save generated outputs under /uploads."
            )
        if context:
            extra_sections.append(f"Context:\n{context.strip()}")
        if resolved_variables:
            pretty_vars = json.dumps(resolved_variables, indent=2, sort_keys=True)
            extra_sections.append(f"Inputs:\n{pretty_vars}")
        if extra_sections:
            prompt = f"{prompt}\n\n" + "\n\n".join(extra_sections)

        prompt_preview = " ".join(prompt.split())
        if len(prompt_preview) > 160:
            prompt_preview = f"{prompt_preview[:157]}..."

        model_to_use = model or self.runner.args.model
        template_to_use = template or self.runner.args.template
        dialogue_limit_to_use = (
            dialogue_limit if dialogue_limit is not None else self.runner.args.dialogue_limit
        )
        max_turns_to_use = max_turns if max_turns is not None else self.runner.args.max_turns
        timeout_to_use = (
            execution_timeout_seconds
            if execution_timeout_seconds is not None
            else getattr(self.runner.env, "execution_timeout_seconds", None)
        )

        request_meta = {
            "query": query,
            "prompt": prompt,
            "context": context,
            "variables": resolved_variables,
            "metadata": metadata,
            "model": model_to_use,
            "template": template_to_use,
            "dialogue_limit": dialogue_limit_to_use,
            "max_turns": max_turns_to_use,
            "execution_timeout_seconds": timeout_to_use,
        }
        request_meta = {k: v for k, v in request_meta.items() if v is not None}

        with self.lock:
            prior_turns = self.runner.args.max_turns
            prior_timeout = getattr(self.runner.env, "execution_timeout_seconds", None)
            task_id = self.task_counter
            before_files = self._snapshot_uploads()
            try:
                if max_turns is not None:
                    self.runner.args.max_turns = max_turns
                if execution_timeout_seconds is not None:
                    self.runner.env.execution_timeout_seconds = execution_timeout_seconds
                self.runner.reconfigure_policy(
                    model=model,
                    template=template,
                    dialogue_limit=dialogue_limit,
                )
                self.logger.info(
                    "Starting task %s | model=%s | template=%s | max_turns=%s | execution_timeout=%s",
                    task_id,
                    self.runner.args.model,
                    self.runner.args.template,
                    self.runner.args.max_turns,
                    getattr(self.runner.env, "execution_timeout_seconds", None),
                )
                if self.verbose:
                    self.logger.debug("Prompt: %s", prompt_preview)
                    if context:
                        self.logger.debug("Context provided (%d chars)", len(context))
                    if variables:
                        self.logger.debug("Variables: %s", pretty_vars)
                record = {"query": prompt}
                if resolved_variables:
                    record["variables"] = resolved_variables
                if metadata:
                    record.update(metadata)
                    self.logger.debug("Metadata: %s", metadata)
                result = self.runner.run_record(
                    record,
                    task_id=self.task_counter,
                    request_meta=request_meta,
                )
                summary = result.get("summary", {})
                turns_taken = summary.get("turns_taken")
                turns_max = summary.get("turns_max")
                max_reward = summary.get("max_reward")
                observations = result.get("turn_history", {}).get("observations", [])
                final_obs = ""
                if observations:
                    final_obs = str(observations[-1]).replace("\n", " ")
                    if len(final_obs) > 120:
                        final_obs = f"{final_obs[:117]}..."
                self.logger.info(
                    "Finished task %s | reward=%s | turns=%s/%s | final_obs='%s'",
                    task_id,
                    max_reward,
                    turns_taken,
                    turns_max,
                    final_obs,
                )
                self.task_counter += 1
                artifact_records = self._collect_new_artifacts(before_files)
                if artifact_records:
                    artifacts_view = [
                        {
                            "file_id": rec["file_id"],
                            "filename": rec["filename"],
                            "container_path": rec["container_path"],
                            "size_bytes": rec["size_bytes"],
                            "sha256": rec["sha256"],
                        }
                        for rec in artifact_records
                    ]
                    result["artifacts"] = artifacts_view
                    # Mirror artifact info into summary for clients that only inspect summary/observation
                    summary = result.get("summary") or {}
                    summary["artifacts"] = artifacts_view
                    result["summary"] = summary
            except Exception as err:
                timeout_exception_cls = getattr(self.runner.env, "timeout_exception_cls", None)
                if timeout_exception_cls and isinstance(err, timeout_exception_cls):
                    timeout_seconds = getattr(err, "timeout_seconds", None)
                    self.logger.warning(
                        "Task %s hit execution timeout; slot %s will be recycled",
                        task_id,
                        self.slot_index,
                    )
                    raise CodeAgentSlotRecycledError(
                        slot_index=self.slot_index,
                        container_name=self.container_name,
                        host_port=self.host_port,
                        timeout_seconds=timeout_seconds,
                    ) from err
                self.logger.exception("Task %s failed", task_id)
                raise
            finally:
                self.runner.args.max_turns = prior_turns
                self.runner.env.execution_timeout_seconds = prior_timeout
                if self.upload_manager and self.upload_manager.delete_after_use and used_uploads:
                    for record in used_uploads.values():
                        self.upload_manager.delete(record["file_id"])
        return result

    def close(self) -> None:
        with self.lock:
            self.logger.info("Closing CodeAgentService")
            self.runner.env.close()
        if self.trace_logger:
            self.trace_logger.close()

    def recycle(self, reason: Optional[str] = None) -> None:
        with self.lock:
            self.logger.warning(
                "Recycling slot %s container `%s` on port %s%s",
                self.slot_index,
                self.container_name,
                self.host_port,
                f" | reason={reason}" if reason else "",
            )
            try:
                self.runner.env.close(remove_container=True)
            except Exception as err:
                self.logger.exception("Failed closing container during recycle: %s", err)
            self._initialize_runner()

    def _snapshot_uploads(self) -> set:
        """Capture the set of files currently in the uploads mount (excluding metadata files)."""
        if not self.upload_manager:
            return set()
        base = Path(self.upload_manager.storage_dir)
        return {
            path.resolve()
            for path in base.rglob("*")
            if path.is_file() and path.name != "metadata.json"
        }

    def _collect_new_artifacts(self, before_files: set) -> list:
        """Register any new files created under uploads during a task run."""
        if not self.upload_manager:
            return []
        after_files = self._snapshot_uploads()
        new_files = [path for path in after_files - before_files if path.is_file()]
        artifacts = []
        for path in new_files:
            try:
                record = self.upload_manager.register_existing_file(path)
                artifacts.append(record)
            except ValueError as err:
                self.logger.warning("Skipping artifact %s: %s", path, err)
        return artifacts

    def _resolve_variables(self, variables: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Expand any *_upload_id references into structured records."""
        if not variables:
            return None, {}
        if not self.upload_manager:
            return variables, {}
        resolved = dict(variables)
        used_records: Dict[str, Dict[str, Any]] = {}
        suffix = "_upload_id"
        for key, value in list(resolved.items()):
            if isinstance(value, str) and key.endswith(suffix):
                try:
                    record = self.upload_manager.get_record(value)
                except KeyError as err:
                    raise ValueError(str(err)) from err
                used_records[value] = record
                record_view = {
                    "file_id": record["file_id"],
                    "filename": record["filename"],
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                    "content_type": record.get("content_type"),
                    "path": record["container_path"],
                    "expires_at": record.get("expires_at"),
                    "metadata": record.get("metadata"),
                }
                resolved[key] = record_view
                base = key[: -len(suffix)] or key
                resolved[f"{base}_path"] = record["container_path"]
        return resolved, used_records

    def _build_env_kwargs(self) -> Dict[str, Any]:
        env_kwargs: Dict[str, Any] = {}
        volumes: Dict[str, Dict[str, str]] = {}
        if self.upload_manager:
            volumes.update(self.upload_manager.volume_mapping)
        if self.workspace_host_dir:
            volumes[self.workspace_host_dir] = {
                "bind": self.workspace_container_dir,
                "mode": "ro" if self.workspace_read_only else "rw",
            }
            env_kwargs["environment"] = {
                PYTHON_SERVER_WORKDIR_ENV: self.workspace_container_dir,
            }
        if volumes:
            env_kwargs["volumes"] = volumes
        if self.container_name:
            env_kwargs["container_name"] = self.container_name
        if self.host_port is not None:
            env_kwargs["host_port"] = self.host_port
        if self.container_nano_cpus is not None:
            env_kwargs["nano_cpus"] = self.container_nano_cpus
        if self.container_mem_limit is not None:
            env_kwargs["mem_limit"] = self.container_mem_limit
        if self.container_pids_limit is not None:
            env_kwargs["pids_limit"] = self.container_pids_limit
        if self.execution_timeout_seconds is not None:
            env_kwargs["execution_timeout_seconds"] = self.execution_timeout_seconds
        return env_kwargs

    def _build_runner_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            data_path=None,
            image_name=self.image_name,
            model=self.model,
            template=self.template,
            dialogue_limit=self.dialogue_limit,
            max_turns=self.max_turns,
            log_dir=None,
            verbose=self.verbose,
            start_index=0,
            num_tasks=1,
            no_auto_submit=not self.auto_submit,
            env_kwargs=self._build_env_kwargs(),
            max_action_chars=self.max_action_chars,
            max_output_chars=self.max_output_chars,
            max_response_tokens=self.max_response_tokens,
        )

    def _initialize_runner(self) -> None:
        args = self._build_runner_args()
        self.runner = CodeAgentRunner(args, trace_logger=self.trace_logger)
        self.runner.set_auto_submit(self.auto_submit)


class CodeAgentSlotRecycledError(RuntimeError):
    def __init__(
        self,
        *,
        slot_index: Optional[int],
        container_name: Optional[str],
        host_port: Optional[int],
        timeout_seconds: Optional[float],
    ):
        self.slot_index = slot_index
        self.container_name = container_name
        self.host_port = host_port
        self.timeout_seconds = timeout_seconds
        timeout_text = "unknown" if timeout_seconds is None else f"{timeout_seconds:g}"
        message = (
            f"Python execution timed out after {timeout_text} seconds. "
            f"Recycled slot {slot_index} ({container_name}, host port {host_port})."
        )
        super().__init__(message)


class CodeAgentServicePool:
    """Bounded pool of isolated CodeAgentService instances."""

    def __init__(
        self,
        size: int = 1,
        *,
        services: Optional[List[CodeAgentService]] = None,
        **service_kwargs,
    ):
        self.load_balancing_strategy = _normalize_load_balancing_strategy(
            service_kwargs.pop("load_balancing_strategy", None)
        )
        self.max_container_cpu_percent = _normalize_non_negative_float(
            service_kwargs.pop("max_container_cpu_percent", None)
        )
        if services is not None:
            if not services:
                raise ValueError("services must not be empty when provided.")
            self.services = list(services)
            self.size = len(self.services)
        else:
            if size <= 0:
                raise ValueError("size must be a positive integer.")
            self.size = size
            self.services = self._build_services(size=size, **service_kwargs)

        self.logger = logging.getLogger(self.__class__.__name__)
        self._status_lock = threading.Lock()
        self._condition = threading.Condition(self._status_lock)
        self._waiting_requests = 0
        self._next_slot_cursor = 0
        self._service_state = {
            id(service): self._initial_slot_state(service, index)
            for index, service in enumerate(self.services)
        }

        first_service = self.services[0]
        self.default_model = first_service.runner.args.model
        self.default_template = first_service.runner.args.template
        self.default_max_turns = first_service.runner.args.max_turns
        self.default_execution_timeout_seconds = getattr(
            first_service.runner.env,
            "execution_timeout_seconds",
            None,
        )

    def run_task(self, **kwargs) -> Dict[str, Any]:
        service = None
        should_return_service = True
        try:
            with self._condition:
                self._waiting_requests += 1
                try:
                    service = self._acquire_service_locked()
                finally:
                    self._waiting_requests -= 1
                self._mark_service_busy(service, kwargs)
            return service.run_task(**kwargs)
        except CodeAgentSlotRecycledError as err:
            with self._condition:
                self._mark_service_idle(service, error=err, outcome="recycled_after_timeout")
                self._record_service_recycle(service, reason=str(err))
            self.logger.warning(
                "Recycling timed-out slot %s (%s)",
                getattr(service, "slot_index", None),
                getattr(service, "container_name", None),
            )
            try:
                service.recycle(str(err))
            except Exception as recycle_err:
                should_return_service = False
                with self._condition:
                    self._mark_service_unavailable(service, error=recycle_err)
                self.logger.exception("Failed to recycle slot %s: %s", getattr(service, "slot_index", None), recycle_err)
            raise
        except Exception as err:
            with self._condition:
                self._mark_service_idle(service, error=err)
            raise
        finally:
            with self._condition:
                state = self._service_state.get(id(service)) if service is not None else None
                if state and state["busy"]:
                    self._mark_service_idle(service)
                if service is not None and should_return_service:
                    self._mark_service_available(service)
                self._condition.notify_all()

    def close(self) -> None:
        errors = []
        for service in self.services:
            try:
                service.close()
            except Exception as err:
                errors.append(err)
                self.logger.exception("Failed closing pooled service: %s", err)
        if errors:
            raise errors[0]

    def get_status(self) -> Dict[str, Any]:
        with self._status_lock:
            waiting_requests = self._waiting_requests
            slot_states = [self._snapshot_slot_state(service) for service in self.services]

        busy_slots = sum(1 for slot in slot_states if slot["busy"])
        available_slots = sum(1 for slot in slot_states if slot["accepting_requests"] and not slot["busy"])
        return {
            "pool_size": self.size,
            "available_slots": available_slots,
            "busy_slots": busy_slots,
            "waiting_requests": waiting_requests,
            "load_balancing_strategy": self.load_balancing_strategy,
            "max_container_cpu_percent": self.max_container_cpu_percent,
            "slots": slot_states,
        }

    def _build_services(self, size: int, **service_kwargs) -> List[CodeAgentService]:
        image_name = service_kwargs.get("image_name", DEFAULT_IMAGE)
        trace_log_path = service_kwargs.get("trace_log_path")
        host_port_base = _normalize_host_port_base(service_kwargs.pop("host_port_base", None))
        services = []
        for slot_index in range(size):
            kwargs = dict(service_kwargs)
            kwargs["slot_index"] = slot_index
            kwargs["container_name"] = _pool_container_name(image_name, slot_index, size)
            kwargs["host_port"] = _pool_host_port(host_port_base, slot_index)
            if size > 1:
                kwargs["trace_log_path"] = _pool_trace_log_path(trace_log_path, slot_index)
            services.append(CodeAgentService(**kwargs))
        return services

    def _initial_slot_state(self, service: CodeAgentService, index: int) -> Dict[str, Any]:
        return {
            "slot_index": getattr(service, "slot_index", index),
            "container_name": getattr(service, "container_name", None),
            "host_port": getattr(service, "host_port", None),
            "container_nano_cpus": getattr(service, "container_nano_cpus", None),
            "container_mem_limit": getattr(service, "container_mem_limit", None),
            "container_pids_limit": getattr(service, "container_pids_limit", None),
            "workspace_host_dir": getattr(service, "workspace_host_dir", None),
            "workspace_container_dir": getattr(service, "workspace_container_dir", None),
            "workspace_read_only": getattr(service, "workspace_read_only", None),
            "accepting_requests": True,
            "busy": False,
            "current_task": None,
            "dispatch_count": 0,
            "completed_tasks": 0,
            "recycle_count": 0,
            "last_recycled_at": None,
            "last_recycle_reason": None,
            "last_assigned_at": None,
            "last_started_at": None,
            "last_finished_at": None,
            "last_duration_seconds": None,
            "last_outcome": None,
            "last_error": None,
        }

    def _mark_service_busy(self, service: CodeAgentService, kwargs: Dict[str, Any]) -> None:
        state = self._service_state[id(service)]
        started_at = _utc_now()
        state["busy"] = True
        state["dispatch_count"] += 1
        state["last_assigned_at"] = started_at.isoformat()
        state["last_started_at"] = started_at.isoformat()
        state["current_task"] = {
            "query_preview": _preview_query(kwargs.get("query")),
            "model": kwargs.get("model") or service.runner.args.model,
            "template": kwargs.get("template") or service.runner.args.template,
            "max_turns": kwargs.get("max_turns") or service.runner.args.max_turns,
            "execution_timeout_seconds": (
                kwargs.get("execution_timeout_seconds")
                if kwargs.get("execution_timeout_seconds") is not None
                else getattr(service.runner.env, "execution_timeout_seconds", None)
            ),
            "started_at": state["last_started_at"],
            "_started_monotonic": time.monotonic(),
        }

    def _mark_service_idle(
        self,
        service: CodeAgentService,
        error: Optional[Exception] = None,
        outcome: Optional[str] = None,
    ) -> None:
        state = self._service_state[id(service)]
        current_task = state.get("current_task") or {}
        started_monotonic = current_task.get("_started_monotonic")
        finished_at = _utc_now().isoformat()
        state["busy"] = False
        state["current_task"] = None
        state["last_finished_at"] = finished_at
        state["completed_tasks"] += 1
        state["last_outcome"] = outcome or ("failed" if error is not None else "completed")
        state["last_error"] = str(error) if error is not None else None
        if started_monotonic is not None:
            state["last_duration_seconds"] = round(time.monotonic() - started_monotonic, 3)
        else:
            state["last_duration_seconds"] = None

    def _record_service_recycle(self, service: CodeAgentService, reason: str) -> None:
        state = self._service_state[id(service)]
        state["recycle_count"] += 1
        state["last_recycled_at"] = _utc_now().isoformat()
        state["last_recycle_reason"] = reason

    def _mark_service_available(self, service: CodeAgentService) -> None:
        state = self._service_state[id(service)]
        state["accepting_requests"] = True

    def _mark_service_unavailable(self, service: CodeAgentService, error: Optional[Exception] = None) -> None:
        state = self._service_state[id(service)]
        state["accepting_requests"] = False
        if error is not None:
            state["last_error"] = str(error)

    def _acquire_service_locked(self) -> CodeAgentService:
        while True:
            service = self._select_service_locked()
            if service is not None:
                return service
            self._condition.wait()

    def _select_service_locked(self) -> Optional[CodeAgentService]:
        ordered_candidates: List[Tuple[int, CodeAgentService]] = []
        for offset in range(self.size):
            service_index = (self._next_slot_cursor + offset) % self.size
            service = self.services[service_index]
            state = self._service_state[id(service)]
            if state["busy"] or not state["accepting_requests"]:
                continue
            ordered_candidates.append((service_index, service))

        if not ordered_candidates:
            return None

        if self.load_balancing_strategy == "least_loaded" or self.max_container_cpu_percent is not None:
            load_candidates = [
                (service_index, service, _container_cpu_percent(service))
                for service_index, service in ordered_candidates
            ]
            if self.max_container_cpu_percent is not None:
                eligible_candidates = [
                    candidate
                    for candidate in load_candidates
                    if candidate[2] is None or candidate[2] <= self.max_container_cpu_percent
                ]
                if eligible_candidates:
                    load_candidates = eligible_candidates
            if self.load_balancing_strategy == "least_loaded":
                chosen_index, chosen_service, _ = min(
                    load_candidates,
                    key=lambda candidate: _cpu_percent_sort_key(candidate[2]),
                )
            else:
                chosen_index, chosen_service, _ = load_candidates[0]
        else:
            chosen_index, chosen_service = ordered_candidates[0]

        self._next_slot_cursor = (chosen_index + 1) % self.size
        return chosen_service

    def _snapshot_slot_state(self, service: CodeAgentService) -> Dict[str, Any]:
        state = dict(self._service_state[id(service)])
        current_task = state.get("current_task")
        if current_task is not None:
            current_task = dict(current_task)
            started_monotonic = current_task.pop("_started_monotonic", None)
            if started_monotonic is not None:
                current_task["running_for_seconds"] = round(time.monotonic() - started_monotonic, 3)
            state["current_task"] = current_task
        state["container_status"] = _container_status(service)
        state["container_cpu_percent"] = _container_cpu_percent(service)
        return state


def _pool_container_name(image_name: str, slot_index: int, pool_size: int) -> str:
    base = f"{image_name}_ic_ctr"
    if pool_size == 1:
        return base
    return f"{base}_{slot_index}"


def _pool_trace_log_path(trace_log_path: Optional[str], slot_index: int) -> str:
    base_path = Path(trace_log_path or os.path.join("logs", "code_agent_trace.jsonl"))
    slot_name = f"{base_path.stem}_slot_{slot_index}{base_path.suffix}"
    return str(base_path.with_name(slot_name))


def _pool_host_port(host_port_base: int, slot_index: int) -> int:
    return host_port_base + slot_index


def _normalize_host_port_base(value: Optional[int]) -> int:
    if value is None or value == "":
        return 3006
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 3006
    return parsed if parsed > 0 else 3006


def _normalize_workspace_host_dir(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Workspace host directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Workspace host path is not a directory: {path}")
    return str(path)


def _normalize_workspace_container_dir(value: Optional[str]) -> str:
    text = str(value or DEFAULT_WORKSPACE_CONTAINER_DIR).strip()
    if not text:
        text = DEFAULT_WORKSPACE_CONTAINER_DIR
    if not text.startswith("/"):
        raise ValueError(f"Workspace container directory must be absolute: {text}")
    return text.rstrip("/") or "/"


def _container_status(service: CodeAgentService) -> Optional[str]:
    container = getattr(getattr(getattr(service, "runner", None), "env", None), "container", None)
    if container is None:
        return None
    try:
        container.reload()
        return container.status
    except Exception:
        return "unknown"


def _container_cpu_percent(service: CodeAgentService) -> Optional[float]:
    container = getattr(getattr(getattr(service, "runner", None), "env", None), "container", None)
    if container is None:
        return None
    try:
        stats = container.stats(stream=False)
    except Exception:
        return None
    return _docker_cpu_percent(stats)


def _docker_cpu_percent(stats: Dict[str, Any]) -> Optional[float]:
    cpu_stats = stats.get("cpu_stats") or {}
    precpu_stats = stats.get("precpu_stats") or {}
    cpu_usage = cpu_stats.get("cpu_usage") or {}
    precpu_usage = precpu_stats.get("cpu_usage") or {}

    total_usage = cpu_usage.get("total_usage")
    previous_total_usage = precpu_usage.get("total_usage")
    system_cpu_usage = cpu_stats.get("system_cpu_usage")
    previous_system_cpu_usage = precpu_stats.get("system_cpu_usage")
    if None in (total_usage, previous_total_usage, system_cpu_usage, previous_system_cpu_usage):
        return None

    cpu_delta = total_usage - previous_total_usage
    system_delta = system_cpu_usage - previous_system_cpu_usage
    if cpu_delta <= 0 or system_delta <= 0:
        return None

    online_cpus = cpu_stats.get("online_cpus")
    if not online_cpus:
        online_cpus = len(cpu_usage.get("percpu_usage") or []) or 1

    return round((cpu_delta / system_delta) * online_cpus * 100, 2)


def _cpu_percent_sort_key(value: Optional[float]) -> float:
    if value is None:
        return float("inf")
    return value


def _preview_query(query: Optional[str], limit: int = 120) -> str:
    if not query:
        return ""
    single_line = " ".join(query.strip().split())
    return single_line[:limit] + ("..." if len(single_line) > limit else "")


def _normalize_load_balancing_strategy(value: Optional[str]) -> str:
    if value is None or value == "":
        return "round_robin"
    normalized = str(value).strip().lower()
    if normalized in {"round_robin", "least_loaded"}:
        return normalized
    return "round_robin"


def _normalize_non_negative_float(value: Optional[Any]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


MathAgentService = CodeAgentService
MathAgentServicePool = CodeAgentServicePool
MathAgentSlotRecycledError = CodeAgentSlotRecycledError
