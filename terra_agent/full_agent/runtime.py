"""Runtime orchestration for single-request and CSV full-agent runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

from terra_agent.full_agent.agent_types import (
    MathAgentObservationStopError,
    _MathAgentStopState,
    _StopAwareReActAgent,
    _ToolStepCompressionReActAgent,
)
from terra_agent.full_agent.config import BASE_DIR, LOGGER, _env_value
from terra_agent.full_agent.csv_requests import (
    _build_request_for_mode,
    _build_row_destination,
    _find_latest_trace,
    _load_csv_rows,
    _resolve_selected_rows,
    _write_question_input,
)
from terra_agent.full_agent.history import (
    _build_history_compression_config,
    _summarize_current_trace_for_prompt,
)
from terra_agent.full_agent.prompts import (
    _fill_prompt_template,
    _load_trace_for_prompt,
    _resolve_path,
    load_system_prompt,
)
from terra_agent.full_agent.toolkit import create_toolkit
from terra_agent.full_agent.tracing.logger import TraceLogger
from terra_agent.full_agent.tracing.writer import TraceWriter
from terra_agent.full_agent.utils import (
    ReportingChatModel,
    TokenUsageReporter,
    _build_chat_backend,
    _record_dialog,
)

DEFAULT_USER_REQUEST = (
    "Use the available TerraBench tools to answer the climate or Earth-system "
    "analysis request supplied for this run."
)


async def _run_single_interaction(
    *,
    tool_mode: str,
    tool_base_url: str,
    user_request: str,
    model_provider: str | None,
    current_trace_path: str | None,
    extra_message: str | None,
    agent_dest: str | None,
    report_detail: bool,
    compress_history: bool,
    compress_current_trace: bool,
    stop_on_math_agent_status_codes: set[int] | None = None,
    number_json_filename: str | None = "number_ground_truth.json",
) -> Path:
    current_path = _resolve_path(current_trace_path)
    current_trace = _load_trace_for_prompt(current_path, prefer_reasoning=False) if current_path else None

    prompt_path, sys_prompt_template = load_system_prompt()

    agent_dest_dir = _resolve_path(agent_dest)
    if agent_dest_dir:
        agent_dest_dir.mkdir(parents=True, exist_ok=True)

    agent_data_dir = None
    if agent_dest_dir:
        agent_data_dir = agent_dest_dir / "data"
        agent_data_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WEB_ARTIFACT_ROOT"] = str(agent_data_dir / "web_artifacts")
        os.environ.setdefault("WEB_ARCHIVE_DEFAULT", "true")

    math_agent_stop_state = _MathAgentStopState(stop_on_math_agent_status_codes)
    toolkit = create_toolkit(
        tool_mode,
        tool_base_url,
        agent_data_dir=agent_data_dir,
        math_agent_stop_state=math_agent_stop_state,
    )

    provider = (model_provider or _env_value("CLIMATE_AGENT_MODEL_PROVIDER", "openai")).strip() or "openai"
    model, formatter = _build_chat_backend(provider)
    reporter = None
    if report_detail:
        reporter = TokenUsageReporter(logger=LOGGER)
        model = ReportingChatModel(model, reporter, report_detail=True, logger=LOGGER)

    current_trace_for_prompt = current_trace
    if current_trace is not None and compress_current_trace:
        compressed_trace = await _summarize_current_trace_for_prompt(
            current_trace=current_trace,
            user_request=user_request,
            model=model,
            formatter=formatter,
        )
        if compressed_trace:
            current_trace_for_prompt = compressed_trace

    sys_prompt = _fill_prompt_template(
        sys_prompt_template,
        current_trace=current_trace_for_prompt,
        extra_message=extra_message,
    )
    compression_config = _build_history_compression_config(
        enabled=compress_history,
        provider=provider,
        model=model,
        formatter=formatter,
    )

    user_msg = Msg(name="user", content=user_request, role="user")

    trace_output_dir = agent_dest_dir or _env_value("CLIMATE_AGENT_TRACE_DIR", "./logs/trajectories")
    trace_writer = TraceWriter(
        output_dir=trace_output_dir,
        file_prefix="single_agent",
        number_json_filename=number_json_filename,
    )

    dialog_trace_path = (
        str(agent_dest_dir / "dialog_traces.jsonl")
        if agent_dest_dir
        else _env_value("CLIMATE_AGENT_DIALOG_TRACE", "./logs/dialog_traces.jsonl")
    )
    tool_switch_guidance = (
        "Tool schemas are grouped by domain to conserve context. Only a small bootstrap tool set is "
        "equipped by default. Before specialized work, call `reset_equipped_tools` to activate the "
        "minimum groups you need from: `core`, `era5_obs`, `air_quality`, `forecast_models`, "
        "`ensemble_verify`, `seasonal`, `geo_osm`, `satellite`, `visualization`, `data_crawl`, "
        "`simulators`. "
        "Treat non-basic tools as a hard budget: keep exactly one specialist group active at a time, "
        "so each step has `basic` plus exactly one additional active group when specialized work is needed. "
        "If the task asks for `MeteoAQ` or point air-quality pollutant retrieval, activate `air_quality` first. "
        "Every `reset_equipped_tools` call should enable one and only one specialist group. Do not send "
        "multi-group payloads such as enabling `core`, `era5_obs`, `air_quality`, `satellite`, `geo_osm`, and "
        "`visualization` together. If the next step needs a different domain, call `reset_equipped_tools` "
        "again first so the old specialist group is turned off and only the new one remains active. "
        "Switch groups serially; do not pre-activate future groups."
    )
    sys_prompt += f"\n\n{tool_switch_guidance}\n"
    if agent_dest_dir:
        destination_prompt = f"Every new data should be saved to {agent_dest_dir}/data/"
        sys_prompt += f"\n\n{destination_prompt}\n"

    if compression_config:
        agent_cls = _ToolStepCompressionReActAgent
    elif math_agent_stop_state.enabled:
        agent_cls = _StopAwareReActAgent
    else:
        agent_cls = ReActAgent
    agent_kwargs: dict[str, Any] = {
        "name": "TerraAgent",
        "sys_prompt": sys_prompt,
        "model": model,
        "formatter": formatter,
        "toolkit": toolkit,
        "memory": InMemoryMemory(),
        "enable_meta_tool": True,
        "parallel_tool_calls": False,
        "max_iters": 100,
        "compression_config": compression_config,
    }
    if math_agent_stop_state.enabled:
        agent_kwargs["math_agent_stop_state"] = math_agent_stop_state
    agent = agent_cls(
        **agent_kwargs,
    )

    dialog_logger = TraceLogger(dialog_trace_path, agent=agent.name)
    dialog_logger.run_started(
        task=user_request,
        model=model.model_name,
        model_provider=provider,
        params={
            "tool_mode": tool_mode,
            "model_provider": provider,
            "compress_history": compress_history,
            "compress_current_trace": compress_current_trace,
        },
    )

    trace_writer.start_run(
        prompt_path=str(prompt_path),
        user_request=user_request,
        extra_metadata={
            "agent_name": agent.name,
            "model": model.model_name,
            "model_provider": provider,
            "tool_mode": tool_mode,
            "tool_base_url": tool_base_url if tool_mode.lower() == "http" else None,
            "current_trace_path": str(current_path) if current_path else None,
            "compress_history": compress_history,
            "compress_current_trace": compress_current_trace,
        },
    )

    try:
        await agent(user_msg)
    finally:
        messages = await agent.memory.get_memory()
        trace_writer.add_messages(messages)
        trace_path = trace_writer.finalize()
        print(f"[TraceWriter] Trajectory saved to {trace_path}")
        if reporter:
            reporter.log_summary()
        try:
            final_answer = _record_dialog(dialog_logger, messages)
            dialog_logger.final(
                answer=final_answer or "",
                artifacts=[str(trace_path)],
                metrics={},
            )
        finally:
            dialog_logger.run_finished()
            dialog_logger.close()
    return trace_path


async def main(
    *,
    tool_mode: str = "local",
    tool_base_url: str = "http://127.0.0.1:8080",
    user_request: str = DEFAULT_USER_REQUEST,
    model_provider: str | None = None,
    current_trace_path: str | None = None,
    extra_message: str | None = None,
    agent_csv: str | None = None,
    agent_index: int | None = None,
    agent_start_index: int | None = None,
    agent_end_index: int | None = None,
    agent_dest: str | None = None,
    report_detail: bool = False,
    auto_resume_from_dest: bool = False,
    compress_history: bool = False,
    compress_current_trace: bool = True,
    csv_request_mode: str = "full",
    stop_on_math_agent_status_codes: set[int] | None = None,
    stop_batch_on_math_agent_error: bool = False,
    number_json_filename: str | None = "number_ground_truth.json",
) -> None:
    # (optional) wire up tracing now or later (AgentScope Studio / OTLP)
    # agentscope.init(
    #     project="Climate_Copilot",
    #     name="test_analysis_2025_11_07",
    #     logging_path="./logs/climate_copilot.log",
    #     logging_level="INFO" # For visualization
    # )                  # Studio
    # agentscope.init(tracing_url="https://your-otel-collector/v1/traces")  # OTLP
    # (See “Tracing” below.)

    if not agent_csv:
        if agent_index is not None or agent_start_index is not None or agent_end_index is not None:
            raise ValueError("CSV row selection flags require --agent-csv.")
        selected_current_trace = current_trace_path
        if auto_resume_from_dest and not selected_current_trace and agent_dest:
            dest_dir = _resolve_path(agent_dest)
            if dest_dir and dest_dir.exists():
                latest_trace = _find_latest_trace(dest_dir)
                if latest_trace:
                    selected_current_trace = str(latest_trace)
                    print(f"[resume] Using current trace from {latest_trace}")
        await _run_single_interaction(
            tool_mode=tool_mode,
            tool_base_url=tool_base_url,
            user_request=user_request,
            model_provider=model_provider,
            current_trace_path=selected_current_trace,
            extra_message=extra_message,
            agent_dest=agent_dest,
            report_detail=report_detail,
            compress_history=compress_history,
            compress_current_trace=compress_current_trace,
            stop_on_math_agent_status_codes=stop_on_math_agent_status_codes,
            number_json_filename=number_json_filename,
        )
        return

    csv_path = _resolve_path(agent_csv)
    if not csv_path:
        raise ValueError("Unable to resolve --agent-csv path.")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = _load_csv_rows(csv_path)
    selected_rows = _resolve_selected_rows(
        len(rows),
        index=agent_index,
        start_index=agent_start_index,
        end_index=agent_end_index,
    )
    if not selected_rows:
        raise ValueError(f"No rows found in {csv_path}")
    if current_trace_path and len(selected_rows) != 1:
        raise ValueError("--current-trace can only be used when selecting exactly one CSV row.")

    base_dest = _resolve_path(agent_dest) or (BASE_DIR / "logs" / "agent_answers")
    base_dest.mkdir(parents=True, exist_ok=True)
    print(f"[csv] Loaded {len(rows)} row(s); running {len(selected_rows)} row(s) from {csv_path}")
    print(f"[csv] Output root: {base_dest}")

    succeeded = 0
    failed = 0
    for row_index in selected_rows:
        row = rows[row_index - 1]
        row_request = _build_request_for_mode(row, request_mode=csv_request_mode)
        row_dest = _build_row_destination(base_dest, row_index=row_index)
        row_dest.mkdir(parents=True, exist_ok=True)
        _write_question_input(row_dest, row_index=row_index, row=row, user_request=row_request)

        row_current_trace = current_trace_path
        if auto_resume_from_dest and not row_current_trace:
            latest_trace = _find_latest_trace(row_dest)
            if latest_trace:
                row_current_trace = str(latest_trace)
                print(f"[{row_index}] Resuming from {latest_trace.name}")

        print(f"[{row_index}] Running question -> {row_dest}")
        try:
            trace_path = await _run_single_interaction(
                tool_mode=tool_mode,
                tool_base_url=tool_base_url,
                user_request=row_request,
                model_provider=model_provider,
                current_trace_path=row_current_trace,
                extra_message=extra_message,
                agent_dest=str(row_dest),
                report_detail=report_detail,
                compress_history=compress_history,
                compress_current_trace=compress_current_trace,
                stop_on_math_agent_status_codes=stop_on_math_agent_status_codes,
                number_json_filename=number_json_filename,
            )
            print(f"[{row_index}] Completed -> {trace_path}")
            succeeded += 1
        except MathAgentObservationStopError as exc:
            LOGGER.exception("Row %s stopped after fatal math-agent observation", row_index)
            print(f"[{row_index}] Stopped on math-agent HTTP error: {exc}")
            failed += 1
            if stop_batch_on_math_agent_error:
                print(f"[csv] Stopping batch at row {row_index}. Resume with --agent-start-index {row_index}.")
                break
        except Exception as exc:
            LOGGER.exception("Row %s failed", row_index)
            print(f"[{row_index}] Failed: {exc}")
            failed += 1

    print(f"[csv] Done. succeeded={succeeded} failed={failed}")
