#!/usr/bin/env python3
"""
Manual CLI runner for the standalone climate model tools.

This script wraps the Aurora and Pangu-Weather tool functions so they can be
executed directly from the command line without spinning up the full agent or
HTTP deployment server.

Examples:
    python manual_tool_runner.py aurora \
        --surface-path ./data/era5/era5_single_levels_a69a0f9a7869.nc \
        --pressure-path ./data/era5/era5_pressure_levels_cdd426d70d6e.nc \
        --static-path ./data/era5/era5_single_levels_98ece744637f.nc \
        --device cuda:3

    python manual_tool_runner.py pangu \
        --surface-path data/era5/era5_single_levels_873d380baa31.nc \
        --pressure-path data/era5/era5_pressure_levels_f1a9e631f8fb.nc \
        --valid-time 2019-12-26T00:00:00 \
        --output-dir ./data/model_forecasts \
        --model-path Pangu_weight/pangu_weather_6.onnx \
        --device cuda:3
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from agentscope.tool import ToolResponse

from terra_agent.tools.climate.aurora_tool import run_aurora_forecast
from terra_agent.tools.climate.math_agent_tool import run_math_agent
from terra_agent.tools.climate.math_agent_data_stream import stream_data_to_math_agent
from terra_agent.tools.climate.pangu_tool import run_pangu_forecast


def _json_argument(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - argument parsing
        raise argparse.ArgumentTypeError(f"Invalid JSON payload: {exc}") from exc


def _float_or_none(value: str) -> Optional[float]:
    if value.lower() in {"none", "null"}:
        return None
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - argument parsing
        raise argparse.ArgumentTypeError(f"Expected float or 'none', got '{value}'.") from exc


def _json_default(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - numpy is an optional dep
        np = None  # type: ignore

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (np.datetime64, np.timedelta64)):
            return value.astype("datetime64[s]").astype(int)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - fallback for unexpected objects
            pass
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _tool_response_to_payload(name: str, response: ToolResponse) -> Dict[str, Any]:
    metadata = response.metadata or {}
    sanitized = json.loads(json.dumps(metadata, default=_json_default))
    trace_value = sanitized.get("trace")
    if isinstance(trace_value, str):
        sanitized["trace_lines"] = trace_value.rstrip("\n").splitlines()
    messages: List[str] = []
    for block in response.content or []:
        text = getattr(block, "text", None)
        if text:
            messages.append(text)
    ok = not sanitized.get("error")
    return {
        "tool": name,
        "ok": bool(ok),
        "metadata": sanitized,
        "messages": messages,
    }


def _merge_extra(kwargs: Dict[str, Any], extra: Any, *, context: str) -> None:
    if extra is None:
        return
    if not isinstance(extra, dict):
        raise ValueError(f"{context}: --extra must be a JSON object mapping.")
    kwargs.update(extra)


def _add_extra_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--extra",
        type=_json_argument,
        help=(
            "Optional JSON object with additional keyword arguments forwarded to the tool. "
            "Useful for advanced parameters not explicitly exposed as CLI flags."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Aurora or Pangu-Weather tools from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="tool", required=True)

    aurora = subparsers.add_parser(
        "aurora",
        help="Run the Aurora forecast tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    aurora.add_argument("--surface-path", required=True, dest="surface_path", help="Path to ERA5 surface NetCDF.")
    aurora.add_argument("--pressure-path", required=True, dest="pressure_path", help="Path to ERA5 pressure NetCDF.")
    aurora.add_argument("--static-path", required=True, dest="static_path", help="Path to ERA5 static fields NetCDF.")
    aurora.add_argument(
        "--selected-times",
        nargs="+",
        dest="selected_times",
        help="Explicit ISO timestamps used as Aurora inputs.",
    )
    aurora.add_argument(
        "--history-steps",
        type=int,
        dest="history_steps",
        help="Number of historical timesteps to consider when selecting inputs.",
    )
    aurora.add_argument(
        "--time-block",
        type=int,
        dest="time_block",
        help="Number of most recent steps from the history window to feed the model.",
    )
    aurora.add_argument(
        "--rollout-steps",
        type=int,
        default=4,
        dest="rollout_steps",
        help="How many forecast steps to generate.",
    )
    aurora.add_argument(
        "--forecast-interval-hours",
        type=_float_or_none,
        default=6.0,
        dest="forecast_interval_hours",
        help="Lead-time spacing between successive outputs ('none' to keep the metadata time).",
    )
    aurora.add_argument(
        "--analysis-time",
        dest="analysis_time",
        help="ISO timestamp used for metadata; defaults to last input time.",
    )
    aurora.add_argument(
        "--output-dir",
        default="./data/model_forecasts",
        dest="output_dir",
        help="Directory where the NetCDF output will be stored.",
    )
    aurora.add_argument(
        "--model-repo",
        default="microsoft/aurora",
        dest="model_repo",
        help="Aurora checkpoint repository or local path.",
    )
    aurora.add_argument(
        "--checkpoint-name",
        default="aurora-0.25-pretrained.ckpt",
        dest="checkpoint_name",
        help="Checkpoint filename inside the model repository.",
    )
    aurora.add_argument(
        "--model-kwargs",
        type=_json_argument,
        dest="model_kwargs",
        help="JSON object forwarded to the Aurora model constructor.",
    )
    aurora.add_argument("--device", default="cuda:0", help="Torch device for the Aurora model (e.g., cuda:0 or cpu).")
    aurora.add_argument(
        "--input-device",
        default="cpu",
        dest="input_device",
        help="Device where the ERA5 tensors are loaded.",
    )
    _add_extra_argument(aurora)

    pangu = subparsers.add_parser(
        "pangu",
        help="Run the Pangu-Weather ONNX tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pangu.add_argument("--surface-path", required=True, dest="surface_path", help="Path to ERA5 surface NetCDF.")
    pangu.add_argument("--pressure-path", required=True, dest="pressure_path", help="Path to ERA5 pressure NetCDF.")
    pangu.add_argument("--valid-time", required=True, dest="valid_time", help="ERA5 analysis time (ISO format).")
    pangu.add_argument(
        "--output-dir",
        default="./data/model_forecasts",
        dest="output_dir",
        help="Directory where the NetCDF output will be stored.",
    )
    pangu.add_argument(
        "--model-path",
        default="Pangu_weather/pangu_weight/pangu_weather_6.onnx",
        dest="model_path",
        help="Path to the Pangu-Weather ONNX checkpoint.",
    )
    pangu.add_argument("--device", default="cuda:0", help="Execution device string (cuda:<index> or cpu).")
    pangu.add_argument(
        "--upper-input-name",
        default="input",
        dest="upper_input_name",
        help="Name of the upper-air input tensor in the ONNX graph.",
    )
    pangu.add_argument(
        "--surface-input-name",
        default="input_surface",
        dest="surface_input_name",
        help="Name of the surface input tensor in the ONNX graph.",
    )
    pangu.add_argument(
        "--lead-hours",
        nargs="+",
        type=float,
        dest="lead_hours",
        help="Explicit lead times (hours) for successive outputs.",
    )
    _add_extra_argument(pangu)

    math_agent = subparsers.add_parser(
        "math_agent",
        help="Forward requests to the local Intercode math agent service.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    math_agent.add_argument("--query", required=True, help="Natural language question or task for the math agent.")
    math_agent.add_argument(
        "--variables",
        type=_json_argument,
        dest="variables",
        help="Optional JSON object with variable bindings forwarded to the Intercode agent.",
    )
    math_agent.add_argument(
        "--max-turns",
        type=int,
        dest="max_turns",
        help="Maximum number of chat turns the Intercode agent may use.",
    )
    math_agent.add_argument(
        "--base-url",
        dest="base_url",
        help="Base URL for the Intercode service (defaults to MATH_AGENT_BASE_URL or http://127.0.0.1:8000).",
    )
    math_agent.add_argument(
        "--route",
        dest="route",
        help="HTTP route for the math agent endpoint (defaults to MATH_AGENT_ROUTE or /math-agent).",
    )
    math_agent.add_argument(
        "--headers",
        type=_json_argument,
        dest="headers",
        help="Optional JSON object of extra HTTP headers (e.g., authentication tokens).",
    )
    math_agent.add_argument(
        "--data-paths",
        nargs="+",
        dest="data_paths",
        help="One or more local data files (NetCDF/JSON/CSV/text) such as ERA5 or model outputs to attach.",
    )
    math_agent.add_argument(
        "--model-output-key",
        dest="model_output_key",
        default="model_outputs",
        help="Variable name used when attaching loaded model outputs.",
    )
    math_agent.add_argument(
        "--model-output-max-text-chars",
        type=int,
        dest="model_output_max_text_chars",
        default=20000,
        help="Maximum characters to read from each text file when forwarding outputs.",
    )
    math_agent.add_argument(
        "--model-output-max-csv-rows",
        type=int,
        dest="model_output_max_csv_rows",
        default=2000,
        help="Maximum CSV rows to include per file.",
    )
    math_agent.add_argument(
        "--model-output-max-values",
        type=int,
        dest="model_output_max_values",
        default=2048,
        help="Maximum flattened values to sample per NetCDF variable.",
    )
    math_agent.add_argument(
        "--model-output-max-variables",
        type=int,
        dest="model_output_max_variables",
        default=5,
        help="Maximum number of NetCDF variables to sample per file.",
    )

    stream_upload = subparsers.add_parser(
        "stream_math_data",
        help="Upload a local data file to the math agent staging endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    stream_upload.add_argument("--file-path", required=True, dest="file_path", help="Local file to upload.")
    stream_upload.add_argument(
        "--upload-url",
        dest="upload_url",
        help="Override the math agent upload URL (defaults to MATH_AGENT_UPLOAD_URL or base URL + /math-agent/upload).",
    )
    stream_upload.add_argument(
        "--metadata",
        type=_json_argument,
        dest="metadata",
        help="Optional JSON metadata blob stored alongside the upload (e.g., variable descriptions).",
    )
    stream_upload.add_argument(
        "--headers",
        type=_json_argument,
        dest="upload_headers",
        help="Optional HTTP headers (JSON object) for authentication when uploading.",
    )
    stream_upload.add_argument(
        "--timeout",
        type=int,
        default=300,
        dest="upload_timeout",
        help="Timeout in seconds for the upload request.",
    )

    return parser


def _invoke_aurora(args: argparse.Namespace) -> ToolResponse:
    kwargs = {
        "surface_path": args.surface_path,
        "pressure_path": args.pressure_path,
        "static_path": args.static_path,
        "selected_times": args.selected_times,
        "history_steps": args.history_steps,
        "time_block": args.time_block,
        "rollout_steps": args.rollout_steps,
        "forecast_interval_hours": args.forecast_interval_hours,
        "analysis_time": args.analysis_time,
        "output_dir": args.output_dir,
        "model_repo": args.model_repo,
        "checkpoint_name": args.checkpoint_name,
        "model_kwargs": args.model_kwargs,
        "device": args.device,
        "input_device": args.input_device,
    }
    _merge_extra(kwargs, args.extra, context="aurora")
    return run_aurora_forecast(**kwargs)


def _invoke_pangu(args: argparse.Namespace) -> ToolResponse:
    kwargs = {
        "surface_path": args.surface_path,
        "pressure_path": args.pressure_path,
        "valid_time": args.valid_time,
        "output_dir": args.output_dir,
        "model_path": args.model_path,
        "device": args.device,
        "upper_input_name": args.upper_input_name,
        "surface_input_name": args.surface_input_name,
        "lead_hours": args.lead_hours,
    }
    _merge_extra(kwargs, args.extra, context="pangu")
    return run_pangu_forecast(**kwargs)


def _invoke_math_agent(args: argparse.Namespace) -> ToolResponse:
    kwargs = {
        "query": args.query,
        "variables": args.variables,
        "max_turns": args.max_turns,
        "base_url": args.base_url,
        "route": args.route,
        "headers": args.headers,
        "data_paths": args.data_paths,
        "model_output_key": args.model_output_key,
        "max_model_output_text_chars": args.model_output_max_text_chars,
        "max_model_output_csv_rows": args.model_output_max_csv_rows,
        "max_model_output_values": args.model_output_max_values,
        "max_model_output_variables": args.model_output_max_variables,
    }
    return run_math_agent(**kwargs)


def _invoke_stream_upload(args: argparse.Namespace) -> ToolResponse:
    kwargs = {
        "file_path": args.file_path,
        "upload_url": args.upload_url,
        "metadata": args.metadata,
        "headers": args.upload_headers,
        "timeout": args.upload_timeout,
    }
    return stream_data_to_math_agent(**kwargs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.tool == "aurora":
            response = _invoke_aurora(args)
        elif args.tool == "pangu":
            response = _invoke_pangu(args)
        elif args.tool == "math_agent":
            response = _invoke_math_agent(args)
        elif args.tool == "stream_math_data":
            response = _invoke_stream_upload(args)
        else:  # pragma: no cover - argparse enforces allowed values
            parser.error(f"Unknown tool '{args.tool}'.")
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError("parser.error should exit")  # pragma: no cover

    payload = _tool_response_to_payload(args.tool, response)
    json.dump(payload, sys.stdout, indent=2, default=_json_default)
    sys.stdout.write("\n")
    trace_lines = payload.get("metadata", {}).get("trace_lines")
    if isinstance(trace_lines, list) and trace_lines:
        sys.stderr.write("\nTraceback (most recent call last):\n")
        sys.stderr.write("\n".join(trace_lines))
        sys.stderr.write("\n")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
