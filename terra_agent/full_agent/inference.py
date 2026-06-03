"""Batch inference runner for CSV files using Context + Question + Output Template."""

from __future__ import annotations

import argparse
import asyncio
import os

from terra_agent.full_agent.config import (
    PROMPT_DIR,
    _env_bool,
    _env_value,
    _parse_bool_arg,
)
from terra_agent.full_agent.prompts import _resolve_path
from terra_agent.full_agent.runtime import main as run_single_agent_main


DEFAULT_STOP_STATUSES = (404, 500, 443)
DEFAULT_PROMPT_PATH = PROMPT_DIR / "Inference_prompt.yaml"
DEFAULT_EVAL_NUMBER_JSON_FILENAME = "number_prediction.json"


def _parse_status_codes(value: str) -> set[int]:
    codes: set[int] = set()
    for chunk in value.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            codes.add(int(text))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid HTTP status code {text!r}. Use a comma-separated list like 404,500,504."
            ) from exc
    if not codes:
        raise argparse.ArgumentTypeError("Provide at least one HTTP status code.")
    return codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CSV inference with Context + Question + Output Template and Q-index output folders."
    )
    parser.add_argument(
        "--tool-mode",
        choices=("local", "http"),
        default=_env_value("CLIMATE_AGENT_TOOL_MODE", "local"),
        help="Select 'local' to use in-process tools or 'http' to call remote deployment endpoints.",
    )
    parser.add_argument(
        "--tool-base-url",
        default=_env_value("CLIMATE_AGENT_TOOL_BASE_URL", "http://127.0.0.1:8080"),
        help="Base URL for remote tool endpoints when using --tool-mode=http.",
    )
    parser.add_argument(
        "--model-provider",
        choices=("openai", "anthropic", "deepseek", "gemini"),
        default=_env_value("CLIMATE_AGENT_MODEL_PROVIDER", "openai"),
        help="Model backend used by the agent runtime.",
    )
    parser.add_argument(
        "--agent-csv",
        required=True,
        help="CSV file to run in batch inference mode.",
    )
    parser.add_argument(
        "--agent-index",
        type=int,
        help="1-based row index used with --agent-csv to select a single question.",
    )
    parser.add_argument(
        "--agent-start-index",
        type=int,
        help="1-based start row (inclusive) when running a CSV range.",
    )
    parser.add_argument(
        "--agent-end-index",
        type=int,
        help="1-based end row (inclusive) when running a CSV range.",
    )
    parser.add_argument(
        "--agent-dest",
        help="Destination folder. Defaults to the CSV's parent directory so outputs appear as Q1, Q2, ... beside the CSV.",
    )
    parser.add_argument(
        "--current-trace",
        dest="current_trace",
        help="Path to the current/partial trace JSON to continue from when selecting exactly one row.",
    )
    parser.add_argument(
        "--extra-message",
        dest="extra_message",
        help="Optional extra guidance appended into the prompt template.",
    )
    parser.add_argument(
        "--auto-resume-from-dest",
        action="store_true",
        help="Auto-load the latest trace from each Q-folder before resuming that row.",
    )
    parser.add_argument(
        "--report-detail",
        action="store_true",
        help="Log per-call token usage plus prompt/tool payload sizes for each request.",
    )
    parser.add_argument(
        "--compress-history",
        nargs="?",
        const=True,
        type=_parse_bool_arg,
        default=_env_bool("CLIMATE_AGENT_COMPRESS_HISTORY", False),
        help="Enable AgentScope history compression. Accepts true/false.",
    )
    parser.add_argument(
        "--compress-current-trace",
        nargs="?",
        const=True,
        type=_parse_bool_arg,
        default=_env_bool("CLIMATE_AGENT_COMPRESS_CURRENT_TRACE", True),
        help="Summarize --current-trace before injecting it into the prompt. Accepts true/false.",
    )
    parser.add_argument(
        "--stop-on-math-agent-status",
        type=_parse_status_codes,
        default=set(DEFAULT_STOP_STATUSES),
        help="Comma-separated HTTP status codes that should stop the batch after the fatal math-agent observation is recorded.",
    )
    parser.add_argument(
        "--evaluation-mode",
        action="store_true",
        help=(
            "Write the extracted <final_json> sidecar as "
            f"{DEFAULT_EVAL_NUMBER_JSON_FILENAME} instead of number_ground_truth.json."
        ),
    )
    parser.add_argument(
        "--number-json-filename",
        default="",
        help=(
            "Optional explicit filename for the extracted <final_json> sidecar. "
            "Overrides --evaluation-mode when provided."
        ),
    )
    return parser.parse_args()


def _default_agent_dest(agent_csv: str) -> str:
    csv_path = _resolve_path(agent_csv)
    if not csv_path:
        raise ValueError("Unable to resolve --agent-csv path.")
    return str(csv_path.parent)


if __name__ == "__main__":
    args = parse_args()
    os.environ.setdefault("TERRABENCH_PROMPT_PATH", str(DEFAULT_PROMPT_PATH))
    number_json_filename = args.number_json_filename or (
        DEFAULT_EVAL_NUMBER_JSON_FILENAME if args.evaluation_mode else "number_ground_truth.json"
    )
    asyncio.run(
        run_single_agent_main(
            tool_mode=args.tool_mode,
            tool_base_url=args.tool_base_url,
            model_provider=args.model_provider,
            current_trace_path=args.current_trace,
            extra_message=args.extra_message,
            agent_csv=args.agent_csv,
            agent_index=args.agent_index,
            agent_start_index=args.agent_start_index,
            agent_end_index=args.agent_end_index,
            agent_dest=args.agent_dest or _default_agent_dest(args.agent_csv),
            report_detail=args.report_detail,
            auto_resume_from_dest=args.auto_resume_from_dest,
            compress_history=args.compress_history,
            compress_current_trace=args.compress_current_trace,
            csv_request_mode="inference",
            stop_on_math_agent_status_codes=args.stop_on_math_agent_status,
            stop_batch_on_math_agent_error=True,
            number_json_filename=number_json_filename,
        )
    )
