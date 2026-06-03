"""Command-line interface for the full TerraBench agent."""

from __future__ import annotations

import argparse
import asyncio

from terra_agent.full_agent.config import _env_bool, _env_value, _parse_bool_arg
from terra_agent.full_agent.runtime import DEFAULT_USER_REQUEST, main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TerraBench single-agent workflow.")
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
        "--user-request",
        default=_env_value("CLIMATE_AGENT_DEFAULT_REQUEST", DEFAULT_USER_REQUEST),
        help="Initial user request prompt passed to the agent.",
    )
    parser.add_argument(
        "--model-provider",
        choices=("openai", "anthropic", "deepseek", "gemini"),
        default=_env_value("CLIMATE_AGENT_MODEL_PROVIDER", "openai"),
        help=(
            "Select 'openai' for OpenAI-compatible endpoints, 'deepseek' for DeepSeek, "
            "'anthropic' for Claude, or 'gemini' for Google Gemini."
        ),
    )
    parser.add_argument(
        "--current-trace",
        dest="current_trace",
        help="Path to the current/partial trace JSON to continue from.",
    )
    parser.add_argument(
        "--extra-message",
        dest="extra_message",
        help="Optional extra guidance appended into the prompt template.",
    )
    parser.add_argument(
        "--agent-csv",
        dest="agent_csv",
        help="Optional CSV path to load the question text for agent mode.",
    )
    parser.add_argument(
        "--agent-index",
        dest="agent_index",
        type=int,
        help="1-based row index used with --agent-csv to select the question.",
    )
    parser.add_argument(
        "--agent-start-index",
        dest="agent_start_index",
        type=int,
        help="1-based start row (inclusive) when running a CSV range.",
    )
    parser.add_argument(
        "--agent-end-index",
        dest="agent_end_index",
        type=int,
        help="1-based end row (inclusive) when running a CSV range.",
    )
    parser.add_argument(
        "--agent-dest",
        dest="agent_dest",
        help="Destination folder for outputs. In CSV mode, one subfolder is created per row.",
    )
    parser.add_argument(
        "--auto-resume-from-dest",
        action="store_true",
        help="When enabled, auto-load the latest trace in each destination folder as --current-trace.",
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
        help=(
            "Enable AgentScope memory compression to summarize older conversation turns and reduce "
            "large model history payloads. Accepts true/false. Threshold can be tuned with "
            "CLIMATE_AGENT_COMPRESS_HISTORY_TRIGGER_TOKENS and keep_recent with "
            "CLIMATE_AGENT_COMPRESS_HISTORY_KEEP_RECENT."
        ),
    )
    parser.add_argument(
        "--compress-current-trace",
        nargs="?",
        const=True,
        type=_parse_bool_arg,
        default=_env_bool("CLIMATE_AGENT_COMPRESS_CURRENT_TRACE", True),
        help=(
            "When --current-trace is provided, summarize that trace once into a gap-analysis bootstrap "
            "summary before the main run starts, instead of injecting the full raw trace into the prompt. "
            "Accepts true/false."
        ),
    )
    parser.add_argument(
        "--number-json-filename",
        default=_env_value("CLIMATE_AGENT_NUMBER_JSON_FILENAME", "number_ground_truth.json"),
        help=(
            "Filename for the extracted <final_json> sidecar written beside each trace. "
            "Set to a different name for prediction/evaluation runs."
        ),
    )
    return parser.parse_args()


def cli_main() -> None:
    args = parse_args()
    asyncio.run(
        main(
            tool_mode=args.tool_mode,
            tool_base_url=args.tool_base_url,
            user_request=args.user_request,
            model_provider=args.model_provider,
            current_trace_path=args.current_trace,
            extra_message=args.extra_message,
            agent_csv=args.agent_csv,
            agent_index=args.agent_index,
            agent_start_index=args.agent_start_index,
            agent_end_index=args.agent_end_index,
            agent_dest=args.agent_dest,
            report_detail=args.report_detail,
            auto_resume_from_dest=args.auto_resume_from_dest,
            compress_history=args.compress_history,
            compress_current_trace=args.compress_current_trace,
            number_json_filename=args.number_json_filename,
        )
    )


if __name__ == "__main__":
    cli_main()
