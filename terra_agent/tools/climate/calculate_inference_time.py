"""Summarise model inference time from ClimateAgent experiment traces.

The script walks an Experiments directory, finds every ``single_agent_*.json``
trace, and sums each run's own start/end duration. Summing per-run durations is
intentional: it avoids counting idle wall-clock gaps when a model batch was
paused and resumed later.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENTS_DIR = Path("Experiments")


@dataclass(frozen=True)
class RunTiming:
    model_folder: str
    path: Path
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    timestamp_source: str
    metadata_model: str | None = None


@dataclass
class InvalidTrace:
    path: Path
    reason: str


@dataclass
class ModelSummary:
    model_folder: str
    runs: list[RunTiming] = field(default_factory=list)
    invalid_traces: list[InvalidTrace] = field(default_factory=list)

    @property
    def total_inference_seconds(self) -> float:
        return sum(run.duration_seconds for run in self.runs)

    @property
    def average_run_seconds(self) -> float:
        if not self.runs:
            return 0.0
        return self.total_inference_seconds / len(self.runs)

    @property
    def first_started_at(self) -> datetime | None:
        if not self.runs:
            return None
        return min(run.started_at for run in self.runs)

    @property
    def last_finished_at(self) -> datetime | None:
        if not self.runs:
            return None
        return max(run.finished_at for run in self.runs)

    @property
    def wall_span_seconds(self) -> float:
        first = self.first_started_at
        last = self.last_finished_at
        if not first or not last:
            return 0.0
        return max(0.0, (last - first).total_seconds())

    @property
    def idle_gap_seconds(self) -> float:
        return max(0.0, self.wall_span_seconds - self.total_inference_seconds)

    @property
    def metadata_models(self) -> list[str]:
        return sorted({run.metadata_model for run in self.runs if run.metadata_model})

    def long_gap_seconds(self, threshold_seconds: float) -> float:
        """Return inter-run gaps larger than threshold_seconds.

        This estimates manual stop/resume deadtime while ignoring short launcher
        delays between adjacent questions.
        """
        if len(self.runs) < 2:
            return 0.0

        total = 0.0
        previous_end: datetime | None = None
        for run in sorted(self.runs, key=lambda item: item.started_at):
            if previous_end is not None:
                gap = (run.started_at - previous_end).total_seconds()
                if gap > threshold_seconds:
                    total += gap
            if previous_end is None or run.finished_at > previous_end:
                previous_end = run.finished_at
        return max(0.0, total)

    def active_session_seconds(self, threshold_seconds: float) -> float:
        return max(0.0, self.wall_span_seconds - self.long_gap_seconds(threshold_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate total inference time per model from single_agent trace files. "
            "Totals are sums of per-run durations, so paused periods between runs are excluded."
        )
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_DIR,
        help="Root Experiments directory to scan. Default: %(default)s",
    )
    parser.add_argument(
        "--deadtime-gap-minutes",
        type=float,
        default=30.0,
        help=(
            "Inter-run gap threshold used only for the reported long-deadtime column. "
            "The primary total always sums per-run durations. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path for a CSV summary.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for a JSON summary.",
    )
    parser.add_argument(
        "--sort",
        choices=("model", "total", "runs"),
        default="model",
        help="Sort table by model name, total inference seconds, or run count. Default: %(default)s",
    )
    parser.add_argument(
        "--show-invalid",
        action="store_true",
        help="Print trace paths that could not be timed.",
    )
    return parser.parse_args()


def parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()
    candidates = [text]
    if text.endswith("Z"):
        candidates.append(f"{text[:-1]}+00:00")
    if " " in text and "T" not in text:
        candidates.append(text.replace(" ", "T"))

    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("trace JSON root is not an object")
    return payload


def trajectory_bounds(payload: dict[str, Any]) -> tuple[datetime, datetime] | None:
    timestamps: list[datetime] = []
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list):
        return None

    for record in trajectory:
        if not isinstance(record, dict):
            continue
        parsed = parse_datetime(record.get("timestamp"))
        if parsed:
            timestamps.append(parsed)

    if not timestamps:
        return None
    return min(timestamps), max(timestamps)


def timing_from_trace(experiments_dir: Path, path: Path) -> RunTiming | InvalidTrace:
    try:
        rel_path = path.relative_to(experiments_dir)
    except ValueError:
        rel_path = path
    model_folder = rel_path.parts[0] if rel_path.parts else "(root)"

    try:
        payload = load_json(path)
    except Exception as exc:  # pragma: no cover - defensive CLI handling
        return InvalidTrace(path=path, reason=f"failed to read JSON: {exc}")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    started_at = parse_datetime(metadata.get("started_at"))
    finished_at = parse_datetime(metadata.get("finished_at"))
    timestamp_source = "metadata"

    if not started_at or not finished_at:
        bounds = trajectory_bounds(payload)
        if not bounds:
            return InvalidTrace(path=path, reason="missing started_at/finished_at and trajectory timestamps")
        started_at, finished_at = bounds
        timestamp_source = "trajectory"

    duration_seconds = (finished_at - started_at).total_seconds()
    if duration_seconds < 0:
        return InvalidTrace(path=path, reason="finished_at is before started_at")

    metadata_model = metadata.get("model")
    if not isinstance(metadata_model, str) or not metadata_model.strip():
        metadata_model = None

    return RunTiming(
        model_folder=model_folder,
        path=path,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        timestamp_source=timestamp_source,
        metadata_model=metadata_model,
    )


def collect_summaries(experiments_dir: Path) -> dict[str, ModelSummary]:
    summaries: dict[str, ModelSummary] = {}

    for path in sorted(experiments_dir.rglob("single_agent_*.json")):
        timing = timing_from_trace(experiments_dir, path)
        model_folder = timing.model_folder if isinstance(timing, RunTiming) else _model_from_path(experiments_dir, path)
        summary = summaries.setdefault(model_folder, ModelSummary(model_folder=model_folder))
        if isinstance(timing, RunTiming):
            summary.runs.append(timing)
        else:
            summary.invalid_traces.append(timing)

    return summaries


def _model_from_path(experiments_dir: Path, path: Path) -> str:
    try:
        rel_path = path.relative_to(experiments_dir)
    except ValueError:
        rel_path = path
    return rel_path.parts[0] if rel_path.parts else "(root)"


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_datetime(value: datetime | None) -> str:
    return value.isoformat(sep=" ") if value else ""


def sorted_summaries(summaries: dict[str, ModelSummary], sort_key: str) -> list[ModelSummary]:
    values = list(summaries.values())
    if sort_key == "total":
        return sorted(values, key=lambda item: (-item.total_inference_seconds, item.model_folder.lower()))
    if sort_key == "runs":
        return sorted(values, key=lambda item: (-len(item.runs), item.model_folder.lower()))
    return sorted(values, key=lambda item: item.model_folder.lower())


def rows_for_output(
    summaries: list[ModelSummary],
    *,
    gap_threshold_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.append(
            {
                "model_folder": summary.model_folder,
                "run_count": len(summary.runs),
                "invalid_trace_count": len(summary.invalid_traces),
                "total_inference_seconds": round(summary.total_inference_seconds, 3),
                "total_inference_hms": format_seconds(summary.total_inference_seconds),
                "average_run_seconds": round(summary.average_run_seconds, 3),
                "average_run_hms": format_seconds(summary.average_run_seconds),
                "wall_span_seconds": round(summary.wall_span_seconds, 3),
                "wall_span_hms": format_seconds(summary.wall_span_seconds),
                "idle_gap_seconds": round(summary.idle_gap_seconds, 3),
                "idle_gap_hms": format_seconds(summary.idle_gap_seconds),
                "long_deadtime_seconds": round(summary.long_gap_seconds(gap_threshold_seconds), 3),
                "long_deadtime_hms": format_seconds(summary.long_gap_seconds(gap_threshold_seconds)),
                "active_session_seconds": round(summary.active_session_seconds(gap_threshold_seconds), 3),
                "active_session_hms": format_seconds(summary.active_session_seconds(gap_threshold_seconds)),
                "first_started_at": format_datetime(summary.first_started_at),
                "last_finished_at": format_datetime(summary.last_finished_at),
                "metadata_models": "; ".join(summary.metadata_models),
            }
        )
    return rows


def print_table(rows: list[dict[str, Any]], *, gap_minutes: float) -> None:
    if not rows:
        print("No single_agent_*.json traces found.")
        return

    columns = [
        ("Model", "model_folder", 30),
        ("Runs", "run_count", 6),
        ("Total inference", "total_inference_hms", 16),
        ("Avg/run", "average_run_hms", 10),
        ("Wall span", "wall_span_hms", 14),
        (f"Long deadtime >{gap_minutes:g}m", "long_deadtime_hms", 20),
        ("Invalid", "invalid_trace_count", 8),
    ]
    header = " ".join(title.ljust(width) for title, _, width in columns)
    print(header.rstrip())
    print("-" * len(header.rstrip()))
    for row in rows:
        values = []
        for _, key, width in columns:
            text = str(row[key])
            if len(text) > width:
                text = text[: max(0, width - 1)] + "…"
            values.append(text.ljust(width))
        print(" ".join(values).rstrip())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def print_invalid_traces(summaries: list[ModelSummary]) -> None:
    invalid = [
        invalid_trace
        for summary in summaries
        for invalid_trace in summary.invalid_traces
    ]
    if not invalid:
        return

    print("\nInvalid traces:")
    for item in invalid:
        print(f"- {item.path}: {item.reason}")


def main() -> int:
    args = parse_args()
    experiments_dir = args.experiments_dir.resolve()
    if not experiments_dir.exists():
        raise SystemExit(f"Experiments directory does not exist: {experiments_dir}")

    summaries_by_model = collect_summaries(experiments_dir)
    summaries = sorted_summaries(summaries_by_model, args.sort)
    gap_threshold_seconds = max(0.0, args.deadtime_gap_minutes * 60.0)
    rows = rows_for_output(summaries, gap_threshold_seconds=gap_threshold_seconds)

    print_table(rows, gap_minutes=args.deadtime_gap_minutes)

    total_runs = sum(row["run_count"] for row in rows)
    total_seconds = sum(row["total_inference_seconds"] for row in rows)
    total_invalid = sum(row["invalid_trace_count"] for row in rows)
    print(
        "\nGrand total: "
        f"{format_seconds(total_seconds)} across {total_runs} runs "
        f"({round(total_seconds, 3)} seconds)."
    )
    if total_invalid:
        print(f"Skipped {total_invalid} invalid trace(s). Use --show-invalid for details.")

    if args.show_invalid:
        print_invalid_traces(summaries)

    if args.output_csv:
        write_csv(args.output_csv.resolve(), rows)
        print(f"Wrote CSV summary: {args.output_csv.resolve()}")
    if args.output_json:
        write_json(args.output_json.resolve(), rows)
        print(f"Wrote JSON summary: {args.output_json.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
