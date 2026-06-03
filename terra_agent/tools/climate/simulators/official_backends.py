"""
Optional official simulator backends.

These adapters preserve the ClimateAgent tool response contract while delegating
the numerical work to installed official packages or command-line executables.
They intentionally fail fast when the required official backend is unavailable.
"""
from __future__ import annotations

import csv
import os
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency for standalone imports
    load_dotenv = None

from .common import (
    DEFAULT_OUTPUT_DIR,
    _artifact_record,
    _as_float,
    _clamp,
    _error,
    _load_rows,
    _metric,
    _parse_datetime_from_row,
    _percentile,
    _read_json,
    _safe_slug,
    _success,
    _tool_run_dir,
    _write_csv,
    _write_json,
)


OFFICIAL_BACKEND_ENV = "CLIMATE_SIMULATOR_BACKEND"
OFFICIAL = "official"
PROXY = "proxy"


def _load_backend_env() -> None:
    if load_dotenv is not None:
        load_dotenv()
        return

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


_load_backend_env()


def _backend_mode(backend: str | None) -> str:
    requested = backend if backend is not None else os.getenv(OFFICIAL_BACKEND_ENV, OFFICIAL)
    normalized = str(requested or OFFICIAL).strip().lower()
    aliases = {
        "local": PROXY,
        "deterministic": PROXY,
        "deterministic_proxy": PROXY,
        "simulator": OFFICIAL,
        "official_api": OFFICIAL,
    }
    return aliases.get(normalized, normalized)


def wants_official_backend(backend: str | None) -> bool:
    return _backend_mode(backend) == OFFICIAL


def _official_run_dir(tool_name: str, run_id: str, scenario_name: str, output_dir: str | Path) -> Path:
    return _tool_run_dir(output_dir, f"{tool_name}_official", run_id, scenario_name)


def _resolve_executable(explicit: str | None, env_name: str, default_name: str) -> str | None:
    candidate = explicit or os.getenv(env_name)
    if candidate:
        parts = shlex.split(candidate)
        return parts[0] if parts else None
    return shutil.which(default_name)


def _split_command(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    return [str(item) for item in command]


def _looks_like_placeholder(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and (("<" in text and ">" in text) or text.startswith("/path/to/"))


def _configured_value(*values: Any) -> Any:
    for value in values:
        if value is not None and not _looks_like_placeholder(value):
            return value
    return None


def _run_command(
    command: str | Sequence[str],
    *,
    cwd: str | Path | None,
    timeout_seconds: int,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = _split_command(command)
    if not cmd:
        raise ValueError("Official backend command is empty.")
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=merged_env,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr = f"{stderr}\nTimeoutExpired: command timed out after {exc.timeout} seconds".strip()
        return subprocess.CompletedProcess(cmd, 124, stdout, stderr)
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", f"{type(exc).__name__}: {exc}")


def _write_command_log(run_dir: Path, result: subprocess.CompletedProcess[str], *, command: Sequence[str]) -> Path:
    log_path = run_dir / "official_backend_run.log"
    payload = [
        "command: " + " ".join(shlex.quote(part) for part in command),
        f"returncode: {result.returncode}",
        "",
        "[stdout]",
        result.stdout or "",
        "",
        "[stderr]",
        result.stderr or "",
    ]
    log_path.write_text("\n".join(payload), encoding="utf-8")
    return log_path


_ERROR_MARKERS = (
    "severe",
    "fatal",
    "error",
    "failed",
    "failure",
    "exception",
    "traceback",
    "not found",
    "no such file",
    "cannot",
    "unable",
    "forrtl",
    "segmentation",
)
_WARNING_MARKERS = ("warning", "warn")
_NON_ISSUE_MARKERS = (
    "0 severe errors",
    "0 fatal errors",
    "0 warning",
    "0 errors",
    "no errors",
    "completed successfully",
)
_DIAGNOSTIC_SUFFIXES = {".log", ".err", ".out", ".txt"}


def _short_log_line(value: str, *, limit: int = 500) -> str:
    text = " ".join(str(value).strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _tail_text(value: str, *, max_chars: int = 2000) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def _line_is_issue(line: str, markers: Sequence[str]) -> bool:
    lower = line.lower()
    if any(marker in lower for marker in _NON_ISSUE_MARKERS):
        return False
    return any(marker in lower for marker in markers)


def _issue_records_from_text(
    text: str,
    *,
    source: str,
    max_errors: int,
    max_warnings: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(str(text or "").splitlines(), start=1):
        line = _short_log_line(raw_line)
        if not line:
            continue
        if len(errors) < max_errors and _line_is_issue(line, _ERROR_MARKERS):
            errors.append({"source": source, "line": line_no, "text": line})
            continue
        if len(warnings) < max_warnings and _line_is_issue(line, _WARNING_MARKERS):
            warnings.append({"source": source, "line": line_no, "text": line})
        if len(errors) >= max_errors and len(warnings) >= max_warnings:
            break
    return errors, warnings


def _read_text_tail(path: Path, *, max_chars: int = 120_000) -> str:
    try:
        with path.open("rb") as handle:
            size = path.stat().st_size
            if size > max_chars:
                handle.seek(-max_chars, os.SEEK_END)
            data = handle.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="ignore")


def _collect_log_diagnostics(
    run_dir: Path,
    *,
    result: subprocess.CompletedProcess[str] | None = None,
    log_path: Path | None = None,
    max_errors: int = 12,
    max_warnings: int = 8,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    if result is not None:
        diagnostics["returncode"] = result.returncode
        stdout_tail = _tail_text(result.stdout or "")
        stderr_tail = _tail_text(result.stderr or "")
        if stdout_tail:
            diagnostics["stdout_tail"] = stdout_tail
        if stderr_tail:
            diagnostics["stderr_tail"] = stderr_tail

    if log_path is not None:
        diagnostics["command_log_path"] = str(log_path.resolve())

    error_lines: list[dict[str, Any]] = []
    warning_lines: list[dict[str, Any]] = []
    if result is not None:
        for stream_name, stream_text in (("stdout", result.stdout or ""), ("stderr", result.stderr or "")):
            errors, warnings = _issue_records_from_text(
                stream_text,
                source=stream_name,
                max_errors=max_errors - len(error_lines),
                max_warnings=max_warnings - len(warning_lines),
            )
            error_lines.extend(errors)
            warning_lines.extend(warnings)

    for path in sorted(run_dir.rglob("*")):
        if len(error_lines) >= max_errors and len(warning_lines) >= max_warnings:
            break
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in _DIAGNOSTIC_SUFFIXES:
            continue
        text = _read_text_tail(path)
        if not text:
            continue
        errors, warnings = _issue_records_from_text(
            text,
            source=str(path.resolve()),
            max_errors=max_errors - len(error_lines),
            max_warnings=max_warnings - len(warning_lines),
        )
        error_lines.extend(errors)
        warning_lines.extend(warnings)

    if error_lines:
        diagnostics["error_lines"] = error_lines
    elif result is not None and result.returncode:
        diagnostics["error_lines"] = [
            {
                "source": "returncode",
                "line": None,
                "text": f"Process exited with code {result.returncode}; no stderr error text was emitted.",
            }
        ]
    if warning_lines:
        diagnostics["warning_lines"] = warning_lines
    return diagnostics


def _failed_command_response(
    tool_name: str,
    *,
    message: str,
    start_ts: float,
    run_context: Mapping[str, Any],
    warnings: list[str],
    run_dir: Path,
    result: subprocess.CompletedProcess[str],
    log_path: Path,
) -> Any:
    diagnostics = _collect_log_diagnostics(run_dir, result=result, log_path=log_path)
    diagnostic_warnings = list(dict.fromkeys(record["text"] for record in diagnostics.get("warning_lines", [])[:3]))
    diagnostic_errors = list(dict.fromkeys(record["text"] for record in diagnostics.get("error_lines", [])[:5]))
    command = " ".join(shlex.quote(str(part)) for part in _split_command(getattr(result, "args", [])))
    return _error(
        tool_name,
        message,
        start_ts=start_ts,
        run_context=run_context,
        warnings=[*warnings, *diagnostic_warnings],
        output_artifacts=_collect_artifacts(run_dir),
        provenance_errors=diagnostic_errors,
        provenance_command=command or f"inprocess:{tool_name}",
        extra={"log_diagnostics": diagnostics},
    )


def _collect_artifacts(run_dir: Path, *, descriptions: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    descriptions = dict(descriptions or {})
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        desc = descriptions.get(path.name, f"Official backend artifact: {path.name}")
        records.append(_artifact_record(path, desc))
    return records


def _load_metrics_from_assumptions(assumptions: Mapping[str, Any] | None) -> dict[str, Any]:
    if not assumptions:
        return {}
    metrics_path = assumptions.get("official_metrics_path") or assumptions.get("metrics_path")
    if not metrics_path:
        return {}
    payload = _read_json(str(metrics_path))
    if not isinstance(payload, dict):
        raise ValueError("official_metrics_path must contain a JSON object.")
    metrics: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict) and "value" in value:
            metrics[str(key)] = value
        else:
            metrics[str(key)] = _metric(_as_float(value), "unknown")
    return metrics


def _generic_command_backend(
    *,
    tool_name: str,
    command_env: str,
    default_executable: str,
    explicit_command: str | Sequence[str] | None,
    run_id: str,
    scenario_name: str,
    output_dir: str | Path,
    start_ts: float,
    run_context: Mapping[str, Any],
    timeout_seconds: int,
    assumptions: Mapping[str, Any] | None,
    input_env: Mapping[str, str],
    warnings: list[str],
) -> Any:
    run_dir = _official_run_dir(tool_name, run_id, scenario_name, output_dir)
    command: str | Sequence[str] | None = explicit_command or os.getenv(command_env)
    if command is None:
        exe = _resolve_executable(None, command_env, default_executable)
        if exe is None:
            return _error(
                tool_name,
                f"Official backend requested, but no command was configured. Set {command_env} or pass assumptions.official_command.",
                start_ts=start_ts,
                run_context=run_context,
                warnings=warnings,
            )
        command = [exe]

    env = {**input_env, "CLIMATE_AGENT_OFFICIAL_RUN_DIR": str(run_dir)}
    result = _run_command(command, cwd=run_dir, timeout_seconds=timeout_seconds, env=env)
    command_list = _split_command(command)
    log_path = _write_command_log(run_dir, result, command=command_list)
    if result.returncode != 0:
        return _failed_command_response(
            tool_name,
            message=f"Official backend command failed with exit code {result.returncode}. See {log_path}.",
            start_ts=start_ts,
            run_context=run_context,
            warnings=warnings,
            run_dir=run_dir,
            result=result,
            log_path=log_path,
        )

    metrics = _load_metrics_from_assumptions(assumptions)
    if not metrics:
        warnings.append("Official command completed, but no official_metrics_path was supplied; returning raw artifacts only.")
    return _success(
        tool_name=tool_name,
        start_ts=start_ts,
        metrics=metrics,
        derived_metrics={"official_artifact_count": _metric(float(len(list(run_dir.rglob('*')))), "count")},
        output_artifacts=_collect_artifacts(run_dir),
        run_context=run_context,
        warnings=warnings,
        agentic_summary={
            "headline": f"Official backend completed for {tool_name}.",
            "primary_signal": {"metric": "official_artifact_count", "value": float(len(list(run_dir.rglob('*')))), "unit": "count"},
            "risk_level": "none",
            "comparison_ready": True,
            "recommended_next_tools": [],
        },
        provenance_command=" ".join(shlex.quote(part) for part in command_list),
    )


def run_aquacrop_official(
    *,
    weather_daily_path: str,
    soil_path: str,
    crop_path: str,
    management_path: str,
    run_id: str,
    scenario_name: str,
    output_dir: str | Path,
    start_ts: float,
    run_context: Mapping[str, Any],
    timeout_seconds: int,
    assumptions: Mapping[str, Any] | None,
    warnings: list[str],
) -> Any:
    assumptions = dict(assumptions or {})
    return _generic_command_backend(
        tool_name="impact_aquacrop_run",
        command_env="AQUACROP_COMMAND",
        default_executable="aquacrop",
        explicit_command=assumptions.get("official_command"),
        run_id=run_id,
        scenario_name=scenario_name,
        output_dir=output_dir,
        start_ts=start_ts,
        run_context=run_context,
        timeout_seconds=timeout_seconds,
        assumptions=assumptions,
        input_env={
            "AQUACROP_WEATHER_DAILY_PATH": weather_daily_path,
            "AQUACROP_SOIL_PATH": soil_path,
            "AQUACROP_CROP_PATH": crop_path,
            "AQUACROP_MANAGEMENT_PATH": management_path,
        },
        warnings=warnings,
    )


def run_dssat_official(
    *,
    weather_daily_path: str,
    soil_profile_path: str,
    cultivar_path: str,
    treatment_path: str,
    run_id: str,
    scenario_name: str,
    output_dir: str | Path,
    start_ts: float,
    run_context: Mapping[str, Any],
    timeout_seconds: int,
    assumptions: Mapping[str, Any] | None,
    warnings: list[str],
) -> Any:
    assumptions = dict(assumptions or {})
    command = _configured_value(assumptions.get("official_command"), os.getenv("DSSAT_COMMAND"))
    run_folder = _configured_value(
        assumptions.get("dssat_run_folder"),
        os.getenv("DSSAT_RUN_FOLDER"),
        os.getenv("DSSAT_RUN_DIR"),
    )
    metrics_path = _configured_value(
        assumptions.get("official_metrics_path"),
        assumptions.get("metrics_path"),
        os.getenv("DSSAT_METRICS_PATH"),
        os.getenv("DSSAT_OFFICIAL_METRICS_PATH"),
    )
    if metrics_path is not None:
        assumptions["official_metrics_path"] = str(metrics_path)

    raw_command = assumptions.get("official_command") or os.getenv("DSSAT_COMMAND")
    if command is None and _looks_like_placeholder(raw_command):
        return _error(
            "impact_dssat_run",
            "Official DSSAT backend needs a real DSCSM executable command; replace the placeholder "
            "or set DSSAT_COMMAND in .env with something like '/absolute/path/to/dscsm048 A EXPERIMENT.MZX'.",
            start_ts=start_ts,
            run_context=run_context,
            warnings=warnings,
            provenance_errors=[f"Invalid DSSAT command: {raw_command!r}"],
        )
    raw_run_folder = assumptions.get("dssat_run_folder") or os.getenv("DSSAT_RUN_FOLDER") or os.getenv("DSSAT_RUN_DIR")
    if run_folder is None and _looks_like_placeholder(raw_run_folder):
        return _error(
            "impact_dssat_run",
            "Official DSSAT backend needs a real prepared DSSAT run folder; replace the placeholder "
            "or set DSSAT_RUN_FOLDER in .env to an existing directory containing the experiment file and DSSAT inputs.",
            start_ts=start_ts,
            run_context=run_context,
            warnings=warnings,
            provenance_errors=[f"Invalid DSSAT run folder: {raw_run_folder!r}"],
            provenance_command=str(command),
        )
    cwd = Path(str(run_folder)).expanduser() if run_folder else _official_run_dir("impact_dssat_run", run_id, scenario_name, output_dir)
    cwd.mkdir(parents=True, exist_ok=True)
    if command is None:
        exe = _resolve_executable(None, "DSSAT_COMMAND", "dscsm048")
        if exe is None:
            return _error(
                "impact_dssat_run",
                "Official DSSAT backend requested, but DSSAT is not configured. Set DSSAT_COMMAND or assumptions.official_command.",
                start_ts=start_ts,
                run_context=run_context,
                warnings=warnings,
            )
        command = [exe]
    result = _run_command(
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        env={
            "DSSAT_WEATHER_DAILY_PATH": weather_daily_path,
            "DSSAT_SOIL_PROFILE_PATH": soil_profile_path,
            "DSSAT_CULTIVAR_PATH": cultivar_path,
            "DSSAT_TREATMENT_PATH": treatment_path,
            "CLIMATE_AGENT_OFFICIAL_RUN_DIR": str(cwd),
        },
    )
    command_list = _split_command(command)
    log_path = _write_command_log(cwd, result, command=command_list)
    if result.returncode != 0:
        return _failed_command_response(
            "impact_dssat_run",
            message=f"Official DSSAT command failed with exit code {result.returncode}. See {log_path}.",
            start_ts=start_ts,
            run_context=run_context,
            warnings=warnings,
            run_dir=cwd,
            result=result,
            log_path=log_path,
        )
    metrics = _load_metrics_from_assumptions(assumptions)
    if not metrics:
        warnings.append("DSSAT completed, but no official_metrics_path was supplied; raw DSSAT artifacts are captured.")
    archive_path = cwd / "dssat_official_run_folder.tar.gz"
    shutil.make_archive(str(archive_path.with_suffix("").with_suffix("")), "gztar", root_dir=cwd)
    outputs = _collect_artifacts(cwd)
    return _success(
        tool_name="impact_dssat_run",
        start_ts=start_ts,
        metrics=metrics,
        derived_metrics={"official_artifact_count": _metric(float(len(outputs)), "count")},
        output_artifacts=outputs,
        run_context=run_context,
        warnings=warnings,
        agentic_summary={
            "headline": "Official DSSAT backend completed.",
            "primary_signal": {"metric": "official_artifact_count", "value": float(len(outputs)), "unit": "count"},
            "risk_level": "none",
            "comparison_ready": True,
            "recommended_next_tools": [],
        },
        provenance_command=" ".join(shlex.quote(part) for part in command_list),
    )


def _load_climada_builtin_impact_functions(
    ImpactFuncSet: Any,
    *,
    builtin_name: str,
) -> tuple[Any, str]:
    normalized = builtin_name.strip().lower().replace("-", "_").replace(" ", "_")
    tc_aliases = {
        "tc",
        "tropical_cyclone",
        "tropical_cyclone_emanuel_usa",
        "tc_emanuel_usa",
        "emanuel_usa",
        "builtin:tropical_cyclone",
        "builtin:tropical_cyclone_emanuel_usa",
        "builtin:tc",
        "builtin:tc_emanuel_usa",
    }
    if normalized in tc_aliases:
        from climada.entity import ImpfTropCyclone

        return (
            ImpactFuncSet([ImpfTropCyclone.from_emanuel_usa()]),
            "python:climada.entity.ImpfTropCyclone.from_emanuel_usa",
        )
    raise ValueError(
        f"Unsupported built-in CLIMADA impact function source {builtin_name!r}. "
        "Supported built-ins: builtin:tropical_cyclone_emanuel_usa."
    )


def _impact_func_set_from_json(
    ImpactFuncSet: Any,
    ImpactFunc: Any,
    payload: Any,
    *,
    assumptions: Mapping[str, Any] | None,
    warnings: list[str],
) -> Any:
    if not isinstance(payload, Mapping):
        raise ValueError("CLIMADA JSON impact-function spec must contain a JSON object.")

    builtin = payload.get("builtin") or payload.get("source")
    if builtin:
        impact_funcs, _ = _load_climada_builtin_impact_functions(ImpactFuncSet, builtin_name=str(builtin))
        return impact_funcs

    assumptions = dict(assumptions or {})
    default_haz_type = str(
        payload.get("haz_type")
        or payload.get("hazard_type")
        or assumptions.get("haz_type")
        or assumptions.get("hazard_type")
        or "TC"
    )
    default_unit = str(payload.get("intensity_unit") or assumptions.get("intensity_unit") or "")

    raw_functions = payload.get("functions")
    if isinstance(raw_functions, Mapping):
        specs = []
        for key, value in raw_functions.items():
            if isinstance(value, Mapping):
                merged = dict(value)
                merged.setdefault("id", key)
                specs.append(merged)
    elif isinstance(raw_functions, Sequence) and not isinstance(raw_functions, (str, bytes)):
        specs = [dict(item) for item in raw_functions if isinstance(item, Mapping)]
    else:
        default_spec = payload.get("default", payload)
        specs = [dict(default_spec)] if isinstance(default_spec, Mapping) else []

    if not specs:
        raise ValueError("CLIMADA JSON impact-function spec did not define any functions.")

    funcs: list[Any] = []
    for idx, spec in enumerate(specs, start=1):
        raw_id = spec.get("id", idx)
        try:
            func_id = int(raw_id)
        except (TypeError, ValueError):
            func_id = idx
        haz_type = str(spec.get("haz_type") or spec.get("hazard_type") or default_haz_type)
        intensity_unit = str(spec.get("intensity_unit") or default_unit)
        name = str(spec.get("name") or f"json_impact_function_{func_id}")

        intensity_raw = spec.get("intensity")
        mdd_raw = spec.get("mdd") or spec.get("mean_damage_degree")
        paa_raw = spec.get("paa") or spec.get("percentage_affected")
        if isinstance(intensity_raw, Sequence) and isinstance(mdd_raw, Sequence) and not isinstance(intensity_raw, (str, bytes)):
            intensity = [_as_float(value) for value in intensity_raw]
            mdd = [_clamp(_as_float(value), 0.0, 1.0) for value in mdd_raw]
            if paa_raw is None:
                paa = [1.0 for _ in intensity]
            elif isinstance(paa_raw, Sequence) and not isinstance(paa_raw, (str, bytes)):
                paa = [_clamp(_as_float(value), 0.0, 1.0) for value in paa_raw]
            else:
                raise ValueError(f"Impact function {name!r} has non-list paa.")
        else:
            threshold = _as_float(spec.get("threshold"), 0.0)
            saturation = max(threshold + 1e-6, _as_float(spec.get("saturation"), 100.0))
            beta = max(0.1, _as_float(spec.get("beta"), 1.4))
            min_damage = _clamp(_as_float(spec.get("min_damage"), 0.0), 0.0, 1.0)
            midpoint = threshold + 0.5 * (saturation - threshold)
            intensity = [0.0, threshold, midpoint, saturation]
            mdd = []
            for value in intensity:
                if value <= threshold:
                    mdd.append(min_damage)
                else:
                    ratio = _clamp((value - threshold) / (saturation - threshold), 0.0, 1.0)
                    mdd.append(_clamp(min_damage + (1.0 - min_damage) * (ratio**beta), 0.0, 1.0))
            paa = [1.0 for _ in intensity]
            warnings.append(
                f"Generated CLIMADA impact function {name!r} from threshold/saturation JSON parameters."
            )

        if not (len(intensity) == len(mdd) == len(paa)):
            raise ValueError(f"Impact function {name!r} has mismatched intensity/mdd/paa lengths.")

        funcs.append(
            ImpactFunc(
                id=func_id,
                name=name,
                haz_type=haz_type,
                intensity_unit=intensity_unit,
                intensity=intensity,
                mdd=mdd,
                paa=paa,
            )
        )

    impact_funcs = ImpactFuncSet(funcs)
    impact_funcs.check()
    return impact_funcs


def _load_climada_impact_functions(
    ImpactFuncSet: Any,
    ImpactFunc: Any,
    impact_functions_path: str,
    *,
    assumptions: Mapping[str, Any] | None,
    warnings: list[str],
) -> tuple[Any, str]:
    assumptions = dict(assumptions or {})
    source = (
        assumptions.get("impact_function_source")
        or assumptions.get("impact_functions_source")
        or assumptions.get("impact_functions_builtin")
        or impact_functions_path
    )
    source_text = str(source or "").strip()
    if not source_text:
        raise ValueError(
            "No CLIMADA impact-function source was provided. Pass an Excel/MAT/JSON file, "
            "an HDF5 file if your CLIMADA version supports ImpactFuncSet.from_hdf5, or "
            "impact_functions_path='builtin:tropical_cyclone_emanuel_usa'."
        )

    if source_text.lower().startswith("builtin:") or source_text.lower() in {
        "tc",
        "tropical_cyclone",
        "tropical_cyclone_emanuel_usa",
        "tc_emanuel_usa",
        "emanuel_usa",
    }:
        return _load_climada_builtin_impact_functions(ImpactFuncSet, builtin_name=source_text)

    path = Path(source_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"CLIMADA impact-function source was not found: {path}. "
            "Use an existing Excel/MAT/JSON file or pass 'builtin:tropical_cyclone_emanuel_usa'."
        )

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return ImpactFuncSet.from_excel(path), f"python:climada.entity.ImpactFuncSet.from_excel:{path}"
    if suffix == ".mat":
        return ImpactFuncSet.from_mat(path), f"python:climada.entity.ImpactFuncSet.from_mat:{path}"
    if suffix == ".json":
        impact_funcs = _impact_func_set_from_json(
            ImpactFuncSet,
            ImpactFunc,
            _read_json(path),
            assumptions=assumptions,
            warnings=warnings,
        )
        return impact_funcs, f"json:{path}"
    if suffix in {".h5", ".hdf5", ".hdf"}:
        from_hdf5 = getattr(ImpactFuncSet, "from_hdf5", None)
        if from_hdf5 is None:
            raise ValueError(
                "This CLIMADA version does not provide ImpactFuncSet.from_hdf5. "
                "Use an Excel/MAT/JSON impact-function file or "
                "impact_functions_path='builtin:tropical_cyclone_emanuel_usa'."
            )
        return from_hdf5(path), f"python:climada.entity.ImpactFuncSet.from_hdf5:{path}"

    raise ValueError(
        f"Unsupported CLIMADA impact-function file extension {suffix!r}. "
        "Supported inputs: .xlsx, .xls, .mat, .json, .h5/.hdf5 when supported by CLIMADA, "
        "or builtin:tropical_cyclone_emanuel_usa."
    )


def _configure_climada_projection_runtime(warnings: list[str]) -> None:
    existing_proj_dir = os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    candidates: list[Path] = []
    if existing_proj_dir:
        candidates.append(Path(existing_proj_dir).expanduser())
    if conda_prefix:
        candidates.append(Path(conda_prefix).expanduser() / "share" / "proj")
    for proj_dir in candidates:
        if (proj_dir / "proj.db").exists():
            os.environ.setdefault("PROJ_LIB", str(proj_dir))
            os.environ.setdefault("PROJ_DATA", str(proj_dir))
            try:
                import warnings as py_warnings

                with py_warnings.catch_warnings():
                    py_warnings.filterwarnings("ignore", message="pyproj unable to set PROJ database path.*")
                    from pyproj import datadir

                datadir.set_data_dir(str(proj_dir))
            except Exception as exc:
                warnings.append(f"Unable to set pyproj data directory to {proj_dir}: {exc}")
            return


def _resolve_climada_demo_path(source: str, *, kind: str) -> Path | None:
    normalized = source.strip().lower().replace("-", "_").replace(" ", "_")
    hazard_aliases = {
        "demo:hazard",
        "demo:tropical_cyclone",
        "demo:tc",
        "demo:tc_fl_1990_2004",
    }
    exposure_aliases = {
        "demo:exposure",
        "demo:exposures",
        "demo:exp_demo_today",
        "demo:today",
    }
    if kind == "hazard" and normalized in hazard_aliases:
        from climada.util.constants import HAZ_DEMO_H5

        return Path(HAZ_DEMO_H5)
    if kind == "exposure" and normalized in exposure_aliases:
        from climada.util.constants import EXP_DEMO_H5

        return Path(EXP_DEMO_H5)
    return None


def _load_climada_hazard(Hazard: Any, hazard_event_set_path: str) -> tuple[Any, str]:
    demo_path = _resolve_climada_demo_path(hazard_event_set_path, kind="hazard")
    if demo_path is not None:
        return Hazard.from_hdf5(demo_path), f"python:climada.util.constants.HAZ_DEMO_H5:{demo_path}"
    path = Path(hazard_event_set_path).expanduser()
    return Hazard.from_hdf5(path), f"python:climada.hazard.Hazard.from_hdf5:{path}"


def _load_climada_exposures(Exposures: Any, exposure_path: str) -> tuple[Any, str]:
    demo_path = _resolve_climada_demo_path(exposure_path, kind="exposure")
    if demo_path is not None:
        return Exposures.from_hdf5(demo_path), f"python:climada.util.constants.EXP_DEMO_H5:{demo_path}"
    path = Path(exposure_path).expanduser()
    return Exposures.from_hdf5(path), f"python:climada.entity.Exposures.from_hdf5:{path}"


def run_climada_official(
    *,
    hazard_event_set_path: str,
    exposure_path: str,
    impact_functions_path: str,
    run_id: str,
    scenario_name: str,
    output_dir: str | Path,
    start_ts: float,
    run_context: Mapping[str, Any],
    assumptions: Mapping[str, Any] | None,
    warnings: list[str],
) -> Any:
    run_dir = _official_run_dir("impact_climada_run", run_id, scenario_name, output_dir)
    home = Path(os.environ.get("HOME", "")).expanduser()
    if not home.exists() or not os.access(home, os.W_OK):
        fallback_home = run_dir / "_home"
        fallback_home.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(fallback_home)
        os.environ.setdefault("MPLCONFIGDIR", str(fallback_home / ".config" / "matplotlib"))
        warnings.append(f"HOME was not writable; using {fallback_home} for CLIMADA runtime files.")
    _configure_climada_projection_runtime(warnings)

    try:
        from climada.engine import ImpactCalc
        from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
        from climada.hazard import Hazard
    except Exception as exc:
        return _error(
            "impact_climada_run",
            f"Official CLIMADA backend requested, but the climada Python package is unavailable: {exc}",
            start_ts=start_ts,
            run_context=run_context,
            warnings=warnings,
        )

    try:
        hazard, hazard_provenance = _load_climada_hazard(Hazard, hazard_event_set_path)
        exposures, exposure_provenance = _load_climada_exposures(Exposures, exposure_path)
        impact_funcs, impact_func_provenance = _load_climada_impact_functions(
            ImpactFuncSet,
            ImpactFunc,
            impact_functions_path,
            assumptions=assumptions,
            warnings=warnings,
        )
        impact = ImpactCalc(exposures, impact_funcs, hazard).impact(save_mat=True)
        event_csv_path = run_dir / "climada_impact_at_event.csv"
        risk_curve_path = run_dir / "climada_risk_curve.json"
        at_event = getattr(impact, "at_event", None)
        event_values = list(at_event) if at_event is not None else []
        rows = [
            {"event_index": idx, "event_impact": float(value)}
            for idx, value in enumerate(event_values)
        ]
        _write_csv(event_csv_path, rows)
        total_impact = float(getattr(impact, "aai_agg", 0.0))
        worst_event = max((row["event_impact"] for row in rows), default=0.0)
        _write_json(risk_curve_path, {"events": len(rows), "aai_agg": total_impact, "worst_event_impact": worst_event})
        outputs = _collect_artifacts(run_dir)
        metrics = {
            "total_impact": _metric(total_impact, "exposure_unit_per_year"),
            "expected_event_impact": _metric(total_impact, "exposure_unit_per_year"),
            "worst_event_impact": _metric(worst_event, "exposure_unit"),
        }
        return _success(
            tool_name="impact_climada_run",
            start_ts=start_ts,
            metrics=metrics,
            derived_metrics={"events_count": _metric(float(len(rows)), "count")},
            output_artifacts=outputs,
            run_context=run_context,
            warnings=warnings,
            agentic_summary={
                "headline": f"Official CLIMADA annual average impact is {total_impact:.2f}.",
                "primary_signal": {"metric": "total_impact", **metrics["total_impact"]},
                "risk_level": "unknown",
                "comparison_ready": True,
                "recommended_next_tools": [],
            },
            provenance_command=(
                "python:climada.ImpactCalc; "
                f"hazard={hazard_provenance}; exposure={exposure_provenance}; "
                f"impact_functions={impact_func_provenance}"
            ),
        )
    except Exception as exc:
        return _error(
            "impact_climada_run",
            "Official CLIMADA backend requires CLIMADA-compatible HDF5 hazard/exposure files and "
            "Excel, MAT, JSON, HDF5, or built-in CLIMADA impact functions. "
            f"Failed with: {exc}",
            start_ts=start_ts,
            run_context=run_context,
            warnings=warnings,
        )


def _parse_energyplus_metrics(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    derived: dict[str, Any] = {}
    csv_path = run_dir / "eplusout.csv"
    if not csv_path.exists():
        warnings.append("EnergyPlus completed, but eplusout.csv was not found; add Output:Variable/Output:Meter requests to the IDF.")
        return metrics, derived, warnings
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        warnings.append("EnergyPlus eplusout.csv was empty.")
        return metrics, derived, warnings
    electricity_cols = [name for name in rows[0] if "electric" in name.lower()]
    demand_cols = [name for name in electricity_cols if "demand" in name.lower() or "power" in name.lower() or "[w]" in name.lower()]
    energy_cols = [
        name
        for name in electricity_cols
        if "energy" in name.lower() or "[j]" in name.lower() or "[kj]" in name.lower() or "[wh]" in name.lower()
    ]
    facility_energy_cols = [name for name in energy_cols if "facility" in name.lower()]
    if facility_energy_cols:
        energy_cols = facility_energy_cols
    if energy_cols:
        total_raw = sum(sum(_as_float(row.get(col), 0.0) for row in rows) for col in energy_cols)
        unit = "raw_energy_output"
        scale = 1.0
        if any("[j]" in col.lower() for col in energy_cols):
            unit = "kWh"
            scale = 1.0 / 3_600_000.0
        metrics["total_kwh"] = _metric(total_raw * scale, unit)
    if demand_cols:
        peak_raw = max(max(_as_float(row.get(col), 0.0) for row in rows) for col in demand_cols)
        scale = 1.0 / 1000.0 if any("[w]" in col.lower() for col in demand_cols) else 1.0
        metrics["peak_kw"] = _metric(peak_raw * scale, "kW")
    derived["energyplus_output_rows"] = _metric(float(len(rows)), "count")
    if not metrics:
        warnings.append("No standard electricity energy/demand columns were detected in eplusout.csv.")
    return metrics, derived, warnings


def run_energyplus_official(
    *,
    building_idf_path: str,
    weather_epw_path: str,
    run_id: str,
    scenario_name: str,
    output_dir: str | Path,
    start_ts: float,
    run_context: Mapping[str, Any],
    timeout_seconds: int,
    warnings: list[str],
) -> Any:
    exe = _resolve_executable(None, "ENERGYPLUS_EXE", "energyplus")
    if exe is None:
        return _error(
            "impact_energyplus_run",
            "Official EnergyPlus backend requested, but EnergyPlus is not installed or ENERGYPLUS_EXE is unset.",
            start_ts=start_ts,
            run_context=run_context,
            warnings=warnings,
        )
    run_dir = _official_run_dir("impact_energyplus_run", run_id, scenario_name, output_dir)
    command = [exe, "-r", "-w", str(Path(weather_epw_path).expanduser()), "-d", str(run_dir), str(Path(building_idf_path).expanduser())]
    result = _run_command(command, cwd=None, timeout_seconds=timeout_seconds)
    log_path = _write_command_log(run_dir, result, command=command)
    if result.returncode != 0:
        return _failed_command_response(
            "impact_energyplus_run",
            message=f"EnergyPlus failed with exit code {result.returncode}. See {log_path}.",
            start_ts=start_ts,
            run_context=run_context,
            warnings=warnings,
            run_dir=run_dir,
            result=result,
            log_path=log_path,
        )
    metrics, derived, parse_warnings = _parse_energyplus_metrics(run_dir)
    warnings.extend(parse_warnings)
    outputs = _collect_artifacts(run_dir)
    return _success(
        tool_name="impact_energyplus_run",
        start_ts=start_ts,
        metrics=metrics,
        derived_metrics=derived,
        output_artifacts=outputs,
        run_context=run_context,
        warnings=warnings,
        agentic_summary={
            "headline": "Official EnergyPlus backend completed.",
            "primary_signal": {"metric": next(iter(metrics), "official_artifact_count"), "value": float(len(outputs)), "unit": "count"},
            "risk_level": "unknown",
            "comparison_ready": bool(metrics),
            "recommended_next_tools": [],
        },
        provenance_command=" ".join(shlex.quote(part) for part in command),
    )


def _sumo_lane_speed_map(network_path: str | Path) -> dict[str, tuple[list[str], float]]:
    tree = ET.parse(Path(network_path).expanduser())
    root = tree.getroot()
    result: dict[str, tuple[list[str], float]] = {}
    for edge in root.findall(".//edge"):
        edge_id = edge.get("id")
        if not edge_id or edge.get("function") == "internal":
            continue
        lanes: list[str] = []
        speeds: list[float] = []
        for lane in edge.findall("lane"):
            lane_id = lane.get("id")
            if lane_id:
                lanes.append(lane_id)
                speeds.append(max(0.1, _as_float(lane.get("speed"), 13.9)))
        if lanes:
            result[edge_id] = (lanes, sum(speeds) / len(speeds))
    return result


def _write_sumo_additional(network_path: str, disruptions_path: str, run_dir: Path) -> Path | None:
    disruptions = _read_json(disruptions_path)
    if not isinstance(disruptions, dict):
        return None
    lane_map = _sumo_lane_speed_map(network_path)
    speed_by_edge: dict[str, float] = {}
    for edge in disruptions.get("edge_closures", []):
        speed_by_edge[str(edge)] = 0.1
    for row in disruptions.get("speed_reductions", []):
        if not isinstance(row, dict):
            continue
        edge_id = str(row.get("edge_id") or "")
        if not edge_id or edge_id not in lane_map:
            continue
        _, base_speed = lane_map[edge_id]
        speed_by_edge[edge_id] = min(speed_by_edge.get(edge_id, base_speed), base_speed * _clamp(_as_float(row.get("speed_factor"), 1.0), 0.05, 1.0))
    if not speed_by_edge:
        return None
    additional_path = run_dir / "official_disruptions.add.xml"
    root = ET.Element("additional")
    for edge_id, speed in sorted(speed_by_edge.items()):
        lanes, _ = lane_map.get(edge_id, ([], 13.9))
        if not lanes:
            continue
        vss = ET.SubElement(root, "variableSpeedSign", id=f"vss_{_safe_slug(edge_id)}", lanes=" ".join(lanes))
        ET.SubElement(vss, "step", time="0", speed=f"{max(0.1, speed):.4f}")
        ET.SubElement(vss, "step", time="864000", speed=f"{max(0.1, speed):.4f}")
    ET.ElementTree(root).write(additional_path, encoding="utf-8", xml_declaration=True)
    return additional_path


def _parse_sumo_tripinfo(tripinfo_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not tripinfo_path.exists():
        return {}, {}
    root = ET.parse(tripinfo_path).getroot()
    durations: list[float] = []
    time_losses: list[float] = []
    for elem in root.findall(".//tripinfo"):
        durations.append(_as_float(elem.get("duration"), 0.0))
        time_losses.append(_as_float(elem.get("timeLoss"), 0.0))
    metrics = {
        "avg_travel_time_s": _metric(sum(durations) / max(1, len(durations)), "s"),
        "p95_travel_time_s": _metric(_percentile(durations, 0.95), "s"),
        "total_delay_s": _metric(sum(time_losses), "s"),
        "throughput_vehicles": _metric(float(len(durations)), "vehicles"),
    }
    derived = {"completion_ratio": _metric(1.0 if durations else 0.0, "ratio")}
    return metrics, derived


def run_sumo_official(
    *,
    network_path: str,
    routes_path: str,
    disruptions_path: str,
    run_id: str,
    scenario_name: str,
    output_dir: str | Path,
    start_ts: float,
    run_context: Mapping[str, Any],
    timeout_seconds: int,
    warnings: list[str],
) -> Any:
    exe = _resolve_executable(None, "SUMO_BINARY", "sumo")
    if exe is None:
        return _error(
            "impact_sumo_run",
            "Official SUMO backend requested, but SUMO is not installed or SUMO_BINARY is unset.",
            start_ts=start_ts,
            run_context=run_context,
            warnings=warnings,
        )
    run_dir = _official_run_dir("impact_sumo_run", run_id, scenario_name, output_dir)
    tripinfo_path = run_dir / "tripinfo.xml"
    summary_path = run_dir / "summary.xml"
    add_path = _write_sumo_additional(network_path, disruptions_path, run_dir)
    command = [
        exe,
        "-n",
        str(Path(network_path).expanduser()),
        "-r",
        str(Path(routes_path).expanduser()),
        "--tripinfo-output",
        str(tripinfo_path),
        "--summary-output",
        str(summary_path),
        "--duration-log.statistics",
        "true",
        "--no-step-log",
        "true",
    ]
    if add_path is not None:
        command.extend(["-a", str(add_path)])
    result = _run_command(command, cwd=None, timeout_seconds=timeout_seconds)
    log_path = _write_command_log(run_dir, result, command=command)
    if result.returncode != 0:
        return _failed_command_response(
            "impact_sumo_run",
            message=f"SUMO failed with exit code {result.returncode}. See {log_path}.",
            start_ts=start_ts,
            run_context=run_context,
            warnings=warnings,
            run_dir=run_dir,
            result=result,
            log_path=log_path,
        )
    metrics, derived = _parse_sumo_tripinfo(tripinfo_path)
    if not metrics:
        warnings.append("SUMO completed, but tripinfo.xml was missing or empty.")
    outputs = _collect_artifacts(run_dir)
    return _success(
        tool_name="impact_sumo_run",
        start_ts=start_ts,
        metrics=metrics,
        derived_metrics=derived,
        output_artifacts=outputs,
        run_context=run_context,
        warnings=warnings,
        agentic_summary={
            "headline": "Official SUMO backend completed.",
            "primary_signal": {"metric": "throughput_vehicles", **metrics.get("throughput_vehicles", _metric(float(len(outputs)), "count"))},
            "risk_level": "unknown",
            "comparison_ready": bool(metrics),
            "recommended_next_tools": [],
        },
        provenance_command=" ".join(shlex.quote(part) for part in command),
    )


def _extract_utci_value(result: Any) -> float:
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, dict):
        for key in ("utci", "utci_c", "value"):
            if key in result:
                return _as_float(result[key])
    for attr in ("utci", "utci_c", "value"):
        if hasattr(result, attr):
            return _as_float(getattr(result, attr))
    return _as_float(result)


def run_utci_official(
    *,
    met_timeseries_path: str,
    run_id: str,
    scenario_name: str,
    output_dir: str | Path,
    start_ts: float,
    run_context: Mapping[str, Any],
    strong_stress_threshold_c: float,
    warnings: list[str],
) -> Any:
    run_dir = _official_run_dir("impact_health_utci", run_id, scenario_name, output_dir)
    runtime_dir = run_dir / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    home = Path(os.environ.get("HOME", "")).expanduser()
    if not home.exists() or not os.access(home, os.W_OK):
        fallback_home = runtime_dir / "home"
        fallback_home.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(fallback_home)
        os.environ.setdefault("MPLCONFIGDIR", str(fallback_home / ".config" / "matplotlib"))
        warnings.append(f"HOME was not writable; using {fallback_home} for UTCI runtime files.")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(runtime_dir / "numba_cache"))

    try:
        from pythermalcomfort.models import utci as pythermalcomfort_utci
    except Exception as exc:
        return _error(
            "impact_health_utci",
            "Official/validated UTCI backend requested, but pythermalcomfort is unavailable. "
            f"Install pythermalcomfort or use backend='proxy'. Import error: {exc}",
            start_ts=start_ts,
            run_context=run_context,
            warnings=warnings,
        )
    rows = _load_rows(met_timeseries_path)
    if not rows:
        return _error("impact_health_utci", "Meteorological input is empty.", start_ts=start_ts, run_context=run_context, warnings=warnings)
    out_rows: list[dict[str, Any]] = []
    values: list[float] = []
    categories: dict[str, int] = {}
    for idx, row in enumerate(rows):
        timestamp = _parse_datetime_from_row(row)
        tair = _as_float(row.get("tair_c") or row.get("temperature_c") or row.get("t2m_c"), 25.0)
        rh = _clamp(_as_float(row.get("rh_pct") or row.get("humidity_pct"), 55.0), 1.0, 100.0)
        wind = max(0.0, _as_float(row.get("wind_m_s") or row.get("wind"), 1.5))
        tmrt = _as_float(row.get("tmrt_c"), tair + 1.8)
        value = _extract_utci_value(pythermalcomfort_utci(tdb=tair, tr=tmrt, v=wind, rh=rh))
        values.append(value)
        category = _utci_category(value)
        categories[category] = categories.get(category, 0) + 1
        out_rows.append(
            {
                "time": timestamp.isoformat() if timestamp else str(idx),
                "tair_c": round(tair, 4),
                "rh_pct": round(rh, 4),
                "wind_m_s": round(wind, 4),
                "tmrt_c": round(tmrt, 4),
                "utci_c": round(value, 4),
                "category": category,
            }
        )
    timeseries_path = run_dir / "utci_timeseries.csv"
    summary_path = run_dir / "utci_summary.json"
    _write_csv(timeseries_path, out_rows)
    strong_hours = sum(1 for value in values if value >= strong_stress_threshold_c)
    _write_json(summary_path, {"utci_mean_c": sum(values) / len(values), "utci_p95_c": _percentile(values, 0.95), "hours_above_strong_stress": strong_hours, "category_counts": categories})
    outputs = _collect_artifacts(run_dir)
    metrics = {
        "utci_mean_c": _metric(sum(values) / len(values), "degC"),
        "utci_p95_c": _metric(_percentile(values, 0.95), "degC"),
        "hours_above_strong_stress": _metric(float(strong_hours), "hours"),
    }
    return _success(
        tool_name="impact_health_utci",
        start_ts=start_ts,
        metrics=metrics,
        derived_metrics={"strong_stress_share": _metric(strong_hours / max(1, len(values)), "ratio")},
        output_artifacts=outputs,
        run_context=run_context,
        warnings=warnings,
        agentic_summary={
            "headline": f"Official UTCI calculation found {strong_hours} strong-stress hours.",
            "primary_signal": {"metric": "hours_above_strong_stress", **metrics["hours_above_strong_stress"]},
            "risk_level": "unknown",
            "comparison_ready": True,
            "recommended_next_tools": [],
        },
        extra={"category_counts": categories},
        provenance_command="python:pythermalcomfort.models.utci",
    )


def _utci_category(value: float) -> str:
    if value >= 46.0:
        return "extreme_heat"
    if value >= 38.0:
        return "very_strong_heat"
    if value >= 32.0:
        return "strong_heat"
    if value >= 26.0:
        return "moderate_heat"
    if value >= 9.0:
        return "no_thermal_stress"
    if value >= 0.0:
        return "slight_cold"
    if value >= -13.0:
        return "moderate_cold"
    return "strong_cold"


def run_benmap_official(
    *,
    exposure_series_path: str,
    population_exposed_path: str,
    erf_spec_path: str,
    run_id: str,
    scenario_name: str,
    output_dir: str | Path,
    start_ts: float,
    run_context: Mapping[str, Any],
    timeout_seconds: int,
    assumptions: Mapping[str, Any] | None,
    warnings: list[str],
) -> Any:
    assumptions = dict(assumptions or {})
    return _generic_command_backend(
        tool_name="impact_health_erf",
        command_env="BENMAP_COMMAND",
        default_executable="benmap",
        explicit_command=assumptions.get("official_command"),
        run_id=run_id,
        scenario_name=scenario_name,
        output_dir=output_dir,
        start_ts=start_ts,
        run_context=run_context,
        timeout_seconds=timeout_seconds,
        assumptions=assumptions,
        input_env={
            "BENMAP_EXPOSURE_SERIES_PATH": exposure_series_path,
            "BENMAP_POPULATION_EXPOSED_PATH": population_exposed_path,
            "BENMAP_ERF_SPEC_PATH": erf_spec_path,
        },
        warnings=warnings,
    )
