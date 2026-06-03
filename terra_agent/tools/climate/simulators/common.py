
"""
Shared deterministic utilities for simulator and glue tool modules.
"""
from __future__ import annotations

import calendar
import csv
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

DEFAULT_OUTPUT_DIR = Path("./data/simulators")

_CONTENT_TYPE_BY_SUFFIX = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".geojson": "application/geo+json",
    ".nc": "application/x-netcdf",
    ".epw": "text/plain",
    ".xml": "application/xml",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".sql": "application/sql",
    ".gz": "application/gzip",
    ".tar": "application/x-tar",
}


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _ok(payload: Mapping[str, Any]) -> ToolResponse:
    sanitized = _sanitize(dict(payload))
    return ToolResponse(
        content=[TextBlock(type="text", text=json.dumps(sanitized, ensure_ascii=False))],
        metadata=sanitized,
    )


def _error(
    tool_name: str,
    message: str,
    *,
    start_ts: float | None = None,
    run_context: Mapping[str, Any] | None = None,
    warnings: Sequence[str] | None = None,
    output_artifacts: Sequence[dict[str, Any]] | None = None,
    provenance_errors: Sequence[str] | None = None,
    provenance_command: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> ToolResponse:
    elapsed = max(0.0, time.perf_counter() - start_ts) if start_ts is not None else 0.0
    errors = [message]
    errors.extend(str(item) for item in provenance_errors or [] if str(item) != message)
    payload = {
        "status": "error",
        "metrics": {},
        "derived_metrics": {},
        "outputs_artifacts": list(output_artifacts or []),
        "provenance": {
            "command": provenance_command or f"inprocess:{tool_name}",
            "runtime_seconds": elapsed,
            "warnings": list(warnings or []),
            "errors": errors,
        },
        "run_context": dict(run_context or {}),
        "error": True,
        "message": message,
    }
    if extra is not None:
        payload.update(dict(extra))
    return _ok(payload)


def _metric(value: float, unit: str) -> dict[str, Any]:
    return {"value": float(value), "unit": unit}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _safe_slug(value: str) -> str:
    text = (value or "default").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "default"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guess_content_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tar.gz"):
        return "application/gzip"
    return _CONTENT_TYPE_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")


def _artifact_record(path: str | Path, description: str) -> dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    return {
        "path": str(artifact_path),
        "sha256": _sha256_file(artifact_path),
        "content_type": _guess_content_type(artifact_path),
        "description": description,
    }


def _collect_input_artifacts(paths: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path_str, description in paths:
        path = Path(path_str).expanduser()
        if not path.exists() or not path.is_file():
            continue
        artifacts.append(_artifact_record(path, description))
    return artifacts


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_output_path(path: str | Path, *, fallback_dir: Path, filename: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        _ensure_dir(candidate)
        return candidate / filename
    if candidate.suffix:
        _ensure_dir(candidate.parent)
        return candidate
    _ensure_dir(candidate)
    return candidate / filename


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path).expanduser()
    _ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize(dict(payload)), handle, ensure_ascii=False, indent=2)
    return out


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    out = Path(path).expanduser()
    _ensure_dir(out.parent)
    if not rows:
        fieldnames = fieldnames or []
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
        return out

    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(str(key))
        fieldnames = keys

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return out


def _read_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _parse_date(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y%m%d", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt in {"%Y-%m", "%Y/%m"}:
                return parsed.replace(day=1)
            return parsed
        except ValueError:
            continue
    return None


def _parse_datetime_from_row(row: Mapping[str, Any]) -> datetime | None:
    for key in ("datetime", "timestamp", "time", "date", "valid_time"):
        if key in row:
            parsed = _parse_date(row.get(key))
            if parsed is not None:
                return parsed
    if "year" in row and "month" in row:
        year = int(_as_float(row.get("year"), 0))
        month = int(_as_float(row.get("month"), 1))
        day = int(_as_float(row.get("day"), 1))
        hour = int(_as_float(row.get("hour"), 0))
        minute = int(_as_float(row.get("minute"), 0))
        if year >= 1:
            return datetime(year=year, month=max(1, min(12, month)), day=max(1, min(28, day)), hour=hour, minute=minute)
    return None


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    q = _clamp(q, 0.0, 1.0)
    sorted_vals = sorted(float(v) for v in values)
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path).expanduser()
    suffix = file_path.suffix.lower()
    if suffix in {".json", ".geojson"}:
        payload = _read_json(file_path)
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            if isinstance(payload.get("rows"), list):
                return [dict(item) for item in payload["rows"] if isinstance(item, dict)]
            if isinstance(payload.get("events"), list):
                return [dict(item) for item in payload["events"] if isinstance(item, dict)]
            if isinstance(payload.get("features"), list):
                rows: list[dict[str, Any]] = []
                for feature in payload["features"]:
                    if not isinstance(feature, dict):
                        continue
                    item: dict[str, Any] = {}
                    props = feature.get("properties")
                    if isinstance(props, dict):
                        item.update(props)
                    geom = feature.get("geometry")
                    if geom is not None:
                        item["geometry"] = geom
                    rows.append(item)
                if rows:
                    return rows
            return [dict(payload)]
        raise ValueError(f"Unsupported JSON structure in {file_path}")
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _tool_run_dir(output_dir: str | Path, tool_name: str, run_id: str, scenario_name: str) -> Path:
    root = _ensure_dir(output_dir)
    run_dir = root / _safe_slug(tool_name) / _safe_slug(run_id) / _safe_slug(scenario_name or "baseline")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _risk_level(value: float, *, low: float, high: float, inverse: bool = False) -> str:
    if inverse:
        if value <= low:
            return "high"
        if value <= high:
            return "medium"
        return "low"
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


def _build_run_context(
    *,
    run_id: str,
    instance_id: str,
    scenario_name: str,
    scenario_description: str,
    start_iso: str | None,
    end_iso: str | None,
    timestep: str,
    assumptions: Mapping[str, Any] | None,
    seed: int,
    model_version: str,
    container_image: str | None,
    threads: int,
    cpu_cores: int,
    memory_gb: float,
    timeout_seconds: int,
    inputs: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "instance_id": instance_id,
        "scenario": {"name": scenario_name, "description": scenario_description or ""},
        "time": {"start_iso": start_iso, "end_iso": end_iso, "timestep": timestep},
        "assumptions": dict(assumptions or {}),
        "determinism": {
            "seed": int(seed),
            "model_version": model_version,
            "container_image": container_image,
            "threads": int(threads),
        },
        "resources": {
            "cpu_cores": int(cpu_cores),
            "memory_gb": float(memory_gb),
            "timeout_seconds": int(timeout_seconds),
        },
        "inputs_artifacts": _collect_input_artifacts(inputs),
    }


def _success(
    *,
    tool_name: str,
    start_ts: float,
    metrics: Mapping[str, Any],
    derived_metrics: Mapping[str, Any],
    output_artifacts: Sequence[dict[str, Any]],
    run_context: Mapping[str, Any],
    warnings: Sequence[str] | None = None,
    agentic_summary: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    provenance_command: str | None = None,
) -> ToolResponse:
    payload: dict[str, Any] = {
        "status": "success",
        "metrics": dict(metrics),
        "derived_metrics": dict(derived_metrics),
        "outputs_artifacts": list(output_artifacts),
        "provenance": {
            "command": provenance_command or f"inprocess:{tool_name}",
            "runtime_seconds": max(0.0, time.perf_counter() - start_ts),
            "warnings": list(warnings or []),
            "errors": [],
        },
        "run_context": dict(run_context),
    }
    if agentic_summary is not None:
        payload["agentic_summary"] = dict(agentic_summary)
    if extra is not None:
        payload.update(dict(extra))
    return _ok(payload)


def _read_weather_daily(path: str | Path) -> list[dict[str, Any]]:
    rows = _load_rows(path)
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        timestamp = _parse_datetime_from_row(row)
        if timestamp is None:
            raise ValueError(f"Unable to parse date for weather row {idx + 1} in {path}")
        precip = _as_float(row.get("precip_mm") or row.get("precip") or row.get("rain_mm"), 0.0)
        tmin = _as_float(row.get("tmin_c") or row.get("tmin") or row.get("tasmin"), 0.0)
        tmax = _as_float(row.get("tmax_c") or row.get("tmax") or row.get("tasmax"), tmin)
        et0 = _as_float(row.get("et0_mm") or row.get("eto_mm") or row.get("et_mm"), max(0.0, (tmax - tmin) * 0.22))
        normalized.append(
            {
                "date": timestamp.date().isoformat(),
                "timestamp": timestamp,
                "precip_mm": max(0.0, precip),
                "tmin_c": min(tmin, tmax),
                "tmax_c": max(tmin, tmax),
                "et0_mm": max(0.0, et0),
            }
        )
    normalized.sort(key=lambda item: item["timestamp"])
    return normalized


def _read_monthly_weather(path: str | Path) -> list[dict[str, Any]]:
    rows = _load_rows(path)
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        parsed = _parse_datetime_from_row(row)
        if parsed is None:
            raise ValueError(f"Unable to parse month for row {idx + 1} in {path}")
        year = parsed.year
        month = parsed.month
        precip = _as_float(row.get("precip_mm") or row.get("precip") or row.get("rain_mm"), 0.0)
        tmin = _as_float(row.get("tmin_c") or row.get("tmin"), 0.0)
        tmax = _as_float(row.get("tmax_c") or row.get("tmax"), tmin + 6.0)
        et0 = _as_float(row.get("et0_mm") or row.get("eto_mm") or row.get("et_mm"), max(0.0, (tmax - tmin) * 3.0))
        normalized.append(
            {
                "year": year,
                "month": month,
                "precip_mm": max(0.0, precip),
                "tmin_c": min(tmin, tmax),
                "tmax_c": max(tmin, tmax),
                "et0_mm": max(0.0, et0),
            }
        )
    normalized.sort(key=lambda item: (item["year"], item["month"]))
    return normalized


def _run_aquacrop_proxy(
    weather_rows: Sequence[Mapping[str, Any]],
    *,
    soil: Mapping[str, Any],
    crop: Mapping[str, Any],
    management: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not weather_rows:
        raise ValueError("Weather series is empty.")
    n_days = len(weather_rows)
    available_water_mm = _as_float(soil.get("available_water_mm"), 150.0)
    runoff_coeff = _clamp(_as_float(soil.get("runoff_coeff"), 0.08), 0.0, 0.45)
    kc = _clamp(_as_float(crop.get("kc"), 1.05), 0.6, 1.4)
    optimal_temp_c = _as_float(crop.get("optimal_temp_c"), 25.0)
    wp_biomass = _as_float(crop.get("water_productivity_kg_ha_per_mm"), 34.0)
    harvest_index = _clamp(_as_float(crop.get("harvest_index"), 0.45), 0.15, 0.7)
    irrigation_mm = max(
        0.0,
        _as_float(management.get("seasonal_irrigation_mm"), _as_float(management.get("irrigation_mm"), 0.0)),
    )
    irrigation_efficiency = _clamp(_as_float(management.get("irrigation_efficiency"), 0.85), 0.4, 1.0)
    daily_irrig = irrigation_mm / n_days

    soil_store = available_water_mm * 0.55
    biomass_kg_ha = 0.0
    seasonal_et_mm = 0.0
    stress_values: list[float] = []
    output_rows: list[dict[str, Any]] = []

    for row in weather_rows:
        tmean = (_as_float(row.get("tmin_c")) + _as_float(row.get("tmax_c"))) / 2.0
        et_demand = max(0.0, _as_float(row.get("et0_mm")) * kc)
        effective_rain = max(0.0, _as_float(row.get("precip_mm")) * (1.0 - runoff_coeff))
        effective_irrig = daily_irrig * irrigation_efficiency
        incoming = effective_rain + effective_irrig

        if incoming >= et_demand:
            actual_et = et_demand
            excess = incoming - et_demand
            soil_store = min(available_water_mm, soil_store + excess)
        else:
            deficit = et_demand - incoming
            soil_release = min(soil_store, deficit)
            soil_store -= soil_release
            actual_et = incoming + soil_release

        stress_idx = 0.0 if et_demand <= 1e-6 else _clamp(1.0 - (actual_et / et_demand), 0.0, 1.0)
        temp_penalty = _clamp(abs(tmean - optimal_temp_c) / 18.0, 0.0, 1.0)
        growth_factor = _clamp(1.0 - 0.72 * stress_idx - 0.30 * temp_penalty, 0.0, 1.0)
        biomass_kg_ha += actual_et * wp_biomass * growth_factor
        seasonal_et_mm += actual_et
        stress_values.append(stress_idx)
        output_rows.append(
            {
                "date": row["date"],
                "tmean_c": round(tmean, 3),
                "water_demand_mm": round(et_demand, 3),
                "water_supply_mm": round(actual_et, 3),
                "stress_index": round(stress_idx, 5),
                "biomass_kg_ha": round(biomass_kg_ha, 4),
            }
        )

    biomass_t_ha = biomass_kg_ha / 1000.0
    yield_t_ha = biomass_t_ha * harvest_index
    stress_mean = statistics.fmean(stress_values) if stress_values else 0.0
    return (
        {
            "yield_t_ha": yield_t_ha,
            "biomass_t_ha": biomass_t_ha,
            "seasonal_et_mm": seasonal_et_mm,
            "seasonal_irrigation_mm": irrigation_mm,
            "water_stress_index": stress_mean,
        },
        output_rows,
    )


def _run_dssat_proxy(
    weather_rows: Sequence[Mapping[str, Any]],
    *,
    cultivar: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not weather_rows:
        raise ValueError("Weather series is empty.")
    n_days = len(weather_rows)
    base_temp = _as_float(cultivar.get("base_temp_c"), 8.0)
    optimum_gdd = max(200.0, _as_float(cultivar.get("optimum_gdd"), 1450.0))
    potential_yield = max(100.0, _as_float(cultivar.get("potential_yield_kg_ha"), 7200.0))
    kc = _clamp(_as_float(cultivar.get("kc"), 1.0), 0.5, 1.5)
    fert_factor = _clamp(_as_float(treatment.get("fertility_factor"), 1.0), 0.35, 1.25)
    irrigation_mm = max(
        0.0,
        _as_float(treatment.get("seasonal_irrigation_mm"), _as_float(treatment.get("irrigation_mm"), 0.0)),
    )
    daily_irrig = irrigation_mm / n_days

    gdd_total = 0.0
    seasonal_et = 0.0
    stress_days = 0
    output_rows: list[dict[str, Any]] = []
    for row in weather_rows:
        tmean = (_as_float(row.get("tmin_c")) + _as_float(row.get("tmax_c"))) / 2.0
        gdd = max(0.0, tmean - base_temp)
        gdd_total += gdd
        water_supply = max(0.0, _as_float(row.get("precip_mm"))) + daily_irrig
        et = max(0.0, _as_float(row.get("et0_mm")) * kc)
        seasonal_et += min(water_supply, et)
        stress = water_supply < et
        if stress:
            stress_days += 1
        output_rows.append(
            {
                "date": row["date"],
                "gdd": round(gdd, 4),
                "water_supply_mm": round(water_supply, 4),
                "water_demand_mm": round(et, 4),
                "stress_day": int(stress),
            }
        )

    gdd_factor = _clamp(gdd_total / optimum_gdd, 0.45, 1.25)
    water_penalty = _clamp(1.0 - 0.65 * (stress_days / max(1, n_days)), 0.15, 1.0)
    yield_kg_ha = potential_yield * gdd_factor * fert_factor * water_penalty
    emergence_offset = max(5, int(0.06 * n_days))
    flowering_offset = max(emergence_offset + 15, int(0.45 * n_days))
    maturity_offset = max(flowering_offset + 20, int(0.92 * n_days))
    first_day = _parse_date(weather_rows[0].get("date")) or datetime.utcnow()
    phenology = {
        "emergence_date": (first_day + timedelta(days=emergence_offset)).date().isoformat(),
        "flowering_date": (first_day + timedelta(days=flowering_offset)).date().isoformat(),
        "maturity_date": (first_day + timedelta(days=maturity_offset)).date().isoformat(),
    }
    return (
        {
            "yield_kg_ha": yield_kg_ha,
            "water_stress_days": float(stress_days),
            "seasonal_et_mm": seasonal_et,
            "gdd_total": gdd_total,
            "seasonal_irrigation_mm": irrigation_mm,
            "phenology": phenology,
        },
        output_rows,
    )


def _load_population(population_exposed_path: str) -> float:
    payload = _read_json(population_exposed_path) if population_exposed_path.lower().endswith(".json") else _load_rows(population_exposed_path)
    if isinstance(payload, dict):
        if "population" in payload:
            return _as_float(payload["population"], 0.0)
        if "total_population" in payload:
            return _as_float(payload["total_population"], 0.0)
        if "rows" in payload and isinstance(payload["rows"], list):
            return sum(_as_float(row.get("population"), 0.0) for row in payload["rows"] if isinstance(row, dict))
        return 0.0
    if isinstance(payload, list):
        return sum(_as_float(row.get("population") or row.get("value") or row.get("count"), 0.0) for row in payload if isinstance(row, dict))
    return 0.0


def _read_epw_weather(path: str | Path) -> list[dict[str, Any]]:
    epw_path = Path(path).expanduser()
    lines = epw_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 9:
        raise ValueError(f"EPW file {epw_path} is too short.")
    weather_lines = lines[8:]
    rows: list[dict[str, Any]] = []
    for line in weather_lines:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 22:
            continue
        year = int(_as_float(parts[0], 2000))
        month = int(_as_float(parts[1], 1))
        day = int(_as_float(parts[2], 1))
        hour = max(1, int(_as_float(parts[3], 1)))
        minute = int(_as_float(parts[4], 60))
        dry_bulb = _as_float(parts[6], 20.0)
        rh = _as_float(parts[8], 55.0)
        wind_speed = _as_float(parts[21], 1.5)
        timestamp = datetime(year, max(1, min(12, month)), max(1, min(28, day)), hour=hour - 1, minute=0 if minute == 60 else minute)
        rows.append(
            {
                "time": timestamp,
                "dry_bulb_c": dry_bulb,
                "rh_pct": _clamp(rh, 1.0, 100.0),
                "wind_m_s": max(0.0, wind_speed),
            }
        )
    return rows


def _infer_floor_area_m2(idf_path: str | Path, default_floor_area_m2: float) -> float:
    content = Path(idf_path).expanduser().read_text(encoding="utf-8", errors="ignore")
    patterns = [
        r"(?i)floor\s*area[^0-9\\-]*([0-9]+(?:\\.[0-9]+)?)",
        r"(?i)zone\s*floor\s*area[^0-9\\-]*([0-9]+(?:\\.[0-9]+)?)",
        r"(?i)area[^0-9\\-]*([0-9]+(?:\\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            value = _as_float(match.group(1), default_floor_area_m2)
            if value > 0:
                return value
    return default_floor_area_m2


def _parse_sumo_network(network_path: str | Path) -> dict[str, dict[str, float]]:
    tree = ET.parse(Path(network_path).expanduser())
    root = tree.getroot()
    network: dict[str, dict[str, float]] = {}
    for edge in root.findall(".//edge"):
        edge_id = edge.get("id")
        if not edge_id:
            continue
        if edge.get("function") == "internal":
            continue
        lane_lengths: list[float] = []
        lane_speeds: list[float] = []
        for lane in edge.findall("lane"):
            lane_lengths.append(max(1.0, _as_float(lane.get("length"), 100.0)))
            lane_speeds.append(max(0.5, _as_float(lane.get("speed"), 13.9)))
        if lane_lengths and lane_speeds:
            length = statistics.fmean(lane_lengths)
            speed = statistics.fmean(lane_speeds)
        else:
            length = max(1.0, _as_float(edge.get("length"), 300.0))
            speed = max(0.5, _as_float(edge.get("speed"), 13.9))
        network[edge_id] = {"length_m": length, "speed_m_s": speed}
    return network


def _parse_sumo_routes(routes_path: str | Path) -> list[dict[str, Any]]:
    tree = ET.parse(Path(routes_path).expanduser())
    root = tree.getroot()
    route_defs: dict[str, list[str]] = {}
    for route in root.findall(".//route"):
        rid = route.get("id")
        edges_raw = route.get("edges", "")
        if rid and edges_raw:
            route_defs[rid] = [edge for edge in edges_raw.split() if edge]

    trips: list[dict[str, Any]] = []
    for idx, vehicle in enumerate(root.findall(".//vehicle")):
        vehicle_id = vehicle.get("id") or f"veh_{idx + 1}"
        route_elem = vehicle.find("route")
        if route_elem is not None and route_elem.get("edges"):
            edges = [edge for edge in route_elem.get("edges", "").split() if edge]
        else:
            route_ref = vehicle.get("route")
            edges = route_defs.get(route_ref, [])
        trips.append({"vehicle_id": vehicle_id, "edges": edges})

    for idx, trip in enumerate(root.findall(".//trip")):
        trip_id = trip.get("id") or f"trip_{idx + 1}"
        frm = trip.get("from")
        to = trip.get("to")
        edges: list[str] = []
        if frm:
            edges.append(frm)
        if to and to != frm:
            edges.append(to)
        trips.append({"vehicle_id": trip_id, "edges": edges})

    for idx, flow in enumerate(root.findall(".//flow")):
        flow_id = flow.get("id") or f"flow_{idx + 1}"
        number = int(_as_float(flow.get("number"), 1))
        number = max(1, min(number, 500))
        route_ref = flow.get("route")
        edges = route_defs.get(route_ref, [])
        if not edges:
            frm = flow.get("from")
            to = flow.get("to")
            if frm:
                edges = [frm]
                if to and to != frm:
                    edges.append(to)
        for i in range(number):
            trips.append({"vehicle_id": f"{flow_id}_{i + 1}", "edges": list(edges)})
    return trips




__all__ = [name for name in globals() if not name.startswith("__")]
