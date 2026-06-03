"""
AquaCrop-OSPy backend for the AquaCrop impact tool.

This backend is intentionally separate from the FAO stand-alone executable
backend. AquaCrop-OSPy is a Python implementation of AquaCrop-OS, not the
official FAO AquaCrop executable.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import (
    _artifact_record,
    _as_float,
    _clamp,
    _error,
    _load_rows,
    _metric,
    _read_json,
    _read_weather_daily,
    _risk_level,
    _success,
    _tool_run_dir,
    _write_json,
)


AQUACROP_BACKEND_ENV = "AQUACROP_BACKEND"
OSPY = "ospy"
OSPY_ALIASES = {"ospy", "aquacrop_ospy", "aquacrop-ospy", "aquacropos", "os-py"}


def wants_aquacrop_ospy_backend(backend: str | None) -> bool:
    requested = backend if backend is not None else os.getenv(AQUACROP_BACKEND_ENV)
    return str(requested or "").strip().lower() in OSPY_ALIASES


def _format_ospy_date(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("AquaCrop-OSPy date value is empty.")
    if "-" in text and len(text.split("-")[0]) == 4:
        year, month, day = text.split("-")[:3]
        return f"{int(year):04d}/{int(month):02d}/{int(day):02d}"
    if "/" in text and len(text.split("/")[0]) == 4:
        year, month, day = text.split("/")[:3]
        return f"{int(year):04d}/{int(month):02d}/{int(day):02d}"
    raise ValueError(f"AquaCrop-OSPy simulation dates must be YYYY-MM-DD or YYYY/MM/DD, got {text!r}.")


def _format_planting_date(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("AquaCrop-OSPy planting date is empty.")
    if "-" in text and len(text.split("-")[0]) == 4:
        _, month, day = text.split("-")[:3]
        return f"{int(month):02d}/{int(day):02d}"
    if "/" in text:
        parts = text.split("/")
        if len(parts[0]) == 4:
            _, month, day = parts[:3]
        else:
            month, day = parts[:2]
        return f"{int(month):02d}/{int(day):02d}"
    raise ValueError(f"AquaCrop-OSPy planting/harvest dates must be MM/DD or YYYY-MM-DD, got {text!r}.")


def _dataframe_to_csv(frame: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(frame, "to_csv"):
        frame.to_csv(path, index=False)
    else:
        path.write_text(str(frame), encoding="utf-8")
    return path


def _last_numeric(frame: Any, candidates: Sequence[str]) -> float | None:
    if frame is None or frame is False or not hasattr(frame, "columns"):
        return None
    columns = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        column = columns.get(candidate.strip().lower())
        if column is None:
            continue
        values = frame[column].dropna()
        if len(values) == 0:
            continue
        return _as_float(values.iloc[-1])
    return None


def _sum_numeric(frame: Any, candidates: Sequence[str]) -> float | None:
    if frame is None or frame is False or not hasattr(frame, "columns"):
        return None
    columns = {str(column).strip().lower(): column for column in frame.columns}
    total = 0.0
    found = False
    for candidate in candidates:
        column = columns.get(candidate.strip().lower())
        if column is None:
            continue
        total += float(frame[column].fillna(0.0).sum())
        found = True
    return total if found else None


def _max_numeric(frame: Any, candidates: Sequence[str]) -> float | None:
    if frame is None or frame is False or not hasattr(frame, "columns"):
        return None
    columns = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        column = columns.get(candidate.strip().lower())
        if column is None:
            continue
        values = frame[column].dropna()
        if len(values) == 0:
            continue
        return _as_float(values.max())
    return None


def _nested_mapping(payload: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            merged.update(dict(value))
    return merged


def _first_present(*lookups: tuple[Mapping[str, Any], str]) -> Any:
    for payload, key in lookups:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _build_weather_frame(weather_rows: Sequence[Mapping[str, Any]], pd: Any) -> Any:
    if not weather_rows:
        raise ValueError("AquaCrop-OSPy weather series is empty.")
    records = []
    for row in weather_rows:
        records.append(
            {
                "MinTemp": _as_float(row.get("tmin_c")),
                "MaxTemp": _as_float(row.get("tmax_c")),
                "Precipitation": max(0.0, _as_float(row.get("precip_mm"))),
                "ReferenceET": max(0.1, _as_float(row.get("et0_mm"))),
                "Date": row["date"],
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame["Date"] = pd.to_datetime(frame["Date"])
    # AquaCrop-OSPy uses the Date column for clipping, then stores
    # weather_df.values. Its timestep solver expects columns 0-3 to be
    # MinTemp, MaxTemp, Precipitation, and ReferenceET in that order.
    return frame[["MinTemp", "MaxTemp", "Precipitation", "ReferenceET", "Date"]]


def _build_irrigation_management(management: Mapping[str, Any], n_days: int, IrrigationManagement: Any, pd: Any) -> Any:
    irrigation_cfg = _nested_mapping(management, "irrigation", "ospy_irrigation", "aquacrop_ospy_irrigation")
    method_value = _first_present(
        (irrigation_cfg, "irrigation_method"),
        (irrigation_cfg, "IrrMethod"),
        (management, "irrigation_method"),
        (management, "IrrMethod"),
    )
    seasonal_irrigation = _as_float(
        irrigation_cfg.get("seasonal_irrigation_mm")
        or management.get("seasonal_irrigation_mm")
        or management.get("irrigation_mm"),
        0.0,
    )
    if method_value is None:
        method = 5 if seasonal_irrigation > 0 else 0
    else:
        method = int(method_value)

    kwargs: dict[str, Any] = {}
    key_map = {
        "WetSurf": ("WetSurf", "wet_surface_pct"),
        "AppEff": ("AppEff", "application_efficiency_pct"),
        "MaxIrr": ("MaxIrr", "max_irrigation_mm"),
        "MaxIrrSeason": ("MaxIrrSeason", "seasonal_irrigation_mm"),
        "SMT": ("SMT", "soil_moisture_targets"),
        "IrrInterval": ("IrrInterval", "irrigation_interval_days"),
        "NetIrrSMT": ("NetIrrSMT", "net_irrigation_smt"),
        "depth": ("depth", "daily_irrigation_mm"),
    }
    for target, aliases in key_map.items():
        for alias in aliases:
            if alias in irrigation_cfg:
                kwargs[target] = irrigation_cfg[alias]
                break
            if alias in management:
                kwargs[target] = management[alias]
                break

    efficiency = irrigation_cfg.get("irrigation_efficiency", management.get("irrigation_efficiency"))
    if efficiency is not None and "AppEff" not in kwargs:
        value = _as_float(efficiency)
        kwargs["AppEff"] = value * 100.0 if value <= 1.0 else value
    if seasonal_irrigation > 0:
        kwargs.setdefault("MaxIrrSeason", seasonal_irrigation)
        if method == 5:
            kwargs.setdefault("depth", seasonal_irrigation / max(1, n_days))

    schedule_path = irrigation_cfg.get("schedule_path") or management.get("irrigation_schedule_path")
    if schedule_path:
        rows = []
        for row in _load_rows(schedule_path):
            timestamp = row.get("date") or row.get("Date") or row.get("timestamp") or row.get("time")
            depth = (
                row.get("depth_mm")
                or row.get("Depth")
                or row.get("irrigation_mm")
                or row.get("amount_mm")
                or row.get("precip_mm")
            )
            if timestamp is None or depth is None:
                raise ValueError("Irrigation schedule rows must include date and depth_mm columns.")
            rows.append({"Date": pd.to_datetime(timestamp), "Depth": _as_float(depth)})
        kwargs["Schedule"] = pd.DataFrame.from_records(rows, columns=["Date", "Depth"])

    return IrrigationManagement(irrigation_method=method, **kwargs)


def run_aquacrop_ospy(
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
    assumptions: Mapping[str, Any] | None,
    warnings: list[str],
) -> Any:
    tool_name = "impact_aquacrop_run"
    assumptions = dict(assumptions or {})
    if assumptions.get("ospy_development_mode") or os.getenv("AQUACROP_OSPY_DEVELOPMENT"):
        os.environ.setdefault("DEVELOPMENT", "DEVELOPMENT")

    try:
        import pandas as pd
        from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement, Soil
    except ImportError as exc:
        return _error(
            tool_name,
            "AquaCrop-OSPy backend requested, but the Python package is not importable. "
            "Install it in this environment with `pip install aquacrop`.",
            start_ts=start_ts,
            run_context=run_context,
            warnings=warnings,
        )

    try:
        weather_rows = _read_weather_daily(weather_daily_path)
        soil_cfg = _read_json(soil_path) if soil_path else {}
        crop_cfg = _read_json(crop_path) if crop_path else {}
        management_cfg = _read_json(management_path) if management_path else {}
        if not isinstance(soil_cfg, dict) or not isinstance(crop_cfg, dict) or not isinstance(management_cfg, dict):
            return _error(
                tool_name,
                "AquaCrop-OSPy soil, crop, and management inputs must be JSON objects.",
                start_ts=start_ts,
                run_context=run_context,
                warnings=warnings,
            )

        weather_df = _build_weather_frame(weather_rows, pd)
        sim_start = _format_ospy_date(
            assumptions.get("sim_start_time")
            or management_cfg.get("sim_start_time")
            or management_cfg.get("simulation_start_date")
            or weather_rows[0]["date"]
        )
        sim_end = _format_ospy_date(
            assumptions.get("sim_end_time")
            or management_cfg.get("sim_end_time")
            or management_cfg.get("simulation_end_date")
            or weather_rows[-1]["date"]
        )

        soil_type = str(soil_cfg.get("soil_type") or soil_cfg.get("type") or soil_cfg.get("name") or "SandyLoam")
        soil_kwargs = _nested_mapping(soil_cfg, "ospy_kwargs", "aquacrop_ospy_kwargs")
        try:
            soil = Soil(soil_type=soil_type, **soil_kwargs)
        except TypeError:
            soil = Soil(soil_type, **soil_kwargs)

        crop_name = str(crop_cfg.get("crop_name") or crop_cfg.get("name") or crop_cfg.get("c_name") or "Maize")
        planting_date = _format_planting_date(
            crop_cfg.get("planting_date")
            or management_cfg.get("planting_date")
            or assumptions.get("planting_date")
            or "05/01"
        )
        harvest_raw = crop_cfg.get("harvest_date") or management_cfg.get("harvest_date") or assumptions.get("harvest_date")
        harvest_date = _format_planting_date(harvest_raw) if harvest_raw else None
        crop_kwargs = _nested_mapping(crop_cfg, "ospy_kwargs", "aquacrop_ospy_kwargs")
        if harvest_date:
            crop = Crop(crop_name, planting_date=planting_date, harvest_date=harvest_date, **crop_kwargs)
        else:
            crop = Crop(crop_name, planting_date=planting_date, **crop_kwargs)

        iwc_cfg = (
            management_cfg.get("initial_water_content")
            or soil_cfg.get("initial_water_content")
            or assumptions.get("initial_water_content")
            or {"value": ["FC"]}
        )
        if isinstance(iwc_cfg, str):
            iwc_cfg = {"value": [iwc_cfg]}
        if not isinstance(iwc_cfg, Mapping):
            raise ValueError("initial_water_content must be a string or JSON object.")
        depth_layer = iwc_cfg.get("depth_layer", [1])
        if not isinstance(depth_layer, list):
            depth_layer = [depth_layer]
        iwc_value = iwc_cfg.get("value", ["FC"])
        if not isinstance(iwc_value, list):
            iwc_value = [iwc_value]
        init_wc = InitialWaterContent(
            wc_type=iwc_cfg.get("wc_type", "Prop"),
            method=iwc_cfg.get("method", "Layer"),
            depth_layer=depth_layer,
            value=iwc_value,
        )

        irrigation_management = _build_irrigation_management(
            management_cfg,
            len(weather_rows),
            IrrigationManagement,
            pd,
        )

        run_dir = _tool_run_dir(output_dir, f"{tool_name}_ospy", run_id, scenario_name)
        model = AquaCropModel(
            sim_start_time=sim_start,
            sim_end_time=sim_end,
            weather_df=weather_df,
            soil=soil,
            crop=crop,
            initial_water_content=init_wc,
            irrigation_management=irrigation_management,
        )
        model.run_model(till_termination=True)

        final_stats = model.get_simulation_results()
        water_flux = model.get_water_flux()
        water_storage = model.get_water_storage()
        crop_growth = model.get_crop_growth()

        final_stats_path = _dataframe_to_csv(final_stats, run_dir / "aquacrop_ospy_final_stats.csv")
        water_flux_path = _dataframe_to_csv(water_flux, run_dir / "aquacrop_ospy_water_flux.csv")
        water_storage_path = _dataframe_to_csv(water_storage, run_dir / "aquacrop_ospy_water_storage.csv")
        crop_growth_path = _dataframe_to_csv(crop_growth, run_dir / "aquacrop_ospy_crop_growth.csv")

        yield_t_ha = _last_numeric(final_stats, ["Yield (tonne/ha)", "Dry yield (tonne/ha)"])
        seasonal_irrigation_mm = _last_numeric(final_stats, ["Seasonal irrigation (mm)"])
        if seasonal_irrigation_mm is None:
            seasonal_irrigation_mm = _sum_numeric(water_flux, ["IrrDay"])
        seasonal_et_mm = _sum_numeric(water_flux, ["Tr", "Es"])
        biomass_raw = _max_numeric(crop_growth, ["biomass"])
        biomass_t_ha = biomass_raw / 100.0 if biomass_raw is not None else None
        tr = _sum_numeric(water_flux, ["Tr"])
        tr_pot = _sum_numeric(water_flux, ["TrPot"])
        stress = _clamp(1.0 - (tr / tr_pot), 0.0, 1.0) if tr is not None and tr_pot and tr_pot > 0 else None

        metrics: dict[str, Any] = {}
        if yield_t_ha is not None:
            metrics["yield_t_ha"] = _metric(yield_t_ha, "t/ha")
        if biomass_t_ha is not None:
            metrics["biomass_t_ha"] = _metric(biomass_t_ha, "t/ha")
        if seasonal_et_mm is not None:
            metrics["seasonal_et_mm"] = _metric(seasonal_et_mm, "mm")
        if seasonal_irrigation_mm is not None:
            metrics["seasonal_irrigation_mm"] = _metric(seasonal_irrigation_mm, "mm")
        if stress is not None:
            metrics["water_stress_index"] = _metric(stress, "ratio")

        if "yield_t_ha" not in metrics:
            warnings.append("AquaCrop-OSPy completed, but no yield column was found in final_stats.")
        warnings.append("AquaCrop-OSPy is not the official FAO AquaCrop executable; cite it as an AquaCrop-OSPy backend.")
        season_count = float(len(final_stats)) if final_stats is not False and hasattr(final_stats, "__len__") else 0.0

        summary_path = _write_json(
            run_dir / "aquacrop_ospy_summary.json",
            {
                "backend": "aquacrop-ospy",
                "sim_start_time": sim_start,
                "sim_end_time": sim_end,
                "soil_type": soil_type,
                "crop_name": crop_name,
                "planting_date": planting_date,
                "harvest_date": harvest_date,
                "metrics": metrics,
            },
        )

        risk_anchor = metrics.get("water_stress_index", metrics.get("yield_t_ha", {"value": 0.0}))["value"]
        risk_level = _risk_level(float(risk_anchor), low=0.25, high=0.5) if "water_stress_index" in metrics else "none"
        headline = "AquaCrop-OSPy backend completed."
        if "yield_t_ha" in metrics:
            headline = f"AquaCrop-OSPy projected seasonal yield is {metrics['yield_t_ha']['value']:.2f} t/ha."

        return _success(
            tool_name=tool_name,
            start_ts=start_ts,
            metrics=metrics,
            derived_metrics={
                "ospy_season_count": _metric(season_count, "count")
            },
            output_artifacts=[
                _artifact_record(final_stats_path, "AquaCrop-OSPy seasonal final statistics"),
                _artifact_record(water_flux_path, "AquaCrop-OSPy daily water flux outputs"),
                _artifact_record(water_storage_path, "AquaCrop-OSPy daily soil water storage outputs"),
                _artifact_record(crop_growth_path, "AquaCrop-OSPy daily crop growth outputs"),
                _artifact_record(summary_path, "AquaCrop-OSPy standardized summary"),
            ],
            run_context=run_context,
            warnings=warnings,
            agentic_summary={
                "headline": headline,
                "primary_signal": {"metric": "yield_t_ha", **metrics["yield_t_ha"]}
                if "yield_t_ha" in metrics
                else {"metric": "ospy_season_count", "value": season_count, "unit": "count"},
                "risk_level": risk_level,
                "comparison_ready": True,
                "recommended_next_tools": ["impact_dssat_run", "impact_climada_run"],
            },
            provenance_command="inprocess:aquacrop-ospy",
        )
    except Exception as exc:
        elapsed = max(0.0, time.perf_counter() - start_ts)
        warnings.append(f"AquaCrop-OSPy failed after {elapsed:.2f}s.")
        return _error(tool_name, str(exc), start_ts=start_ts, run_context=run_context, warnings=warnings)
