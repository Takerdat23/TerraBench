"""
impact_health_utci simulator tool.
"""

from .common import *  # noqa: F401,F403
from .official_backends import run_utci_official, wants_official_backend

def impact_health_utci(
    met_timeseries_path: str,
    *,
    run_id: str = "run-health-utci",
    instance_id: str = "instance-001",
    scenario_name: str = "baseline",
    scenario_description: str = "",
    assumptions: dict[str, Any] | None = None,
    seed: int = 37,
    model_version: str = "oracle:impact_health_utci@1.0.0",
    container_image: str | None = None,
    threads: int = 1,
    cpu_cores: int = 1,
    memory_gb: float = 1.0,
    timeout_seconds: int = 120,
    strong_stress_threshold_c: float = 32.0,
    output_dir: str = "./data/simulators",
    backend: str | None = None,
) -> ToolResponse:
    """
    Compute UTCI heat stress metrics from meteorological time series.

    By default this uses a validated UTCI library backend. Pass backend="proxy"
    to use the deterministic local UTCI-style approximation.
    """

    tool_name = "impact_health_utci"
    start_ts = time.perf_counter()
    run_context = _build_run_context(
        run_id=run_id,
        instance_id=instance_id,
        scenario_name=scenario_name,
        scenario_description=scenario_description,
        start_iso=None,
        end_iso=None,
        timestep="hourly",
        assumptions=assumptions,
        seed=seed,
        model_version=model_version,
        container_image=container_image,
        threads=threads,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        timeout_seconds=timeout_seconds,
        inputs=[(met_timeseries_path, "meteorological time series for UTCI")],
    )
    warnings: list[str] = []
    if wants_official_backend(backend):
        return run_utci_official(
            met_timeseries_path=met_timeseries_path,
            run_id=run_id,
            scenario_name=scenario_name,
            output_dir=output_dir,
            start_ts=start_ts,
            run_context=run_context,
            strong_stress_threshold_c=strong_stress_threshold_c,
            warnings=warnings,
        )
    try:
        met_rows = _load_rows(met_timeseries_path)
        if not met_rows:
            return _error(tool_name, "Meteorological input is empty.", start_ts=start_ts, run_context=run_context)

        out_rows: list[dict[str, Any]] = []
        utci_values: list[float] = []
        categories = Counter()
        for idx, row in enumerate(met_rows):
            timestamp = _parse_datetime_from_row(row) or datetime(2000, 1, 1) + timedelta(hours=idx)
            tair = _as_float(row.get("tair_c") or row.get("temperature_c") or row.get("t2m_c"), 25.0)
            rh = _clamp(_as_float(row.get("rh_pct") or row.get("humidity_pct"), 55.0), 1.0, 100.0)
            wind = max(0.0, _as_float(row.get("wind_m_s") or row.get("wind"), 1.5))
            tmrt = _as_float(row.get("tmrt_c"), tair + 1.8)
            # A deterministic UTCI-style approximation for fast proxy scoring.
            utci = tair + 0.607562 + 0.8 * (tmrt - tair) - 1.25 * math.sqrt(max(0.1, wind)) - 0.020 * (100.0 - rh)
            utci_values.append(utci)
            if utci >= 46.0:
                category = "extreme_heat"
            elif utci >= 38.0:
                category = "very_strong_heat"
            elif utci >= 32.0:
                category = "strong_heat"
            elif utci >= 26.0:
                category = "moderate_heat"
            elif utci >= 9.0:
                category = "no_thermal_stress"
            elif utci >= 0.0:
                category = "slight_cold"
            elif utci >= -13.0:
                category = "moderate_cold"
            else:
                category = "strong_cold"
            categories[category] += 1
            out_rows.append(
                {
                    "time": timestamp.isoformat(),
                    "tair_c": round(tair, 4),
                    "rh_pct": round(rh, 4),
                    "wind_m_s": round(wind, 4),
                    "tmrt_c": round(tmrt, 4),
                    "utci_c": round(utci, 4),
                    "category": category,
                }
            )

        mean_utci = statistics.fmean(utci_values) if utci_values else 0.0
        p95_utci = _percentile(utci_values, 0.95)
        strong_hours = sum(1 for value in utci_values if value >= strong_stress_threshold_c)

        run_dir = _tool_run_dir(output_dir, tool_name, run_id, scenario_name)
        timeseries_path = run_dir / "utci_timeseries.csv"
        summary_path = run_dir / "utci_summary.json"
        _write_csv(timeseries_path, out_rows)
        _write_json(
            summary_path,
            {
                "utci_mean_c": mean_utci,
                "utci_p95_c": p95_utci,
                "hours_above_strong_stress": strong_hours,
                "category_counts": dict(categories),
            },
        )

        outputs = [
            _artifact_record(timeseries_path, "UTCI proxy timeseries"),
            _artifact_record(summary_path, "UTCI summary statistics"),
        ]
        metrics = {
            "utci_mean_c": _metric(mean_utci, "degC"),
            "utci_p95_c": _metric(p95_utci, "degC"),
            "hours_above_strong_stress": _metric(float(strong_hours), "hours"),
        }
        derived: dict[str, Any] = {
            "strong_stress_share": _metric(strong_hours / max(1, len(out_rows)), "ratio"),
        }
        for category_name, count in sorted(categories.items()):
            derived[f"category_{category_name}_hours"] = _metric(float(count), "hours")
        agentic_summary = {
            "headline": f"UTCI strong-heat exposure totals {strong_hours} hours.",
            "primary_signal": {"metric": "hours_above_strong_stress", **metrics["hours_above_strong_stress"]},
            "risk_level": _risk_level(strong_hours / max(1, len(out_rows)), low=0.1, high=0.25),
            "comparison_ready": True,
            "recommended_next_tools": ["impact_health_erf", "impact_energyplus_run"],
        }
        return _success(
            tool_name=tool_name,
            start_ts=start_ts,
            metrics=metrics,
            derived_metrics=derived,
            output_artifacts=outputs,
            run_context=run_context,
            warnings=warnings,
            agentic_summary=agentic_summary,
            extra={"category_counts": dict(categories)},
        )
    except Exception as exc:
        return _error(tool_name, str(exc), start_ts=start_ts, run_context=run_context, warnings=warnings)


__all__ = ["impact_health_utci"]
