"""
weather_monthly_to_daily simulator tool.
"""

from .common import *  # noqa: F401,F403

def weather_monthly_to_daily(
    monthly_weather_path: str,
    *,
    run_id: str = "run-weather-downscale",
    instance_id: str = "instance-001",
    scenario_name: str = "baseline",
    scenario_description: str = "",
    assumptions: dict[str, Any] | None = None,
    seed: int = 7,
    model_version: str = "oracle:weather_monthly_to_daily@1.0.0",
    container_image: str | None = None,
    threads: int = 1,
    cpu_cores: int = 1,
    memory_gb: float = 1.0,
    timeout_seconds: int = 120,
    wet_day_fraction: float = 0.35,
    output_path: str = "./data/simulators/weather_daily.csv",
) -> ToolResponse:
    """
    Deterministically downscale monthly weather summaries into daily forcing.

    Required input columns:
    - date/month+year
    - precip_mm
    - tmin_c
    - tmax_c
    Optional:
    - et0_mm
    """

    tool_name = "weather_monthly_to_daily"
    start_ts = time.perf_counter()
    run_context = _build_run_context(
        run_id=run_id,
        instance_id=instance_id,
        scenario_name=scenario_name,
        scenario_description=scenario_description,
        start_iso=None,
        end_iso=None,
        timestep="daily",
        assumptions=assumptions,
        seed=seed,
        model_version=model_version,
        container_image=container_image,
        threads=threads,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        timeout_seconds=timeout_seconds,
        inputs=[(monthly_weather_path, "monthly weather forcing")],
    )
    warnings: list[str] = []
    try:
        wet_day_fraction = _clamp(float(wet_day_fraction), 0.05, 1.0)
        monthly_rows = _read_monthly_weather(monthly_weather_path)
        if not monthly_rows:
            return _error(tool_name, "Monthly weather input is empty.", start_ts=start_ts, run_context=run_context)

        rng = random.Random(seed)
        daily_rows: list[dict[str, Any]] = []
        for month_row in monthly_rows:
            year = int(month_row["year"])
            month = int(month_row["month"])
            n_days = calendar.monthrange(year, month)[1]
            wet_days = max(1, int(round(n_days * wet_day_fraction)))
            daily_indices = list(range(n_days))
            rng.shuffle(daily_indices)
            wet_set = set(daily_indices[:wet_days])
            weights = [rng.random() for _ in range(wet_days)]
            weight_sum = sum(weights) or 1.0
            wet_alloc = [month_row["precip_mm"] * (w / weight_sum) for w in weights]
            wet_cursor = 0
            temp_mean = (month_row["tmin_c"] + month_row["tmax_c"]) / 2.0
            temp_span = max(3.5, month_row["tmax_c"] - month_row["tmin_c"])

            for day_idx in range(n_days):
                current_date = date(year, month, day_idx + 1)
                phase = (2.0 * math.pi * day_idx) / max(1, n_days)
                diurnal_wave = 0.8 * math.sin(phase)
                temp_noise = rng.uniform(-0.6, 0.6)
                local_mean = temp_mean + diurnal_wave + temp_noise
                local_span = max(3.0, temp_span + rng.uniform(-1.0, 1.0))
                tmin = local_mean - local_span / 2.0
                tmax = local_mean + local_span / 2.0
                precip = 0.0
                if day_idx in wet_set:
                    precip = wet_alloc[wet_cursor]
                    wet_cursor += 1
                et0 = max(0.0, month_row["et0_mm"] / n_days + rng.uniform(-0.15, 0.15))
                daily_rows.append(
                    {
                        "date": current_date.isoformat(),
                        "precip_mm": round(max(0.0, precip), 5),
                        "tmin_c": round(tmin, 4),
                        "tmax_c": round(max(tmax, tmin), 4),
                        "et0_mm": round(et0, 5),
                    }
                )

        target_dir = _tool_run_dir(DEFAULT_OUTPUT_DIR, tool_name, run_id, scenario_name)
        daily_path = _resolve_output_path(output_path, fallback_dir=target_dir, filename="weather_daily.csv")
        _write_csv(daily_path, daily_rows)
        summary_path = target_dir / "weather_downscale_summary.json"
        summary_payload = {
            "rows": len(daily_rows),
            "seed": seed,
            "wet_day_fraction": wet_day_fraction,
            "source_months": len(monthly_rows),
        }
        _write_json(summary_path, summary_payload)
        outputs = [
            _artifact_record(daily_path, "Downscaled daily weather forcing"),
            _artifact_record(summary_path, "Downscaling metadata and deterministic controls"),
        ]
        metrics = {
            "daily_rows": _metric(float(len(daily_rows)), "count"),
            "precip_total_mm": _metric(sum(_as_float(row["precip_mm"]) for row in daily_rows), "mm"),
            "temperature_mean_c": _metric(
                statistics.fmean((_as_float(row["tmin_c"]) + _as_float(row["tmax_c"])) / 2.0 for row in daily_rows),
                "degC",
            ),
        }
        derived = {
            "wet_days": _metric(float(sum(1 for row in daily_rows if _as_float(row["precip_mm"]) > 0.0)), "days"),
        }
        agentic_summary = {
            "headline": f"Generated {len(daily_rows)} deterministic daily weather rows for {scenario_name}.",
            "primary_signal": {"metric": "precip_total_mm", **metrics["precip_total_mm"]},
            "risk_level": "none",
            "comparison_ready": True,
            "recommended_next_tools": ["impact_aquacrop_run", "impact_dssat_run", "weather_to_epw"],
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
        )
    except Exception as exc:
        return _error(tool_name, str(exc), start_ts=start_ts, run_context=run_context, warnings=warnings)


__all__ = ["weather_monthly_to_daily"]
