"""
impact_aquacrop_run simulator tool.
"""

from .common import *  # noqa: F401,F403
from .official_backends import run_aquacrop_official, wants_official_backend
from .aquacrop_ospy_backend import run_aquacrop_ospy, wants_aquacrop_ospy_backend

def impact_aquacrop_run(
    weather_daily_path: str,
    soil_path: str,
    crop_path: str,
    management_path: str,
    *,
    ensemble_weather_daily_paths: list[str] | None = None,
    run_id: str = "run-aquacrop",
    instance_id: str = "instance-001",
    scenario_name: str = "baseline",
    scenario_description: str = "",
    assumptions: dict[str, Any] | None = None,
    seed: int = 23,
    model_version: str = "oracle:impact_aquacrop@1.0.0",
    container_image: str | None = None,
    threads: int = 1,
    cpu_cores: int = 2,
    memory_gb: float = 2.0,
    timeout_seconds: int = 600,
    yield_threshold_t_ha: float = 2.5,
    output_dir: str = "./data/simulators",
    backend: str | None = None,
) -> ToolResponse:
    """
    Run crop yield and water stress analysis.

    By default this uses the configured official AquaCrop backend. Pass
    backend="ospy" to use AquaCrop-OSPy or backend="proxy" to use the
    deterministic AquaCrop-style local proxy.
    """

    tool_name = "impact_aquacrop_run"
    start_ts = time.perf_counter()
    input_paths: list[tuple[str, str]] = [
        (weather_daily_path, "daily weather forcing"),
        (soil_path, "soil profile"),
        (crop_path, "crop parameters"),
        (management_path, "management settings"),
    ]
    for idx, path in enumerate(ensemble_weather_daily_paths or []):
        input_paths.append((path, f"ensemble weather member {idx + 1}"))
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
        inputs=input_paths,
    )
    warnings: list[str] = []
    if wants_aquacrop_ospy_backend(backend):
        return run_aquacrop_ospy(
            weather_daily_path=weather_daily_path,
            soil_path=soil_path,
            crop_path=crop_path,
            management_path=management_path,
            run_id=run_id,
            scenario_name=scenario_name,
            output_dir=output_dir,
            start_ts=start_ts,
            run_context=run_context,
            assumptions=assumptions,
            warnings=warnings,
        )
    if wants_official_backend(backend):
        return run_aquacrop_official(
            weather_daily_path=weather_daily_path,
            soil_path=soil_path,
            crop_path=crop_path,
            management_path=management_path,
            run_id=run_id,
            scenario_name=scenario_name,
            output_dir=output_dir,
            start_ts=start_ts,
            run_context=run_context,
            timeout_seconds=timeout_seconds,
            assumptions=assumptions,
            warnings=warnings,
        )
    try:
        weather_rows = _read_weather_daily(weather_daily_path)
        soil = _read_json(soil_path) if soil_path else {}
        crop = _read_json(crop_path) if crop_path else {}
        management = _read_json(management_path) if management_path else {}
        if not isinstance(soil, dict) or not isinstance(crop, dict) or not isinstance(management, dict):
            return _error(tool_name, "Soil/crop/management inputs must be JSON objects.", start_ts=start_ts, run_context=run_context)

        base_metrics, detail_rows = _run_aquacrop_proxy(weather_rows, soil=soil, crop=crop, management=management)
        run_dir = _tool_run_dir(output_dir, tool_name, run_id, scenario_name)
        timeseries_path = run_dir / "aquacrop_timeseries.csv"
        _write_csv(timeseries_path, detail_rows)

        derived: dict[str, Any] = {}
        if ensemble_weather_daily_paths:
            member_yields: list[float] = []
            for member_path in ensemble_weather_daily_paths:
                member_rows = _read_weather_daily(member_path)
                member_metrics, _ = _run_aquacrop_proxy(member_rows, soil=soil, crop=crop, management=management)
                member_yields.append(member_metrics["yield_t_ha"])
            if member_yields:
                p10 = _percentile(member_yields, 0.10)
                p50 = _percentile(member_yields, 0.50)
                p90 = _percentile(member_yields, 0.90)
                prob_below = sum(1 for value in member_yields if value < yield_threshold_t_ha) / len(member_yields)
                derived.update(
                    {
                        "ensemble_yield_p10_t_ha": _metric(p10, "t/ha"),
                        "ensemble_yield_p50_t_ha": _metric(p50, "t/ha"),
                        "ensemble_yield_p90_t_ha": _metric(p90, "t/ha"),
                        "probability_yield_below_threshold": _metric(prob_below, "ratio"),
                    }
                )
            else:
                warnings.append("Ensemble weather list was provided but no member produced usable output.")

        summary_path = run_dir / "aquacrop_summary.json"
        _write_json(summary_path, {"base_metrics": base_metrics, "derived_metrics": derived})
        outputs = [
            _artifact_record(timeseries_path, "Daily crop-water simulation trace"),
            _artifact_record(summary_path, "AquaCrop-style summary metrics"),
        ]
        metrics = {
            "yield_t_ha": _metric(base_metrics["yield_t_ha"], "t/ha"),
            "biomass_t_ha": _metric(base_metrics["biomass_t_ha"], "t/ha"),
            "seasonal_et_mm": _metric(base_metrics["seasonal_et_mm"], "mm"),
            "seasonal_irrigation_mm": _metric(base_metrics["seasonal_irrigation_mm"], "mm"),
            "water_stress_index": _metric(base_metrics["water_stress_index"], "ratio"),
        }
        if "probability_yield_below_threshold" in derived:
            risk_anchor = derived["probability_yield_below_threshold"]["value"]
        else:
            risk_anchor = metrics["water_stress_index"]["value"]
        agentic_summary = {
            "headline": f"Projected seasonal yield is {metrics['yield_t_ha']['value']:.2f} t/ha.",
            "primary_signal": {"metric": "yield_t_ha", **metrics["yield_t_ha"]},
            "risk_level": _risk_level(float(risk_anchor), low=0.25, high=0.5),
            "comparison_ready": True,
            "recommended_next_tools": ["impact_dssat_run", "impact_climada_run"],
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


__all__ = ["impact_aquacrop_run"]
