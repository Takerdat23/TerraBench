"""
impact_dssat_run simulator tool.
"""

from .common import *  # noqa: F401,F403
from .official_backends import run_dssat_official, wants_official_backend

def impact_dssat_run(
    weather_daily_path: str,
    soil_profile_path: str,
    cultivar_path: str,
    treatment_path: str,
    *,
    run_id: str = "run-dssat",
    instance_id: str = "instance-001",
    scenario_name: str = "baseline",
    scenario_description: str = "",
    assumptions: dict[str, Any] | None = None,
    seed: int = 29,
    model_version: str = "oracle:impact_dssat@1.0.0",
    container_image: str | None = "container:dssat-proxy@sha256:deterministic",
    threads: int = 1,
    cpu_cores: int = 2,
    memory_gb: float = 2.5,
    timeout_seconds: int = 900,
    output_dir: str = "./data/simulators",
    backend: str | None = None,
) -> ToolResponse:
    """
    Run crop systems analysis.

    By default this uses the configured official DSSAT backend. Pass
    backend="proxy" to use the deterministic DSSAT-like local proxy.
    """

    tool_name = "impact_dssat_run"
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
        inputs=[
            (weather_daily_path, "daily weather forcing"),
            (soil_profile_path, "soil profile"),
            (cultivar_path, "cultivar parameters"),
            (treatment_path, "treatment and management"),
        ],
    )
    warnings: list[str] = []
    if wants_official_backend(backend):
        return run_dssat_official(
            weather_daily_path=weather_daily_path,
            soil_profile_path=soil_profile_path,
            cultivar_path=cultivar_path,
            treatment_path=treatment_path,
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
        cultivar = _read_json(cultivar_path) if cultivar_path else {}
        treatment = _read_json(treatment_path) if treatment_path else {}
        if not isinstance(cultivar, dict) or not isinstance(treatment, dict):
            return _error(tool_name, "Cultivar and treatment inputs must be JSON objects.", start_ts=start_ts, run_context=run_context)

        sim_metrics, detail_rows = _run_dssat_proxy(weather_rows, cultivar=cultivar, treatment=treatment)
        run_dir = _tool_run_dir(output_dir, tool_name, run_id, scenario_name)
        detail_path = run_dir / "dssat_daily_proxy.csv"
        _write_csv(detail_path, detail_rows)

        run_folder = run_dir / "dssat_run_folder"
        run_folder.mkdir(parents=True, exist_ok=True)
        for source in (weather_daily_path, soil_profile_path, cultivar_path, treatment_path):
            src = Path(source).expanduser()
            if src.exists():
                shutil.copy2(src, run_folder / src.name)
        summary_payload = {
            "yield_kg_ha": sim_metrics["yield_kg_ha"],
            "water_stress_days": sim_metrics["water_stress_days"],
            "seasonal_et_mm": sim_metrics["seasonal_et_mm"],
            "gdd_total": sim_metrics["gdd_total"],
            "phenology": sim_metrics["phenology"],
        }
        summary_in_folder = run_folder / "summary.json"
        _write_json(summary_in_folder, summary_payload)

        archive_path = run_dir / "dssat_run_folder.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(run_folder, arcname=run_folder.name)

        outputs = [
            _artifact_record(detail_path, "DSSAT proxy day-level diagnostics"),
            _artifact_record(archive_path, "Archived DSSAT-like run folder for replay"),
            _artifact_record(summary_in_folder, "Run summary with key metrics and phenology"),
        ]
        metrics = {
            "yield_kg_ha": _metric(sim_metrics["yield_kg_ha"], "kg/ha"),
            "water_stress_days": _metric(sim_metrics["water_stress_days"], "days"),
            "seasonal_et_mm": _metric(sim_metrics["seasonal_et_mm"], "mm"),
        }
        derived = {
            "gdd_total": _metric(sim_metrics["gdd_total"], "degC-day"),
        }
        risk_anchor = sim_metrics["water_stress_days"] / max(1, len(detail_rows))
        agentic_summary = {
            "headline": f"DSSAT proxy yield estimate: {metrics['yield_kg_ha']['value']:.0f} kg/ha.",
            "primary_signal": {"metric": "yield_kg_ha", **metrics["yield_kg_ha"]},
            "risk_level": _risk_level(risk_anchor, low=0.18, high=0.35),
            "comparison_ready": True,
            "recommended_next_tools": ["impact_aquacrop_run", "impact_climada_run"],
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
            extra={"phenology": sim_metrics["phenology"]},
        )
    except Exception as exc:
        return _error(tool_name, str(exc), start_ts=start_ts, run_context=run_context, warnings=warnings)


__all__ = ["impact_dssat_run"]
