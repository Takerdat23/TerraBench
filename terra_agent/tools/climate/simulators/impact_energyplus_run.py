"""
impact_energyplus_run simulator tool.
"""

from .common import *  # noqa: F401,F403
from .official_backends import run_energyplus_official, wants_official_backend

def impact_energyplus_run(
    building_idf_path: str,
    weather_epw_path: str,
    *,
    run_id: str = "run-energyplus",
    instance_id: str = "instance-001",
    scenario_name: str = "baseline",
    scenario_description: str = "",
    assumptions: dict[str, Any] | None = None,
    seed: int = 43,
    model_version: str = "oracle:impact_energyplus@1.0.0",
    container_image: str | None = "container:energyplus-proxy@sha256:deterministic",
    threads: int = 1,
    cpu_cores: int = 2,
    memory_gb: float = 3.0,
    timeout_seconds: int = 900,
    default_floor_area_m2: float = 5000.0,
    cooling_setpoint_c: float = 24.0,
    heating_setpoint_c: float = 20.0,
    base_load_w_m2: float = 6.0,
    cooling_sensitivity_w_m2_per_c: float = 4.0,
    heating_sensitivity_w_m2_per_c: float = 3.2,
    output_dir: str = "./data/simulators",
    backend: str | None = None,
) -> ToolResponse:
    """
    Run building demand and discomfort analysis.

    By default this uses the official EnergyPlus executable backend. Pass
    backend="proxy" to use the deterministic EnergyPlus-style local proxy.
    """

    tool_name = "impact_energyplus_run"
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
        inputs=[
            (building_idf_path, "building IDF model"),
            (weather_epw_path, "EPW weather file"),
        ],
    )
    warnings: list[str] = []
    if wants_official_backend(backend):
        return run_energyplus_official(
            building_idf_path=building_idf_path,
            weather_epw_path=weather_epw_path,
            run_id=run_id,
            scenario_name=scenario_name,
            output_dir=output_dir,
            start_ts=start_ts,
            run_context=run_context,
            timeout_seconds=timeout_seconds,
            warnings=warnings,
        )
    try:
        epw_rows = _read_epw_weather(weather_epw_path)
        if not epw_rows:
            return _error(tool_name, "No valid hourly records found in EPW input.", start_ts=start_ts, run_context=run_context)
        floor_area = _infer_floor_area_m2(building_idf_path, default_floor_area_m2)

        hourly_rows: list[dict[str, Any]] = []
        total_kwh = 0.0
        cooling_kwh = 0.0
        heating_kwh = 0.0
        peak_kw = 0.0
        discomfort_hours = 0
        for row in epw_rows:
            temp = row["dry_bulb_c"]
            cooling_w = max(0.0, temp - cooling_setpoint_c) * cooling_sensitivity_w_m2_per_c * floor_area
            heating_w = max(0.0, heating_setpoint_c - temp) * heating_sensitivity_w_m2_per_c * floor_area
            base_w = base_load_w_m2 * floor_area
            total_w = base_w + cooling_w + heating_w
            total_hour_kwh = total_w / 1000.0
            cooling_hour_kwh = cooling_w / 1000.0
            heating_hour_kwh = heating_w / 1000.0
            total_kwh += total_hour_kwh
            cooling_kwh += cooling_hour_kwh
            heating_kwh += heating_hour_kwh
            peak_kw = max(peak_kw, total_w / 1000.0)
            if temp < 18.0 or temp > 27.0:
                discomfort_hours += 1
            hourly_rows.append(
                {
                    "time": row["time"].isoformat(),
                    "dry_bulb_c": round(temp, 4),
                    "total_kw": round(total_w / 1000.0, 4),
                    "cooling_kw": round(cooling_w / 1000.0, 4),
                    "heating_kw": round(heating_w / 1000.0, 4),
                }
            )

        run_dir = _tool_run_dir(output_dir, tool_name, run_id, scenario_name)
        hourly_path = run_dir / "eplus_hourly_proxy.csv"
        summary_path = run_dir / "energyplus_summary.json"
        log_path = run_dir / "energyplus_run.log"
        _write_csv(hourly_path, hourly_rows)
        _write_json(
            summary_path,
            {
                "peak_kw": peak_kw,
                "total_kwh": total_kwh,
                "cooling_kwh": cooling_kwh,
                "heating_kwh": heating_kwh,
                "discomfort_hours": discomfort_hours,
                "floor_area_m2": floor_area,
            },
        )
        log_path.write_text(
            "EnergyPlus proxy run completed successfully.\n"
            f"rows={len(hourly_rows)} floor_area_m2={floor_area:.2f}\n",
            encoding="utf-8",
        )
        outputs = [
            _artifact_record(hourly_path, "Hourly demand proxy outputs"),
            _artifact_record(summary_path, "Aggregated EnergyPlus proxy KPIs"),
            _artifact_record(log_path, "EnergyPlus proxy runtime log"),
        ]
        metrics = {
            "peak_kw": _metric(peak_kw, "kW"),
            "total_kwh": _metric(total_kwh, "kWh"),
            "cooling_kwh": _metric(cooling_kwh, "kWh"),
            "heating_kwh": _metric(heating_kwh, "kWh"),
            "discomfort_hours": _metric(float(discomfort_hours), "hours"),
        }
        derived = {
            "floor_area_m2": _metric(floor_area, "m2"),
            "specific_energy_kwh_m2": _metric(total_kwh / max(1e-6, floor_area), "kWh/m2"),
        }
        agentic_summary = {
            "headline": f"Proxy building demand total is {metrics['total_kwh']['value']:.1f} kWh with peak {metrics['peak_kw']['value']:.2f} kW.",
            "primary_signal": {"metric": "total_kwh", **metrics["total_kwh"]},
            "risk_level": _risk_level(discomfort_hours / max(1, len(hourly_rows)), low=0.1, high=0.3),
            "comparison_ready": True,
            "recommended_next_tools": ["impact_health_utci", "impact_climada_run"],
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


__all__ = ["impact_energyplus_run"]
