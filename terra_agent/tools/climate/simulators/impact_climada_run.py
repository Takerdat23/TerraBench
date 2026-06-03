"""
impact_climada_run simulator tool.
"""

from .common import *  # noqa: F401,F403
from .official_backends import run_climada_official, wants_official_backend

def impact_climada_run(
    hazard_event_set_path: str,
    exposure_path: str,
    impact_functions_path: str,
    *,
    run_id: str = "run-climada",
    instance_id: str = "instance-001",
    scenario_name: str = "baseline",
    scenario_description: str = "",
    assumptions: dict[str, Any] | None = None,
    seed: int = 31,
    model_version: str = "oracle:impact_climada@1.0.0",
    container_image: str | None = None,
    threads: int = 1,
    cpu_cores: int = 2,
    memory_gb: float = 3.0,
    timeout_seconds: int = 900,
    output_dir: str = "./data/simulators",
    backend: str | None = None,
) -> ToolResponse:
    """
    Compute event impacts from hazard, exposure, and vulnerability data.

    By default this uses the official CLIMADA Python backend. Pass
    backend="proxy" to use the deterministic local risk proxy.
    """

    tool_name = "impact_climada_run"
    start_ts = time.perf_counter()
    run_context = _build_run_context(
        run_id=run_id,
        instance_id=instance_id,
        scenario_name=scenario_name,
        scenario_description=scenario_description,
        start_iso=None,
        end_iso=None,
        timestep="event",
        assumptions=assumptions,
        seed=seed,
        model_version=model_version,
        container_image=container_image,
        threads=threads,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        timeout_seconds=timeout_seconds,
        inputs=[
            (hazard_event_set_path, "hazard event set"),
            (exposure_path, "exposure values and geometry ids"),
            (impact_functions_path, "vulnerability/impact functions"),
        ],
    )
    warnings: list[str] = []
    if wants_official_backend(backend):
        return run_climada_official(
            hazard_event_set_path=hazard_event_set_path,
            exposure_path=exposure_path,
            impact_functions_path=impact_functions_path,
            run_id=run_id,
            scenario_name=scenario_name,
            output_dir=output_dir,
            start_ts=start_ts,
            run_context=run_context,
            assumptions=assumptions,
            warnings=warnings,
        )
    try:
        hazard_rows = _load_rows(hazard_event_set_path)
        exposure_rows = _load_rows(exposure_path)
        impact_spec = _read_json(impact_functions_path)
        if not isinstance(impact_spec, dict):
            return _error(tool_name, "impact_functions_path must contain a JSON object.", start_ts=start_ts, run_context=run_context)
        if not hazard_rows or not exposure_rows:
            return _error(tool_name, "Hazard or exposure input is empty.", start_ts=start_ts, run_context=run_context)

        curve_defaults = impact_spec.get("default", impact_spec)
        curve_by_id = impact_spec.get("functions") if isinstance(impact_spec.get("functions"), dict) else {}

        def pick_curve(vuln_id: str) -> Mapping[str, Any]:
            if isinstance(curve_by_id, dict) and vuln_id in curve_by_id and isinstance(curve_by_id[vuln_id], dict):
                return curve_by_id[vuln_id]
            return curve_defaults if isinstance(curve_defaults, dict) else {}

        event_impacts: list[dict[str, Any]] = []
        asset_impacts_rows: list[dict[str, Any]] = []
        total_exposure_value = 0.0
        exposure_values: list[dict[str, Any]] = []
        for idx, exposure in enumerate(exposure_rows):
            exposure_id = str(exposure.get("exposure_id") or exposure.get("id") or f"exp_{idx + 1}")
            value = _as_float(exposure.get("value") or exposure.get("asset_value_usd") or exposure.get("population"), 0.0)
            vuln_id = str(exposure.get("impact_function_id") or exposure.get("vulnerability_id") or "default")
            total_exposure_value += value
            exposure_values.append({"exposure_id": exposure_id, "value": value, "vuln_id": vuln_id})

        for idx, event in enumerate(hazard_rows):
            event_id = str(event.get("event_id") or event.get("id") or f"event_{idx + 1}")
            intensity = _as_float(event.get("intensity") or event.get("hazard_intensity") or event.get("value"), 0.0)
            event_total = 0.0
            mdr_values: list[float] = []
            for exposure in exposure_values:
                curve = pick_curve(exposure["vuln_id"])
                threshold = _as_float(curve.get("threshold"), 0.1)
                saturation = max(threshold + 1e-6, _as_float(curve.get("saturation"), 1.0))
                beta = max(0.1, _as_float(curve.get("beta"), 1.4))
                min_damage = _clamp(_as_float(curve.get("min_damage"), 0.0), 0.0, 1.0)
                if intensity <= threshold:
                    mdr = min_damage
                else:
                    ratio = _clamp((intensity - threshold) / (saturation - threshold), 0.0, 1.0)
                    mdr = _clamp(min_damage + (1.0 - min_damage) * (ratio ** beta), 0.0, 1.0)
                impact_value = exposure["value"] * mdr
                event_total += impact_value
                mdr_values.append(mdr)
                asset_impacts_rows.append(
                    {
                        "event_id": event_id,
                        "exposure_id": exposure["exposure_id"],
                        "intensity": round(intensity, 5),
                        "mdr": round(mdr, 6),
                        "impact": round(impact_value, 4),
                    }
                )
            mean_mdr = statistics.fmean(mdr_values) if mdr_values else 0.0
            event_impacts.append(
                {
                    "event_id": event_id,
                    "intensity": round(intensity, 6),
                    "event_impact": round(event_total, 4),
                    "mean_damage_ratio": round(mean_mdr, 6),
                }
            )

        total_impact = sum(row["event_impact"] for row in event_impacts)
        expected_event_impact = total_impact / max(1, len(event_impacts))
        worst_event_impact = max((row["event_impact"] for row in event_impacts), default=0.0)

        sorted_impacts = sorted((row["event_impact"] for row in event_impacts), reverse=True)
        risk_curve = []
        for rank, loss in enumerate(sorted_impacts, start=1):
            exceedance_prob = rank / (len(sorted_impacts) + 1.0)
            return_period = 1.0 / exceedance_prob if exceedance_prob > 0 else None
            risk_curve.append({"rank": rank, "loss": round(loss, 4), "exceedance_probability": round(exceedance_prob, 6), "return_period_years": return_period})

        run_dir = _tool_run_dir(output_dir, tool_name, run_id, scenario_name)
        event_csv_path = run_dir / "impact_by_event.csv"
        assets_csv_path = run_dir / "impact_by_asset.csv"
        risk_curve_path = run_dir / "risk_curve.json"
        _write_csv(event_csv_path, event_impacts)
        _write_csv(assets_csv_path, asset_impacts_rows)
        _write_json(risk_curve_path, {"risk_curve": risk_curve, "events": len(event_impacts)})

        outputs = [
            _artifact_record(event_csv_path, "Impact totals per hazard event"),
            _artifact_record(assets_csv_path, "Impact decomposition per event and exposure"),
            _artifact_record(risk_curve_path, "Risk curve for exceedance analysis"),
        ]
        metrics = {
            "total_impact": _metric(total_impact, "exposure_unit"),
            "expected_event_impact": _metric(expected_event_impact, "exposure_unit"),
            "worst_event_impact": _metric(worst_event_impact, "exposure_unit"),
        }
        derived = {
            "events_count": _metric(float(len(event_impacts)), "count"),
            "exposure_points": _metric(float(len(exposure_values)), "count"),
        }
        loss_ratio = total_impact / max(1e-9, total_exposure_value)
        agentic_summary = {
            "headline": f"Estimated total impact is {metrics['total_impact']['value']:.2f} (same unit as exposure values).",
            "primary_signal": {"metric": "total_impact", **metrics["total_impact"]},
            "risk_level": _risk_level(loss_ratio, low=0.05, high=0.18),
            "comparison_ready": True,
            "recommended_next_tools": ["impact_sumo_run", "impact_health_erf", "extract_exposure_layers"],
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


__all__ = ["impact_climada_run"]
