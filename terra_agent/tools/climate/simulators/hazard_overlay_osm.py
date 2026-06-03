"""
hazard_overlay_osm simulator tool.
"""

from .common import *  # noqa: F401,F403

def hazard_overlay_osm(
    hazard_path: str,
    osm_edges_path: str,
    *,
    run_id: str = "run-hazard-overlay",
    instance_id: str = "instance-001",
    scenario_name: str = "baseline",
    scenario_description: str = "",
    assumptions: dict[str, Any] | None = None,
    seed: int = 19,
    model_version: str = "oracle:hazard_overlay_osm@1.0.0",
    container_image: str | None = None,
    threads: int = 1,
    cpu_cores: int = 1,
    memory_gb: float = 1.0,
    timeout_seconds: int = 120,
    disruption_threshold: float = 0.4,
    closure_threshold: float = 0.9,
    output_overlay_path: str = "./data/simulators/hazard_overlay.csv",
    output_disruptions_path: str = "./data/simulators/disruptions.json",
) -> ToolResponse:
    """
    Apply hazard intensity footprints onto OSM edge-like assets.
    """

    tool_name = "hazard_overlay_osm"
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
            (hazard_path, "hazard intensities"),
            (osm_edges_path, "OSM-derived edges"),
        ],
    )
    warnings: list[str] = []
    try:
        hazard_rows = _load_rows(hazard_path)
        edge_rows = _load_rows(osm_edges_path)
        if not hazard_rows:
            return _error(tool_name, "Hazard input is empty.", start_ts=start_ts, run_context=run_context)
        if not edge_rows:
            return _error(tool_name, "OSM edge input is empty.", start_ts=start_ts, run_context=run_context)

        hazard_by_key: dict[str, float] = {}
        for idx, row in enumerate(hazard_rows):
            key = str(row.get("edge_id") or row.get("asset_id") or row.get("id") or f"edge_{idx + 1}")
            intensity = _clamp(_as_float(row.get("intensity") or row.get("hazard_intensity") or row.get("value"), 0.0), 0.0, 5.0)
            hazard_by_key[key] = intensity

        overlay_rows: list[dict[str, Any]] = []
        closures: list[str] = []
        speed_reductions: list[dict[str, Any]] = []
        disruption_threshold = _clamp(disruption_threshold, 0.0, 1.0)
        closure_threshold = _clamp(closure_threshold, disruption_threshold, 5.0)
        for idx, edge in enumerate(edge_rows):
            edge_id = str(edge.get("edge_id") or edge.get("id") or edge.get("osm_id") or f"edge_{idx + 1}")
            intensity = hazard_by_key.get(edge_id, _as_float(edge.get("intensity"), 0.0))
            severity = _clamp((intensity - disruption_threshold) / max(1e-6, closure_threshold - disruption_threshold), 0.0, 1.0)
            speed_factor = _clamp(1.0 - 0.75 * severity, 0.1, 1.0)
            is_closed = intensity >= closure_threshold
            overlay_rows.append(
                {
                    "edge_id": edge_id,
                    "hazard_intensity": round(intensity, 5),
                    "severity_index": round(severity, 5),
                    "speed_factor": round(0.0 if is_closed else speed_factor, 5),
                    "status": "closed" if is_closed else "open",
                }
            )
            if is_closed:
                closures.append(edge_id)
            elif severity > 0.0:
                speed_reductions.append({"edge_id": edge_id, "speed_factor": round(speed_factor, 5)})

        run_dir = _tool_run_dir(DEFAULT_OUTPUT_DIR, tool_name, run_id, scenario_name)
        overlay_path = _resolve_output_path(output_overlay_path, fallback_dir=run_dir, filename="hazard_overlay.csv")
        disruptions_path = _resolve_output_path(output_disruptions_path, fallback_dir=run_dir, filename="disruptions.json")
        _write_csv(overlay_path, overlay_rows)
        disruptions_payload = {
            "edge_closures": closures,
            "speed_reductions": speed_reductions,
            "seed": seed,
            "scenario_name": scenario_name,
        }
        _write_json(disruptions_path, disruptions_payload)

        outputs = [
            _artifact_record(overlay_path, "Hazard intensity overlaid on edges"),
            _artifact_record(disruptions_path, "Deterministic disruptions for SUMO-style simulation"),
        ]
        mean_speed_factor = statistics.fmean(
            row["speed_factor"] for row in overlay_rows if row["status"] != "closed"
        ) if overlay_rows else 1.0
        metrics = {
            "disrupted_edges_count": _metric(float(len(speed_reductions) + len(closures)), "count"),
            "closed_edges_count": _metric(float(len(closures)), "count"),
            "mean_speed_factor": _metric(mean_speed_factor, "ratio"),
        }
        derived = {
            "total_edges": _metric(float(len(overlay_rows)), "count"),
        }
        risk_ref = len(closures) / max(1, len(overlay_rows))
        agentic_summary = {
            "headline": f"Hazard overlay produced {len(closures)} closures and {len(speed_reductions)} speed reductions.",
            "primary_signal": {"metric": "closed_edges_count", **metrics["closed_edges_count"]},
            "risk_level": _risk_level(risk_ref, low=0.05, high=0.15),
            "comparison_ready": True,
            "recommended_next_tools": ["impact_sumo_run", "impact_climada_run"],
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


__all__ = ["hazard_overlay_osm"]
