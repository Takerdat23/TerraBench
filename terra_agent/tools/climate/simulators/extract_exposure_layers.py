"""
extract_exposure_layers simulator tool.
"""

from .common import *  # noqa: F401,F403

def extract_exposure_layers(
    osm_assets_path: str,
    *,
    sentinel_stats_path: str | None = None,
    population_path: str | None = None,
    run_id: str = "run-exposure",
    instance_id: str = "instance-001",
    scenario_name: str = "baseline",
    scenario_description: str = "",
    assumptions: dict[str, Any] | None = None,
    seed: int = 17,
    model_version: str = "oracle:extract_exposure_layers@1.0.0",
    container_image: str | None = None,
    threads: int = 1,
    cpu_cores: int = 1,
    memory_gb: float = 1.0,
    timeout_seconds: int = 120,
    default_asset_value_usd: float = 100000.0,
    output_path: str = "./data/simulators/exposure_layers.csv",
) -> ToolResponse:
    """
    Build deterministic exposure layers from OSM-like assets and optional ancillary data.
    """

    tool_name = "extract_exposure_layers"
    start_ts = time.perf_counter()
    input_paths: list[tuple[str, str]] = [(osm_assets_path, "OSM-derived assets/POIs")]
    if sentinel_stats_path:
        input_paths.append((sentinel_stats_path, "Sentinel-2 derived statistics"))
    if population_path:
        input_paths.append((population_path, "Population exposure source"))

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
        inputs=input_paths,
    )

    warnings: list[str] = []
    try:
        assets = _load_rows(osm_assets_path)
        if not assets:
            return _error(tool_name, "Asset input is empty.", start_ts=start_ts, run_context=run_context)
        exposure_rows: list[dict[str, Any]] = []
        total_asset_value = 0.0
        for idx, row in enumerate(assets):
            asset_id = row.get("asset_id") or row.get("id") or row.get("osm_id") or f"asset_{idx + 1}"
            asset_type = str(row.get("asset_type") or row.get("type") or "generic")
            asset_value = _as_float(
                row.get("value") or row.get("asset_value_usd") or row.get("replacement_cost_usd"),
                default_asset_value_usd,
            )
            total_asset_value += asset_value
            exposure_rows.append(
                {
                    "exposure_id": str(asset_id),
                    "asset_type": asset_type,
                    "value": round(asset_value, 2),
                    "source": "osm",
                }
            )

        cropland_fraction = 0.5
        urban_fraction = 0.3
        if sentinel_stats_path:
            sentinel_rows = _load_rows(sentinel_stats_path)
            if sentinel_rows:
                crop_values = [
                    _as_float(row.get("crop_fraction") or row.get("cropland_fraction") or row.get("ndvi"), float("nan"))
                    for row in sentinel_rows
                ]
                crop_values = [value for value in crop_values if math.isfinite(value)]
                if crop_values:
                    raw = statistics.fmean(crop_values)
                    cropland_fraction = _clamp(raw if raw <= 1.0 else raw / 100.0, 0.0, 1.0)
                urban_values = [_as_float(row.get("urban_fraction"), float("nan")) for row in sentinel_rows]
                urban_values = [value for value in urban_values if math.isfinite(value)]
                if urban_values:
                    raw_u = statistics.fmean(urban_values)
                    urban_fraction = _clamp(raw_u if raw_u <= 1.0 else raw_u / 100.0, 0.0, 1.0)

        exposed_population = 0.0
        if population_path:
            pop_rows = _load_rows(population_path)
            if not pop_rows:
                warnings.append("Population source provided but empty; falling back to zero population.")
            else:
                for row in pop_rows:
                    exposed_population += _as_float(row.get("population") or row.get("value") or row.get("count"), 0.0)

        run_dir = _tool_run_dir(DEFAULT_OUTPUT_DIR, tool_name, run_id, scenario_name)
        exposure_path = _resolve_output_path(output_path, fallback_dir=run_dir, filename="exposure_layers.csv")
        _write_csv(exposure_path, exposure_rows)
        summary_path = run_dir / "exposure_summary.json"
        summary_payload = {
            "total_asset_value_usd": round(total_asset_value, 2),
            "asset_count": len(exposure_rows),
            "cropland_fraction": round(cropland_fraction, 5),
            "urban_fraction": round(urban_fraction, 5),
            "exposed_population": round(exposed_population, 2),
        }
        _write_json(summary_path, summary_payload)
        outputs = [
            _artifact_record(exposure_path, "Exposure layer rows assembled from assets"),
            _artifact_record(summary_path, "Exposure summary statistics"),
        ]
        metrics = {
            "total_asset_value_usd": _metric(total_asset_value, "USD"),
            "exposed_population": _metric(exposed_population, "people"),
            "cropland_fraction": _metric(cropland_fraction, "ratio"),
        }
        derived = {
            "asset_count": _metric(float(len(exposure_rows)), "count"),
            "urban_fraction": _metric(urban_fraction, "ratio"),
        }
        agentic_summary = {
            "headline": "Constructed exposure layers for downstream hazard-impact simulation.",
            "primary_signal": {"metric": "total_asset_value_usd", **metrics["total_asset_value_usd"]},
            "risk_level": "none",
            "comparison_ready": True,
            "recommended_next_tools": ["impact_climada_run", "impact_sumo_run", "impact_health_erf"],
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


__all__ = ["extract_exposure_layers"]
