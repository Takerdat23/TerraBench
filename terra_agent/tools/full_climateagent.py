"""Lazy TerraAgent adapters for the full ClimateAgent-derived tool surface."""

from __future__ import annotations

import importlib
import json
from typing import Any

from terra_agent.tools.base import BaseTool, ToolArtifact, ToolResult
from terra_agent.tools.registry import register_tool


ToolSpec = tuple[str, str, str, str]


TOOL_SPECS: tuple[ToolSpec, ...] = (
    # Basic/runtime tools.
    ("gis_resolve_region", "terra_agent.tools.climate.osm_tools", "gis_resolve_region", "geo_osm"),
    ("stream_data_to_math_agent", "terra_agent.tools.climate.math_agent_data_stream", "stream_data_to_math_agent", "basic"),
    ("run_math_agent", "terra_agent.tools.climate.math_agent_tool", "run_math_agent", "basic"),
    ("crawl_web_site", "terra_agent.tools.climate.web_tools", "crawl_web_site", "web_search"),
    # Core data inspection.
    ("describe_netcdf_variables", "terra_agent.tools.climate.preprocess_era5", "describe_netcdf_variables", "core"),
    ("inspect_netcdf", "terra_agent.tools.climate.netcdf_tools", "inspect_netcdf", "core"),
    ("read_netcdf_data", "terra_agent.tools.climate.netcdf_tools", "read_netcdf_data", "core"),
    # Reanalysis, preprocessing, and diagnostics.
    ("fetch_era5", "terra_agent.tools.climate.fetch_era5", "fetch_era5", "era5_obs"),
    ("preprocess_era5", "terra_agent.tools.climate.preprocess_era5", "preprocess_era5", "era5_obs"),
    ("build_era5_climatology", "terra_agent.tools.climate.historical_stats", "build_era5_climatology", "era5_obs"),
    ("count_era5_events", "terra_agent.tools.climate.historical_stats", "count_era5_events", "era5_obs"),
    ("temporal_aggregate", "terra_agent.tools.climate.temporal_aggregate", "temporal_aggregate", "era5_obs"),
    ("winds_basic", "terra_agent.tools.climate.diagnostic_tools", "winds_basic", "era5_obs"),
    ("humidity_basic", "terra_agent.tools.climate.diagnostic_tools", "humidity_basic", "era5_obs"),
    ("moisture_integrals", "terra_agent.tools.climate.diagnostic_tools", "moisture_integrals", "era5_obs"),
    ("kinematics", "terra_agent.tools.climate.diagnostic_tools", "kinematics", "era5_obs"),
    ("vertical_shear", "terra_agent.tools.climate.diagnostic_tools", "vertical_shear", "era5_obs"),
    ("region_reduce", "terra_agent.tools.climate.diagnostic_tools", "region_reduce", "era5_obs"),
    # Data crawl helpers. Aliases are included because older prompts mention them.
    ("MeteoAQ", "terra_agent.tools.climate.data_crawl", "MeteoAQ", "air_quality"),
    ("crawl_meteo_aq", "terra_agent.tools.climate.data_crawl", "MeteoAQ", "air_quality"),
    ("ERA5", "terra_agent.tools.climate.data_crawl", "ERA5", "data_crawl"),
    ("crawl_era5", "terra_agent.tools.climate.data_crawl", "ERA5", "data_crawl"),
    ("CMIP6", "terra_agent.tools.climate.data_crawl", "CMIP6", "data_crawl"),
    ("crawl_cmip6", "terra_agent.tools.climate.data_crawl", "CMIP6", "data_crawl"),
    ("MSWEP", "terra_agent.tools.climate.data_crawl", "MSWEP", "data_crawl"),
    ("crawl_mswep", "terra_agent.tools.climate.data_crawl", "MSWEP", "data_crawl"),
    ("fetch_cmip6", "terra_agent.tools.climate.fetch_cmip6", "fetch_cmip6", "data_crawl"),
    # Ensemble, calibration, events, and verification.
    ("harmonize_grid", "terra_agent.tools.climate.ensemble_toolkit", "harmonize_grid", "ensemble_verify"),
    ("anomaly_and_climo", "terra_agent.tools.climate.ensemble_toolkit", "anomaly_and_climo", "ensemble_verify"),
    ("smooth_gaussian_field", "terra_agent.tools.climate.ensemble_toolkit", "smooth_gaussian_field", "ensemble_verify"),
    ("aggregate_ensemble", "terra_agent.tools.climate.ensemble_toolkit", "aggregate_ensemble", "ensemble_verify"),
    ("ensemble_spread", "terra_agent.tools.climate.ensemble_toolkit", "ensemble_spread", "ensemble_verify"),
    ("normalized_spread", "terra_agent.tools.climate.ensemble_toolkit", "normalized_spread", "ensemble_verify"),
    ("event_probability", "terra_agent.tools.climate.ensemble_toolkit", "event_probability", "ensemble_verify"),
    ("emos_calibration", "terra_agent.tools.climate.ensemble_toolkit", "emos_calibration", "ensemble_verify"),
    ("confidence_index", "terra_agent.tools.climate.ensemble_toolkit", "confidence_index", "ensemble_verify"),
    ("lagged_ic_ensemble", "terra_agent.tools.climate.ensemble_toolkit", "lagged_ic_ensemble", "ensemble_verify"),
    ("derive_event_mask", "terra_agent.tools.climate.ensemble_toolkit", "derive_event_mask", "ensemble_verify"),
    ("ehf_heatwave", "terra_agent.tools.climate.ensemble_toolkit", "ehf_heatwave", "ensemble_verify"),
    ("verify", "terra_agent.tools.climate.ensemble_toolkit", "verify", "ensemble_verify"),
    ("bias_correct_cmip6", "terra_agent.tools.climate.ensemble_toolkit", "bias_correct_cmip6", "ensemble_verify"),
    ("scenario_exceedance", "terra_agent.tools.climate.ensemble_toolkit", "scenario_exceedance", "ensemble_verify"),
    # Seasonal forecast utilities.
    ("fetch_c3s_seasonal", "terra_agent.tools.climate.seasonal_core", "fetch_c3s_seasonal", "seasonal"),
    ("progressive_cds_constraints", "terra_agent.tools.climate.seasonal_core", "progressive_cds_constraints", "seasonal"),
    ("standardize_seasonal_files", "terra_agent.tools.climate.seasonal_core", "standardize_seasonal_files", "seasonal"),
    ("aggregate_to_season", "terra_agent.tools.climate.seasonal_core", "aggregate_to_season", "seasonal"),
    ("fit_bias_correction", "terra_agent.tools.climate.seasonal_core", "fit_bias_correction", "seasonal"),
    ("apply_bias_correction", "terra_agent.tools.climate.seasonal_core", "apply_bias_correction", "seasonal"),
    ("category_probabilities", "terra_agent.tools.climate.seasonal_core", "category_probabilities", "seasonal"),
    ("blend_multi_model", "terra_agent.tools.climate.seasonal_core", "blend_multi_model", "seasonal"),
    ("forecast_anomaly_from_model_climo", "terra_agent.tools.climate.seasonal_core", "forecast_anomaly_from_model_climo", "seasonal"),
    ("compute_enso_indices", "terra_agent.tools.climate.seasonal_core", "compute_enso_indices", "seasonal"),
    ("region_mask", "terra_agent.tools.climate.seasonal_core", "region_mask", "seasonal"),
    ("seasonal_region_mask", "terra_agent.tools.climate.seasonal_core", "region_mask", "seasonal"),
    ("seasonal_region_table", "terra_agent.tools.climate.seasonal_core", "seasonal_region_table", "seasonal"),
    # GIS and routing.
    ("gis_feature_query", "terra_agent.tools.climate.osm_tools", "gis_feature_query", "geo_osm"),
    ("gis_feature_stats", "terra_agent.tools.climate.osm_tools", "gis_feature_stats", "geo_osm"),
    ("gis_distance_matrix", "terra_agent.tools.climate.osm_tools", "gis_distance_matrix", "geo_osm"),
    ("gis_overlay", "terra_agent.tools.climate.osm_tools", "gis_overlay", "geo_osm"),
    ("gis_source_metadata", "terra_agent.tools.climate.osm_tools", "gis_source_metadata", "geo_osm"),
    ("overpass_poi_search", "terra_agent.tools.climate.osm_tools", "overpass_poi_search", "geo_osm"),
    ("osm_travel_time_matrix", "terra_agent.tools.climate.routing_tools", "osm_travel_time_matrix", "geo_osm"),
    # Satellite and image interpretation. GEE tools are native too; these names are
    # registered there and intentionally not duplicated here.
    ("describe_image_vlm", "terra_agent.tools.climate.image_caption_tool", "describe_image_vlm", "satellite"),
    ("draw_geometries_on_image", "terra_agent.tools.climate.image_overlay_tool", "draw_geometries_on_image", "satellite"),
    # Visualization.
    ("plot_era5_map", "terra_agent.tools.climate.map_rendering", "plot_era5_map", "visualization"),
    ("plot_world_map", "terra_agent.tools.climate.map_rendering", "plot_world_map", "visualization"),
    ("plot_region_map", "terra_agent.tools.climate.map_rendering", "plot_region_map", "visualization"),
    ("plot_ensemble_mean_map", "terra_agent.tools.climate.map_rendering", "plot_ensemble_mean_map", "visualization"),
    ("plot_spread_map", "terra_agent.tools.climate.map_rendering", "plot_spread_map", "visualization"),
    ("plot_confidence_map", "terra_agent.tools.climate.map_rendering", "plot_confidence_map", "visualization"),
    ("plot_spaghetti_map", "terra_agent.tools.climate.map_rendering", "plot_spaghetti_map", "visualization"),
    ("plot_tercile_map", "terra_agent.tools.climate.map_rendering", "plot_tercile_map", "visualization"),
    ("plot_anomaly_map", "terra_agent.tools.climate.map_rendering", "plot_anomaly_map", "visualization"),
    ("plot_skill_map", "terra_agent.tools.climate.map_rendering", "plot_skill_map", "visualization"),
    # Forecast model execution.
    ("run_aurora_forecast", "terra_agent.tools.climate.aurora_tool", "run_aurora_forecast", "forecast_models"),
    ("run_pangu_forecast", "terra_agent.tools.climate.pangu_tool", "run_pangu_forecast", "forecast_models"),
    ("regrid_to_target_grid", "terra_agent.tools.climate.regrid_tools", "regrid_to_target_grid", "forecast_models"),
    ("run_tc_tracks_from_era5", "terra_agent.tools.climate.tc_tracks_from_era5", "run_tc_tracks_from_era5", "forecast_models"),
    # Simulator adapters.
    ("impact_aquacrop_run", "terra_agent.tools.climate.simulators", "impact_aquacrop_run", "simulators"),
    ("impact_dssat_run", "terra_agent.tools.climate.simulators", "impact_dssat_run", "simulators"),
    ("impact_climada_run", "terra_agent.tools.climate.simulators", "impact_climada_run", "simulators"),
    ("impact_health_utci", "terra_agent.tools.climate.simulators", "impact_health_utci", "simulators"),
    ("impact_health_erf", "terra_agent.tools.climate.simulators", "impact_health_erf", "simulators"),
    ("impact_energyplus_run", "terra_agent.tools.climate.simulators", "impact_energyplus_run", "simulators"),
    ("impact_sumo_run", "terra_agent.tools.climate.simulators", "impact_sumo_run", "simulators"),
    ("weather_monthly_to_daily", "terra_agent.tools.climate.simulators", "weather_monthly_to_daily", "simulators"),
    ("weather_to_epw", "terra_agent.tools.climate.simulators", "weather_to_epw", "simulators"),
    ("hazard_overlay_osm", "terra_agent.tools.climate.simulators", "hazard_overlay_osm", "simulators"),
    ("extract_exposure_layers", "terra_agent.tools.climate.simulators", "extract_exposure_layers", "simulators"),
)


class ClimateAgentCompatTool(BaseTool):
    module_name: str = ""
    attr_name: str = ""
    deterministic = False

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        if not isinstance(inputs, dict):
            raise ValueError(f"{self.name} inputs must be a JSON object.")

    def run(self, inputs: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        self.validate_inputs(inputs)
        try:
            module = importlib.import_module(self.module_name)
            func = getattr(module, self.attr_name)
        except Exception as exc:
            return ToolResult(
                status="error",
                summary=f"{self.name} is unavailable: {exc}",
                metadata={
                    "module": self.module_name,
                    "attribute": self.attr_name,
                    "error": True,
                },
            )

        try:
            response = func(**inputs)
        except TypeError as exc:
            return ToolResult(
                status="error",
                summary=f"{self.name} argument error: {exc}",
                metadata={"error": True, "module": self.module_name, "attribute": self.attr_name},
            )
        except Exception as exc:
            return ToolResult(
                status="error",
                summary=f"{self.name} failed: {exc}",
                metadata={"error": True, "module": self.module_name, "attribute": self.attr_name},
            )

        return _to_tool_result(response, default_summary=f"{self.name} completed.")


def _to_tool_result(response: Any, *, default_summary: str) -> ToolResult:
    metadata = dict(getattr(response, "metadata", None) or {})
    status = "error" if metadata.get("error") else "success"

    text_blocks = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            text_blocks.append(str(text))

    if text_blocks:
        summary = text_blocks[0]
    elif isinstance(response, dict):
        summary = _compact_json(response)
        metadata = response
    elif response is None:
        summary = default_summary
    else:
        summary = str(response)

    artifacts = _extract_artifacts(metadata)
    data = metadata if metadata else {"result": response}
    return ToolResult(status=status, summary=summary, data=data, artifacts=artifacts, metadata=metadata)


def _compact_json(value: Any, *, limit: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _extract_artifacts(metadata: dict[str, Any]) -> list[ToolArtifact]:
    artifacts: list[ToolArtifact] = []
    candidates = metadata.get("artifacts")
    if not isinstance(candidates, list):
        return artifacts
    for item in candidates:
        if not isinstance(item, dict):
            continue
        path = item.get("path") or item.get("saved_path") or item.get("file_path")
        if not path:
            continue
        artifacts.append(
            ToolArtifact(
                path=str(path),
                type=str(item.get("type") or item.get("kind") or "file"),
                description=item.get("description") or item.get("label"),
            )
        )
    return artifacts


def _class_name(tool_name: str) -> str:
    return "".join(part.capitalize() for part in tool_name.replace("-", "_").split("_")) + "Tool"


def _register_spec(tool_name: str, module_name: str, attr_name: str, group: str) -> None:
    cls = type(
        _class_name(tool_name),
        (ClimateAgentCompatTool,),
        {
            "name": tool_name,
            "group": group,
            "description": f"Compatibility wrapper for {module_name}.{attr_name}.",
            "module_name": module_name,
            "attr_name": attr_name,
        },
    )
    register_tool(cls)


for spec in TOOL_SPECS:
    _register_spec(*spec)

