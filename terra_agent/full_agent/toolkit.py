"""AgentScope toolkit construction for the full TerraBench agent."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable

from agentscope.tool import Toolkit

from terra_agent.full_agent.agent_types import _MathAgentStopState
from terra_agent.full_agent.utils import log_tool_errors, make_http_tool
from terra_agent.tools.climate.fetch_era5 import fetch_era5
from terra_agent.tools.climate.aurora_tool import run_aurora_forecast
from terra_agent.tools.climate.math_agent_tool import run_math_agent
from terra_agent.tools.climate.math_agent_data_stream import stream_data_to_math_agent
from terra_agent.tools.climate.pangu_tool import run_pangu_forecast
from terra_agent.tools.climate.temporal_aggregate import temporal_aggregate
from terra_agent.tools.climate.diagnostic_tools import (
    winds_basic,
    humidity_basic,
    moisture_integrals,
    kinematics,
    vertical_shear,
    region_reduce,
)
from terra_agent.tools.climate.netcdf_tools import inspect_netcdf, read_netcdf_data
from terra_agent.tools.climate.seasonal_core import (
    fetch_c3s_seasonal,
    progressive_cds_constraints,
    standardize_seasonal_files,
    aggregate_to_season,
    fit_bias_correction,
    apply_bias_correction,
    category_probabilities,
    blend_multi_model,
    forecast_anomaly_from_model_climo,
    compute_enso_indices,
    region_mask as seasonal_region_mask,
    seasonal_region_table,
)
from terra_agent.tools.climate.historical_stats import build_era5_climatology, count_era5_events
from terra_agent.tools.climate.map_rendering import (
    plot_region_map,
    plot_ensemble_mean_map,
    plot_spread_map,
    plot_confidence_map,
    plot_spaghetti_map,
    plot_tercile_map,
    plot_anomaly_map,
    plot_skill_map,
)
from terra_agent.tools.climate.preprocess_era5 import describe_netcdf_variables
from terra_agent.tools.climate.osm_tools import (
    gis_resolve_region,
    gis_feature_query,
    gis_feature_stats,
    gis_distance_matrix,
    gis_overlay,
    gis_source_metadata,
)
from terra_agent.tools.climate.web_tools import web_search_serper, summarize_web_page
from terra_agent.tools.climate.data_crawl import CMIP6 as crawl_cmip6
from terra_agent.tools.climate.data_crawl import ERA5 as crawl_era5
from terra_agent.tools.climate.data_crawl import MeteoAQ as crawl_meteo_aq
from terra_agent.tools.climate.data_crawl import MSWEP as crawl_mswep
from terra_agent.tools.climate.gee_satellite_tools import (
    composite_satellite_environmental_image,
    gee_fetch_sentinel2_indices,
    gee_index_area_stats,
    gee_fetch_truecolor_thumbnail,
)
from terra_agent.tools.climate.image_caption_tool import describe_image_vlm
from terra_agent.tools.climate.ensemble_toolkit import (
    harmonize_grid,
    anomaly_and_climo,
    smooth_gaussian_field,
    aggregate_ensemble,
    ensemble_spread,
    normalized_spread,
    event_probability,
    emos_calibration,
    confidence_index,
    lagged_ic_ensemble,
    ehf_heatwave,
    verify,
)
from terra_agent.tools.climate.regrid_tools import regrid_to_target_grid
from terra_agent.tools.climate.tc_tracks_from_era5 import run_tc_tracks_from_era5
from terra_agent.tools.climate.image_overlay_tool import draw_geometries_on_image
from terra_agent.tools.climate.simulators import (
    impact_aquacrop_run,
    impact_dssat_run,
    impact_climada_run,
    impact_health_utci,
    impact_health_erf,
    impact_energyplus_run,
    impact_sumo_run,
    weather_monthly_to_daily,
    weather_to_epw,
    hazard_overlay_osm,
    extract_exposure_layers,
)


def _wrap_math_agent_runner(
    *,
    agent_data_dir: Path | None = None,
    stop_state: _MathAgentStopState | None = None,
) -> Callable[..., Any]:
    @functools.wraps(run_math_agent)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if agent_data_dir and ("artifact_dir" not in kwargs or not kwargs["artifact_dir"]):
            kwargs["artifact_dir"] = str(agent_data_dir)
        response = run_math_agent(*args, **kwargs)
        if stop_state:
            stop_state.capture(response)
        return response

    return wrapper


def create_toolkit(
    tool_mode: str,
    tool_base_url: str,
    *,
    agent_data_dir: Path | None = None,
    math_agent_stop_state: _MathAgentStopState | None = None,
) -> Toolkit:
    """Register agent tools using either local implementations or HTTP proxies."""
    toolkit = Toolkit()

    math_agent_runner = run_math_agent
    if agent_data_dir or (math_agent_stop_state and math_agent_stop_state.enabled):
        math_agent_runner = _wrap_math_agent_runner(
            agent_data_dir=agent_data_dir,
            stop_state=math_agent_stop_state,
        )

    basic_tools: tuple[Callable[..., Any], ...] = (
        gis_resolve_region,
        web_search_serper,
        summarize_web_page,
        stream_data_to_math_agent,
        math_agent_runner,
    )

    core_tools: tuple[Callable[..., Any], ...] = (
        describe_netcdf_variables,
        inspect_netcdf,
        read_netcdf_data,
    )

    era5_obs_tools: tuple[Callable[..., Any], ...] = (
        fetch_era5,
        build_era5_climatology,
        count_era5_events,
        temporal_aggregate,
        winds_basic,
        humidity_basic,
        moisture_integrals,
        kinematics,
        vertical_shear,
        region_reduce,
    )

    air_quality_tools: tuple[Callable[..., Any], ...] = (
        crawl_meteo_aq,
    )

    ensemble_verify_tools: tuple[Callable[..., Any], ...] = (
        harmonize_grid,
        anomaly_and_climo,
        smooth_gaussian_field,
        aggregate_ensemble,
        ensemble_spread,
        normalized_spread,
        event_probability,
        emos_calibration,
        confidence_index,
        lagged_ic_ensemble,
        ehf_heatwave,
        verify,
    )

    seasonal_tools: tuple[Callable[..., Any], ...] = (
        fetch_c3s_seasonal,
        progressive_cds_constraints,
        standardize_seasonal_files,
        aggregate_to_season,
        fit_bias_correction,
        apply_bias_correction,
        category_probabilities,
        blend_multi_model,
        forecast_anomaly_from_model_climo,
        compute_enso_indices,
        seasonal_region_mask,
        seasonal_region_table,
    )

    geo_osm_tools: tuple[Callable[..., Any], ...] = (
        gis_feature_query,
        gis_feature_stats,
        gis_distance_matrix,
        gis_overlay,
        gis_source_metadata,
    )

    satellite_tools: tuple[Callable[..., Any], ...] = (
        composite_satellite_environmental_image,
        gee_fetch_sentinel2_indices,
        gee_index_area_stats,
        gee_fetch_truecolor_thumbnail,
        describe_image_vlm,
        draw_geometries_on_image,
    )

    visualization_tools: tuple[Callable[..., Any], ...] = (
        plot_region_map,
        plot_ensemble_mean_map,
        plot_spread_map,
        plot_confidence_map,
        plot_spaghetti_map,
        plot_tercile_map,
        plot_anomaly_map,
        plot_skill_map,
    )

    forecast_tools: tuple[tuple[str | None, Callable[..., Any]], ...] = (
        ("aurora", run_aurora_forecast),
        ("pangu", run_pangu_forecast),
        (None, regrid_to_target_grid),
        (None, run_tc_tracks_from_era5),
    )

    crawl_helpers = (
        crawl_cmip6,
        crawl_mswep,
    )

    simulator_tools: tuple[Callable[..., Any], ...] = (
        impact_aquacrop_run,
        impact_dssat_run,
        impact_climada_run,
        impact_health_utci,
        impact_health_erf,
        impact_energyplus_run,
        impact_sumo_run,
        weather_monthly_to_daily,
        weather_to_epw,
        hazard_overlay_osm,
        extract_exposure_layers,
    )

    toolkit.create_tool_group(
        group_name="core",
        description="Core data inspection tools for local files and NetCDF structure.",
        notes=(
            "Use these tools to inspect files, variables, dimensions, and sample values before "
            "requesting domain-specific analysis."
        ),
    )
    toolkit.create_tool_group(
        group_name="era5_obs",
        description="ERA5 retrieval plus observation-side meteorological diagnostics and reductions.",
        notes=(
            "Activate this group when you need ERA5 data retrieval, climatologies, event counts, "
            "or atmospheric diagnostics over observed/reanalysis data."
        ),
    )
    toolkit.create_tool_group(
        group_name="forecast_models",
        description="Numerical forecast model execution and forecast-grid preparation.",
        notes=(
            "Activate this group for Aurora, Pangu, regridding forecast outputs, or tropical-cyclone "
            "track derivation from model data."
        ),
    )
    toolkit.create_tool_group(
        group_name="air_quality",
        description="Air-quality retrieval tools for pollutant time series and AQI sampling.",
        notes=(
            "Activate this group when you need PM2.5, PM10, NO2, O3, or related air-quality time series. "
            "Use `MeteoAQ` here for Open-Meteo air-quality retrieval at a point location."
        ),
    )
    toolkit.create_tool_group(
        group_name="ensemble_verify",
        description="Ensemble statistics, calibration, event probability, and verification utilities.",
        notes=(
            "Activate this group for ensemble post-processing, spread/probability metrics, calibration, "
            "heatwave indices, or forecast verification."
        ),
    )
    toolkit.create_tool_group(
        group_name="seasonal",
        description="Seasonal forecast retrieval, bias correction, anomalies, and ENSO-related utilities.",
        notes=(
            "Activate this group for C3S seasonal workflows, seasonal aggregation, bias correction, "
            "tercile/category probabilities, and ENSO context."
        ),
    )
    toolkit.create_tool_group(
        group_name="geo_osm",
        description="Geospatial feature lookup, overlay, statistics, and OSM-derived context.",
        notes=(
            "Activate this group for infrastructure/population feature queries, geometry overlays, "
            "distance calculations, and source metadata."
        ),
    )
    toolkit.create_tool_group(
        group_name="satellite",
        description="Remote-sensing composites, Earth Engine imagery, and image interpretation tools.",
        notes=(
            "Activate this group for Sentinel-2 products, area statistics, environmental image composites, "
            "thumbnail generation, and visual interpretation."
        ),
    )
    toolkit.create_tool_group(
        group_name="visualization",
        description="Map rendering and forecast/analysis visualization tools.",
        notes=(
            "Activate this group only when you need figures or maps; keep it off during pure data "
            "collection and numeric analysis to conserve context."
        ),
    )
    toolkit.create_tool_group(
        group_name="data_crawl",
        description="External dataset discovery and crawl helpers.",
        notes=(
            "Activate this group when the needed source data is not already available locally and you "
            "need dataset crawling or source discovery."
        ),
    )
    toolkit.create_tool_group(
        group_name="simulators",
        description="Deterministic sectoral impact simulators and weather/exposure glue tools.",
        notes=(
            "Use these tools for counterfactual impact estimates. "
            "Each output is deterministic and artifact-backed; compare baseline vs "
            "counterfactual using matched run settings."
        ),
    )

    def register(
        tool: Callable[..., Any],
        *,
        group_name: str = "basic",
    ) -> None:
        #Set to false if want to save tokens
        toolkit.register_tool_function(
            log_tool_errors(tool),
            group_name=group_name,
            include_long_description=True,
        )

    for tool in basic_tools:
        register(tool)

    for tool in core_tools:
        register(tool, group_name="core")

    for tool in era5_obs_tools:
        register(tool, group_name="era5_obs")

    for tool in air_quality_tools:
        register(tool, group_name="air_quality")

    for tool in ensemble_verify_tools:
        register(tool, group_name="ensemble_verify")

    for tool in seasonal_tools:
        register(tool, group_name="seasonal")

    for tool in geo_osm_tools:
        register(tool, group_name="geo_osm")

    for tool in satellite_tools:
        register(tool, group_name="satellite")

    for tool in visualization_tools:
        register(tool, group_name="visualization")

    for tool_name, tool in forecast_tools:
        if tool_mode.lower() == "http" and tool_name:
            remote = make_http_tool(tool, tool_name=tool_name, base_url=tool_base_url)
            register(remote, group_name="forecast_models")
        else:
            register(tool, group_name="forecast_models")

    for tool in crawl_helpers:
        register(tool, group_name="data_crawl")

    for tool in simulator_tools:
        register(tool, group_name="simulators")

    return toolkit
