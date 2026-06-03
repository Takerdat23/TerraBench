"""TerraAgent wrappers for Google Earth Engine satellite tools."""

from __future__ import annotations

from typing import Any, Callable

from terra_agent.tools.base import BaseTool, ToolResult
from terra_agent.tools.registry import register_tool


@register_tool
class GeeFetchSentinel2IndicesTool(BaseTool):
    name = "gee_fetch_sentinel2_indices"
    group = "satellite"
    description = "Fetch Sentinel-2 indices from Google Earth Engine for an AOI/time range."
    requires_credentials = False
    deterministic = False

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        _require_bbox_dates(inputs)

    def run(self, inputs: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        self.validate_inputs(inputs)
        func = _load_gee_function("gee_fetch_sentinel2_indices")
        payload = _filter_kwargs(
            inputs,
            {
                "bbox",
                "start_date",
                "end_date",
                "max_cloud",
                "indices",
                "scale",
                "dimensions",
                "output_dir",
                "project_id",
                "include_truecolor",
                "include_geotiff",
                "geotiff_scale",
            },
        )
        return _to_tool_result(func(**payload), default_summary="GEE Sentinel-2 indices fetched.")


@register_tool
class GeeIndexAreaStatsTool(BaseTool):
    name = "gee_index_area_stats"
    group = "satellite"
    description = "Compute Sentinel-2 index-based area fractions over an AOI."
    requires_credentials = False
    deterministic = False

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        _require_bbox_dates(inputs)

    def run(self, inputs: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        self.validate_inputs(inputs)
        func = _load_gee_function("gee_index_area_stats")
        payload = _filter_kwargs(
            inputs,
            {
                "bbox",
                "start_date",
                "end_date",
                "max_cloud",
                "scale",
                "class_thresholds",
                "project_id",
            },
        )
        return _to_tool_result(func(**payload), default_summary="GEE index area stats computed.")


@register_tool
class GeeFetchTruecolorThumbnailTool(BaseTool):
    name = "gee_fetch_truecolor_thumbnail"
    group = "satellite"
    description = "Fetch a Sentinel-2 truecolor thumbnail and optional GeoTIFF from GEE."
    requires_credentials = False
    deterministic = False

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        _require_bbox_dates(inputs)

    def run(self, inputs: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        self.validate_inputs(inputs)
        func = _load_gee_function("gee_fetch_truecolor_thumbnail")
        payload = _filter_kwargs(
            inputs,
            {
                "bbox",
                "start_date",
                "end_date",
                "max_cloud",
                "dimensions",
                "output_dir",
                "overlay_geometries",
                "draw_aoi_outline",
                "overlay_color",
                "overlay_width",
                "overlay_opacity",
                "overlay_fill",
                "project_id",
                "include_geotiff",
                "geotiff_scale",
            },
        )
        return _to_tool_result(func(**payload), default_summary="GEE truecolor thumbnail fetched.")


@register_tool
class CompositeSatelliteEnvironmentalImageTool(BaseTool):
    name = "composite_satellite_environmental_image"
    group = "satellite"
    description = "Blend a GEE satellite GeoTIFF with an environmental NetCDF field."
    requires_credentials = False
    deterministic = True

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        if "satellite_image" not in inputs:
            raise ValueError("composite_satellite_environmental_image requires 'satellite_image'.")
        anomaly_path = inputs.get("anomaly_path")
        if not isinstance(anomaly_path, str) or not anomaly_path.strip():
            raise ValueError("composite_satellite_environmental_image requires 'anomaly_path'.")

    def run(self, inputs: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        self.validate_inputs(inputs)
        func = _load_gee_function("composite_satellite_environmental_image")
        payload = _filter_kwargs(
            inputs,
            {
                "satellite_image",
                "anomaly_path",
                "anomaly_variable",
                "anomaly_time_index",
                "anomaly_aggregate",
                "mode",
                "right_field_path",
                "right_field_variable",
                "right_field_time_index",
                "right_field_aggregate",
                "right_label",
                "output_path",
                "overlay_alpha",
                "overlay_cmap",
                "overlay_center_zero",
                "overlay_vmin",
                "overlay_vmax",
                "right_cmap",
                "right_vmin",
                "right_vmax",
            },
        )
        return _to_tool_result(func(**payload), default_summary="Satellite/environment composite created.")


def _load_gee_function(name: str) -> Callable[..., Any]:
    from terra_agent.tools.satellite import gee_satellite_tools

    return getattr(gee_satellite_tools, name)


def _require_bbox_dates(inputs: dict[str, Any]) -> None:
    bbox = inputs.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("GEE satellite tools require bbox=[north, west, south, east].")
    for key in ("start_date", "end_date"):
        value = inputs.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"GEE satellite tools require '{key}' as YYYY-MM-DD.")


def _filter_kwargs(inputs: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    return {key: value for key, value in inputs.items() if key in allowed and value is not None}


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
    summary = text_blocks[0] if text_blocks else metadata.get("message") or default_summary
    return ToolResult(status=status, summary=summary, data=metadata, metadata=metadata)
