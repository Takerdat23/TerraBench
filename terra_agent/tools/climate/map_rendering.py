"""
Map rendering utilities (consolidated).

This module groups the plotting helpers that were previously split across
``plot_era5_map.py``, ``plot_world_map.py``, ``plot_region_map.py``,
``ensemble_plot_tools.py``, and ``seasonal_plotting.py`` into a single entry
point for maps, ensemble diagnostics, and seasonal visualisations.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zipfile import ZipFile, is_zipfile

import matplotlib
import numpy as np
import xarray as xr
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:  # cartopy gives nicer coastlines if available
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:  # pragma: no cover - optional dependency
    ccrs = None
    cfeature = None

__all__ = [
    "plot_era5_map",
    "plot_world_map",
    "plot_region_map",
    "plot_ensemble_mean_map",
    "plot_spread_map",
    "plot_confidence_map",
    "plot_spaghetti_map",
    "plot_tercile_map",
    "plot_anomaly_map",
    "plot_skill_map",
    "_prepare_latlon_field",
    "_format_filename",
]


# ----- Shared helpers --------------------------------------------------------
def _ok(payload: Dict[str, Any]) -> ToolResponse:
    sanitized = _sanitize(payload)
    return ToolResponse(
        content=[TextBlock(type="text", text=json.dumps(sanitized))],
        metadata=sanitized,
    )


def _error(msg: str) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {msg}")],
        metadata={"error": True, "message": msg},
    )


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_)):
        return bool(value)
    return value


def _format_coord_value(value: Any) -> str:
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.datetime64):
        return str(arr.astype("datetime64[m]"))
    if arr.size == 1:
        return str(arr.item())
    return str(arr)


@dataclass
class LatLonField:
    data: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    var_name: str
    units: str
    time_label: str
    source_path: Path
    lon_name: str
    lat_name: str


def _coord_in_dataset(ds: xr.Dataset, name: str) -> bool:
    return name in ds.coords or name in ds.dims


def _coord_values(ds: xr.Dataset, name: str) -> Optional[np.ndarray]:
    if not _coord_in_dataset(ds, name):
        return None
    try:
        return np.asarray(ds[name].values)
    except Exception:
        return None


def _coord_units(ds: xr.Dataset, name: str) -> str:
    if not _coord_in_dataset(ds, name):
        return ""
    try:
        return str(ds[name].attrs.get("units", "")).lower()
    except Exception:
        return ""


def _looks_like_lat(values: np.ndarray, units: str) -> bool:
    if "degree" in units and "north" in units:
        return True
    if values.ndim != 1:
        return False
    if not np.isfinite(values).any():
        return False
    return float(np.nanmin(values)) >= -90.5 and float(np.nanmax(values)) <= 90.5


def _looks_like_lon(values: np.ndarray, units: str) -> bool:
    if "degree" in units and "east" in units:
        return True
    if values.ndim != 1:
        return False
    if not np.isfinite(values).any():
        return False
    return float(np.nanmin(values)) >= -180.5 and float(np.nanmax(values)) <= 180.5


def _ensure_latlon(ds: xr.Dataset) -> tuple[str, str]:
    for lat_name in ("latitude", "lat", "Latitude", "LAT"):
        if _coord_in_dataset(ds, lat_name):
            break
    else:
        lat_name = ""

    for lon_name in ("longitude", "lon", "Longitude", "LON"):
        if _coord_in_dataset(ds, lon_name):
            break
    else:
        lon_name = ""

    if lat_name and lon_name:
        return lat_name, lon_name

    # Fallback for regridded GeoTIFFs that expose x/y in degrees.
    if _coord_in_dataset(ds, "y") and _coord_in_dataset(ds, "x"):
        y_vals = _coord_values(ds, "y")
        x_vals = _coord_values(ds, "x")
        if y_vals is not None and x_vals is not None:
            if _looks_like_lat(y_vals, _coord_units(ds, "y")) and _looks_like_lon(
                x_vals, _coord_units(ds, "x")
            ):
                return "y", "x"

    raise ValueError(
        "Dataset must include a latitude/longitude coordinate. "
        "If the data uses x/y, ensure they are geographic degrees (EPSG:4326) "
        "or rename them to latitude/longitude."
    )


def _sort_lat(ds: xr.Dataset, lat_name: str) -> xr.Dataset:
    lat_vals = ds[lat_name].values
    if lat_vals.ndim == 1 and np.all(np.diff(lat_vals) < 0):
        return ds.sortby(lat_name)
    return ds


def _resolve_variable(ds: xr.Dataset, variable: Optional[str]) -> str:
    if variable:
        if variable not in ds.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset.")
        return variable
    if len(ds.data_vars) == 0:
        raise ValueError("Dataset has no data variables to plot.")
    return list(ds.data_vars)[0]


def _find_time_dim(da: xr.DataArray) -> Optional[str]:
    for dim in da.dims:
        if "time" in dim.lower():
            return dim
    return None


def _format_filename(base: Path, var_name: str, suffix: str) -> str:
    stem = base.stem
    if stem.endswith(".nc"):
        stem = stem[:-3]
    return f"{stem}_{var_name}{suffix}"


def _squeeze_time(da: xr.DataArray, time_dim: str) -> xr.DataArray:
    if da.sizes.get(time_dim, 0) == 1:
        return da.isel({time_dim: 0}, drop=True)
    return da


def _prepare_latlon_field(
    *,
    netcdf_path: str,
    variable: Optional[str],
    time_index: Optional[int],
    aggregate: Optional[str],
    auto_time_mean_threshold: Optional[int] = None,
    reduce_dims: Optional[Sequence[str]] = None,
    reduce_mode: str = "mean",
) -> LatLonField:
    path = Path(netcdf_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"NetCDF path not found: {path}")

    if is_zipfile(path):
        with ZipFile(path) as zf:
            names = [name for name in zf.namelist() if name.endswith(".nc")]
            if not names:
                raise ValueError("Zip archive does not contain a NetCDF file.")
            with zf.open(names[0]) as fh:
                ds = xr.open_dataset(fh)
    else:
        ds = xr.open_dataset(path)

    lat_name, lon_name = _ensure_latlon(ds)
    ds = _sort_lat(ds, lat_name)
    var_name = _resolve_variable(ds, variable)
    da = ds[var_name]

    time_label = "no-time-dimension"
    time_dim = _find_time_dim(da)
    if time_dim:
        time_size = da.sizes[time_dim]
        time_label = f"{time_dim} size {time_size}"
        if time_size > 0:
            if time_index is not None:
                coord = da.coords.get(time_dim)
                coord_val = coord.isel({time_dim: time_index}).values if coord is not None else None
                da = da.isel({time_dim: time_index}, drop=True)
                time_label = (
                    _format_coord_value(coord_val)
                    if coord_val is not None
                    else f"{time_dim} index {time_index}"
                )
            else:
                agg = aggregate.lower() if isinstance(aggregate, str) else None
                if agg == "auto" or aggregate is None:
                    if (
                        auto_time_mean_threshold is not None
                        and time_size > auto_time_mean_threshold
                    ):
                        da = da.mean(time_dim, keep_attrs=True)
                        time_label = f"{time_dim} mean over {time_size} steps"
                    else:
                        coord = da.coords.get(time_dim)
                        coord_val = coord.isel({time_dim: 0}).values if coord is not None else None
                        da = da.isel({time_dim: 0}, drop=True)
                        time_label = (
                            _format_coord_value(coord_val)
                            if coord_val is not None
                            else f"{time_dim} index 0"
                        )
                elif agg:
                    if agg == "mean":
                        da = da.mean(time_dim, keep_attrs=True)
                        time_label = f"{time_dim} mean"
                    elif agg == "sum":
                        da = da.sum(time_dim, keep_attrs=True)
                        time_label = f"{time_dim} sum"
                    elif agg == "auto":
                        coord = da.coords.get(time_dim)
                        coord_val = coord.isel({time_dim: 0}).values if coord is not None else None
                        da = da.isel({time_dim: 0}, drop=True)
                        time_label = (
                            _format_coord_value(coord_val)
                            if coord_val is not None
                            else f"{time_dim} index 0"
                        )
                    else:
                        raise ValueError("aggregate must be one of {'mean','sum','auto',None}.")

    for dim in list(da.dims):
        if dim not in (lat_name, lon_name) and da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)

    # Reduce any remaining non-lat/lon dims (e.g., forecastMonth, number) to avoid plotting failures.
    remaining_dims = set(da.dims)
    extras = [dim for dim in remaining_dims if dim not in (lat_name, lon_name)]
    if extras:
        dims_to_reduce = extras if reduce_dims is None else [d for d in extras if d in reduce_dims]
        if not dims_to_reduce and reduce_dims:
            # If user asked to reduce a subset but none matched, fall back to all extras.
            dims_to_reduce = extras
        for dim in dims_to_reduce:
            if reduce_mode == "sum":
                da = da.sum(dim, keep_attrs=True)
                time_label = f"{time_label}; {dim} sum"
            else:
                da = da.mean(dim, keep_attrs=True)
                time_label = f"{time_label}; {dim} mean"

    remaining_dims = set(da.dims)
    expected_dims = {lat_name, lon_name}
    if remaining_dims != expected_dims:
        missing = expected_dims - remaining_dims
        extras = remaining_dims - expected_dims
        raise ValueError(
            "Unable to reduce data to 2D lat/lon field. "
            f"Missing: {missing}, extra dims: {extras}. "
            "Specify a variable, time_index, aggregation, or reduce_dims to resolve."
        )

    da = da.transpose(lat_name, lon_name)

    if da[lon_name].ndim != 1 or da[lat_name].ndim != 1:
        raise ValueError("Latitude and longitude coordinates must be 1D.")

    da = da.load()
    lon_vals = da[lon_name].values
    lon_norm = ((lon_vals + 180.0) % 360.0) - 180.0
    if not np.allclose(lon_norm, lon_vals):
        da = da.assign_coords({lon_name: lon_norm}).sortby(lon_name)

    da = da.load()
    data = da.values
    if np.all(np.isnan(data)):
        raise ValueError("Selected field contains only NaN values.")

    units = da.attrs.get("units", "")

    return LatLonField(
        data=data,
        lon=da[lon_name].values,
        lat=da[lat_name].values,
        var_name=var_name,
        units=units,
        time_label=time_label,
        source_path=path,
        lon_name=lon_name,
        lat_name=lat_name,
    )


# ----- Core map rendering ----------------------------------------------------
def plot_era5_map(
    netcdf_path: str,
    variable: Optional[str] = None,
    time_index: Optional[int] = None,
    aggregate: Optional[str] = "mean",
    reduce_dims: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    output_dir: str = "./plots",
    filename: Optional[str] = None,
) -> ToolResponse:
    """Render an ERA5 field on a global map and save it as an image."""
    try:
        field = _prepare_latlon_field(
            netcdf_path=netcdf_path,
            variable=variable,
            time_index=time_index,
            aggregate=aggregate,
            reduce_dims=reduce_dims,
        )
    except FileNotFoundError:
        path = Path(netcdf_path).expanduser()
        return _error(f"NetCDF file not found: {path}")
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:  # pragma: no cover - safety net
        return _error(f"Unexpected failure: {exc}")

    try:
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        suffix = ""
        if time_index is not None:
            suffix = f"_t{time_index}"
        elif aggregate:
            suffix = f"_{aggregate.lower()}"

        fig_name = filename or _format_filename(field.source_path, field.var_name, suffix + ".png")
        fig_path = output_path / fig_name

        if ccrs is not None:
            proj = ccrs.Robinson()
            data_crs = ccrs.PlateCarree()
            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes(projection=proj)
            mesh = ax.pcolormesh(
                field.lon,
                field.lat,
                field.data,
                transform=data_crs,
                cmap=cmap,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.coastlines(color="black", linewidth=0.8)
            if cfeature is not None:
                ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.6)
            ax.set_global()
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            mesh = ax.pcolormesh(
                field.lon,
                field.lat,
                field.data,
                cmap=cmap,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Global map (no cartopy installed)")
            ax.set_xlim(field.lon.min(), field.lon.max())
            ax.set_ylim(field.lat.min(), field.lat.max())

        cbar = plt.colorbar(mesh, orientation="horizontal", pad=0.05, aspect=50)
        cbar.set_label(field.units)

        title_text = title or f"{field.var_name} ({field.time_label})"
        plt.title(title_text)
        plt.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        metadata = {
            "figure_path": str(fig_path),
            "variable": field.var_name,
            "time_descriptor": field.time_label,
            "aggregate": aggregate,
            "vmin": vmin,
            "vmax": vmax,
        }
        return _ok(metadata)
    except Exception as exc:  # pragma: no cover - capture plotting issues
        return _error(f"Failed to render map: {exc}")


def plot_world_map(
    netcdf_path: str,
    variable: Optional[str] = None,
    time_index: Optional[int] = None,
    aggregate: Optional[str] = "auto",
    reduce_dims: Optional[Sequence[str]] = None,
    time_mean_threshold: Optional[int] = 12,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    cmap: str = "RdBu_r",
    center_zero: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    output_dir: str = "./plots",
    filename: Optional[str] = None,
    coastline_color: str = "#555555",
    land_color: str = "#f5f5f5",
    ocean_color: str = "#e8eef2",
) -> ToolResponse:
    """Render a global map with optional time aggregation."""
    try:
        field = _prepare_latlon_field(
            netcdf_path=netcdf_path,
            variable=variable,
            time_index=time_index,
            aggregate=aggregate,
            auto_time_mean_threshold=time_mean_threshold,
            reduce_dims=reduce_dims,
        )
    except FileNotFoundError:
        path = Path(netcdf_path).expanduser()
        return _error(f"NetCDF file not found: {path}")
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:  # pragma: no cover - safety net
        return _error(f"Unexpected failure: {exc}")

    scale_vmin = vmin
    scale_vmax = vmax
    if center_zero and (scale_vmin is None or scale_vmax is None):
        max_abs = np.nanmax(np.abs(field.data))
        if np.isfinite(max_abs) and max_abs > 0:
            scale_vmin = -max_abs
            scale_vmax = max_abs

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    suffix = ""
    if time_index is not None:
        suffix = f"_t{time_index}"
    elif aggregate and aggregate.lower() not in ("auto",):
        suffix = f"_{aggregate.lower()}"

    if filename:
        fig_name = filename if Path(filename).suffix else f"{filename}.png"
    else:
        fig_name = _format_filename(field.source_path, field.var_name, suffix + ".png")
    fig_path = output_path / fig_name

    try:
        figsize = (12, 6)
        if ccrs is not None:
            fig = plt.figure(figsize=figsize)
            proj = ccrs.PlateCarree()
            ax = plt.axes(projection=proj)
            ax.set_facecolor(ocean_color)
            mesh = ax.pcolormesh(
                field.lon,
                field.lat,
                field.data,
                transform=proj,
                cmap=cmap,
                shading="auto",
                vmin=scale_vmin,
                vmax=scale_vmax,
            )
            if cfeature is not None:
                ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor="none")
                ax.add_feature(
                    cfeature.COASTLINE,
                    linewidth=0.6,
                    edgecolor=coastline_color,
                )
                ax.add_feature(
                    cfeature.BORDERS,
                    linewidth=0.4,
                    edgecolor=coastline_color,
                    alpha=0.7,
                )
            ax.set_global()
            ax.set_extent([-180, 180, -90, 90], crs=proj)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_facecolor(ocean_color)
            mesh = ax.pcolormesh(
                field.lon,
                field.lat,
                field.data,
                cmap=cmap,
                shading="auto",
                vmin=scale_vmin,
                vmax=scale_vmax,
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim(field.lon.min(), field.lon.max())
            ax.set_ylim(field.lat.min(), field.lat.max())
            for spine in ax.spines.values():
                spine.set_visible(False)

        cbar = plt.colorbar(mesh, orientation="horizontal", pad=0.05, aspect=50)
        cbar.set_label(field.units or "Units")

        if title:
            fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
        if subtitle or field.time_label:
            fig.text(0.5, 0.92, subtitle or field.time_label, ha="center", fontsize=12)
        plt.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        metadata = {
            "figure_path": str(fig_path),
            "variable": field.var_name,
            "time_descriptor": field.time_label,
            "vmin": scale_vmin,
            "vmax": scale_vmax,
        }
        return _ok(metadata)
    except Exception as exc:  # pragma: no cover
        return _error(f"plot_world_map failed: {exc}")


def plot_region_map(
    netcdf_path: str,
    *,
    variable: Optional[str] = None,
    time_index: Optional[int] = None,
    aggregate: Optional[str] = None,
    reduce_dims: Optional[Sequence[str]] = None,
    region: Sequence[float] | None = None,
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    output_dir: str = "./plots",
    filename: Optional[str] = None,
) -> ToolResponse:
    """Plot a specified region (N,W,S,E) subset of a NetCDF variable."""
    try:
        field = _prepare_latlon_field(
            netcdf_path=netcdf_path,
            variable=variable,
            time_index=time_index,
            aggregate=aggregate,
            auto_time_mean_threshold=12,
            reduce_dims=reduce_dims,
        )
    except FileNotFoundError as exc:
        return _error(str(exc))
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:  # pragma: no cover
        return _error(f"plot_region_map failed to prepare data: {exc}")

    data = field.data
    lats = field.lat
    lons = field.lon

    if region:
        north, west, south, east = region
        lat_mask = (lats >= south) & (lats <= north)
        lon_mask = (lons >= west) & (lons <= east)
        if not lat_mask.any() or not lon_mask.any():
            return _error("Region does not overlap the data grid.")
        data = data[np.ix_(lat_mask, lon_mask)]
        lats = lats[lat_mask]
        lons = lons[lon_mask]

    if not np.isfinite(data).any():
        return _error("No numeric data to plot after subsetting/reduction.")

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_name = filename if filename else _format_filename(field.source_path, field.var_name, ".png")
    if not Path(fig_name).suffix:
        fig_name = f"{fig_name}.png"
    fig_path = out_dir / fig_name

    proj = ccrs.PlateCarree() if ccrs else None
    fig_kwargs = {"figsize": (8, 6)}
    ax_kwargs = {"projection": proj} if proj else {}
    fig, ax = plt.subplots(**fig_kwargs, subplot_kw=ax_kwargs)

    mesh = ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        shading="auto",
        transform=proj,
    )

    if proj:
        ax.set_extent([float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())], crs=proj)
        ax.coastlines()
        if cfeature:
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, fraction=0.05)
    cbar.set_label(field.units or "")

    title_text = title or field.var_name
    subtitle = field.time_label
    if title_text and subtitle:
        ax.set_title(f"{title_text}\n{subtitle}", fontsize=12)
    elif title_text:
        ax.set_title(title_text, fontsize=12)
    elif subtitle:
        ax.set_title(subtitle, fontsize=12)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    return _ok(
        {
            "figure_path": str(fig_path),
            "variable": field.var_name,
            "time_descriptor": field.time_label,
            "region": region if region else [float(lats.max()), float(lons.min()), float(lats.min()), float(lons.max())],
        }
    )


# ----- Ensemble diagnostics --------------------------------------------------
@dataclass
class OverlayConfig:
    data: np.ndarray
    threshold: float
    hatch: str
    label: str
    facecolor: str = "none"


def _render_map(
    field: LatLonField,
    *,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    title: Optional[str],
    subtitle: Optional[str],
    colorbar_label: str,
    output_dir: str,
    filename: Optional[str],
    center_zero: bool,
    overlay: Optional[OverlayConfig],
    contours: Optional[Sequence[float]],
    contours_label: Optional[str],
) -> Dict[str, Any]:
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    fig_name = filename if filename else _format_filename(field.source_path, field.var_name, ".png")
    if not Path(fig_name).suffix:
        fig_name = f"{fig_name}.png"
    fig_path = output_path / fig_name

    scale_vmin = vmin
    scale_vmax = vmax
    if center_zero and (scale_vmin is None or scale_vmax is None):
        max_abs = np.nanmax(np.abs(field.data))
        if np.isfinite(max_abs) and max_abs > 0:
            scale_vmin = -max_abs
            scale_vmax = max_abs

    proj = ccrs.PlateCarree() if ccrs is not None else None
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=proj) if proj else plt.axes()

    if proj:
        ax.set_global()
        ax.set_extent([-180, 180, -90, 90], crs=proj)
        ax.set_facecolor("#e8eef2")
    else:
        ax.set_xlim(field.lon.min(), field.lon.max())
        ax.set_ylim(field.lat.min(), field.lat.max())
        ax.set_facecolor("#e8eef2")

    mesh = ax.pcolormesh(
        field.lon,
        field.lat,
        field.data,
        cmap=cmap,
        shading="auto",
        vmin=scale_vmin,
        vmax=scale_vmax,
        transform=proj,
    )

    if proj and cfeature is not None:
        ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555555")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#666666", alpha=0.7)
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        for spine in ax.spines.values():
            spine.set_visible(False)

    if overlay:
        mask = np.where(overlay.data >= overlay.threshold, 1.0, np.nan)
        levels = [0, 0.5, 1]
        ax.contourf(
            field.lon,
            field.lat,
            mask,
            levels=levels,
            hatches=[None, overlay.hatch],
            colors=[overlay.facecolor],
            transform=proj,
            alpha=0.0,
        )

    if contours:
        cs = ax.contour(
            field.lon,
            field.lat,
            field.data,
            levels=contours,
            colors="black",
            linewidths=0.6,
            transform=proj,
        )
        if contours_label:
            ax.clabel(cs, fmt="%.2f", fontsize=8)

    cbar = plt.colorbar(mesh, orientation="horizontal", pad=0.04, aspect=40)
    cbar.set_label(colorbar_label)

    top_margin = 0.92
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
        top_margin = 0.9
    if subtitle:
        fig.text(0.5, top_margin, subtitle, ha="center", va="center", fontsize=12)
        top_margin -= 0.04

    plt.tight_layout(rect=(0, 0, 1, top_margin))
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure_path": str(fig_path),
        "variable": field.var_name,
        "time_descriptor": field.time_label,
        "vmin": scale_vmin,
        "vmax": scale_vmax,
        "colorbar_label": colorbar_label,
    }
    if overlay:
        metadata["overlay"] = {
            "threshold": overlay.threshold,
            "hatch": overlay.hatch,
            "label": overlay.label,
        }
    if contours:
        metadata["contours"] = {
            "levels": list(contours),
            "label": contours_label,
        }
    return metadata


def _prepare_field(
    path: str,
    variable: Optional[str],
    time_index: Optional[int],
    aggregate: Optional[str],
    reduce_dims: Optional[Sequence[str]] = None,
) -> LatLonField:
    return _prepare_latlon_field(
        netcdf_path=path,
        variable=variable,
        time_index=time_index,
        aggregate=aggregate,
        auto_time_mean_threshold=12,
        reduce_dims=reduce_dims,
    )


def _ensure_same_grid(base: LatLonField, overlay: LatLonField) -> None:
    if base.data.shape != overlay.data.shape:
        raise ValueError("Overlay field must match the ensemble mean grid.")
    if not np.allclose(base.lat, overlay.lat) or not np.allclose(base.lon, overlay.lon):
        raise ValueError("Overlay field does not share the same lat/lon coordinates as the target field.")


def plot_ensemble_mean_map(
    netcdf_path: str,
    variable: str = "ensemble_mean",
    *,
    time_index: Optional[int] = None,
    aggregate: Optional[str] = "auto",
    reduce_dims: Optional[Sequence[str]] = None,
    cmap: str = "RdBu_r",
    center_zero: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    confidence_path: Optional[str] = None,
    confidence_variable: str = "confidence",
    confidence_threshold: float = 0.7,
    hatch_pattern: str = "///",
    title: Optional[str] = "Ensemble Mean",
    subtitle: Optional[str] = None,
    output_dir: str = "./plots",
    filename: Optional[str] = None,
) -> ToolResponse:
    """Plot the ensemble mean field with optional hatched overlay from confidence data."""
    try:
        field = _prepare_field(netcdf_path, variable, time_index, aggregate, reduce_dims)
        overlay = None
        if confidence_path:
            confidence = _prepare_field(confidence_path, confidence_variable, time_index, aggregate, reduce_dims)
            _ensure_same_grid(field, confidence)
            overlay = OverlayConfig(
                data=confidence.data,
                threshold=confidence_threshold,
                hatch=hatch_pattern,
                label=f"Confidence ≥ {confidence_threshold:.2f}",
            )
        metadata = _render_map(
            field,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=title,
            subtitle=subtitle or field.time_label,
            colorbar_label=field.units or "Units",
            output_dir=output_dir,
            filename=filename,
            center_zero=center_zero,
            overlay=overlay,
            contours=None,
            contours_label=None,
        )
        return _ok(metadata)
    except FileNotFoundError as exc:
        return _error(str(exc))
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:  # pragma: no cover
        return _error(f"plot_ensemble_mean_map failed: {exc}")


def plot_spread_map(
    netcdf_path: str,
    variable: str = "ensemble_spread",
    *,
    time_index: Optional[int] = None,
    aggregate: Optional[str] = "auto",
    reduce_dims: Optional[Sequence[str]] = None,
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    title: Optional[str] = "Ensemble Spread",
    subtitle: Optional[str] = None,
    output_dir: str = "./plots",
    filename: Optional[str] = None,
) -> ToolResponse:
    """Plot a standard deviation (spread) field with linear colorbar."""
    try:
        field = _prepare_field(netcdf_path, variable, time_index, aggregate, reduce_dims)
        metadata = _render_map(
            field,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=title,
            subtitle=subtitle or field.time_label,
            colorbar_label=field.units or "Spread",
            output_dir=output_dir,
            filename=filename,
            center_zero=False,
            overlay=None,
            contours=None,
            contours_label=None,
        )
        return _ok(metadata)
    except FileNotFoundError as exc:
        return _error(str(exc))
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:  # pragma: no cover
        return _error(f"plot_spread_map failed: {exc}")


def plot_confidence_map(
    netcdf_path: str,
    variable: str = "confidence",
    *,
    time_index: Optional[int] = None,
    aggregate: Optional[str] = "auto",
    reduce_dims: Optional[Sequence[str]] = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    contour_levels: Optional[Sequence[float]] = (0.4, 0.6, 0.8),
    title: Optional[str] = "Confidence (0–1)",
    subtitle: Optional[str] = None,
    output_dir: str = "./plots",
    filename: Optional[str] = None,
) -> ToolResponse:
    """Plot the 0–1 confidence index with optional contour lines."""
    try:
        field = _prepare_field(netcdf_path, variable, time_index, aggregate, reduce_dims)
        metadata = _render_map(
            field,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=title,
            subtitle=subtitle or field.time_label,
            colorbar_label="Confidence",
            output_dir=output_dir,
            filename=filename,
            center_zero=False,
            overlay=None,
            contours=contour_levels,
            contours_label="Confidence",
        )
        return _ok(metadata)
    except FileNotFoundError as exc:
        return _error(str(exc))
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:  # pragma: no cover
        return _error(f"plot_confidence_map failed: {exc}")


def plot_spaghetti_map(
    member_paths: Sequence[str],
    variable: str,
    *,
    member_labels: Optional[Sequence[str]] = None,
    time_index: Optional[int] = None,
    aggregate: Optional[str] = "auto",
    reduce_dims: Optional[Sequence[str]] = None,
    contour_levels: Optional[Sequence[float]] = None,
    linewidth: float = 1.0,
    alpha: float = 0.8,
    colors: Optional[Sequence[str]] = None,
    title: Optional[str] = "Ensemble Spaghetti Plot",
    subtitle: Optional[str] = None,
    output_dir: str = "./plots",
    filename: Optional[str] = None,
) -> ToolResponse:
    """Overlay each ensemble member as contour lines to visualise spatial spread."""
    try:
        if not member_paths:
            return _error("member_paths must contain at least one NetCDF path.")
        paths = [Path(p).expanduser() for p in member_paths]
        if member_labels and len(member_labels) != len(paths):
            return _error("member_labels must match the number of member_paths.")

        fields = []
        for idx, path in enumerate(paths):
            field = _prepare_field(str(path), variable, time_index, aggregate, reduce_dims)
            if idx > 0:
                _ensure_same_grid(fields[0], field)
            fields.append(field)

        global_min = min(float(np.nanmin(f.data)) for f in fields)
        global_max = max(float(np.nanmax(f.data)) for f in fields)
        if contour_levels:
            levels = list(contour_levels)
        else:
            if not np.isfinite(global_min) or not np.isfinite(global_max):
                levels = [0.0]
            elif np.isclose(global_min, global_max):
                span = abs(global_min) * 0.1 or 1.0
                levels = [global_min - span, global_min + span]
            else:
                levels = list(np.linspace(global_min, global_max, 6))

        color_cycle = list(colors) if colors else list(
            plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        )
        if not color_cycle:
            color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        fig = plt.figure(figsize=(12, 6))
        proj = ccrs.PlateCarree() if ccrs is not None else None
        ax = plt.axes(projection=proj) if proj else plt.axes()

        base_field = fields[0]
        if proj:
            ax.set_global()
            ax.set_extent([-180, 180, -90, 90], crs=proj)
            ax.set_facecolor("#e8eef2")
        else:
            ax.set_xlim(base_field.lon.min(), base_field.lon.max())
            ax.set_ylim(base_field.lat.min(), base_field.lat.max())
            ax.set_facecolor("#e8eef2")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            for spine in ax.spines.values():
                spine.set_visible(False)

        if proj and cfeature is not None:
            ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555555")
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#666666", alpha=0.7)

        legend_handles: List[Line2D] = []
        for idx, field in enumerate(fields):
            color = color_cycle[idx % len(color_cycle)]
            label = (
                member_labels[idx]
                if member_labels
                else paths[idx].stem
            )
            cs = ax.contour(
                field.lon,
                field.lat,
                field.data,
                levels=levels,
                colors=color,
                linewidths=linewidth,
                alpha=alpha,
                transform=proj,
            )
            if cs.collections:
                legend_handles.append(Line2D([0], [0], color=color, lw=linewidth, alpha=alpha, label=label))

        if legend_handles:
            ax.legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=False)

        if title:
            fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
        fig.text(0.5, 0.92, subtitle or base_field.time_label, ha="center", fontsize=12)

        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        fig_name = filename if filename else _format_filename(base_field.source_path, variable, ".png")
        if not Path(fig_name).suffix:
            fig_name = f"{fig_name}.png"
        fig_path = output_path / fig_name
        plt.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        metadata = {
            "figure_path": str(fig_path),
            "variable": variable,
            "member_count": len(fields),
            "contour_levels": [float(level) for level in levels],
            "time_descriptor": base_field.time_label,
        }
        if member_labels:
            metadata["members"] = list(member_labels)
        return _ok(metadata)
    except FileNotFoundError as exc:
        return _error(str(exc))
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:  # pragma: no cover
        return _error(f"plot_spaghetti_map failed: {exc}")


# ----- Seasonal plotting -----------------------------------------------------
def plot_tercile_map(
    prob_path: str,
    *,
    category: str = "above",
    title: Optional[str] = None,
    output_path: str = "./plots/tercile_map.png",
) -> ToolResponse:
    """Plot tercile probability map."""
    try:
        ds = xr.load_dataset(Path(prob_path).expanduser())
        var = ds.get(f"p_{category}")
        if var is None:
            return _error(f"Category {category} not found.")
        fig = plt.figure(figsize=(8, 5))
        ax = plt.axes(projection=ccrs.PlateCarree() if ccrs else None)
        var.plot(
            ax=ax,
            transform=ccrs.PlateCarree() if ccrs else None,
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar_kwargs={"label": "Probability"},
        )
        if ccrs:
            ax.coastlines()
        ax.set_title(title or f"{category} probability")
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return _ok({"output_path": str(out_path)})
    except Exception as exc:
        return _error(f"Plot tercile map failed: {exc}")


def plot_anomaly_map(
    anomaly_path: str,
    *,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
    output_path: str = "./plots/anomaly_map.png",
) -> ToolResponse:
    """Plot a seasonal anomaly map."""
    try:
        ds = xr.load_dataset(Path(anomaly_path).expanduser())
        var = ds[list(ds.data_vars)[0]]
        fig = plt.figure(figsize=(8, 5))
        ax = plt.axes(projection=ccrs.PlateCarree() if ccrs else None)
        var.plot(
            ax=ax,
            transform=ccrs.PlateCarree() if ccrs else None,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kwargs={"label": var.attrs.get("units", "")},
        )
        if ccrs:
            ax.coastlines()
        ax.set_title(title or "Anomaly")
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return _ok({"output_path": str(out_path)})
    except Exception as exc:
        return _error(f"Plot anomaly map failed: {exc}")


def plot_skill_map(
    skill_nc_path: str,
    *,
    metric: Optional[str] = None,
    mask_threshold: float = 0.0,
    cmap: str = "viridis",
    output_path: str = "./plots/skill_map.png",
) -> ToolResponse:
    """Plot model skill map."""
    try:
        ds = xr.load_dataset(Path(skill_nc_path).expanduser())
        var_name = metric if metric else list(ds.data_vars)[0]
        if var_name not in ds:
            return _error(f"Metric '{var_name}' not found.")
        var = ds[var_name]
        if mask_threshold > 0:
            var = var.where(var >= mask_threshold)
        fig = plt.figure(figsize=(8, 5))
        ax = plt.axes(projection=ccrs.PlateCarree() if ccrs else None)
        var.plot(
            ax=ax,
            transform=ccrs.PlateCarree() if ccrs else None,
            cmap=cmap,
            cbar_kwargs={"label": var.attrs.get("units", "")},
        )
        if ccrs:
            ax.coastlines()
        ax.set_title(metric or "Skill")
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return _ok({"output_path": str(out_path)})
    except Exception as exc:
        return _error(f"Plot skill map failed: {exc}")
