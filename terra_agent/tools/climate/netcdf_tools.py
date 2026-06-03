"""
NetCDF inspection and subset utilities (combined).

This module merges the previous ``inspect_netcdf`` and ``read_netcdf_data`` tools
into a single file so inspection and light-weight slicing live together.
"""

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import xarray as xr
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

__all__ = ["inspect_netcdf", "read_netcdf_data"]


def _ok(payload: Dict[str, Any]) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload))], metadata=payload)


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
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


# ----- Introspection helpers -------------------------------------------------
def _expand_paths(path: Path) -> Sequence[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No NetCDF files found in directory: {path}")
        return files
    return [path]


def inspect_netcdf(
    netcdf_path: str,
    *,
    variables: Optional[list[str]] = None,
    coord_preview: int = 3,
) -> ToolResponse:
    """
    Summarize variables, dimensions, units, and value ranges for a NetCDF file.
    """
    path = Path(netcdf_path).expanduser()
    if not path.exists():
        return _error(f"NetCDF path not found: {path}")

    cleanup_dir: Optional[Path] = None
    try:
        if path.suffix == ".zip":
            cleanup_dir = Path(tempfile.mkdtemp(prefix="nc_zip_"))
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(cleanup_dir)
            path = cleanup_dir

        targets = _expand_paths(path)
        ds_ctx = (
            xr.open_mfdataset([str(p) for p in targets], combine="by_coords")
            if len(targets) > 1
            else xr.open_dataset(targets[0])
        )
    except Exception as exc:
        return _error(f"Failed to open {path}: {exc}")

    summary: Dict[str, Any] = {
        "paths": [str(p) for p in targets],
        "dims": {},
        "coords": {},
        "variables": {},
        "global_attrs": {},
    }

    with ds_ctx as ds:
        summary["dims"] = {name: int(size) for name, size in ds.dims.items()}
        summary["global_attrs"] = {k: str(v) for k, v in ds.attrs.items()}

        for coord_name, coord in ds.coords.items():
            preview = coord.values[:coord_preview].tolist() if coord.size >= coord_preview else coord.values.tolist()
            summary["coords"][coord_name] = {
                "size": int(coord.size),
                "dtype": str(coord.dtype),
                "preview": preview,
            }

        var_names = variables or list(ds.data_vars)
        for name in var_names:
            if name not in ds:
                summary["variables"][name] = {"error": "variable not found"}
                continue
            da = ds[name]
            data = da.values
            finite = np.isfinite(data)
            summary["variables"][name] = {
                "dims": {dim: int(da.sizes[dim]) for dim in da.dims},
                "dtype": str(da.dtype),
                "attrs": {k: str(v) for k, v in da.attrs.items()},
                "min": float(np.nanmin(data)) if finite.any() else None,
                "max": float(np.nanmax(data)) if finite.any() else None,
                "mean": float(np.nanmean(data)) if finite.any() else None,
                "std": float(np.nanstd(data)) if finite.any() else None,
    }

    if cleanup_dir:
        try:
            for f in cleanup_dir.glob("*"):
                f.unlink(missing_ok=True)
            cleanup_dir.rmdir()
        except Exception:
            pass

    return _ok(summary)


# ----- Read/subset helpers ---------------------------------------------------
def _first_numeric_var(ds: xr.Dataset, preferred: Optional[str]) -> str:
    if preferred and preferred in ds and np.issubdtype(ds[preferred].dtype, np.number):
        return preferred
    numeric = [name for name, da in ds.data_vars.items() if np.issubdtype(da.dtype, np.number)]
    if not numeric:
        raise KeyError("No numeric variables found in dataset.")
    return sorted(numeric)[0]


def _coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> Optional[str]:
    for cand in candidates:
        if cand in ds.coords or cand in ds.dims:
            return cand
    return None


def _apply_lat_slice(da: xr.DataArray, lat_name: str, lat_range: Sequence[float]) -> xr.DataArray:
    lo, hi = float(lat_range[0]), float(lat_range[1])
    start, end = (hi, lo) if hi < lo else (lo, hi)
    return da.sel({lat_name: slice(start, end)})


def _apply_lon_slice(da: xr.DataArray, lon_name: str, lon_range: Sequence[float]) -> xr.DataArray:
    lo, hi = float(lon_range[0]), float(lon_range[1])
    start, end = (hi, lo) if hi < lo else (lo, hi)
    return da.sel({lon_name: slice(start, end)})


def _time_summary(da: xr.DataArray, time_name: str) -> Dict[str, Any]:
    coord = da[time_name]
    values = np.asarray(coord.values)
    if values.size == 0:
        return {"name": time_name, "count": 0}
    return {
        "name": time_name,
        "count": int(values.size),
        "start": str(values[0]),
        "end": str(values[-1]),
    }


def read_netcdf_data(
    netcdf_path: str,
    variable: Optional[str] = None,
    time_indices: Optional[Sequence[int]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    years: Optional[Sequence[int]] = None,
    months: Optional[Sequence[int]] = None,
    days: Optional[Sequence[int]] = None,
    hours: Optional[Sequence[int]] = None,
    lat_range: Optional[Sequence[float]] = None,
    lon_range: Optional[Sequence[float]] = None,
    lat_indices: Optional[Sequence[int]] = None,
    lon_indices: Optional[Sequence[int]] = None,
    sample_limit: int = 20,
    include_stats: bool = True,
) -> ToolResponse:
    """
    Read a NetCDF file, subset, and report a compact summary plus sample values.
    """
    src = Path(netcdf_path).expanduser()
    if not src.exists():
        return _error(f"NetCDF file not found: {src}")

    cleanup_dir: Optional[Path] = None
    try:
        open_path = src
        if src.suffix == ".zip":
            cleanup_dir = Path(tempfile.mkdtemp(prefix="nc_zip_"))
            with zipfile.ZipFile(src, "r") as zf:
                zf.extractall(cleanup_dir)
            extracted = sorted(cleanup_dir.glob("*.nc"))
            if not extracted:
                return _error(f"No NetCDF files found inside zip: {src}")
            open_path = extracted[0]

        ds = xr.open_dataset(open_path)
    except Exception as exc:  # pragma: no cover - robustness
        return _error(f"Failed to open dataset: {exc}")

    try:
        var_name = _first_numeric_var(ds, variable)
    except Exception as exc:
        return _error(str(exc))

    sel = ds[var_name]
    time_name = _coord_name(ds, ("time", "valid_time", "forecast_time", "step"))
    lat_name = _coord_name(ds, ("latitude", "lat", "Latitude", "LAT", "y"))
    lon_name = _coord_name(ds, ("longitude", "lon", "Longitude", "LON", "x"))

    if time_name and time_indices is not None:
        idx = sorted({int(i) for i in time_indices})
        sel = sel.isel({time_name: idx})
    if time_name and (start_time or end_time):
        sel = sel.sel({time_name: slice(start_time, end_time)})
    if time_name and (years or months or days or hours):
        coord = sel[time_name]
        if hasattr(coord, "dt"):
            if years:
                yr_vals = [int(y) for y in years]
                sel = sel.sel({time_name: coord[coord.dt.year.isin(yr_vals)]})
                coord = sel[time_name]
            if months:
                mo_vals = [int(m) for m in months]
                sel = sel.sel({time_name: coord[coord.dt.month.isin(mo_vals)]})
                coord = sel[time_name]
            if days:
                day_vals = [int(d) for d in days]
                sel = sel.sel({time_name: coord[coord.dt.day.isin(day_vals)]})
                coord = sel[time_name]
            if hours:
                hour_vals = [int(h) for h in hours]
                sel = sel.sel({time_name: coord[coord.dt.hour.isin(hour_vals)]})
        else:
            return _error("Time coordinate is not datetime-like; cannot filter by calendar fields.")

    if lat_name and lat_range is not None and len(lat_range) == 2:
        sel = _apply_lat_slice(sel, lat_name, lat_range)
    if lon_name and lon_range is not None and len(lon_range) == 2:
        sel = _apply_lon_slice(sel, lon_name, lon_range)
    if lat_name and lat_indices is not None:
        idx = sorted({int(i) for i in lat_indices})
        sel = sel.isel({lat_name: idx})
    if lon_name and lon_indices is not None:
        idx = sorted({int(i) for i in lon_indices})
        sel = sel.isel({lon_name: idx})

    sel = sel.load()
    arr = np.asarray(sel.values)
    flat = arr.ravel()
    sample = flat[: min(flat.size, max(sample_limit, 0))]

    stats = None
    if include_stats and flat.size > 0:
        stats = {
            "min": float(np.nanmin(flat)),
            "max": float(np.nanmax(flat)),
            "mean": float(np.nanmean(flat)),
            "std": float(np.nanstd(flat)),
        }

    coords_summary: Dict[str, Any] = {}
    if time_name:
        coords_summary["time"] = _time_summary(sel, time_name)
    if lat_name and lat_name in sel.coords:
        lat_vals = np.asarray(sel[lat_name].values)
        coords_summary["lat"] = {
            "name": lat_name,
            "count": int(lat_vals.size),
            "min": float(np.nanmin(lat_vals)),
            "max": float(np.nanmax(lat_vals)),
        }
    if lon_name and lon_name in sel.coords:
        lon_vals = np.asarray(sel[lon_name].values)
        coords_summary["lon"] = {
            "name": lon_name,
            "count": int(lon_vals.size),
            "min": float(np.nanmin(lon_vals)),
            "max": float(np.nanmax(lon_vals)),
        }

    payload = {
        "source": str(src),
        "variable": var_name,
        "units": sel.attrs.get("units", ""),
        "dims": {k: int(v) for k, v in sel.sizes.items()},
        "shape": list(sel.shape),
        "coords": coords_summary or None,
        "stats": stats,
        "sample_values": _sanitize(sample),
        "selection_applied": {
            "time_indices": list(time_indices) if time_indices is not None else None,
            "start_time": start_time,
            "end_time": end_time,
            "years": list(years) if years is not None else None,
            "months": list(months) if months is not None else None,
            "days": list(days) if days is not None else None,
            "hours": list(hours) if hours is not None else None,
            "lat_range": list(lat_range) if lat_range is not None else None,
            "lon_range": list(lon_range) if lon_range is not None else None,
            "lat_indices": list(lat_indices) if lat_indices is not None else None,
            "lon_indices": list(lon_indices) if lon_indices is not None else None,
        },
        "note": "Values are flattened; sample is limited to avoid large responses.",
    }

    if cleanup_dir:
        payload["extracted_from_zip"] = str(src)
        try:
            for f in cleanup_dir.glob("*"):
                f.unlink(missing_ok=True)
            cleanup_dir.rmdir()
        except Exception:
            pass

    return _ok(_sanitize(payload))
