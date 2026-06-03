"""Small NetCDF lat/lon field helper used by satellite composite tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
from zipfile import ZipFile, is_zipfile

import numpy as np
import xarray as xr


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


def _format_coord_value(value: Any) -> str:
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.datetime64):
        return str(arr.astype("datetime64[m]"))
    if arr.size == 1:
        return str(arr.item())
    return str(arr)


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

    if _coord_in_dataset(ds, "y") and _coord_in_dataset(ds, "x"):
        y_vals = _coord_values(ds, "y")
        x_vals = _coord_values(ds, "x")
        if y_vals is not None and x_vals is not None:
            if _looks_like_lat(y_vals, _coord_units(ds, "y")) and _looks_like_lon(
                x_vals, _coord_units(ds, "x")
            ):
                return "y", "x"

    raise ValueError(
        "Dataset must include a latitude/longitude coordinate. If the data uses x/y, "
        "ensure they are geographic degrees (EPSG:4326) or rename them to latitude/longitude."
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
                    if auto_time_mean_threshold is not None and time_size > auto_time_mean_threshold:
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
                elif agg == "mean":
                    da = da.mean(time_dim, keep_attrs=True)
                    time_label = f"{time_dim} mean"
                elif agg == "sum":
                    da = da.sum(time_dim, keep_attrs=True)
                    time_label = f"{time_dim} sum"
                else:
                    raise ValueError("aggregate must be one of {'mean','sum','auto',None}.")

    for dim in list(da.dims):
        if dim not in (lat_name, lon_name) and da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)

    extras = [dim for dim in set(da.dims) if dim not in (lat_name, lon_name)]
    if extras:
        dims_to_reduce = extras if reduce_dims is None else [d for d in extras if d in reduce_dims]
        if not dims_to_reduce and reduce_dims:
            dims_to_reduce = extras
        for dim in dims_to_reduce:
            if reduce_mode == "sum":
                da = da.sum(dim, keep_attrs=True)
                time_label = f"{time_label}; {dim} sum"
            else:
                da = da.mean(dim, keep_attrs=True)
                time_label = f"{time_label}; {dim} mean"

    expected_dims = {lat_name, lon_name}
    remaining_dims = set(da.dims)
    if remaining_dims != expected_dims:
        raise ValueError(
            "Unable to reduce data to 2D lat/lon field. "
            f"Missing: {expected_dims - remaining_dims}, extra dims: {remaining_dims - expected_dims}."
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

    return LatLonField(
        data=data,
        lon=da[lon_name].values,
        lat=da[lat_name].values,
        var_name=var_name,
        units=da.attrs.get("units", ""),
        time_label=time_label,
        source_path=path,
        lon_name=lon_name,
        lat_name=lat_name,
    )
