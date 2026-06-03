"""
Diagnostic utilities: winds, humidity, moisture integrals, kinematics, shear, region stats.
"""

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import xarray as xr
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

R_EARTH = 6_371_000.0
G = 9.80665
OMEGA = 7.2921159e-5


def _ok(payload: Dict) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload))], metadata=payload)


def _error(msg: str) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=f"Error: {msg}")], metadata={"error": True, "message": msg})


def _load_ds(path: str) -> xr.Dataset:
    return xr.load_dataset(Path(path).expanduser())


def _lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    lat = next((c for c in ("latitude", "lat", "y") if c in ds.coords), None)
    lon = next((c for c in ("longitude", "lon", "x") if c in ds.coords), None)
    if not lat or not lon:
        raise ValueError("Latitude/longitude coordinates not found.")
    return lat, lon


def winds_basic(
    netcdf_path: str,
    *,
    u: str = "u10",
    v: str = "v10",
    levels: Optional[Sequence[float]] = None,
    output_path: str = "./data/diagnostics/winds_basic.nc",
) -> ToolResponse:
    ds = _load_ds(netcdf_path)
    try:
        u_da = ds[u]
        v_da = ds[v]
    except KeyError as exc:
        return _error(f"Missing wind component: {exc}")
    if levels is not None:
        level_dim = next((d for d in u_da.dims if "lev" in d.lower() or d == "level"), None)
        if not level_dim:
            return _error("levels specified but no level dimension found in wind fields.")
        u_da = u_da.sel({level_dim: levels}, method="nearest")
        v_da = v_da.sel({level_dim: levels}, method="nearest")
    speed = np.sqrt(u_da ** 2 + v_da ** 2)
    direction = (np.rad2deg(np.arctan2(-u_da, -v_da)) + 360.0) % 360.0
    out = xr.Dataset({"wind_speed": speed.astype(np.float32), "wind_dir": direction.astype(np.float32)})
    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    return _ok({"output_path": str(out_path), "variables": ["wind_speed", "wind_dir"]})


def _dewpoint_from_vapor_pressure(e_hpa: xr.DataArray) -> xr.DataArray:
    ln_ratio = np.log(e_hpa / 6.112)
    Td = 243.5 * ln_ratio / (17.67 - ln_ratio)
    return (Td + 273.15).astype(np.float32)


def humidity_basic(
    netcdf_path: str,
    *,
    T: str = "t2m",
    q: Optional[str] = "q",
    d2m: Optional[str] = None,
    p: str = "sp",
    output_path: str = "./data/diagnostics/humidity_basic.nc",
) -> ToolResponse:
    ds = _load_ds(netcdf_path)
    if q is None and d2m is None:
        return _error("Provide either specific humidity 'q' or dewpoint 'd2m'.")
    try:
        temp = ds[T]
        pressure = ds[p]
    except KeyError as exc:
        return _error(f"Missing variable: {exc}")
    if q and q in ds:
        q_da = ds[q]
        e = (q_da * pressure) / (0.622 + (1 - 0.622) * q_da)
        Td = _dewpoint_from_vapor_pressure(e / 100.0)
    elif d2m and d2m in ds:
        Td = ds[d2m]
        e = 6.112 * np.exp(17.67 * (Td - 273.15) / (Td - 29.65)) * 100.0
    else:
        return _error("Specified humidity variable not found.")

    es = 6.112 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65)) * 100.0
    rh = np.clip(e / es, 0.0, 1.2)
    mixing_ratio = 0.622 * e / (pressure - e)
    theta = temp * (1000.0 / (pressure / 100.0)) ** 0.286
    out = xr.Dataset(
        {
            "dewpoint": Td.astype(np.float32),
            "relative_humidity": rh.astype(np.float32),
            "mixing_ratio": mixing_ratio.astype(np.float32),
            "potential_temperature": theta.astype(np.float32),
        }
    )
    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    return _ok({"output_path": str(out_path), "variables": list(out.data_vars)})


def moisture_integrals(
    netcdf_path: str,
    *,
    q: str = "q",
    u: str = "u",
    v: str = "v",
    level_dim: str = "level",
    levels: Optional[Sequence[float]] = None,
    output_path: str = "./data/diagnostics/moisture_integrals.nc",
) -> ToolResponse:
    ds = _load_ds(netcdf_path)
    try:
        q_da = ds[q]
        u_da = ds[u]
        v_da = ds[v]
    except KeyError as exc:
        return _error(f"Missing variable: {exc}")
    if level_dim not in q_da.dims:
        return _error(f"Specific humidity missing '{level_dim}' dimension.")
    if levels is not None:
        q_da = q_da.sel({level_dim: levels}, method="nearest")
        u_da = u_da.sel({level_dim: levels}, method="nearest")
        v_da = v_da.sel({level_dim: levels}, method="nearest")
    if q_da[level_dim].max() <= 2000:
        pres = q_da[level_dim] * 100.0
    else:
        pres = q_da[level_dim]
    q_da = q_da.assign_coords({level_dim: pres})
    u_da = u_da.assign_coords({level_dim: pres})
    v_da = v_da.assign_coords({level_dim: pres})

    iwv = (1.0 / G) * q_da.integrate(level_dim)
    ivt_u = (1.0 / G) * (q_da * u_da).integrate(level_dim)
    ivt_v = (1.0 / G) * (q_da * v_da).integrate(level_dim)
    ivt = np.sqrt(ivt_u ** 2 + ivt_v ** 2)
    out = xr.Dataset(
        {
            "IWV": iwv.astype(np.float32),
            "IVT_u": ivt_u.astype(np.float32),
            "IVT_v": ivt_v.astype(np.float32),
            "IVT": ivt.astype(np.float32),
        }
    )
    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    return _ok({"output_path": str(out_path), "variables": list(out.data_vars)})


def _metric_derivative(da: xr.DataArray, coord: str) -> xr.DataArray:
    return da.differentiate(coord, edge_order=2)


def kinematics(
    netcdf_path: str,
    *,
    u: str = "u",
    v: str = "v",
    level: Optional[float] = None,
    geopotential: Optional[str] = "z",
    pressure: Optional[str] = "msl",
    output_path: str = "./data/diagnostics/kinematics.nc",
) -> ToolResponse:
    ds = _load_ds(netcdf_path)
    lat_name, lon_name = _lat_lon_names(ds)
    try:
        u_da = ds[u]
        v_da = ds[v]
    except KeyError as exc:
        return _error(f"Missing variable: {exc}")
    if level is not None:
        level_dim = next((d for d in u_da.dims if "lev" in d.lower() or d == "level"), None)
        if not level_dim:
            return _error("Level selection requested but no level dimension found.")
        u_da = u_da.sel({level_dim: level}, method="nearest")
        v_da = v_da.sel({level_dim: level}, method="nearest")
    lat_rad = np.deg2rad(u_da[lat_name])
    coslat = np.cos(lat_rad)

    dudx = _metric_derivative(u_da, lon_name) * np.deg2rad(1.0) / (R_EARTH * coslat)
    dvdy = _metric_derivative(v_da, lat_name) * np.deg2rad(1.0) / R_EARTH
    dvdx = _metric_derivative(v_da, lon_name) * np.deg2rad(1.0) / (R_EARTH * coslat)
    dudy = _metric_derivative(u_da, lat_name) * np.deg2rad(1.0) / R_EARTH
    vorticity = dvdx - dudy
    divergence = dudx + dvdy

    data = {"vorticity": vorticity.astype(np.float32), "divergence": divergence.astype(np.float32)}

    if geopotential and geopotential in ds:
        phi = G * ds[geopotential]
    elif pressure and pressure in ds:
        phi = ds[pressure]
    else:
        phi = None

    if phi is not None:
        dphidx = _metric_derivative(phi, lon_name) * np.deg2rad(1.0) / (R_EARTH * coslat)
        dphidy = _metric_derivative(phi, lat_name) * np.deg2rad(1.0) / R_EARTH
        f = 2 * OMEGA * np.sin(lat_rad)
        f = xr.where(np.abs(f) < 1e-5, np.nan, f)
        ug = -dphidy / f
        vg = dphidx / f
        data["geostrophic_u"] = ug.astype(np.float32)
        data["geostrophic_v"] = vg.astype(np.float32)

    out = xr.Dataset(data)
    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    return _ok({"output_path": str(out_path), "variables": list(out.data_vars)})


def vertical_shear(
    netcdf_path: str,
    *,
    top: float = 500.0,
    bottom: float = 850.0,
    u: str = "u",
    v: str = "v",
    level_dim: str = "level",
    output_path: str = "./data/diagnostics/vertical_shear.nc",
) -> ToolResponse:
    ds = _load_ds(netcdf_path)
    try:
        u_da = ds[u]
        v_da = ds[v]
    except KeyError as exc:
        return _error(f"Missing wind variable: {exc}")
    if level_dim not in u_da.dims:
        return _error("No vertical dimension found for shear calculation.")
    u_top = u_da.sel({level_dim: top}, method="nearest")
    u_bot = u_da.sel({level_dim: bottom}, method="nearest")
    v_top = v_da.sel({level_dim: top}, method="nearest")
    v_bot = v_da.sel({level_dim: bottom}, method="nearest")
    shear_u = u_top - u_bot
    shear_v = v_top - v_bot
    shear_mag = np.sqrt(shear_u ** 2 + shear_v ** 2)
    out = xr.Dataset(
        {"shear_u": shear_u.astype(np.float32), "shear_v": shear_v.astype(np.float32), "shear_mag": shear_mag.astype(np.float32)}
    )
    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    return _ok({"output_path": str(out_path), "variables": list(out.data_vars)})


def region_reduce(
    netcdf_path: str,
    *,
    variable: str,
    mask_path: Optional[str] = None,
    mask_variable: Optional[str] = None,
    region_name: Optional[str] = None,
    stats: Sequence[str] = ("mean", "max", "p95"),
    threshold: Optional[float] = None,
) -> ToolResponse:
    ds = _load_ds(netcdf_path)
    if variable not in ds:
        return _error(f"Variable '{variable}' not found.")
    data = ds[variable]
    lat_name = next((c for c in ("latitude", "lat") if c in data.coords), None)
    if mask_path:
        mask_ds = xr.load_dataset(Path(mask_path).expanduser())
        mask_var = mask_variable or next(iter(mask_ds.data_vars))
        mask = mask_ds[mask_var]
        data = data.where(mask > 0.5)

    weights = np.cos(np.deg2rad(data[lat_name])) if lat_name else None
    results = {}
    if "mean" in stats:
        results["mean"] = float(data.weighted(weights).mean().item()) if weights is not None else float(data.mean().item())
    if "max" in stats:
        results["max"] = float(data.max().item())
    if "min" in stats:
        results["min"] = float(data.min().item())
    if any(stat.startswith("p") for stat in stats):
        for stat in stats:
            if stat.startswith("p"):
                try:
                    pct = float(stat[1:])
                except ValueError:
                    continue
                results[stat] = float(data.quantile(pct / 100.0).item())
    if any("area_frac" in stat for stat in stats) and threshold is not None:
        frac = float((data >= threshold).sum().item()) / float(data.count().item())
        results["area_frac"] = frac

    return _ok(results)


__all__ = [
    "winds_basic",
    "humidity_basic",
    "moisture_integrals",
    "kinematics",
    "vertical_shear",
    "region_reduce",
]
