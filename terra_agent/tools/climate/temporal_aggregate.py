"""
Generic temporal aggregation helper (daily/max, rolling totals, etc.).
"""
import json
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd
import xarray as xr
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse


def _ok(payload: Dict) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload))], metadata=payload)


def _error(msg: str) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=f"Error: {msg}")], metadata={"error": True, "message": msg})


def temporal_aggregate(
    netcdf_path: str,
    spec: Mapping[str, str],
    *,
    align: str | None = "00:00",
    output_path: str = "./data/aggregations/temporal_aggregate.nc",
) -> ToolResponse:
    """
    Aggregate variables using pandas-style frequency strings.

    spec example: {"t2m": "1D:max", "tp": "1D:sum"}
    align: optional anchor/offset for windows (e.g., "06:00" to align daily windows at 06Z).
    """
    src = Path(netcdf_path).expanduser()
    if not src.exists():
        return _error(f"NetCDF file not found: {src}")
    try:
        ds = xr.load_dataset(src)
    except Exception as exc:
        return _error(f"Failed to open {src}: {exc}")

    time_name = None
    for cand in ("time", "valid_time", "forecast_time", "step"):
        if cand in ds.coords or cand in ds.dims:
            time_name = cand
            break
    if not time_name:
        return _error("No recognizable time coordinate ('time', 'valid_time', 'forecast_time', or 'step').")

    try:
        align_td = None if align is None or str(align).strip() == "" else pd.to_timedelta(align)
    except Exception as exc:
        return _error(f"Invalid align value {align!r}; must be timedelta-like (e.g., '6H', '06:00'): {exc}")
    # pandas/xarray treat zero offset as None; keep None to avoid errors like "offset should be Timedelta"
    if align_td is not None and align_td == pd.Timedelta(0):
        align_td = None

    out_vars = {}
    for var, rule in spec.items():
        if var not in ds:
            return _error(f"Variable '{var}' not found in dataset.")
        pieces = rule.split(":")
        if len(pieces) != 2:
            return _error(f"Invalid spec for '{var}' ({rule}); use '<freq>:<agg>' (e.g., '1D:max').")
        freq, func = (pieces[0].strip(), pieces[1].strip())
        if not freq:
            return _error(f"Invalid frequency for '{var}' in spec '{rule}'.")
        if not func:
            return _error(f"Invalid aggregation function for '{var}' in spec '{rule}'.")
        data = ds[var]
        if time_name in data.coords:
            coord = data[time_name]
        elif time_name in ds.coords:
            coord = ds[time_name]
            data = data.assign_coords({time_name: coord})
        else:
            return _error(f"Time coordinate '{time_name}' missing for variable '{var}'.")
        if not np.issubdtype(coord.dtype, np.datetime64):
            try:
                decoded = pd.to_datetime(coord.values)
                data = data.assign_coords({time_name: decoded})
            except Exception as exc:
                return _error(f"Time coordinate '{time_name}' is not datetime-like and could not be converted: {exc}")
        if not pd.Index(data[time_name].values).is_monotonic_increasing:
            data = data.sortby(time_name)
        try:
            resample_kwargs = {time_name: freq}
            resampled = (
                data.resample(resample_kwargs, offset=align_td)
                if align_td is not None
                else data.resample(resample_kwargs)
            )
            if func == "sum":
                agg = resampled.sum(skipna=True)
            elif func == "mean":
                agg = resampled.mean(skipna=True)
            elif func == "max":
                agg = resampled.max(skipna=True)
            elif func == "min":
                agg = resampled.min(skipna=True)
            elif func == "median":
                agg = resampled.median(skipna=True)
            else:
                return _error(f"Unsupported aggregation func '{func}' for '{var}'.")
            out_vars[f"{var}_{freq}_{func}"] = agg
        except Exception as exc:
            return _error(f"Failed to aggregate '{var}' with rule '{rule}': {exc}")

    out = xr.Dataset(out_vars)
    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.to_netcdf(out_path)
    except Exception as exc:
        return _error(f"Failed to write aggregated dataset: {exc}")

    summary = {name: str(data.shape) for name, data in out_vars.items()}
    return _ok(
        {
            "output_path": str(out_path),
            "variables": summary,
            "align": align,
            "source": str(src),
        }
    )


__all__ = ["temporal_aggregate"]
