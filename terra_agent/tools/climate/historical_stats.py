# tools/historical_stats.py
"""Utilities for long-range ERA5 analysis (custom climatology + event counts)."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse


def _error(msg: str) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {msg}")],
        metadata={"error": True, "message": msg},
    )


def _ok(payload: Dict[str, Any]) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=json.dumps(payload))],
        metadata=payload,
    )


def _detect_time_coord(da: xr.DataArray) -> str:
    for cand in ("time", "valid_time"):
        if cand in da.coords:
            return cand
    raise ValueError("Could not find a time coordinate (expected one of: time, valid_time)")


def _load_dataarray(path: str, variable: str, time_coord: str | None = None) -> tuple[xr.DataArray, str]:
    ds = xr.open_dataset(path)
    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in {path}")
    da = ds[variable]
    tcoord = time_coord or _detect_time_coord(da)
    if tcoord not in da.coords:
        raise ValueError(f"Detected time coordinate '{tcoord}' missing in data")
    return da, tcoord


def build_era5_climatology(
    data_path: str,
    variable: str,
    start_year: int,
    end_year: int,
    time_coord: Optional[str] = None,
    output_dir: str = "./data/era5_climo",
) -> ToolResponse:
    """
    Build a monthly climatology (mean + std) for a variable over a custom year range.

    Parameters
    ----------
    data_path : str
        Path to ERA5 NetCDF containing the target variable.
    variable : str
        Variable name inside the file.
    start_year : int
        First year to include (e.g., 1950).
    end_year : int
        Last year to include (inclusive).
    time_coord : str, optional
        Time coordinate name if not standard ("time" or "valid_time").
    output_dir : str
        Directory to write the climatology NetCDF.
    """
    try:
        da, tcoord = _load_dataarray(data_path, variable, time_coord)
        window = da.sel({tcoord: slice(f"{start_year}-01-01", f"{end_year}-12-31")})
        if window.size == 0:
            return _error("Selected window has no data; check years or time coordinate.")

        monthly_mean = window.groupby(f"{tcoord}.month").mean(dim=tcoord, skipna=True)
        monthly_std = window.groupby(f"{tcoord}.month").std(dim=tcoord, skipna=True)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(data_path).stem}_climo_{start_year}_{end_year}.nc"
        xr.Dataset(
            {
                f"{variable}_climatology_mean": monthly_mean.astype(np.float32),
                f"{variable}_climatology_std": monthly_std.astype(np.float32),
            }
        ).to_netcdf(out_path)

        return _ok(
            {
                "output_path": str(out_path),
                "variable": variable,
                "time_coord": tcoord,
                "years": {"start": start_year, "end": end_year},
                "stats": ["monthly_mean", "monthly_std"],
            }
        )
    except Exception as exc:
        return _error(f"build_era5_climatology failed: {exc}")


def count_era5_events(
    data_path: str,
    variable: str,
    start_year: int,
    end_year: int,
    threshold: float,
    comparison: str = ">=",
    time_coord: Optional[str] = None,
    output_dir: str = "./data/era5_events",
) -> ToolResponse:
    """
    Count threshold exceedance events over a historical window (e.g., 1950–2000).

    Parameters
    ----------
    data_path : str
        Path to ERA5 NetCDF containing the target variable.
    variable : str
        Variable name inside the file.
    start_year : int
        First year to include.
    end_year : int
        Last year to include (inclusive).
    threshold : float
        Threshold to compare against.
    comparison : str
        One of {">", ">=", "<", "<="}.
    time_coord : str, optional
        Time coordinate name if not standard ("time" or "valid_time").
    output_dir : str
        Directory to write the event count NetCDF.
    """
    ops = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
    }
    if comparison not in ops:
        return _error("comparison must be one of: >, >=, <, <=")

    try:
        da, tcoord = _load_dataarray(data_path, variable, time_coord)
        window = da.sel({tcoord: slice(f"{start_year}-01-01", f"{end_year}-12-31")})
        if window.size == 0:
            return _error("Selected window has no data; check years or time coordinate.")

        mask = ops[comparison](window, threshold)
        event_counts = mask.sum(dim=tcoord, skipna=True).astype(np.int32)
        yearly = mask.groupby(f"{tcoord}.year").sum(dim=tcoord, skipna=True)
        dims_to_collapse = [d for d in yearly.dims if d != "year"]
        if dims_to_collapse:
            by_year = yearly.sum(dim=dims_to_collapse)
        else:
            by_year = yearly
        total_events = int(event_counts.sum().item())
        yearly_counts = {int(k): int(v) for k, v in zip(by_year.year.values.tolist(), by_year.values.tolist())}

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(data_path).stem}_events_{comparison}_{threshold}_{start_year}_{end_year}.nc"
        event_counts.to_dataset(name=f"{variable}_event_count").to_netcdf(out_path)

        return _ok(
            {
                "output_path": str(out_path),
                "variable": variable,
                "time_coord": tcoord,
                "years": {"start": start_year, "end": end_year},
                "comparison": comparison,
                "threshold": threshold,
                "total_events": total_events,
                "yearly_counts": yearly_counts,
            }
        )
    except Exception as exc:
        return _error(f"count_era5_events failed: {exc}")
