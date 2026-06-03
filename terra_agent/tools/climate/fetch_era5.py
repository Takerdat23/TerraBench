# tools/fetch_era5.py
import os, json, hashlib, pathlib, datetime as dt, shutil, zipfile
from typing import Any, Dict, List, Optional, Tuple, Union
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

import xarray as xr
import numpy as np

# ---------- small helpers ----------

def _error(msg: str) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {msg}")],
        metadata={"error": True, "message": msg},
    )

def _ok(payload: dict) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=json.dumps(payload))],
        metadata=payload,
    )


REGIONS: Dict[str, List[float]] = {
    "global": [90, -180, -90, 180],
    "norcal": [42.5, -125.0, 36.5, -118.5],
    "europe": [72, -25, 34, 45],
}

DATASET_KINDS = {
    # Single levels, hourly
    "era5_single_levels": "reanalysis-era5-single-levels",
    # Monthly means of single levels
    "era5_single_levels_monthly": "reanalysis-era5-single-levels-monthly-means",
    # Pressure levels, hourly
    "era5_pressure_levels": "reanalysis-era5-pressure-levels",
    # ERA5-Land hourly single levels
    "era5_land": "reanalysis-era5-land",
}

# maps short var keys to CDS variable names per collection
SINGLE_VARS = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "tp": "total_precipitation",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "lsm": "land_sea_mask",
    "slt": "soil_type",
    "oro": "geopotential",
    "tcc": "total_cloud_cover",
}
PRESSURE_VARS = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "r": "relative_humidity",
    "w": "vertical_velocity",
}

def _resolve_region(region: Optional[Union[str, List[float]]]) -> List[float]:
    if region is None:
        return REGIONS["global"]
    if isinstance(region, str):
        if region not in REGIONS:
            raise ValueError(f"Unknown region preset '{region}'. Known: {list(REGIONS)}")
        return REGIONS[region]
    # assume [N, W, S, E]
    if len(region) != 4:
        raise ValueError("region bbox must be [N, W, S, E]")
    return [float(region[0]), float(region[1]), float(region[2]), float(region[3])]

def _hash_name(parts: Dict[str, Any]) -> str:
    s = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _ensure_sequence(value: Any) -> List[Any]:
    """
    Coerce scalar inputs (int/float/str) into a one-element list while leaving
    list/tuple inputs untouched. This lets callers provide either a single
    month/day/hour or an explicit list without triggering type errors.
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return [value]


def _ensure_dir(p: Union[str, pathlib.Path]) -> str:
    p = pathlib.Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def _normalize_era5(ds: xr.Dataset) -> xr.Dataset:
    if "expver" in ds.dims:
        try:
            ds = ds.sel(expver=ds["expver"].max())
        except Exception:
            ds = ds.isel(expver=-1)
    if "latitude" in ds.coords:
        lat = ds["latitude"].values
        if lat.ndim == 1 and np.all(np.diff(lat) < 0):
            ds = ds.sortby("latitude")
    return ds


def _ensure_unzipped_nc(path: pathlib.Path) -> None:
    """Some CDS downloads arrive as Zip archives even when requested as .nc."""
    if not path.exists() or not zipfile.is_zipfile(path):
        return
    with zipfile.ZipFile(path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if not members:
            raise ValueError(f"Zip archive {path} contains no files.")
        members.sort(key=lambda info: (not info.filename.lower().endswith(".nc"), -info.file_size))
        member = members[0]
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with archive.open(member) as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    path.unlink()
    tmp_path.replace(path)

def _timecoord(ds: xr.Dataset) -> str:
    for cand in ("time", "valid_time"):
        if cand in ds.coords or cand in ds.dims:
            return cand
    raise KeyError(f"No time or valid_time coordinate; have dims={list(ds.dims)} coords={list(ds.coords)}")


def _levels_match(ds: xr.Dataset, levels: Optional[List[int]]) -> bool:
    if not levels:
        return True
    if "pressure_level" not in ds.coords:
        return False
    have_levels = {int(lvl) for lvl in ds["pressure_level"].values}
    want_levels = {int(lvl) for lvl in levels}
    return want_levels.issubset(have_levels)


def _select_time_window(
    ds: xr.Dataset,
    exact_slice: Optional[Tuple[dt.datetime, dt.datetime]],
) -> xr.Dataset:
    if exact_slice is None:
        return ds
    tname = _timecoord(ds)
    start, end = exact_slice
    return ds.sel({tname: slice(start, end)})


def _find_cached_dataset(
    persist_dir: pathlib.Path,
    dataset_kind: str,
    vars_resolved: List[str],
    exact_slice: Optional[Tuple[dt.datetime, dt.datetime]],
    expected_steps: Optional[int],
    levels: Optional[List[int]],
    outfile: pathlib.Path,
    area: List[float],
    grid: Optional[float],
) -> Optional[pathlib.Path]:
    """Search for an existing NetCDF file that satisfies the current request."""
    pattern = f"{dataset_kind}_*.nc"
    north, west, south, east = area
    tol = 1e-3
    for candidate in sorted(persist_dir.glob(pattern)):
        if not candidate.is_file() or candidate == outfile:
            continue
        try:
            with xr.open_dataset(candidate, engine="netcdf4") as ds:
                ds = _normalize_era5(ds)
                if not all(var in ds.data_vars for var in vars_resolved):
                    continue
                if not _levels_match(ds, levels):
                    continue
                lat_name = "latitude" if "latitude" in ds.coords else "lat"
                lon_name = "longitude" if "longitude" in ds.coords else "lon"
                if lat_name not in ds.coords or lon_name not in ds.coords:
                    continue
                lat_vals = np.asarray(ds[lat_name].values, dtype=float)
                lon_vals = np.asarray(ds[lon_name].values, dtype=float)
                if lat_vals.size == 0 or lon_vals.size == 0:
                    continue
                lat_max = float(lat_vals.max())
                lat_min = float(lat_vals.min())
                if abs(lat_max - north) > tol or abs(lat_min - south) > tol:
                    continue
                if grid is not None and lat_vals.size > 1:
                    expected_lat = int(round((north - south) / grid)) + 1
                    if lat_vals.size != expected_lat:
                        continue
                slice_ds = _select_time_window(ds, exact_slice)
                if exact_slice is not None:
                    tname = _timecoord(ds)
                    size = slice_ds.sizes.get(tname, 0)
                    if size == 0:
                        continue
                    if expected_steps is not None and size < expected_steps:
                        continue
                slice_ds.to_netcdf(outfile)
                return candidate
        except Exception:
            continue
    return None

# ---------- the tool ----------

def _fetch_era5_impl(
    dataset_kind: str,
    variables: list[str],
    time_window: dict[str, Any],
    region: Optional[object] = None,
    levels: Optional[list[int]] = None,
    backend: str = "cds",
    format: str = "netcdf",
    grid: Optional[float] = None,
    persist_dir: Optional[str] = "./data/era5",
    dry_run: bool = False,
) -> ToolResponse:
    """
    Retrieve ERA5 (or ERA5-Land) data for downstream tools. Designed for LLM planning:
    the model decides dataset, vars, time range, region, etc.

    Args (LLM-friendly)
    -------------------
    dataset_kind : str
        One of:
          - "era5_single_levels"  -> hourly single levels
          - "era5_single_levels_monthly" -> monthly means
          - "era5_pressure_levels" -> hourly on pressure levels
          - "era5_land" -> ERA5-Land hourly single levels
    variables : List[str]
        Short codes the tool maps to CDS variable names.
        For single-level kinds:  keys in {t2m, d2m, tp, u10, v10, msl, lsm, slt, oro, tcc}
        For pressure-level kind: keys in {z, t, u, v, q, r, w}
    time_window : dict
        One of:
          - {"start": "YYYY-MM-DDTHH:MMZ", "hours": <int>}  # will request per-day hours
          - {"year": 2020, "month": [1,2], "day": [1,2,...], "hours": [0,6,12,18]}
          - {"years": [1991,2020], "months": [1,2,...]}  # for monthly means
    region : str | [N,W,S,E]
        Region preset (global, norcal, europe) or bounding box.
    levels : List[int]
        Pressure levels in hPa (e.g., [1000, 925, 850, 700, 500, 300]) when dataset_kind is "era5_pressure_levels".
    backend : str
        "cds" (Copernicus CDS API) or "aws" (open S3 era5-pds for some single-levels).
    format : str
        "netcdf" or "grib" (dataset dependent; default netcdf).
    grid : float
        Optional output grid spacing in degrees (e.g., 0.25). If provided, adds CDS "grid" param.
    persist_dir : str
        Directory to write results. Created if missing.
    dry_run : bool
        If true, don't download; just return the resolved request dictionary.

    Returns (JSON-serializable)
    ---------------------------
    {
      "request": {...},            # exact dataset + request sent
      "backend": "cds"|"aws",
      "files": [{"path": "...", "size_bytes": 12345, "cached": false}],
      "dataset_kind": "...",
      "variables_resolved": ["2m_temperature", ...],
      "cached": false,             # true if an existing file was reused
      "coords": {"time_name": "time"|"valid_time", "lat_desc": "..."},
    }

    Notes for the planner
    ---------------------
    - Prefer hourly single levels for short-range tasks (t2m, tp, u10/v10, msl).
    - Use pressure levels (e.g., 850hPa, 500hPa) for shear/CAPE workflows.
    - Use monthly means for climatology/baselines (1991–2020).
    """
    time_window = dict(time_window)

    if dataset_kind not in DATASET_KINDS:
        raise ValueError(f"dataset_kind must be one of {list(DATASET_KINDS)}")
    ds_name = DATASET_KINDS[dataset_kind]

    # map variable keys
    if dataset_kind in ("era5_single_levels", "era5_single_levels_monthly", "era5_land"):
        mapping = SINGLE_VARS
    else:
        mapping = PRESSURE_VARS
    try:
        vars_resolved = [mapping[v] for v in variables]
    except KeyError as e:
        raise ValueError(f"Unknown variable key {e}. Allowed: {list(mapping)}")

    area = _resolve_region(region)
    persist = _ensure_dir(persist_dir or "./data/era5")

    # build the request dict in CDS schema terms
    req: Dict[str, Any] = {"variable": vars_resolved}
    if dataset_kind == "era5_pressure_levels":
        if not levels:
            # sensible default pressure levels if model didn't specify
            levels = [1000, 925, 850, 700, 500, 300, 250, 200]
        req["pressure_level"] = [str(l) for l in levels]

    # time semantics
    def _h24(): return [f"{h:02d}:00" for h in range(24)]

    expected_steps: Optional[int] = None

    def _parse_start(ts: str) -> dt.datetime:
        return dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).replace(tzinfo=None)

    def _duration_hours(start_dt: dt.datetime, end_dt: dt.datetime) -> int:
        delta = end_dt - start_dt
        seconds = delta.total_seconds()
        if seconds < 0:
            raise ValueError("time_window end must be after start")
        hours_float = seconds / 3600.0
        if abs(hours_float - round(hours_float)) > 1e-6:
            raise ValueError("time_window start/end must align to whole hours")
        hours = int(round(hours_float))
        # include both endpoints (e.g., 00Z–23Z -> 24 hours)
        return hours + 1
    expected_steps: Optional[int] = None
    static_flag = bool(time_window.pop("static", False))
    static_timestamp = time_window.pop("timestamp", None) if static_flag else None

    if static_flag:
        static_dt = _parse_start(static_timestamp or "2023-01-01T00:00")
        req.update({
            "year": [static_dt.strftime("%Y")],
            "month": [static_dt.strftime("%m")],
            "day": [static_dt.strftime("%d")],
            "time": [static_dt.strftime("%H:00")],
        })
        exact_slice = None
        expected_steps = 1
    elif "start" in time_window:
        start = _parse_start(time_window["start"])
        if "hours" in time_window:
            hours = int(time_window["hours"])
            if hours <= 0:
                raise ValueError("time_window['hours'] must be positive")
            end = start + dt.timedelta(hours=hours - 1)
            expected_steps = hours
        elif "end" in time_window:
            end = _parse_start(time_window["end"])
            expected_steps = _duration_hours(start, end)
        else:
            raise ValueError("time_window with 'start' must include 'hours' or 'end'")
        # build unique YYYY-MM-DD list
        day0 = dt.datetime(start.year, start.month, start.day)
        day1 = dt.datetime(end.year, end.month, end.day)
        days = (day1 - day0).days
        date_list = [(day0 + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days + 1)]
        req.update({"date": date_list, "time": _h24()})
        # subset later with xarray on exact slice
        exact_slice = (start, end)
    elif {"year", "month", "day", "hours"}.issubset(time_window.keys()):
        years = _ensure_sequence(time_window["year"])
        months = _ensure_sequence(time_window["month"])
        days = _ensure_sequence(time_window["day"])
        hours = _ensure_sequence(time_window["hours"])
        req.update({
            "year": [str(y) for y in years],
            "month": [f"{int(m):02d}" for m in months],
            "day": [f"{int(d):02d}" for d in days],
            "time": [f"{int(h):02d}:00" for h in hours],
        })
        expected_steps = len(hours)
        exact_slice = None
    elif {"years", "months"}.issubset(time_window.keys()):
        # monthly means
        years = _ensure_sequence(time_window["years"])
        months = _ensure_sequence(time_window["months"])
        req.update({
            "product_type": "monthly_averaged_reanalysis",
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in months],
            "time": "00:00",
        })
        exact_slice = None
    else:
        raise ValueError("time_window must be one of: {'start','hours'} OR {'start','end'} OR {'year','month','day','hours'} OR {'years','months'}")

    req["area"] = area
    if grid is not None:
        req["grid"] = [grid, grid]
    # product_type default
    if "product_type" not in req:
        req["product_type"] = "reanalysis"

    # format
    if format not in ("netcdf", "grib"):
        raise ValueError("format must be 'netcdf' or 'grib'")
    req["format"] = format

    # Plan-only?
    if dry_run:
        return _ok({
            "request": {"dataset": ds_name, **req},
            "backend": backend,
            "files": [],
            "dataset_kind": dataset_kind,
            "variables_resolved": vars_resolved,
            "coords": {},
        })

    # Execute
    files: List[Dict[str, Any]] = []
    tag = _hash_name({"dataset": ds_name, **req})
    outname = f"{dataset_kind}_{tag}.{ 'nc' if format=='netcdf' else 'grib' }"
    outfile_path = pathlib.Path(persist) / outname
    existing_file = outfile_path.exists() and outfile_path.stat().st_size > 0
    reused_from: Optional[pathlib.Path] = None
    slice_applied = False

    if backend == "aws":
        # minimal AWS path (works for many single-level variables; not universal)
        # For production, you'd browse intake-esm catalogs or era5-pds layout.
        raise NotImplementedError("AWS backend not implemented in this minimal tool. Use backend='cds'.")
    elif backend == "cds":
        if not existing_file and format == "netcdf":
            reused_from = _find_cached_dataset(
                pathlib.Path(persist),
                dataset_kind,
                vars_resolved,
                exact_slice,
                expected_steps,
                levels,
                outfile_path,
                area,
                grid,
            )
            if reused_from is not None:
                existing_file = True
                slice_applied = exact_slice is not None
        if not existing_file:
            import cdsapi

            c = cdsapi.Client()
            # Some datasets (ERA5-Land) use different parameter names; our schema matches CDS pages.
            c.retrieve(ds_name, req, target=str(outfile_path))
    else:
        raise ValueError("backend must be 'cds' or 'aws'")

    if format == "netcdf":
        _ensure_unzipped_nc(outfile_path)

    size = outfile_path.stat().st_size if outfile_path.exists() else None
    file_meta: Dict[str, Any] = {"path": str(outfile_path), "size_bytes": size, "cached": existing_file}
    if reused_from is not None:
        file_meta["cached_from"] = str(reused_from)
    files.append(file_meta)

    # Quick inspection of coords (optional)
    coords: Dict[str, Any] = {}
    try:
        ds = xr.open_dataset(outfile_path, engine="netcdf4")
        ds = _normalize_era5(ds)
        tname = _timecoord(ds)
        coords["time_name"] = tname
        coords["n_time"] = int(ds.dims.get(tname, 0))
        coords["lat_desc"] = f"{float(ds.latitude.max()):.2f}..{float(ds.latitude.min()):.2f} (n={ds.dims.get('latitude',0)})" if "latitude" in ds.coords else ""
        if exact_slice and not slice_applied:
            # optional: subset and resave a tight window (same file for simplicity)
            start, end = exact_slice
            ds = ds.sel({tname: slice(start, end)})
            ds.to_netcdf(outfile_path)
            files[0]["size_bytes"] = outfile_path.stat().st_size
            files[0]["subset_applied"] = True
        ds.close()
    except Exception as _:
        pass


    result = {
        "request": {"dataset": ds_name, **req},
        "backend": backend,
        "files": files,
        "dataset_kind": dataset_kind,
        "variables_resolved": vars_resolved,
        "cached": existing_file,
        "coords": coords,
    }
    if reused_from is not None:
        result["cached_from"] = str(reused_from)

    # Success response (JSON payload)
    return _ok(result)


def fetch_era5(
    dataset_kind: str,
    variables: list[str],
    time_window: dict[str, Any],
    region: Optional[object] = None,
    levels: Optional[list[int]] = None,
    backend: str = "cds",
    format: str = "netcdf",
    grid: Optional[float] = None,
    persist_dir: Optional[str] = "./data/era5",
    dry_run: bool = False,
) -> ToolResponse:
    """Wrapper that converts raised errors into ToolResponse payloads."""
    try:
        return _fetch_era5_impl(
            dataset_kind=dataset_kind,
            variables=variables,
            time_window=time_window,
            region=region,
            levels=levels,
            backend=backend,
            format=format,
            grid=grid,
            persist_dir=persist_dir,
            dry_run=dry_run,
        )
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Unexpected failure: {exc}")


fetch_era5.__doc__ = _fetch_era5_impl.__doc__
