# tools/fetch_cmip6.py

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse


def _error(msg: str) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {msg}")],
        metadata={"error": True, "message": msg},
    )


def _ok(payload: Dict[str, Any]) -> ToolResponse:
    sanitized = _sanitize(payload)
    return ToolResponse(
        content=[TextBlock(type="text", text=json.dumps(sanitized))],
        metadata=sanitized,
    )


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_)):
        return bool(value)
    if isinstance(value, (np.datetime64,)):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def _ensure_dir(path: Optional[str]) -> Path:
    if not path:
        raise ValueError("persist_dir must be provided when persist=True.")
    out_dir = Path(path).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _first_existing_coord(ds: "xr.Dataset", names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in ds.coords:
            return name
    return None


def _slice_by_bounds(ds: "xr.Dataset", coord_name: str, bounds: Sequence[float]) -> "xr.Dataset":
    if len(bounds) != 2:
        raise ValueError(f"Bounds for {coord_name} must have exactly two values.")
    lo, hi = float(bounds[0]), float(bounds[1])
    coord = ds[coord_name]
    if coord.ndim != 1:
        raise ValueError(f"Coordinate '{coord_name}' must be 1D to slice by bounds.")
    ascending = coord.values[0] < coord.values[-1]
    target_slice = slice(lo, hi) if ascending else slice(hi, lo)
    return ds.sel({coord_name: target_slice})


def _time_subset(ds: "xr.Dataset", request: Dict[str, Any]) -> "xr.Dataset":
    time_coord = _first_existing_coord(ds, ("time", "Time"))
    if not time_coord:
        return ds
    coord = ds[time_coord]

    start = request.get("start")
    end = request.get("end")
    if start or end:
        ds = ds.sel({time_coord: slice(start, end)})
        coord = ds[time_coord]

    years: Optional[Sequence[int]] = request.get("years")
    months: Optional[Sequence[int]] = request.get("months")
    days: Optional[Sequence[int]] = request.get("days")

    # Only attempt calendar filtering when we still have datetime-like data.
    if any(opt is not None for opt in (years, months, days)):
        if not hasattr(coord, "dt"):
            raise ValueError("Time coordinate must be datetime-like to filter by year/month/day.")

        if years is not None:
            year_vals = [int(y) for y in years]
            ds = ds.sel({time_coord: coord[coord.dt.year.isin(year_vals)]})
            coord = ds[time_coord]
        if months is not None:
            month_vals = [int(m) for m in months]
            ds = ds.sel({time_coord: coord[coord.dt.month.isin(month_vals)]})
            coord = ds[time_coord]
        if days is not None:
            day_vals = [int(d) for d in days]
            ds = ds.sel({time_coord: coord[coord.dt.day.isin(day_vals)]})
            coord = ds[time_coord]
    return ds


def _describe_time(coord: "xr.DataArray") -> Dict[str, Any]:
    if coord.size == 0:
        return {"size": 0}
    values = np.asarray(coord.values)
    summary: Dict[str, Any] = {"size": int(coord.size)}
    if np.issubdtype(values.dtype, np.datetime64):
        summary["start"] = str(values[0])
        summary["end"] = str(values[-1])
    else:
        summary["min"] = float(np.nanmin(values))
        summary["max"] = float(np.nanmax(values))
    return summary


def _build_default_name(
    variable_id: str,
    experiment_id: str,
    source_id: Optional[str],
    member_id: Optional[str],
    ext: str = ".nc",
) -> str:
    parts = [variable_id, experiment_id]
    if source_id:
        parts.append(source_id)
    if member_id:
        parts.append(member_id)
    return "_".join(parts) + ext


def fetch_cmip6(
    variable_id: str,
    experiment_id: str,
    table_id: str,
    activity_id: str = "ScenarioMIP",
    source_id: Optional[str] = None,
    member_id: Optional[str] = None,
    grid_label: Optional[str] = None,
    dataset_key: Optional[str] = None,
    catalog_url: str = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json",
    storage_options: Optional[Dict[str, Any]] = None,
    zarr_kwargs: Optional[Dict[str, Any]] = None,
    time_subset: Optional[Dict[str, Any]] = None,
    lat_range: Optional[Sequence[float]] = None,
    lon_range: Optional[Sequence[float]] = None,
    drop_variables: Optional[Sequence[str]] = None,
    chunks: Optional[Dict[str, int]] = None,
    decode_times: bool = True,
    persist: bool = True,
    persist_dir: Optional[str] = "./data/cmip6",
    filename: Optional[str] = None,
    dry_run: bool = False,
) -> ToolResponse:
    """
    Retrieve CMIP6 data from the Pangeo intake-esm catalog.

    Args:
        variable_id: CMIP6 variable short name (e.g., "tas").
        experiment_id: Experiment identifier (e.g., "historical", "ssp245").
        table_id: CMIP6 table (e.g., "Amon", "day").
        activity_id: CMIP6 activity (default "ScenarioMIP").
        source_id: Specific model identifier (e.g., "CESM2").
        member_id: Ensemble member (e.g., "r1i1p1f1").
        grid_label: Optional grid label constraint (e.g., "gn").
        dataset_key: Optional explicit dataset key returned by intake; otherwise the first match is used.
        catalog_url: Location of the CMIP6 intake-esm JSON catalog.
        storage_options: Passed to fsspec when opening the zarr store (defaults to anon access).
        zarr_kwargs: Keyword overrides for xarray.open_zarr (e.g., consolidated, use_cftime).
        time_subset: Dict supporting keys {"start", "end", "years", "months", "days"} for filtering.
        lat_range: Two-element sequence [min_lat, max_lat] for spatial subset.
        lon_range: Two-element sequence [min_lon, max_lon] for spatial subset.
        drop_variables: Optional list of variable names to drop before persisting.
        chunks: Optional mapping of dimension name to chunk size before persistence.
        decode_times: Toggle CF-time decoding (passed to zarr open).
        persist: If True, save subset to NetCDF on disk and return file info.
        persist_dir: Target directory for NetCDF output (ignored when persist=False and filename=None).
        filename: Optional explicit output filename. If relative, resolved under persist_dir.
        dry_run: When True, do not hit the catalog; return the resolved request payload only.

    Returns:
        ToolResponse with metadata describing the matched catalog entries, selected dataset,
        optional output file path, and coordinate summaries.
    """
    request_filters: Dict[str, Any] = {
        "variable_id": variable_id,
        "experiment_id": experiment_id,
        "table_id": table_id,
        "activity_id": activity_id,
    }
    if source_id:
        request_filters["source_id"] = source_id
    if member_id:
        request_filters["member_id"] = member_id
    if grid_label:
        request_filters["grid_label"] = grid_label

    planned_output: Optional[str] = None
    if persist or filename:
        out_dir = _ensure_dir(persist_dir)
        if filename:
            out_path = Path(filename)
            if not out_path.is_absolute():
                out_path = out_dir / out_path
        else:
            out_path = out_dir / _build_default_name(
                variable_id=variable_id,
                experiment_id=experiment_id,
                source_id=source_id,
                member_id=member_id,
            )
        planned_output = str(out_path.expanduser())

    if dry_run:
        return _ok(
            {
                "request": request_filters,
                "catalog_url": catalog_url,
                "time_subset": time_subset,
                "lat_range": lat_range,
                "lon_range": lon_range,
                "drop_variables": list(drop_variables) if drop_variables else None,
                "chunks": chunks,
                "decode_times": decode_times,
                "persist": persist,
                "output_path": planned_output,
            }
        )

    try:
        import intake  # type: ignore
    except ImportError as exc:
        return _error(
            "intake and intake-esm are required for fetch_cmip6. "
            "Install with `pip install intake-esm`."
        )

    try:
        import xarray as xr  # type: ignore
    except ImportError:
        return _error("xarray is required for fetch_cmip6.")

    zarr_kwargs_final: Dict[str, Any] = {
        "consolidated": True,
        "use_cftime": True,
    }
    if zarr_kwargs:
        zarr_kwargs_final.update(zarr_kwargs)
    if chunks:
        zarr_kwargs_final["chunks"] = chunks
    if not decode_times:
        zarr_kwargs_final["decode_times"] = False

    storage_opts = {"token": "anon"}
    if storage_options:
        storage_opts.update(storage_options)

    try:
        col = intake.open_esm_datastore(catalog_url)  # type: ignore[attr-defined]
        cat = col.search(**request_filters)  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - network/catalog issues
        return _error(f"Unable to query CMIP6 catalog: {exc}")

    if cat.df.empty:
        return _error(f"No CMIP6 entries matched the filters: {request_filters}")

    available_keys = list(cat.keys())

    if dataset_key and dataset_key not in available_keys:
        return _error(
            f"dataset_key '{dataset_key}' not among available matches: {available_keys}"
        )

    try:
        datasets = cat.to_dataset_dict(
            zarr_kwargs=zarr_kwargs_final,
            storage_options=storage_opts,
        )
    except Exception as exc:  # pragma: no cover - remote access issues
        return _error(f"Failed to open CMIP6 datasets: {exc}")

    if not datasets:
        return _error("Catalog search returned keys but no xarray datasets.")

    selected_key = dataset_key or available_keys[0]
    if selected_key not in datasets:
        # Some catalogs re-key the dictionary; fall back to first dataset.
        selected_key = list(datasets.keys())[0]

    ds = datasets[selected_key]
    try:
        if drop_variables:
            ds = ds.drop_vars(drop_variables, errors="ignore")

        if time_subset:
            ds = _time_subset(ds, time_subset)

        if lat_range:
            lat_name = _first_existing_coord(ds, ("lat", "latitude", "nav_lat"))
            if not lat_name:
                raise ValueError("Dataset lacks latitude coordinate; cannot apply lat_range.")
            ds = _slice_by_bounds(ds, lat_name, lat_range)

        if lon_range:
            lon_name = _first_existing_coord(ds, ("lon", "longitude", "nav_lon"))
            if not lon_name:
                raise ValueError("Dataset lacks longitude coordinate; cannot apply lon_range.")
            ds = _slice_by_bounds(ds, lon_name, lon_range)

        if chunks:
            ds = ds.chunk(chunks)

        dataset_summary = {
            "selected_key": selected_key,
            "data_vars": sorted(list(ds.data_vars)),
            "coords": sorted(list(ds.coords)),
            "dims": {k: int(v) for k, v in ds.dims.items()},
            "attributes": {k: str(v) for k, v in ds.attrs.items()},
        }

        time_coord_name = _first_existing_coord(ds, ("time", "Time"))
        if time_coord_name:
            dataset_summary["time_coord"] = _describe_time(ds[time_coord_name])

        files_info: List[Dict[str, Any]] = []
        if persist:
            if not planned_output:
                return _error(
                    "persist=True but no output path could be resolved. "
                    "Provide persist_dir and/or filename."
                )
            out_path = Path(planned_output).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(out_path)
            size_bytes = out_path.stat().st_size if out_path.exists() else None
            files_info.append({"path": str(out_path), "size_bytes": size_bytes})

        try:
            catalog_records = json.loads(
                cat.df.reset_index().to_json(orient="records", date_format="iso")
            )
        except Exception:
            catalog_records = cat.df.reset_index().to_dict(orient="records")

        metadata = {
            "request": request_filters,
            "catalog_url": catalog_url,
            "available_dataset_keys": available_keys,
            "selected_dataset_key": selected_key,
            "catalog_matches": catalog_records,
            "storage_options": storage_opts,
            "zarr_kwargs": zarr_kwargs_final,
            "subset": {
                "time_subset": time_subset,
                "lat_range": lat_range,
                "lon_range": lon_range,
                "drop_variables": list(drop_variables) if drop_variables else None,
            },
            "dataset_summary": dataset_summary,
            "output_files": files_info,
        }
        return _ok(metadata)
    except Exception as exc:
        return _error(f"Unexpected failure during fetch_cmip6: {exc}")
    finally:
        try:
            ds.close()
        except Exception:
            pass
