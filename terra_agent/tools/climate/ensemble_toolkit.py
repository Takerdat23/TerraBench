"""
Ensemble Fusion & Confidence (EFC) toolkit.

This module bundles a suite of tool-call friendly helpers that take raw
single-model forecasts (e.g., Pangu and Aurora) and expose the common ensemble
workflow primitives: harmonising grids/units, computing anomalies versus
climatology, aggregating weighted means, deriving spread, translating spread
into event probabilities, fitting/applying an EMOS calibration, producing
confidence indices, augmenting the ensemble with lagged members, and verifying
skill against ERA5.

All public functions return ``ToolResponse`` objects so they can be registered
in the AgentScope ``Toolkit`` alongside the existing fetch/preprocess helpers.
Each function writes its primary artifact (NetCDF or JSON manifest) to disk and
describes the output path plus lightweight stats in the metadata payload.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

try:  # optional dependency used for smoothing
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - optional
    gaussian_filter = None  # type: ignore

LAT_CANDIDATES = ("latitude", "lat", "Latitude", "LAT", "y")
LON_CANDIDATES = ("longitude", "lon", "Longitude", "LON", "x")
TIME_CANDIDATES = ("time", "valid_time", "step", "forecast_time", "forecast_reference_time")

DEFAULT_OUTPUT_DIR = Path("./data/ensemble")
LOGGER = logging.getLogger("climate_agent.ensemble_toolkit")
CMIP6_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "cmip6"
EVENT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "events"

VARIABLE_SYNONYM_SETS: tuple[set[str], ...] = (
    {"t2m", "2t", "tas", "temperature_2m"},
    {"u10", "10u"},
    {"v10", "10v"},
    {"msl", "mslp", "mean_sea_level_pressure"},
    {"tp", "total_precipitation", "precip", "prcp"},
)

ALIAS_LOOKUP: Dict[str, set[str]] = {}
for group in VARIABLE_SYNONYM_SETS:
    canonical_group = {name.lower() for name in group}
    for name in group:
        ALIAS_LOOKUP.setdefault(name, set()).update(group)
        # keep lowercase lookups as well
        ALIAS_LOOKUP.setdefault(name.lower(), set()).update(group)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, xr.DataArray):
        return value.item() if value.ndim == 0 else str(value.values)
    return value


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


def _ensure_output_dir(output_dir: Optional[str]) -> Path:
    path = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def _ensure_sequence(value: Sequence[str] | str) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _list_numeric_vars(ds: xr.Dataset) -> List[str]:
    return [name for name, var in ds.data_vars.items() if np.issubdtype(var.dtype, np.number)]


def _normalize_member_names(paths: Sequence[Path], names: Optional[Sequence[str]]) -> List[str]:
    """
    Resolve member names; if a mismatch occurs, fall back to path stems with a warning.
    """
    if names:
        if len(names) != len(paths):
            LOGGER.warning(
                "Length of 'names' (%s) did not match number of files (%s); falling back to file stems.",
                len(names),
                len(paths),
            )
            return [path.stem for path in paths]
        return [str(name) for name in names]
    return [path.stem for path in paths]


def _expand_variable_aliases(var_name: str) -> List[str]:
    candidates = []
    lookup_key = var_name if var_name in ALIAS_LOOKUP else var_name.lower()
    if lookup_key in ALIAS_LOOKUP:
        for alias in ALIAS_LOOKUP[lookup_key]:
            if alias not in candidates:
                candidates.append(alias)
    if var_name not in candidates:
        candidates.insert(0, var_name)
    return candidates


def _resolve_variable(ds: xr.Dataset, variable: Optional[str]) -> str:
    numeric_vars = _list_numeric_vars(ds)
    if variable:
        candidates = _expand_variable_aliases(variable)
        # Exact-name search first
        for candidate in candidates:
            if candidate in ds and np.issubdtype(ds[candidate].dtype, np.number):
                return candidate
        # Case-insensitive fallback
        lower_map = {name.lower(): name for name in ds.data_vars}
        for candidate in candidates:
            lower = candidate.lower()
            if lower in lower_map and np.issubdtype(ds[lower_map[lower]].dtype, np.number):
                return lower_map[lower]
        # If user-specified variable is non-numeric, fall back to first numeric option
        if numeric_vars:
            return sorted(numeric_vars)[0]
        raise KeyError(
            f"Variable '{variable}' (aliases: {', '.join(candidates)}) not found as numeric in dataset. "
            f"Numeric candidates: none."
        )
    if not numeric_vars:
        raise ValueError("Dataset does not contain any numeric data variables.")
    # Deterministic choice to avoid ambiguity
    return sorted(numeric_vars)[0]


def _load_dataarray(path: Path, variable: Optional[str]) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        var_name = _resolve_variable(ds, variable)
        data = ds[var_name].load()
    return data


def _ensure_numeric(da: xr.DataArray, path: Path, var_name: str) -> xr.DataArray:
    """
    Guard against selecting non-numeric variables (e.g., datetime) that cannot be averaged.
    """
    if np.issubdtype(da.dtype, np.number):
        return da
    raise ValueError(
        f"Variable '{var_name}' in {path} has non-numeric dtype '{da.dtype}'. "
        "Select a numeric field (e.g., temperature, wind) or supply 'variable_map' to point to the correct variable."
    )


def _detect_coord_name(obj: xr.DataArray | xr.Dataset, candidates: Sequence[str]) -> Optional[str]:
    for cand in candidates:
        if cand in obj.coords:
            return cand
    for cand in candidates:
        if cand in obj.dims:
            return cand
    return None


def _extract_grid(da: xr.DataArray) -> Tuple[str, str, np.ndarray, np.ndarray]:
    lat_name = _detect_coord_name(da, LAT_CANDIDATES)
    lon_name = _detect_coord_name(da, LON_CANDIDATES)
    if not lat_name or not lon_name:
        raise ValueError("Unable to locate latitude/longitude coordinates in the dataset.")
    lat_vals = np.asarray(da.coords[lat_name].values, dtype=float)
    lon_vals = np.asarray(da.coords[lon_name].values, dtype=float)
    if lat_vals.ndim != 1 or lon_vals.ndim != 1:
        raise ValueError("Latitude and longitude coordinates must be 1-D arrays.")
    return lat_name, lon_name, lat_vals, lon_vals


def _normalize_lon_values(lon_vals: np.ndarray, lon_range: str) -> np.ndarray:
    lon_range = lon_range.lower()
    if lon_range in {"-180:180", "negpos", "180"}:
        return ((lon_vals + 180.0) % 360.0) - 180.0
    if lon_range in {"0:360", "0360"}:
        lons = lon_vals % 360.0
        lons[lons == 360.0] = 0.0
        return lons
    # auto: keep native; caller may still sort later
    return lon_vals


def _normalize_grid_values(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    *,
    lon_range: str = "-180:180",
    sort_lats: bool = True,
    sort_lons: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    lons = _normalize_lon_values(lon_vals, lon_range)
    lats = np.asarray(lat_vals, dtype=float)
    if sort_lats:
        lats = np.sort(lats)[::-1]  # descending (north to south)
    if sort_lons:
        lons = np.sort(lons)
    return lats, lons


def _normalize_coords_on_dataarray(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    *,
    lon_range: str = "-180:180",
    sort_lats: bool = True,
    sort_lons: bool = True,
) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    lat_vals = np.asarray(da.coords[lat_name].values, dtype=float)
    lon_vals = np.asarray(da.coords[lon_name].values, dtype=float)
    lons = _normalize_lon_values(lon_vals, lon_range)
    da = da.assign_coords({lon_name: (da[lon_name].dims, lons)})
    if sort_lats:
        da = da.sortby(lat_name, ascending=False)
    if sort_lons:
        da = da.sortby(lon_name)
    lat_vals = np.asarray(da.coords[lat_name].values, dtype=float)
    lon_vals = np.asarray(da.coords[lon_name].values, dtype=float)
    return da, lat_vals, lon_vals


def _grid_resolution(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    diffs = np.abs(np.diff(values))
    if diffs.size == 0:
        return float("nan")
    return float(np.nanmean(diffs))


def _coerce_time_coord(da: xr.DataArray, mode: Optional[str]) -> xr.DataArray:
    """
    Convert datetime time coordinates to numeric to avoid dtype promotion issues.

    mode=None: no change.
    mode='index': replace time coord with 0..N-1.
    mode='lead_hours': replace with hours since first time step (float).
    """
    if mode is None:
        return da
    time_name = _detect_coord_name(da, TIME_CANDIDATES)
    if not time_name or time_name not in da.coords:
        return da
    times = da[time_name].values
    if mode == "index":
        new_time = np.arange(da.sizes[time_name], dtype=np.float32)
    elif mode == "lead_hours":
        ref = times[0]
        deltas = (times - ref) / np.timedelta64(1, "h")
        new_time = np.asarray(deltas, dtype=np.float32)
    else:
        return da
    return da.assign_coords({time_name: new_time})


def _grid_from_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        lat_name = _detect_coord_name(ds, LAT_CANDIDATES)
        lon_name = _detect_coord_name(ds, LON_CANDIDATES)
        if not lat_name or not lon_name:
            raise ValueError(f"Unable to find latitude/longitude coordinates in {path}.")
        lat_vals = np.asarray(ds.coords[lat_name].values, dtype=float)
        lon_vals = np.asarray(ds.coords[lon_name].values, dtype=float)
    return lat_vals, lon_vals


def _interp_to_reference(reference: xr.DataArray, candidate: xr.DataArray) -> xr.DataArray:
    """Resample candidate so it shares the same coordinates as reference."""
    interp_dims: Dict[str, xr.DataArray] = {}
    for dim in reference.dims:
        if dim not in candidate.dims:
            continue
        ref_coord = reference.coords[dim]
        cand_coord = candidate.coords[dim]
        if ref_coord.size != cand_coord.size or not np.allclose(
            np.asarray(ref_coord),
            np.asarray(cand_coord),
            rtol=0.0,
            atol=1e-6,
        ):
            interp_dims[dim] = ref_coord
    if not interp_dims:
        return candidate
    try:
        return candidate.interp({dim: ref for dim, ref in interp_dims.items()}, kwargs={"fill_value": "extrapolate"})
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning(
            "Interpolation to match reference grid failed (%s). Falling back to nearest-neighbour reindex.",
            exc,
        )
        return candidate.reindex({dim: ref for dim, ref in interp_dims.items()}, method="nearest", tolerance=1e-6)


def _apply_accumulation(
    da: xr.DataArray,
    *,
    accumulation: Optional[str],
    accumulation_hours: Optional[int],
) -> xr.DataArray:
    if not accumulation:
        return da
    time_dim = _detect_coord_name(da, TIME_CANDIDATES)
    if not time_dim:
        raise ValueError("Accumulation requested but no time dimension found in the data.")

    accumulation = accumulation.lower()
    if accumulation not in {"sum", "mean", "max"}:
        raise ValueError("accumulation must be one of {'sum','mean','max'}.")

    if accumulation_hours:
        rule = f"{int(accumulation_hours)}H"
        if accumulation == "sum":
            return da.resample({time_dim: rule}).sum(skipna=True, keep_attrs=True)
        if accumulation == "mean":
            return da.resample({time_dim: rule}).mean(skipna=True, keep_attrs=True)
        if accumulation == "max":
            return da.resample({time_dim: rule}).max(skipna=True, keep_attrs=True)

    if accumulation == "sum":
        return da.sum(time_dim, keep_attrs=True)
    if accumulation == "mean":
        return da.mean(time_dim, keep_attrs=True)
    return da.max(time_dim, keep_attrs=True)


def _apply_transform(da: xr.DataArray, transform: Optional[str]) -> xr.DataArray:
    if not transform:
        return da
    transform = transform.lower()
    if transform == "sqrt":
        return xr.apply_ufunc(np.sqrt, da.clip(min=0.0))
    if transform in {"log", "ln"}:
        return xr.apply_ufunc(np.log, da.clip(min=1e-12))
    if transform in {"log1p", "ln1p"}:
        return xr.apply_ufunc(np.log1p, da.clip(min=0.0))
    raise ValueError("transform must be one of {'sqrt','log','log1p'}")


def _write_dataset(data_vars: Mapping[str, xr.DataArray], path: Path) -> None:
    dataset = xr.Dataset({name: array.astype(np.float32) for name, array in data_vars.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)


def _time_dim_or_raise(da: xr.DataArray) -> str:
    tname = _detect_coord_name(da, TIME_CANDIDATES)
    if not tname:
        raise ValueError("Unable to find a time coordinate in the dataset.")
    return tname


def _apply_region_mask(da: xr.DataArray, mask_path: Optional[str], mask_variable: Optional[str]) -> xr.DataArray:
    if not mask_path:
        return da
    mask = _load_dataarray(_to_path(mask_path), mask_variable or None)
    mask, da = xr.align(mask, da, join="inner")
    return da.where(mask > 0.5)


def _persistence_filter(event: xr.DataArray, time_dim: str, persistence: int) -> xr.DataArray:
    if persistence <= 1:
        return event
    rolled = event.astype(np.int32).rolling({time_dim: persistence}, min_periods=persistence).sum()
    return (rolled >= persistence).astype(bool)


def _is_precip_var(var: str) -> bool:
    lower = var.lower()
    return any(key in lower for key in ("tp", "precip", "pr", "rain"))


def _quantile_adjust_np(values: np.ndarray, hist_q: np.ndarray, obs_q: np.ndarray, probs: np.ndarray, method: str, is_precip: bool) -> np.ndarray:
    """Vectorised quantile delta/equidistant mapping for one spatial point."""
    if values.size == 0:
        return values
    method = method.lower()
    method = method if method in {"qdm", "eqm"} else "qdm"
    hist_q = np.asarray(hist_q, dtype=float)
    obs_q = np.asarray(obs_q, dtype=float)
    probs = np.asarray(probs, dtype=float)
    hist_q = np.nan_to_num(hist_q, nan=np.nanmedian(hist_q))
    obs_q = np.nan_to_num(obs_q, nan=np.nanmedian(obs_q))
    # Avoid repeated division by zero for precipitation-style corrections
    safe_hist = np.where(hist_q == 0.0, 1e-6, hist_q)

    def adjust(x: float) -> float:
        prob = np.interp(x, hist_q, probs, left=probs[0], right=probs[-1])
        hist_val = np.interp(prob, probs, hist_q)
        obs_val = np.interp(prob, probs, obs_q)
        if method == "eqm":
            if is_precip:
                return x * (obs_val / max(hist_val, 1e-6))
            return x + (obs_val - hist_val)
        # QDM
        if is_precip:
            return x * (obs_val / max(hist_val, 1e-6))
        return x + (obs_val - hist_val)

    vectorised = np.vectorize(adjust, otypes=[float])
    return vectorised(values)


def harmonize_grid(
    model_files: Sequence[str],
    variable: str,
    *,
    names: Optional[Sequence[str]] = None,
    target_grid_path: Optional[str] = None,
    target_latitudes: Optional[Sequence[float]] = None,
    target_longitudes: Optional[Sequence[float]] = None,
    target_grid_strategy: str = "coarsest",
    lon_range: str = "-180:180",
    sort_lats: bool = True,
    sort_lons: bool = True,
    interp_method: str = "linear",
    accumulation: Optional[str] = None,
    accumulation_hours: Optional[int] = None,
    transform: Optional[str] = None,
    output_dir: str = "./data/ensemble",
) -> ToolResponse:
    """
    Align multiple model fields to a shared grid/units so they can be blended.

    Enhancements:
    - Optional grid strategy picks a target grid (coarsest/finest/first) when none is provided.
    - Coordinate hygiene: convert lon ranges, sort lats (N→S) and lons (W→E) for consistency.
    - Names auto-fallback to file stems when length mismatches.
    """
    try:
        if not model_files:
            return _error("model_files must contain at least one NetCDF path.")
        paths = [_to_path(path) for path in model_files]
        member_names = _normalize_member_names(paths, names)
        target_grid_strategy = (target_grid_strategy or "coarsest").lower()

        # Precompute grid candidates for strategy selection
        grid_candidates: List[Dict[str, Any]] = []
        for path in paths:
            with xr.open_dataset(path) as ds:
                var_name = _resolve_variable(ds, variable)
                da = ds[var_name]
                lat_name, lon_name, lat_vals, lon_vals = _extract_grid(da)
            norm_lats, norm_lons = _normalize_grid_values(
                lat_vals, lon_vals, lon_range=lon_range, sort_lats=sort_lats, sort_lons=sort_lons
            )
            grid_candidates.append(
                {
                    "path": path,
                    "lat_name": lat_name,
                    "lon_name": lon_name,
                    "lats": norm_lats,
                    "lons": norm_lons,
                    "dlat": _grid_resolution(norm_lats),
                    "dlon": _grid_resolution(norm_lons),
                }
            )

        target_lats: Optional[np.ndarray] = None
        target_lons: Optional[np.ndarray] = None
        if target_latitudes is not None:
            target_lats = np.asarray(list(map(float, target_latitudes)))
        if target_longitudes is not None:
            target_lons = np.asarray(list(map(float, target_longitudes)))

        if target_grid_path:
            grid_path = _to_path(target_grid_path)
            lat_vals, lon_vals = _grid_from_dataset(grid_path)
            target_lats, target_lons = _normalize_grid_values(
                lat_vals, lon_vals, lon_range=lon_range, sort_lats=sort_lats, sort_lons=sort_lons
            )
        elif target_lats is None or target_lons is None:
            if grid_candidates:
                if target_grid_strategy == "finest":
                    idx = int(
                        np.nanargmin(
                            [max(gc.get("dlat", np.nan), gc.get("dlon", np.nan)) for gc in grid_candidates]
                        )
                    )
                elif target_grid_strategy == "first":
                    idx = 0
                else:  # default coarsest
                    idx = int(
                        np.nanargmax(
                            [max(gc.get("dlat", np.nan), gc.get("dlon", np.nan)) for gc in grid_candidates]
                        )
                    )
                target_lats = target_lats if target_lats is not None else grid_candidates[idx]["lats"]
                target_lons = target_lons if target_lons is not None else grid_candidates[idx]["lons"]
                LOGGER.info(
                    "harmonize_grid: selected target grid from %s using strategy=%s (dlat=%.4f, dlon=%.4f).",
                    grid_candidates[idx]["path"],
                    target_grid_strategy,
                    grid_candidates[idx]["dlat"],
                    grid_candidates[idx]["dlon"],
                )

        output_base = _ensure_output_dir(output_dir)
        artifacts: List[Dict[str, Any]] = []
        summaries: Dict[str, Any] = {}
        for path, name in zip(paths, member_names):
            da = _load_dataarray(path, variable)
            lat_name, lon_name, lat_vals, lon_vals = _extract_grid(da)
            da, lat_vals, lon_vals = _normalize_coords_on_dataarray(
                da, lat_name, lon_name, lon_range=lon_range, sort_lats=sort_lats, sort_lons=sort_lons
            )
            da = _apply_accumulation(da, accumulation=accumulation, accumulation_hours=accumulation_hours)
            if target_lats is not None and target_lons is not None:
                interp_kwargs = {lat_name: target_lats, lon_name: target_lons}
                da = da.interp(interp_kwargs, method=interp_method, kwargs={"fill_value": "extrapolate"})
            else:
                target_lats, target_lons = lat_vals, lon_vals
            da = _apply_transform(da, transform)
            da = da.astype(np.float32)
            out_path = output_base / f"{name}_{variable}_harmonized.nc"
            _write_dataset({variable: da}, out_path)
            stats = {
                "name": name,
                "path": str(out_path),
                "dims": {dim: int(size) for dim, size in da.sizes.items()},
                "min": float(da.min().item()),
                "max": float(da.max().item()),
                "source_grid": {
                    "lat_count": int(len(lat_vals)),
                    "lon_count": int(len(lon_vals)),
                },
                "target_grid": {
                    "lat_count": int(len(target_lats)) if target_lats is not None else None,
                    "lon_count": int(len(target_lons)) if target_lons is not None else None,
                },
                "lon_convention": lon_range,
            }
            artifacts.append(stats)
            summaries[name] = stats

        grid_summary = {
            "lat_count": int(len(target_lats)) if target_lats is not None else None,
            "lon_count": int(len(target_lons)) if target_lons is not None else None,
            "lon_convention": lon_range,
            "strategy": target_grid_strategy,
        }
        return _ok(
            {
                "artifacts": artifacts,
                "variable": variable,
                "grid": grid_summary,
                "accumulation": accumulation,
                "accumulation_hours": accumulation_hours,
                "transform": transform,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _error(f"harmonize_grid failed: {exc}")


def smooth_gaussian_field(
    field_path: str,
    variable: Optional[str] = None,
    *,
    sigma_pixels: float = 1.0,
    mode: str = "nearest",
    truncate: float = 3.0,
    output_variable: Optional[str] = None,
    output_path: Optional[str] = None,
    drop_singleton_dims: bool = True,
) -> ToolResponse:
    """
    Apply Gaussian smoothing to a gridded field (e.g., ERA5 resampled to satellite grid).

    Smoothing the anomaly/forecast after regridding removes blocky pixel artifacts and
    produces more continuous gradients for visualization or downstream VLM use.

    Args:
        field_path: NetCDF file containing the field to smooth.
        variable: Variable name inside the NetCDF (defaults to first numeric variable).
        sigma_pixels: Gaussian sigma expressed in output-grid pixels.
        mode: Boundary handling for scipy.ndimage.gaussian_filter (default: 'nearest').
        truncate: Kernel truncation in standard deviations.
        output_variable: Optional name for the smoothed variable.
        output_path: Optional explicit output path. Defaults to <stem>_<var>_gauss.nc.
    """
    if gaussian_filter is None:
        return _error("scipy is required for Gaussian smoothing; install scipy and retry.")
    try:
        sigma_val = float(sigma_pixels)
    except Exception:
        return _error("sigma_pixels must be a numeric value.")
    try:
        da = _load_dataarray(_to_path(field_path), variable)
        var_name = da.name or variable or "field"
        time_dim = _detect_coord_name(da, TIME_CANDIDATES)
        lat_name, lon_name, *_ = _extract_grid(da)

        def _smooth_slice(arr: np.ndarray) -> np.ndarray:
            return gaussian_filter(arr, sigma=sigma_val, mode=mode, truncate=truncate)

        # Bring lat/lon to the end, flatten everything else, smooth per slice, then reshape back.
        dim_order: List[str] = [d for d in da.dims if d not in (lat_name, lon_name)] + [lat_name, lon_name]
        transposed = da.transpose(*dim_order)
        shape = transposed.shape
        lead = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
        arr = transposed.values.reshape((lead, shape[-2], shape[-1]))

        smoothed_flat = np.empty_like(arr)
        for idx in range(lead):
            smoothed_flat[idx] = _smooth_slice(arr[idx])

        smoothed = smoothed_flat.reshape(shape)
        smoothed_da = xr.DataArray(smoothed, coords=transposed.coords, dims=dim_order, attrs=da.attrs)
        smoothed_da = smoothed_da.transpose(*da.dims)

        smoothed_da = smoothed_da.astype(np.float32)
        if drop_singleton_dims:
            drop_dims = [d for d, sz in smoothed_da.sizes.items() if sz == 1 and d not in (lat_name, lon_name)]
            if drop_dims:
                smoothed_da = smoothed_da.isel({d: 0 for d in drop_dims}, drop=True)
        out_path = Path(output_path) if output_path else Path(field_path).with_name(
            f"{Path(field_path).stem}_{output_variable or (var_name or 'field')}_gauss.nc"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_dataset({output_variable or (var_name or "smoothed"): smoothed_da}, out_path)

        return _ok(
            {
                "output_path": str(out_path),
                "variable": output_variable or (var_name or "smoothed"),
                "sigma_pixels": sigma_val,
                "mode": mode,
                "truncate": truncate,
                "dims": {dim: int(size) for dim, size in smoothed_da.sizes.items()},
                "min": float(smoothed_da.min().item()),
                "max": float(smoothed_da.max().item()),
                "notes": "Gaussian smoothing applied after regridding to reduce blocky artifacts.",
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _error(f"smooth_gaussian_field failed: {exc}")


def anomaly_and_climo(
    field_path: str,
    variable: str,
    *,
    climo_mean_path: str,
    climo_std_path: Optional[str] = None,
    climo_mean_variable: Optional[str] = None,
    climo_std_variable: Optional[str] = None,
    output_variable: Optional[str] = None,
    emit_climo: bool = True,
    output_dir: str = "./data/ensemble",
) -> ToolResponse:
    """
    Convert a forecast field into anomalies relative to climatology and return sigma_ref.
    """
    try:
        field = _load_dataarray(_to_path(field_path), variable)
        clim_mean = _load_dataarray(_to_path(climo_mean_path), climo_mean_variable or variable)
        clim_std: Optional[xr.DataArray] = None
        if climo_std_path:
            clim_std = _load_dataarray(_to_path(climo_std_path), climo_std_variable or variable)

        aligned = xr.align(field, clim_mean, join="exact")
        field_aligned, clim_aligned = aligned
        if climo_std_path:
            clim_std = xr.align(clim_std, field_aligned, join="exact")[0]
        else:
            time_dim = _detect_coord_name(field_aligned, TIME_CANDIDATES)
            if time_dim:
                clim_std = field_aligned.std(dim=time_dim, keep_attrs=True)
            else:
                clim_std = xr.zeros_like(field_aligned) + np.nan

        anomaly = (field_aligned - clim_aligned).astype(np.float32)
        sigma_ref = clim_std.astype(np.float32)
        out_dir = _ensure_output_dir(output_dir)
        var_name = output_variable or f"{variable}_anomaly"
        dataset_vars = {var_name: anomaly, "sigma_ref": sigma_ref}
        if emit_climo:
            dataset_vars["climatology"] = clim_aligned.astype(np.float32)
        out_path = out_dir / f"{Path(field_path).stem}_{var_name}.nc"
        _write_dataset(dataset_vars, out_path)
        return _ok(
            {
                "output_path": str(out_path),
                "variable": var_name,
                "sigma_variable": "sigma_ref",
                "includes_climatology": emit_climo,
                "source_field": str(field_path),
            }
        )
    except Exception as exc:  # pragma: no cover
        return _error(f"anomaly_and_climo failed: {exc}")


def _stack_members(
    member_paths: Sequence[str],
    variable: str,
    variable_map: Optional[Mapping[str, str]] = None,
    coerce_time: Optional[str] = None,
) -> Tuple[xr.DataArray, List[str]]:
   
    if not member_paths:
        raise ValueError("member_paths must contain at least one NetCDF path.")
    paths = [_to_path(path) for path in member_paths]
    names = _normalize_member_names(paths, None)
   

    dataarrays: List[xr.DataArray] = []
    reference: xr.DataArray | None = None
    for path in paths:
        override_var = None
        if variable_map:
            for key in (path.as_posix(), path.name, path.stem):
                if key in variable_map:
                    override_var = variable_map[key]
                    break
        var_name = override_var or variable
        da = _load_dataarray(path, var_name)
        # print(f"[Ensemble toolkit] Loaded {var_name} from {path} with shape {da.shape} and dtype {da.dtype}")
        da = _ensure_numeric(da, path, var_name or "<unspecified>")
        da = da.astype(np.float32)
        # Auto-coerce datetime-like time coords to numeric index when coerce_time is not provided.
        if coerce_time is None:
            time_name = _detect_coord_name(da, TIME_CANDIDATES)
            if time_name and time_name in da.coords and not np.issubdtype(da[time_name].dtype, np.number):
                da = _coerce_time_coord(da, "index")
        else:
            da = _coerce_time_coord(da, coerce_time)
  
        if reference is None:
            reference = da
        else:
            da = _interp_to_reference(reference, da)
        # print(f"[Ensemble toolkit] Appending: {da.shape}")
        dataarrays.append(da)

    try:
        aligned = xr.align(*dataarrays, join="exact")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning(
            "Exact coordinate match failed for ensemble members (%s). "
            "Cropping to the common intersection with join='inner'.",
            exc,
        )
        aligned = xr.align(*dataarrays, join="inner")

    try:
        stacked = xr.concat(aligned, dim="member")
    except Exception as exc:
        dtypes = {name: str(da.dtype) for name, da in zip(names, aligned)}
        raise TypeError(
            f"Failed to concatenate members due to dtype mismatch {dtypes}. "
            "Ensure all selected variables are numeric and supply variable_map if needed."
        ) from exc
    stacked = stacked.assign_coords(member=("member", names))
    # print("[Ensemble toolkit] Stacked data array shape:", stacked.shape)
    # print("[Ensemble toolkit] Stack member completed")
    return stacked, names


def aggregate_ensemble(
    member_paths: Sequence[str],
    variable: str,
    *,
    variable_map: Optional[Mapping[str, str]] = None,
    coerce_time: Optional[str] = None,
    weights: Optional[Sequence[float]] = None,
    output_variable: str = "ensemble_mean",
    output_path: Optional[str] = None,
    output_dir: str = "./data/ensemble",
) -> ToolResponse:
    """
    Compute the weighted multi-model mean (best-guess field).
    """
    try:
        stacked, member_names = _stack_members(
            member_paths,
            variable,
            variable_map=variable_map,
            coerce_time=coerce_time,
        )
        count = stacked.sizes["member"]
        w = np.ones(count, dtype=float) if weights is None else np.asarray(list(map(float, weights)))
        if w.size != count:
            return _error("weights length must match number of members.")
        if np.any(w < 0):
            return _error("weights must be non-negative.")
        weight_sum = float(w.sum())
        if weight_sum == 0:
            return _error("weights sum to zero.")
        w_norm = w / weight_sum
        weights_da = xr.DataArray(w_norm, dims=("member",), coords={"member": stacked.member})
        mu = (stacked * weights_da).sum(dim="member").astype(np.float32)
        out_dir = _ensure_output_dir(output_dir)
        out_path = Path(output_path).expanduser() if output_path else out_dir / f"{output_variable}.nc"
        _write_dataset({output_variable: mu}, out_path)
        return _ok(
            {
                "output_path": str(out_path),
                "variable": output_variable,
                "weights": {name: float(weight) for name, weight in zip(member_names, w_norm)},
            }
        )
    except Exception as exc:  # pragma: no cover
        return _error(f"aggregate_ensemble failed: {exc}")


def ensemble_spread(
    member_paths: Sequence[str],
    variable: str,
    *,
    variable_map: Optional[Mapping[str, str]] = None,
    coerce_time: Optional[str] = None,
    weights: Optional[Sequence[float]] = None,
    mean_path: Optional[str] = None,
    mean_variable: Optional[str] = None,
    output_variable: str = "ensemble_spread",
    output_path: Optional[str] = None,
    output_dir: str = "./data/ensemble",
) -> ToolResponse:
    """
    Compute the weighted ensemble standard deviation (spread).
    """
    try:
        stacked, _ = _stack_members(
            member_paths,
            variable,
            variable_map=variable_map,
            coerce_time=coerce_time,
        )
        count = stacked.sizes["member"]
        w = np.ones(count, dtype=float) if weights is None else np.asarray(list(map(float, weights)))
        if w.size != count:
            return _error("weights length must match number of members.")
        w_norm = w / w.sum()
        weights_da = xr.DataArray(w_norm, dims=("member",), coords={"member": stacked.member})

        if mean_path:
            mean_da = _load_dataarray(_to_path(mean_path), mean_variable or output_variable)
            if coerce_time is None:
                time_name = _detect_coord_name(mean_da, TIME_CANDIDATES)
                if time_name and time_name in mean_da.coords and not np.issubdtype(mean_da[time_name].dtype, np.number):
                    mean_da = _coerce_time_coord(mean_da, "index")
            elif coerce_time:
                mean_da = _coerce_time_coord(mean_da, coerce_time)
            mean_da, stacked = xr.align(mean_da, stacked, join="exact")
        else:
            mean_da = (stacked * weights_da).sum(dim="member")

        variance = ((stacked - mean_da) ** 2 * weights_da).sum(dim="member")
        spread = np.sqrt(variance.clip(min=0.0)).astype(np.float32)
        out_dir = _ensure_output_dir(output_dir)
        out_path = Path(output_path).expanduser() if output_path else out_dir / f"{output_variable}.nc"
        _write_dataset({output_variable: spread}, out_path)
        return _ok(
            {
                "output_path": str(out_path),
                "variable": output_variable,
            }
        )
    except Exception as exc:  # pragma: no cover
        return _error(f"ensemble_spread failed: {exc}")


def normalized_spread(
    spread_path: str,
    sigma_ref_path: str,
    *,
    spread_variable: str = "ensemble_spread",
    sigma_variable: str = "sigma_ref",
    output_variable: str = "normalized_spread",
    epsilon: float = 1e-6,
    output_path: Optional[str] = None,
    output_dir: str = "./data/ensemble",
) -> ToolResponse:
    """
    Normalize the ensemble spread by the climatological sigma_ref.
    """
    try:
        spread = _load_dataarray(_to_path(spread_path), spread_variable)
        sigma = _load_dataarray(_to_path(sigma_ref_path), sigma_variable)
        spread, sigma = xr.align(spread, sigma, join="exact")
        s_norm = (spread / (sigma + epsilon)).astype(np.float32)
        out_dir = _ensure_output_dir(output_dir)
        out_path = Path(output_path).expanduser() if output_path else out_dir / f"{output_variable}.nc"
        _write_dataset({output_variable: s_norm}, out_path)
        return _ok(
            {
                "output_path": str(out_path),
                "variable": output_variable,
                "epsilon": epsilon,
            }
        )
    except Exception as exc:  # pragma: no cover
        return _error(f"normalized_spread failed: {exc}")


def event_probability(
    member_paths: Sequence[str],
    variable: str,
    *,
    threshold: float,
    mode: str = "votes",
    mean_path: Optional[str] = None,
    mean_variable: str = "ensemble_mean",
    spread_path: Optional[str] = None,
    spread_variable: str = "ensemble_spread",
    min_spread: float = 1e-6,
    output_variable: str = "event_probability",
    output_path: Optional[str] = None,
    output_dir: str = "./data/ensemble",
) -> ToolResponse:
    """
    Convert a tiny ensemble into event probabilities using votes or Gaussian proxy.
    """
    try:
        stacked, _ = _stack_members(member_paths, variable)
        if mode == "votes":
            prob = (stacked >= threshold).mean(dim="member").astype(np.float32)
        elif mode == "gaussian":
            if not mean_path or not spread_path:
                return _error("Gaussian mode requires both mean_path and spread_path.")
            mean_da = _load_dataarray(_to_path(mean_path), mean_variable)
            spread_da = _load_dataarray(_to_path(spread_path), spread_variable)
            sample = stacked.isel(member=0)
            mean_da, spread_da, _ = xr.align(mean_da, spread_da, sample, join="exact")
            z = (threshold - mean_da) / (spread_da + min_spread)
            erfc = xr.apply_ufunc(np.erfc, z / np.sqrt(2.0))
            prob = (0.5 * erfc).astype(np.float32)
        else:
            return _error("mode must be 'votes' or 'gaussian'.")

        out_dir = _ensure_output_dir(output_dir)
        out_path = Path(output_path).expanduser() if output_path else out_dir / f"{output_variable}.nc"
        _write_dataset({output_variable: prob}, out_path)
        return _ok(
            {
                "output_path": str(out_path),
                "variable": output_variable,
                "threshold": threshold,
                "mode": mode,
            }
        )
    except Exception as exc:  # pragma: no cover
        return _error(f"event_probability failed: {exc}")


def _fit_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def emos_calibration(
    member_paths: Sequence[str],
    variable: str,
    *,
    truth_path: Optional[str] = None,
    truth_variable: Optional[str] = None,
    coeffs: Optional[Mapping[str, float]] = None,
    mode: str = "fit",
    spread_path: Optional[str] = None,
    spread_variable: str = "ensemble_spread",
    min_variance: float = 1e-4,
    output_dir: str = "./data/ensemble",
    output_path: Optional[str] = None,
) -> ToolResponse:
    """
    Fit or apply a simple EMOS (NGR) calibration for two-member ensembles.
    """
    try:
        stacked, names = _stack_members(member_paths, variable)
        member_count = stacked.sizes["member"]
        if member_count < 2:
            return _error("EMOS calibration requires at least two members.")

        if mode == "fit":
            if truth_path is None:
                return _error("truth_path is required in fit mode.")
            truth = _load_dataarray(_to_path(truth_path), truth_variable or variable)
            truth, stacked = xr.align(truth, stacked, join="inner")
            member_values = stacked.values
            samples = member_values.reshape(member_count, -1).T  # (n_samples, members)
            target = truth.values.reshape(-1)
            valid = np.isfinite(target)
            valid &= np.all(np.isfinite(samples), axis=1)
            if not np.any(valid):
                return _error("No valid samples available for EMOS fitting.")
            X = np.concatenate([np.ones((valid.sum(), 1)), samples[valid]], axis=1)
            y = target[valid].reshape(-1, 1)
            beta = _fit_linear_regression(X, y)
            residuals = y - X @ beta
            if spread_path:
                spread_da = _load_dataarray(_to_path(spread_path), spread_variable)
                spread_da, truth = xr.align(spread_da, truth, join="inner")
                spread_vals = spread_da.values.reshape(-1)
                S2 = (spread_vals[valid] ** 2).reshape(-1, 1)
            else:
                variance = ((stacked - stacked.mean(dim="member")) ** 2).mean(dim="member")
                variance_vals = variance.values.reshape(-1)
                S2 = variance_vals[valid].reshape(-1, 1)
            spread_design = np.concatenate([np.ones_like(S2), S2], axis=1)
            gamma = _fit_linear_regression(spread_design, residuals ** 2)
            coeffs_payload = {
                "a": float(beta[0, 0]),
                **{f"b{i+1}": float(beta[i + 1, 0]) for i in range(member_count)},
                "c": float(gamma[0, 0]),
                "d": float(gamma[1, 0]),
            }
            return _ok({"coefficients": coeffs_payload, "mode": "fit", "members": names})

        if coeffs is None:
            return _error("coeffs must be supplied when mode='apply'.")
        a = float(coeffs.get("a", 0.0))
        bs = [float(coeffs.get(f"b{i+1}", 0.0)) for i in range(member_count)]
        if len(bs) != member_count:
            return _error("Coefficient count does not match number of members.")
        c = float(coeffs.get("c", 0.0))
        d = float(coeffs.get("d", 0.0))
        weights_da = xr.DataArray(bs, dims=("member",), coords={"member": stacked.member})
        mu_cal = (stacked * weights_da).sum(dim="member") + a
        mu_cal = mu_cal.astype(np.float32)

        if spread_path:
            spread_da = _load_dataarray(_to_path(spread_path), spread_variable)
            spread_da, mu_cal = xr.align(spread_da, mu_cal, join="inner")
            S2 = (spread_da ** 2).astype(np.float32)
        else:
            member_mean = stacked.mean(dim="member")
            variance = ((stacked - member_mean) ** 2).mean(dim="member")
            variance, mu_cal = xr.align(variance, mu_cal, join="inner")
            S2 = variance

        sigma2 = (c + d * S2).clip(min=min_variance)
        sigma = np.sqrt(sigma2).astype(np.float32)
        out_dir = _ensure_output_dir(output_dir)
        out_path = Path(output_path).expanduser() if output_path else out_dir / "emos_calibrated.nc"
        _write_dataset({"mu_cal": mu_cal, "sigma_cal": sigma}, out_path)
        return _ok(
            {
                "output_path": str(out_path),
                "variable_mean": "mu_cal",
                "variable_sigma": "sigma_cal",
                "coefficients": {
                    "a": a,
                    **{f"b{i+1}": b for i, b in enumerate(bs)},
                    "c": c,
                    "d": d,
                },
                "mode": "apply",
            }
        )
    except Exception as exc:  # pragma: no cover
        return _error(f"emos_calibration failed: {exc}")


def confidence_index(
    member_paths: Sequence[str],
    variable: str,
    *,
    coerce_time: Optional[str] = None,
    normalized_spread_path: Optional[str] = None,
    normalized_variable: str = "normalized_spread",
    spread_path: Optional[str] = None,
    spread_variable: str = "ensemble_spread",
    sigma_ref_path: Optional[str] = None,
    sigma_variable: str = "sigma_ref",
    climo_mean_path: Optional[str] = None,
    climo_variable: str = "climatology",
    skill_weight: float = 1.0,
    output_variable: str = "confidence",
    output_path: Optional[str] = None,
    output_dir: str = "./data/ensemble",
) -> ToolResponse:
    """
    Convert normalized spread + sign agreement into a 0–1 confidence index.
    """
    try:
        stacked, _ = _stack_members(member_paths, variable, coerce_time=coerce_time)
        if normalized_spread_path:
            s_norm = _load_dataarray(_to_path(normalized_spread_path), normalized_variable)
            s_norm = _coerce_time_coord(s_norm, coerce_time)
            s_norm, stacked = xr.align(s_norm, stacked, join="inner")
        else:
            if not spread_path or not sigma_ref_path:
                return _error("Provide either normalized_spread_path or both spread_path and sigma_ref_path.")
            spread = _load_dataarray(_to_path(spread_path), spread_variable)
            sigma = _load_dataarray(_to_path(sigma_ref_path), sigma_variable)
            spread = _coerce_time_coord(spread, coerce_time)
            sigma = _coerce_time_coord(sigma, coerce_time)
            spread, sigma, stacked = xr.align(spread, sigma, stacked, join="inner")
            s_norm = spread / (sigma + 1e-6)

        base_conf = xr.apply_ufunc(np.tanh, s_norm)
        base_conf = (1.0 - base_conf).clip(0.0, 1.0)
        if climo_mean_path:
            climo = _load_dataarray(_to_path(climo_mean_path), climo_variable)
            climo = _coerce_time_coord(climo, coerce_time)
            climo, stacked = xr.align(climo, stacked, join="inner")
            anomalies = stacked - climo
            sign_vals = xr.apply_ufunc(np.sign, anomalies)
            agree = xr.where(
                sign_vals.max(dim="member") == sign_vals.min(dim="member"),
                1.0,
                0.0,
            )
            multiplier = xr.where(agree < 0.5, 0.5, agree)
        else:
            multiplier = xr.ones_like(base_conf)

        confidence = (skill_weight * base_conf * multiplier).clip(0.0, 1.0).astype(np.float32)
        out_dir = _ensure_output_dir(output_dir)
        out_path = Path(output_path).expanduser() if output_path else out_dir / f"{output_variable}.nc"
        _write_dataset({output_variable: confidence}, out_path)
        return _ok(
            {
                "output_path": str(out_path),
                "variable": output_variable,
                "skill_weight": skill_weight,
            }
        )
    except Exception as exc:  # pragma: no cover
        return _error(f"confidence_index failed: {exc}")


def lagged_ic_ensemble(
    member_files: Sequence[str],
    *,
    lagged_files: Sequence[str],
    variable: str,
    output_dir: str = "./data/ensemble",
    manifest_name: str = "ensemble_manifest.json",
) -> ToolResponse:
    """
    Combine contemporaneous members with lagged initial-condition runs.
    """
    try:
        base_paths = [_to_path(path) for path in member_files]
        lag_paths = [_to_path(path) for path in lagged_files]
        manifest: List[Dict[str, Any]] = []
        for label, path in [("base", p) for p in base_paths] + [("lagged", p) for p in lag_paths]:
            da = _load_dataarray(path, variable)
            manifest.append(
                {
                    "type": label,
                    "path": str(path),
                    "dims": {dim: int(size) for dim, size in da.sizes.items()},
                }
            )
        out_dir = _ensure_output_dir(output_dir)
        manifest_path = out_dir / manifest_name
        manifest_path.write_text(json.dumps({"members": manifest}, indent=2), encoding="utf-8")
        return _ok({"manifest_path": str(manifest_path), "member_count": len(manifest)})
    except Exception as exc:  # pragma: no cover
        return _error(f"lagged_ic_ensemble failed: {exc}")


def derive_event_mask(
    netcdf_path: str,
    variable: str,
    threshold: float,
    *,
    op: str = ">=",
    persistence_days: int = 1,
    reducer: str = "region_mean",
    mask_path: Optional[str] = None,
    mask_variable: Optional[str] = None,
    output_dir: str = str(EVENT_OUTPUT_DIR),
) -> ToolResponse:
    """
    Build a binary event mask with optional persistence and regional reduction.
    """
    ops = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
    }
    try:
        if op not in ops:
            return _error("op must be one of: >, >=, <, <=")
        da = _load_dataarray(_to_path(netcdf_path), variable)
        da = _apply_region_mask(da, mask_path, mask_variable)
        time_dim = _time_dim_or_raise(da)
        base_event = ops[op](da, threshold)
        persistent = _persistence_filter(base_event, time_dim, int(persistence_days))
        event_mask = persistent.astype(np.float32)

        diag_var: Dict[str, xr.DataArray] = {}
        reducer = (reducer or "").lower()
        space_dims = [d for d in event_mask.dims if d != time_dim]
        if reducer and space_dims:
            if reducer == "region_mean":
                diag = event_mask.mean(dim=space_dims, skipna=True)
            elif reducer == "max":
                diag = event_mask.max(dim=space_dims, skipna=True)
            elif reducer == "area_fraction":
                valid = xr.ones_like(event_mask).where(~np.isnan(event_mask))
                denom = valid.sum(dim=space_dims)
                diag = event_mask.sum(dim=space_dims, skipna=True) / xr.where(denom == 0, np.nan, denom)
            else:
                return _error("reducer must be one of {'region_mean','max','area_fraction'}")
            diag_var["event_diagnostic"] = diag.astype(np.float32)

        out_dir = _ensure_output_dir(output_dir)
        out_path = out_dir / "event_mask.nc"
        _write_dataset({"event_mask": event_mask, **diag_var}, out_path)
        payload: Dict[str, Any] = {
            "output_path": str(out_path),
            "variable": "event_mask",
            "op": op,
            "threshold": threshold,
            "persistence_days": persistence_days,
            "reducer": reducer or None,
        }
        if mask_path:
            payload["mask_path"] = mask_path
        if diag_var:
            payload["diagnostic_variable"] = "event_diagnostic"
        return _ok(payload)
    except Exception as exc:  # pragma: no cover
        return _error(f"derive_event_mask failed: {exc}")


def ehf_heatwave(
    tmax_path: str,
    tmin_path: str,
    baseline_paths: Mapping[str, str],
    *,
    persistence_days: int = 3,
    output_dir: str = str(EVENT_OUTPUT_DIR),
) -> ToolResponse:
    """
    Compute Excess Heat Factor (EHF) and a heatwave mask (EHF>0 with persistence).

    baseline_paths should map:
      - 'tmax95' (or 'tmax_p95'): path to daily/monthly 95th percentile of Tmax
      - 'tmean_clim' (or 'tmean'): path to daily/monthly climatological mean Tmean
    """
    try:
        tmax = _load_dataarray(_to_path(tmax_path), None)
        tmin = _load_dataarray(_to_path(tmin_path), None)
        time_dim = _time_dim_or_raise(tmax)
        tmin = xr.align(tmin, tmax, join="inner")[0]
        tmean = (tmax + tmin) / 2.0
        tmean_3day = tmean.rolling({time_dim: 3}, min_periods=3).mean()

        if not baseline_paths:
            return _error("baseline_paths must provide tmax95 and tmean_clim entries.")
        t95_path = baseline_paths.get("tmax95") or baseline_paths.get("tmax_p95")
        tmean_clim_path = baseline_paths.get("tmean_clim") or baseline_paths.get("tmean")
        if not t95_path or not tmean_clim_path:
            return _error("baseline_paths missing 'tmax95'/'tmax_p95' or 'tmean_clim'/'tmean'.")

        t95 = _load_dataarray(_to_path(t95_path), None)
        tmean_clim = _load_dataarray(_to_path(tmean_clim_path), None)

        if "month" in t95.dims:
            month_index = xr.DataArray(tmean_3day[time_dim].dt.month, coords={time_dim: tmean_3day[time_dim]})
            t95 = t95.sel(month=month_index)
        if "month" in tmean_clim.dims:
            month_index = xr.DataArray(tmean_3day[time_dim].dt.month, coords={time_dim: tmean_3day[time_dim]})
            tmean_clim = tmean_clim.sel(month=month_index)

        t95, tmean_clim, tmean_3day = xr.align(t95, tmean_clim, tmean_3day, join="inner")
        ehi_sig = tmean_3day - t95
        ehi_accl = tmean_3day - tmean_clim
        ehf = ehi_sig * xr.where(ehi_accl > 0, ehi_accl, 1.0)
        heatwave = _persistence_filter(ehf > 0, time_dim, int(persistence_days)).astype(np.float32)

        out_dir = _ensure_output_dir(output_dir)
        ehf_path = out_dir / "ehf.nc"
        mask_path = out_dir / "heatwave_mask.nc"
        _write_dataset({"ehf": ehf.astype(np.float32)}, ehf_path)
        _write_dataset({"heatwave_mask": heatwave}, mask_path)
        return _ok(
            {
                "ehf_path": str(ehf_path),
                "heatwave_mask_path": str(mask_path),
                "persistence_days": persistence_days,
            }
        )
    except Exception as exc:  # pragma: no cover
        return _error(f"ehf_heatwave failed: {exc}")


def verify(
    forecast_path: str,
    truth_path: str,
    variable: str,
    *,
    truth_variable: Optional[str] = None,
    metrics: Optional[Sequence[str]] = None,
    mask_path: Optional[str] = None,
    mask_variable: Optional[str] = None,
) -> ToolResponse:
    """
    Compute quick verification metrics (RMSE, MAE, Bias, ACC) against ERA5.
    """
    try:
        forecast = _load_dataarray(_to_path(forecast_path), variable)
        truth = _load_dataarray(_to_path(truth_path), truth_variable or variable)
        forecast, truth = xr.align(forecast, truth, join="inner")
        if mask_path:
            mask = _load_dataarray(_to_path(mask_path), mask_variable or variable)
            mask, forecast = xr.align(mask, forecast, join="inner")
            forecast = forecast.where(mask > 0.5)
            truth = truth.where(mask > 0.5)

        diff = forecast - truth
        metrics = metrics or ("rmse", "mae", "bias", "acc")
        results: Dict[str, float] = {}
        if "rmse" in metrics:
            results["rmse"] = float(np.sqrt(np.nanmean(diff.values ** 2)))
        if "mae" in metrics:
            results["mae"] = float(np.nanmean(np.abs(diff.values)))
        if "bias" in metrics:
            results["bias"] = float(np.nanmean(diff.values))
        if "acc" in metrics:
            f_anom = forecast - forecast.mean()
            t_anom = truth - truth.mean()
            num = np.nansum(f_anom.values * t_anom.values)
            denom = np.sqrt(np.nansum(f_anom.values ** 2) * np.nansum(t_anom.values ** 2))
            results["acc"] = float(num / denom) if denom else float("nan")

        return _ok(
            {
                "metrics": results,
                "forecast_path": str(forecast_path),
                "truth_path": str(truth_path),
            }
        )
    except Exception as exc:  # pragma: no cover
        return _error(f"verify failed: {exc}")


def bias_correct_cmip6(
    cmip6_paths: Sequence[str],
    era5_paths: Sequence[str],
    variable: str,
    *,
    method: str = "QDM",
    per: str = "month",
    wetday_threshold: float = 1.0,
    output_dir: str = str(CMIP6_OUTPUT_DIR),
) -> ToolResponse:
    """
    Bias-correct CMIP6 fields against ERA5 using simple QDM/EQM per month or all-time.
    """
    try:
        if method.upper() not in {"QDM", "EQM"}:
            return _error("method must be 'QDM' or 'EQM'")
        per = per.lower()
        if per not in {"month", "all"}:
            return _error("per must be 'month' or 'all'")
        quantiles = np.linspace(0.0, 1.0, 21)
        is_precip = _is_precip_var(variable)

        def _collect(paths: Sequence[str]) -> xr.DataArray:
            data = []
            ref_time: Optional[str] = None
            for p in paths:
                da = _load_dataarray(_to_path(p), variable)
                tdim = _time_dim_or_raise(da)
                if ref_time is None:
                    ref_time = tdim
                elif ref_time != tdim:
                    da = da.rename({tdim: ref_time})
                data.append(da)
            if not data:
                raise ValueError("No input paths supplied.")
            return xr.concat(data, dim=_time_dim_or_raise(data[0]))

        obs = _collect(era5_paths)
        cmip_all = _collect(cmip6_paths)
        time_dim = _time_dim_or_raise(cmip_all)
        if is_precip:
            obs = obs.where(obs >= wetday_threshold, 0.0)
            cmip_all = cmip_all.where(cmip_all >= wetday_threshold, 0.0)

        overlap_start = max(obs[time_dim].min().item(), cmip_all[time_dim].min().item())
        overlap_end = min(obs[time_dim].max().item(), cmip_all[time_dim].max().item())
        obs_train = obs.sel({time_dim: slice(overlap_start, overlap_end)})
        cmip_train = cmip_all.sel({time_dim: slice(overlap_start, overlap_end)})
        if obs_train.size == 0 or cmip_train.size == 0:
            return _error("No overlap between ERA5 and CMIP6 time ranges for training.")

        if per == "month":
            obs_q = obs_train.groupby(f"{time_dim}.month").quantile(quantiles, dim=time_dim, skipna=True)
            mod_q = cmip_train.groupby(f"{time_dim}.month").quantile(quantiles, dim=time_dim, skipna=True)
        else:
            obs_q = obs_train.quantile(quantiles, dim=time_dim, skipna=True)
            mod_q = cmip_train.quantile(quantiles, dim=time_dim, skipna=True)

        out_dir = _ensure_output_dir(output_dir)
        mapping_path = out_dir / "bias_mapping.json"
        artifacts: List[Dict[str, Any]] = []

        def _apply_map(target: xr.DataArray) -> xr.DataArray:
            if per == "month":
                pieces = []
                for m in np.unique(target[time_dim].dt.month.values):
                    part = target.where(target[time_dim].dt.month == m, drop=True)
                    if part.size == 0:
                        continue
                    if "month" in mod_q.coords and m not in mod_q["month"]:
                        continue
                    if "month" in obs_q.coords and m not in obs_q["month"]:
                        continue
                    adj = xr.apply_ufunc(
                        _quantile_adjust_np,
                        part,
                        mod_q.sel(month=m),
                        obs_q.sel(month=m),
                        kwargs={"probs": quantiles, "method": method.lower(), "is_precip": is_precip},
                        input_core_dims=[[time_dim], ["quantile"], ["quantile"]],
                        output_core_dims=[[time_dim]],
                        vectorize=True,
                    )
                    pieces.append(adj)
                if not pieces:
                    return target
                return xr.concat(pieces, dim=time_dim).sortby(time_dim)
            return xr.apply_ufunc(
                _quantile_adjust_np,
                target,
                mod_q,
                obs_q,
                kwargs={"probs": quantiles, "method": method.lower(), "is_precip": is_precip},
                input_core_dims=[[time_dim], ["quantile"], ["quantile"]],
                output_core_dims=[[time_dim]],
                vectorize=True,
            )

        for path in cmip6_paths:
            da = _load_dataarray(_to_path(path), variable)
            tdim = _time_dim_or_raise(da)
            if tdim != time_dim:
                da = da.rename({tdim: time_dim})
            corrected = _apply_map(da)
            out_path = out_dir / f"{Path(path).stem}_bias_corrected.nc"
            _write_dataset({variable: corrected.astype(np.float32)}, out_path)
            artifacts.append({"input": str(path), "output": str(out_path)})

        mapping_path.write_text(
            json.dumps(
                {
                    "method": method.upper(),
                    "per": per,
                    "quantiles": quantiles.tolist(),
                    "is_precip": is_precip,
                    "wetday_threshold": wetday_threshold,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return _ok(
            {
                "artifacts": artifacts,
                "mapping_path": str(mapping_path),
                "time_dim": time_dim,
            }
        )
    except Exception as exc:  # pragma: no cover
        return _error(f"bias_correct_cmip6 failed: {exc}")


def scenario_exceedance(
    cmip6_adj_paths: Sequence[str],
    variable: str,
    threshold: float,
    region_mask: Optional[str],
    *,
    period_present: Tuple[str, str] = ("1995", "2014"),
    period_future: Tuple[str, str] = ("2041", "2060"),
    mask_variable: Optional[str] = None,
    output_dir: str = str(CMIP6_OUTPUT_DIR),
) -> ToolResponse:
    """
    Compute present/future exceedance probabilities and risk ratios per model + ensemble.
    """
    try:
        if not cmip6_adj_paths:
            return _error("cmip6_adj_paths must include at least one file.")
        mask_da: Optional[xr.DataArray] = None
        if region_mask:
            mask_da = _load_dataarray(_to_path(region_mask), mask_variable or None)

        present_list: List[xr.DataArray] = []
        future_list: List[xr.DataArray] = []
        summaries: List[Dict[str, Any]] = []

        for path in cmip6_adj_paths:
            da = _load_dataarray(_to_path(path), variable)
            time_dim = _time_dim_or_raise(da)
            if mask_da is not None:
                da, mask_aligned = xr.align(da, mask_da, join="inner")
                da = da.where(mask_aligned > 0.5)
            present = da.sel({time_dim: slice(f"{period_present[0]}-01-01", f"{period_present[1]}-12-31")})
            future = da.sel({time_dim: slice(f"{period_future[0]}-01-01", f"{period_future[1]}-12-31")})
            if present.size == 0 or future.size == 0:
                return _error(f"No data in requested periods for {path}.")
            p_present = (present >= threshold).mean(dim=time_dim, skipna=True)
            p_future = (future >= threshold).mean(dim=time_dim, skipna=True)
            present_list.append(p_present)
            future_list.append(p_future)
            space_dims = [d for d in p_present.dims if d != "quantile" and d != "model"]
            reg_present = float(p_present.mean(dim=space_dims, skipna=True).item()) if space_dims else float(p_present.item())
            reg_future = float(p_future.mean(dim=space_dims, skipna=True).item()) if space_dims else float(p_future.item())
            rr_reg = reg_future / (reg_present + 1e-6)
            summaries.append(
                {
                    "model": Path(path).stem,
                    "P_present": reg_present,
                    "P_future": reg_future,
                    "risk_ratio": rr_reg,
                }
            )

        present_stack = xr.concat(present_list, dim="model")
        future_stack = xr.concat(future_list, dim="model")
        ensemble_present = present_stack.mean(dim="model", skipna=True)
        ensemble_future = future_stack.mean(dim="model", skipna=True)
        ensemble_rr = ensemble_future / (ensemble_present + 1e-6)

        out_dir = _ensure_output_dir(output_dir)
        map_path = out_dir / "scenario_exceedance_maps.nc"
        _write_dataset(
            {
                "P_present_mean": ensemble_present.astype(np.float32),
                "P_future_mean": ensemble_future.astype(np.float32),
                "risk_ratio_mean": ensemble_rr.astype(np.float32),
            },
            map_path,
        )
        summary_path = out_dir / "scenario_exceedance_summary.json"
        summary_payload = {
            "threshold": threshold,
            "period_present": period_present,
            "period_future": period_future,
            "models": summaries,
            "ensemble": {
                "P_present": float(ensemble_present.mean().item()),
                "P_future": float(ensemble_future.mean().item()),
                "risk_ratio": float(ensemble_rr.mean().item()),
            },
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        return _ok({"map_path": str(map_path), "summary_path": str(summary_path)})
    except Exception as exc:  # pragma: no cover
        return _error(f"scenario_exceedance failed: {exc}")


__all__ = [
    "harmonize_grid",
    "anomaly_and_climo",
    "smooth_gaussian_field",
    "aggregate_ensemble",
    "ensemble_spread",
    "normalized_spread",
    "event_probability",
    "emos_calibration",
    "confidence_index",
    "lagged_ic_ensemble",
    "derive_event_mask",
    "ehf_heatwave",
    "bias_correct_cmip6",
    "scenario_exceedance",
    "verify",
]
