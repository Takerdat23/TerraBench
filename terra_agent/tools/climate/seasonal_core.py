"""
Seasonal forecast utilities: CDS fetch, standardisation, aggregation, calibration, probabilities, blending, anomalies, ENSO, region tables.
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

KEY = os.getenv("CDS_API_KEY")

def _ok(payload: Dict[str, Any]) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload))], metadata=payload)


def _error(msg: str) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=f"Error: {msg}")], metadata={"error": True, "message": msg})


def _resolve_coords(ds: xr.Dataset) -> Tuple[str, str]:
    lat = next((c for c in ("latitude", "lat") if c in ds.coords), None)
    lon = next((c for c in ("longitude", "lon") if c in ds.coords), None)
    if not lat or not lon:
        raise ValueError("Latitude/longitude coordinates not found.")
    return lat, lon

def _resolve_key(override: str | None) -> str:
    key = override or KEY or os.getenv("CDSAPI_KEY")
    if not key:
        raise ValueError("CDS API key not provided. Set CDS_API_KEY or pass KEY parameter.")
    return key

_DATASET_ALIASES = {
    "seasonal-monthly-ocean-single-levels": "seasonal-monthly-single-levels",
    "seasonal-monthly-single-level": "seasonal-monthly-single-levels",
    "seasonal-original-single-level": "seasonal-original-single-levels",
    "seasonal-original-ocean-single-levels": "seasonal-original-single-levels",
}

_MONTHLY_DATASETS = {
    "seasonal-monthly-single-levels",
    "seasonal-monthly-pressure-levels",
    "seasonal-monthly-ocean",
    "seasonal-postprocessed-single-levels",
    "seasonal-postprocessed-pressure-levels",
}
_ORIGINAL_DATASETS = {"seasonal-original-single-levels", "seasonal-original-pressure-levels"}

# Known valid product_type values from CDS metadata for the seasonal collections.
_MONTHLY_PRODUCT_TYPES = {
    "ensemble_mean",
    "monthly_mean",
    "hindcast_climate_mean",
    "hindcast_monthly_mean",
    "monthly_minimum",
    "monthly_maximum",
    "monthly_standard_deviation",
}
_ORIGINAL_PRODUCT_TYPES = {
    "hindcast",  # re-forecast back catalogue
    "control_forecast",  # control member (where available)
    "perturbed_forecast",  # ensemble members
    "ensemble_mean",  # ensemble mean where supported
}


def _validate_with_datastore(
    collection_id: str,
    request: Dict[str, Any],
    keys: Sequence[str],
) -> Tuple[Optional[bool], str]:
    """
    Optionally validate values against CDS availability using ecmwf.datastores.

    Returns (status, note):
        status: True if all requested values are available, False if not, None if skipped.
        note: reason/summary.
    """
    try:
        from ecmwf.datastores import Client as DSClient  # type: ignore
    except Exception as exc:
        return None, f"Constraint validation skipped (ecmwf.datastores missing): {exc}"

    try:
        client = DSClient()
        constraints = client.apply_constraints(
            collection_id, {k: request[k] for k in keys if k in request}
        )
    except Exception as exc:
        return None, f"Constraint validation skipped (apply_constraints failed): {exc}"

    for key in keys:
        if key not in request or key not in constraints:
            continue
        req_vals = request[key]
        if not isinstance(req_vals, (list, tuple)):
            req_vals = [req_vals]
        req_vals = [str(v) for v in req_vals]
        valid_vals = constraints[key]
        if isinstance(valid_vals, (list, tuple)):
            valid_vals_str = [str(v) for v in valid_vals]
            missing = [v for v in req_vals if v not in valid_vals_str]
            if missing:
                return False, f"{key} values not available: {missing}; valid examples: {valid_vals_str[:10]}"
    return True, "Constraint validation passed via apply_constraints."



def progressive_cds_constraints(
    collection_id: str,
    request: Mapping[str, Any],
    order: Optional[Sequence[str]] = None,
) -> ToolResponse:
    """
    Progressively apply constraints by adding parameters step-by-step to reveal availability.

    Parameters
    ----------
    collection_id : str
        Dataset/collection id (e.g., 'seasonal-monthly-single-levels', 'projections-cmip6').
    request : Mapping[str, Any]
        Full request payload candidate (variable, model/system, years, months, leadtime, etc.).
    order : Sequence[str], optional
        Order in which to add parameters. Defaults to the order in `request`.
    """
    try:
        from ecmwf.datastores import Client  # type: ignore
    except Exception as exc:
        return _error(f"cdsapi missing: {exc}")
    URL = "https://cds.climate.copernicus.eu/api"
    key = KEY or (Path("~/.cdsapirc").expanduser().exists() and None)
    try:
        client = Client(url=URL, key=_resolve_key(KEY))
    except Exception as exc:
        return _error(f"Failed to init CDS client: {exc}")

    def _default_order(cid: str, req: Mapping[str, Any]) -> list[str]:
        cid_l = cid.lower()
        # Heuristic ordering for CMIP6 / seasonal products to reduce invalid combos.
        if "cmip6" in cid_l or "projection" in cid_l:
            preferred = [
                "variable_id",
                "activity_id",
                "table_id",
                "experiment_id",
                "source_id",
                "member_id",
                "grid_label",
                "years",
                "months",
                "days",
            ]
        elif "seasonal" in cid_l:
            preferred = [
                "variable",
                "originating_centre",
                "system",
                "product_type",
                "year",
                "month",
                "leadtime_month",
            ]
        else:
            preferred = []
        # preserve provided request keys after preferred order
        seen = set()
        ordered = []
        for key in preferred:
            if key in req:
                ordered.append(key)
                seen.add(key)
        for key in req.keys():
            if key not in seen:
                ordered.append(key)
        return ordered

    steps: list[dict[str, Any]] = []
    cumulative: Dict[str, Any] = {}
    key_order = list(order) if order else _default_order(collection_id, request)

    try:
        for key in key_order:
            if key not in request:
                continue
            cumulative[key] = request[key]
            constraints = client.apply_constraints(collection_id, cumulative)

            missing: list[str] = []
            valid_examples: list[Any] = []
            if key in constraints:
                allowed = constraints[key]
                if isinstance(allowed, (list, tuple)):
                    valid_examples = allowed[:10]
                    values = request[key] if isinstance(request[key], (list, tuple)) else [request[key]]
                    values = [str(v) for v in values]
                    allowed_str = [str(v) for v in allowed]
                    missing = [v for v in values if v not in allowed_str]

            steps.append(
                {
                    "step_key": key,
                    "request": dict(cumulative),
                    "constraints": constraints,
                    "missing": missing,
                    "valid_examples": valid_examples,
                    "ok": not missing,
                }
            )
    except Exception as exc:
        return _error(f"Progressive constraint query failed at key '{key}': {exc}")

    return _ok(
        {
            "collection": collection_id,
            "steps": steps,
            "order": key_order,
            "note": "Uses heuristic ordering for CMIP6/seasonal collections to help agents converge on valid parameters.",
        }
    )


def _normalize_c3s_request(dataset: str, product_type: Union[str, Sequence[str]]) -> Tuple[str, List[str], List[str]]:
    """
    Align dataset/product_type with CDS allowed values to avoid MARS 400s.

    Returns normalized (dataset, product_type_list, notes).
    """
    notes: List[str] = []
    ds = _DATASET_ALIASES.get(dataset.strip(), dataset.strip())
    pt_raw_list = [product_type] if isinstance(product_type, str) else list(product_type)
    pt_raw_list = [p.strip().lower() for p in pt_raw_list]
    norm_pt: List[str] = []

    if ds in _MONTHLY_DATASETS:
        # CDS monthly endpoints expect monthly_mean/ensemble_mean variants.
        pt_alias = {"hindcast": "hindcast_climate_mean"}
        for p in pt_raw_list:
            mapped = pt_alias.get(p, p)
            if mapped != p:
                notes.append(f"Mapped product_type '{p}' -> '{mapped}' for monthly dataset.")
            if mapped not in _MONTHLY_PRODUCT_TYPES:
                raise ValueError(
                    f"product_type '{p}' not valid for {ds}. "
                    f"Use one of {sorted(_MONTHLY_PRODUCT_TYPES)}."
                )
            norm_pt.append(mapped)
        return ds, norm_pt, notes

    if ds in _ORIGINAL_DATASETS:
        # Original-resolution endpoints: allow explicit hindcast/ensemble options (no 'forecast' UI item).
        for p in pt_raw_list:
            if p not in _ORIGINAL_PRODUCT_TYPES:
                raise ValueError(
                    f"product_type '{p}' not valid for {ds}. "
                    f"Use one of {sorted(_ORIGINAL_PRODUCT_TYPES)}."
                )
            norm_pt.append(p)
        return ds, norm_pt, notes

    raise ValueError(
        f"Dataset '{dataset}' not recognized. Use one of "
        f"{sorted(_MONTHLY_DATASETS | _ORIGINAL_DATASETS)} or a known alias."
    )


def fetch_c3s_seasonal(
    *,
    dataset: str,
    originating_centre: str,
    system: Optional[str],
    variable: Sequence[str],
    product_type: Union[str, Sequence[str]],
    year: Sequence[str],
    month: Sequence[str],
    leadtime_month: Sequence[str],
    area: Optional[Sequence[float]] = None,
    grid: Optional[float] = None,
    format: str = "netcdf",
    data_format: Optional[str] = None,
    persist_dir: str = "./data/seasonal",
    KEY: Optional[str] = None,
) -> ToolResponse:
    """Download seasonal hindcast/forecast from CDS per centre/system."""
    try:
        import cdsapi  # type: ignore
    except Exception as exc:
        return _error(f"cdsapi missing: {exc}")
    URL = "https://cds.climate.copernicus.eu/api"
    key = KEY or (Path("~/.cdsapirc").expanduser().exists() and None)
    try:
        client = cdsapi.Client(url=URL, key=_resolve_key(KEY))
    except Exception as exc:
        return _error(f"Failed to init CDS client: {exc}")

    out_dir = Path(persist_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Dict[str, Any]] = []
    try:
        norm_dataset, norm_product_type, notes = _normalize_c3s_request(dataset, product_type)
    except ValueError as exc:
        return _error(str(exc))

    validation_status, validation_note = _validate_with_datastore(
        norm_dataset,
        {
            "originating_centre": originating_centre,
            "system": system or "latest",
            "variable": list(variable),
            "product_type": norm_product_type,
            "year": list(year),
            "month": list(month),
            "leadtime_month": list(leadtime_month),
        },
        ("originating_centre", "system", "variable", "product_type", "year", "month", "leadtime_month"),
    )
    if validation_status is False:
        return _error(validation_note)
    if validation_note:
        notes.append(validation_note)

    fmt = data_format or format

    for y in year:
        for m in month:
            fname = f"{originating_centre}_s{system or 'latest'}_{'_'.join(variable)}_{y}_{m}_leads{'-'.join(leadtime_month)}.nc"
            out_path = out_dir / fname
            if out_path.exists():
                files.append({"path": str(out_path), "vars": list(variable), "nbytes": out_path.stat().st_size, "cached": True})
                continue
            request = {
                # CDS UI exposes "data_format"; CDSAPI uses "format" for many datasets.
                "data_format": fmt,
                "originating_centre": originating_centre,
                "system": system or "latest",
                "variable": list(variable),
                "product_type": norm_product_type,
                "year": y,
                "month": m,
                "leadtime_month": list(leadtime_month),
            }
            if area:
                request["area"] = area
            if grid:
                request["grid"] = [grid, grid]
            try:
                client.retrieve(norm_dataset, request, str(out_path))
            except Exception as exc:
                return _error(f"CDS request failed for {y}-{m}: {exc}")
            files.append({"path": str(out_path), "vars": list(variable), "nbytes": out_path.stat().st_size, "cached": False})

    return _ok(
        {
            "files": files,
            "request": {
                "dataset": norm_dataset,
                "centre": originating_centre,
                "system": system,
                "data_format": fmt,
                "year": list(year),
                "month": list(month),
                "product_type": norm_product_type,
                "notes": notes,
            },
        }
    )


def standardize_seasonal_files(
    paths: Sequence[str],
    *,
    unit_map: Optional[Mapping[str, str]] = None,
    rename_map: Optional[Mapping[str, str]] = None,
    output_dir: str = "./data/seasonal/standardized",
) -> ToolResponse:
    """Normalize seasonal files: lat order, time decoding, optional unit conversion/rename."""
    unit_map = unit_map or {}
    rename_map = rename_map or {}
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []
    try:
        for p in paths:
            src = Path(p).expanduser()
            if not src.exists():
                return _error(f"File not found: {src}")
            ds = xr.open_dataset(src)
            lat_name, _ = _resolve_coords(ds)
            lats = ds[lat_name].values
            if np.all(np.diff(lats) > 0):
                ds = ds.sortby(lat_name, ascending=False)
            ds = ds.rename({k: v for k, v in rename_map.items() if k in ds})
            for var, target_unit in unit_map.items():
                if var not in ds:
                    continue
                da = ds[var]
                if target_unit.lower() in ("c", "degc") and "units" in da.attrs and da.attrs["units"].lower() in ("k", "kelvin"):
                    ds[var] = (da - 273.15).astype(np.float32)
                    ds[var].attrs["units"] = "degC"
                elif target_unit.lower() in ("mm",) and "units" in da.attrs and da.attrs["units"] in ("m", "meters", "metre"):
                    ds[var] = (da * 1000.0).astype(np.float32)
                    ds[var].attrs["units"] = "mm"
            out_path = out_dir / src.name
            ds.to_netcdf(out_path)
            outputs.append(str(out_path))
    except Exception as exc:
        return _error(f"Standardization failed: {exc}")
    return _ok({"files": outputs})


_SEASON_MAP = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}


def _infer_init_month(src: Path, ds: xr.Dataset) -> Optional[int]:
    """Best-effort guess of the initialization month from filename or metadata."""
    match = re.search(r"_(\d{4})_(\d{2})_", src.name)
    if match:
        try:
            month = int(match.group(2))
            if 1 <= month <= 12:
                return month
        except Exception:
            pass
    for key in ("start_month", "init_month", "month"):
        if key in ds.attrs:
            try:
                month = int(ds.attrs[key])
                if 1 <= month <= 12:
                    return month
            except Exception:
                continue
    return None


def aggregate_to_season(
    path: str,
    *,
    agg: str = "mean",
    season_def: str = "DJF",
    center_on: str = "middle_month",
    output_path: str = "./data/seasonal/seasonal_agg.nc",
) -> ToolResponse:
    """Aggregate monthly leads to seasonal (3-month) totals/means."""
    src = Path(path).expanduser()
    if not src.exists():
        return _error(f"NetCDF not found: {src}")
    try:
        ds = xr.load_dataset(src)
    except Exception as exc:
        return _error(f"Failed to load dataset: {exc}")

    time_candidates = (
        "time",
        "valid_time",
        "forecast_month",
        "forecastMonth",
        "leadtime_month",
        "leadtime",
        "lead",
        "step",
    )
    time_name = next((c for c in time_candidates if c in ds.coords), None)
    if not time_name:
        time_name = next((c for c in time_candidates if c in ds.dims), None)

    # Fallback: handle files that store each lead as a separate variable (lead1, lead2, ...).
    lead_vars = [v for v in ds.data_vars if re.fullmatch(r"lead\d+", v)]
    if time_name is None and lead_vars:
        lead_vars = sorted(lead_vars, key=lambda v: int(re.findall(r"\d+", v)[0]))
        lead_coord = np.array([int(re.findall(r"\d+", v)[0]) for v in lead_vars], dtype=int)
        stacked = xr.concat([ds[v] for v in lead_vars], dim="lead")
        stacked = stacked.assign_coords(lead=("lead", lead_coord))
        out_da = stacked.mean("lead") if agg == "mean" else stacked.sum("lead")
        name = stacked.name or lead_vars[0] or "season_aggregate"
        out = out_da.to_dataset(name=name)
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(out_path)
        return _ok({"output_path": str(out_path), "season": season_def, "aggregation": agg})

    if not time_name:
        return _error("No time or lead coordinate found for aggregation.")

    # Ensure the time coordinate exists even if it was only a dimension.
    if time_name in ds.coords:
        time_coord = ds[time_name]
    else:
        time_coord = xr.DataArray(np.arange(ds.sizes[time_name]), dims=(time_name,))
        ds = ds.assign_coords({time_name: time_coord})

    time_values = time_coord.values
    init_month = _infer_init_month(src, ds)
    if np.issubdtype(time_values.dtype, np.datetime64):
        months = pd.to_datetime(time_values).month
    else:
        numeric = np.asarray(time_values)
        numeric = numeric.astype(int) if numeric.size else np.array([], dtype=int)
        # If the file encodes lead months as 1,2,3 and we know the init month, map to calendar months.
        is_sequential_lead = numeric.size and np.all(np.diff(numeric) == 1) and numeric[0] in (0, 1)
        if init_month is not None and is_sequential_lead:
            months = ((init_month - 1 + numeric) % 12) + 1
        elif numeric.size and np.nanmin(numeric) >= 1 and np.nanmax(numeric) <= 12:
            months = numeric
        elif init_month is not None and numeric.size:
            months = ((init_month - 1 + numeric) % 12) + 1
        else:
            months = np.arange(len(time_values)) + 1

    if season_def in _SEASON_MAP:
        target = _SEASON_MAP[season_def]
        mask = np.isin(months, target)
        ds = ds.sel({time_name: mask})
        if ds[time_name].size == 0:
            return _error(f"No months matching season {season_def}")
        grouper = xr.full_like(ds[time_name], season_def, dtype=object)
    else:
        try:
            mid = int(season_def)
            target = [(mid - 1) % 12 + 1, mid, (mid % 12) + 1]
        except Exception:
            return _error("season_def must be DJF/MAM/JJA/SON or middle-month number.")
        mask = np.isin(months, target)
        ds = ds.sel({time_name: mask})
        grouper = xr.full_like(ds[time_name], f"{target}", dtype=object)

    if agg == "mean":
        out = ds.groupby(grouper).mean(time_name, skipna=True)
    elif agg == "sum":
        out = ds.groupby(grouper).sum(time_name, skipna=True)
    else:
        return _error("agg must be 'mean' or 'sum'.")
    out = out.rename({time_name: "season"})
    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    return _ok({"output_path": str(out_path), "season": season_def, "aggregation": agg})


def _quantile_map(source: np.ndarray, target: np.ndarray, probs: np.ndarray, method: str) -> np.ndarray:
    source_q = np.nanquantile(source, probs)
    target_q = np.nanquantile(target, probs)
    if method.lower() == "eqm":
        def adj(x):
            p = np.interp(x, source_q, probs, left=probs[0], right=probs[-1])
            s_val = np.interp(p, probs, source_q)
            t_val = np.interp(p, probs, target_q)
            return x + (t_val - s_val)
    else:
        def adj(x):
            p = np.interp(x, source_q, probs, left=probs[0], right=probs[-1])
            s_val = np.interp(p, probs, source_q)
            t_val = np.interp(p, probs, target_q)
            return x * (t_val / s_val) if s_val != 0 else t_val
    return np.vectorize(adj, otypes=[float])(source)


def fit_bias_correction(
    hindcast_paths: Sequence[str],
    obs_paths: Sequence[str],
    *,
    variable: str,
    method: str = "QDM",
    probs: Sequence[float] = tuple(np.linspace(0, 1, 21)),
    output_path: str = "./data/seasonal/bias_map.nc",
) -> ToolResponse:
    try:
        h_ds = xr.open_mfdataset([str(Path(p).expanduser()) for p in hindcast_paths], combine="by_coords")
        o_ds = xr.open_mfdataset([str(Path(p).expanduser()) for p in obs_paths], combine="by_coords")
        h = h_ds[variable]
        o = o_ds[variable]
        h, o = xr.align(h, o, join="inner")
        mapping = xr.apply_ufunc(
            _quantile_map,
            h,
            o,
            kwargs={"probs": np.asarray(probs), "method": method},
            vectorize=True,
        )
        out = xr.Dataset({f"{variable}_bias_map": mapping.astype(np.float32)})
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(out_path)
        return _ok({"bias_map_path": str(out_path), "method": method, "probs": list(probs)})
    except Exception as exc:
        return _error(f"Bias fit failed: {exc}")


def apply_bias_correction(
    forecast_paths: Sequence[str],
    *,
    bias_map_path: str,
    variable: str,
    output_dir: str = "./data/seasonal/calibrated",
) -> ToolResponse:
    try:
        bias_ds = xr.load_dataset(Path(bias_map_path).expanduser())
        bias_map = bias_ds[f"{variable}_bias_map"]
        outputs = []
        out_dir = Path(output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in forecast_paths:
            fc = xr.load_dataset(Path(p).expanduser())
            if variable not in fc:
                return _error(f"Variable '{variable}' not found in {p}")
            data = xr.align(fc[variable], bias_map, join="inner")[0]
            adjusted = xr.apply_ufunc(
                _quantile_map,
                data,
                bias_map,
                kwargs={"probs": np.linspace(0, 1, 21), "method": "QDM"},
                vectorize=True,
            )
            out = fc.copy()
            out[variable] = adjusted.astype(np.float32)
            out_path = out_dir / Path(p).name.replace(".nc", "_calibrated.nc")
            out.to_netcdf(out_path)
            outputs.append(str(out_path))
        return _ok({"files": outputs})
    except Exception as exc:
        return _error(f"Apply bias correction failed: {exc}")


def category_probabilities(
    member_paths: Sequence[str],
    *,
    variable: str,
    terciles_path: str,
    category_names: Sequence[str] = ("below", "normal", "above"),
    output_path: str = "./data/seasonal/category_probs.nc",
) -> ToolResponse:
    try:
        members = [xr.load_dataset(Path(p).expanduser())[variable] for p in member_paths]
        stack = xr.concat(members, dim="member")
        terciles = xr.load_dataset(Path(terciles_path).expanduser())
        lower = terciles.get("tercile_lower") or terciles[list(terciles.data_vars)[0]]
        upper = terciles.get("tercile_upper") or terciles[list(terciles.data_vars)[1]]
        stack, lower, upper = xr.align(stack, lower, upper, join="inner")
        p_below = (stack < lower).mean("member")
        p_above = (stack > upper).mean("member")
        p_normal = 1.0 - p_below - p_above
        out = xr.Dataset(
            {
                "p_below": p_below.astype(np.float32),
                "p_normal": p_normal.astype(np.float32),
                "p_above": p_above.astype(np.float32),
            }
        )
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(out_path)
        return _ok({"output_path": str(out_path), "categories": list(category_names)})
    except Exception as exc:
        return _error(f"Category probabilities failed: {exc}")


def blend_multi_model(
    prob_paths: Sequence[str],
    *,
    weights: Optional[Sequence[float]] = None,
    output_path: str = "./data/seasonal/blended_probs.nc",
) -> ToolResponse:
    try:
        datasets = [xr.load_dataset(Path(p).expanduser()) for p in prob_paths]
        stacked = xr.concat(datasets, dim="model")
        n = stacked.sizes["model"]
        w = np.ones(n) / n if weights is None else np.asarray(weights) / np.sum(weights)
        weights_da = xr.DataArray(w, dims=("model",), coords={"model": stacked.model if "model" in stacked.coords else np.arange(n)})
        out = (stacked * weights_da).sum("model")
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(out_path)
        return _ok({"output_path": str(out_path), "weights": weights_da.values.tolist()})
    except Exception as exc:
        return _error(f"Blend failed: {exc}")


def forecast_anomaly_from_model_climo(
    forecast_path: str,
    model_climo_mean_path: str,
    *,
    variable: str,
    agg: str = "mean",
    output_path: str = "./data/seasonal/forecast_anomaly.nc",
) -> ToolResponse:
    try:
        fc = xr.load_dataset(Path(forecast_path).expanduser())
        climo = xr.load_dataset(Path(model_climo_mean_path).expanduser())
        fc_var = fc[variable]
        climo_var = climo.get(variable) or climo[list(climo.data_vars)[0]]
        fc_var, climo_var = xr.align(fc_var, climo_var, join="inner")
        anomaly = (fc_var - climo_var).astype(np.float32)
        out = xr.Dataset({f"{variable}_anomaly": anomaly})
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(out_path)
        return _ok({"output_path": str(out_path)})
    except Exception as exc:
        return _error(f"Anomaly computation failed: {exc}")


def compute_enso_indices(
    sst_path: str,
    *,
    baseline_years: List[int] = [1991, 2020],
    smooth_months: int = 3,
    output_path: str = "./data/seasonal/enso_indices.csv",
) -> ToolResponse:
    try:
        if len(baseline_years) < 2:
            return _error("baseline_years must include at least [start_year, end_year]")
        start_year, end_year = baseline_years[0], baseline_years[1]
        ds = xr.load_dataset(Path(sst_path).expanduser())
        sst = ds[list(ds.data_vars)[0]]
        time_name = next((c for c in ("time", "valid_time") if c in sst.coords), None)
        lat_name, lon_name = _resolve_coords(ds)
        box = sst.sel({lat_name: slice(5, -5), lon_name: slice(190, 240)})
        clim = box.sel({time_name: slice(f"{start_year}-01-01", f"{end_year}-12-31")}).groupby(f"{time_name}.month").mean()
        anomaly = box.groupby(f"{time_name}.month") - clim
        nino34 = anomaly.mean(dim=(lat_name, lon_name))
        nino34_smooth = nino34.rolling({time_name: smooth_months}, center=True, min_periods=1).mean()
        df = nino34_smooth.to_dataframe(name="nino34")
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path)
        return _ok({"output_path": str(out_path), "records": len(df)})
    except Exception as exc:
        return _error(f"ENSO index computation failed: {exc}")


def region_mask(
    grid_path: str,
    *,
    region: Sequence[float],
    output_path: str = "./data/seasonal/region_mask.nc",
) -> ToolResponse:
    try:
        ds = xr.load_dataset(Path(grid_path).expanduser())
        lat_name, lon_name = _resolve_coords(ds)
        lats = ds[lat_name]
        lons = ds[lon_name]
        north, west, south, east = region
        mask = xr.where((lats >= south) & (lats <= north) & (lons >= west) & (lons <= east), 1, 0)
        out = xr.Dataset({"region_mask": mask.astype(np.uint8)})
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(out_path)
        return _ok({"output_path": str(out_path), "region": region})
    except Exception as exc:
        return _error(f"Region mask failed: {exc}")


def seasonal_region_table(
    prob_path: str,
    *,
    regions: Optional[List[str]] = None,
    mask_paths: Optional[List[str]] = None,
    anomaly_path: Optional[str] = None,
    skill_path: Optional[str] = None,
    output_path: str = "./data/seasonal/region_table.csv",
) -> ToolResponse:
    try:
        prob = xr.load_dataset(Path(prob_path).expanduser())
        p_above = prob.get("p_above")
        p_normal = prob.get("p_normal")
        p_below = prob.get("p_below")
        if p_above is None or p_normal is None or p_below is None:
            return _error("Probability dataset must contain p_above, p_normal, p_below.")

        anomalies = xr.load_dataset(anomaly_path)[list(xr.load_dataset(anomaly_path).data_vars)[0]] if anomaly_path else None
        skill = xr.load_dataset(skill_path)[list(xr.load_dataset(skill_path).data_vars)[0]] if skill_path else None

        rows = []
        targets = mask_paths or []
        for mp in targets:
            mask_ds = xr.load_dataset(Path(mp).expanduser())
            mask_var = mask_ds[list(mask_ds.data_vars)[0]]
            pa = (p_above * mask_var).mean().item()
            pn = (p_normal * mask_var).mean().item()
            pb = (p_below * mask_var).mean().item()
            row = {"region": Path(mp).stem, "p_above": pa, "p_normal": pn, "p_below": pb}
            if anomalies is not None:
                row["anomaly_mean"] = float((anomalies * mask_var).mean().item())
            if skill is not None:
                row["skill"] = float((skill * mask_var).mean().item())
            rows.append(row)
        df = pd.DataFrame(rows)
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return _ok({"output_path": str(out_path), "rows": len(df)})
    except Exception as exc:
        return _error(f"Region table failed: {exc}")


__all__ = [
    "fetch_c3s_seasonal",
    "standardize_seasonal_files",
    "aggregate_to_season",
    "fit_bias_correction",
    "apply_bias_correction",
    "category_probabilities",
    "blend_multi_model",
    "forecast_anomaly_from_model_climo",
    "compute_enso_indices",
    "region_mask",
    "seasonal_region_table",
]
