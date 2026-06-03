"""
ERA5 preprocessing utilities for the ClimateAgent toolchain.

This module exposes a single tool-friendly entrypoint that can merge multiple
NetCDF files (often individual ERA5 variable downloads) into a consolidated
dataset tailored for downstream models such as Aurora or Pangu. The design uses
an operation registry so future filtering, scaling, or regridding steps can be
plugged in without changing the public API.
"""

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import xarray as xr
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from .fetch_era5 import _normalize_era5


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _ok(payload: Dict[str, Any]) -> ToolResponse:
    sanitized = _sanitize(payload)
    return ToolResponse(
        content=[TextBlock(type="text", text=_json_dumps(sanitized))],
        metadata=sanitized,
    )


def _error(msg: str) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {msg}")],
        metadata={"error": True, "message": msg},
    )


def _json_dumps(payload: Dict[str, Any]) -> str:
    try:
        import json

        return json.dumps(payload)
    except Exception:
        return str(payload)


OperationHandler = Callable[[xr.Dataset, Mapping[str, Any]], xr.Dataset]
_OPERATION_REGISTRY: Dict[str, OperationHandler] = {}


def _register_operation(name: str) -> Callable[[OperationHandler], OperationHandler]:
    def decorator(func: OperationHandler) -> OperationHandler:
        _OPERATION_REGISTRY[name] = func
        return func

    return decorator


@_register_operation("rename")
def _op_rename(ds: xr.Dataset, params: Mapping[str, Any]) -> xr.Dataset:
    mapping = params.get("mapping") or params.get("variables")
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("rename operation requires a non-empty 'mapping' dict.")
    return ds.rename(mapping)


@_register_operation("drop")
def _op_drop(ds: xr.Dataset, params: Mapping[str, Any]) -> xr.Dataset:
    vars_to_drop = params.get("variables")
    if not vars_to_drop:
        return ds
    names = [str(name) for name in vars_to_drop]
    return ds.drop_vars(names, errors="ignore")


@_register_operation("scale")
def _op_scale(ds: xr.Dataset, params: Mapping[str, Any]) -> xr.Dataset:
    factors: Mapping[str, Any] | None = params.get("factors")
    if factors is None:
        variables = params.get("variables")
        factor = params.get("factor")
        if not variables or factor is None:
            raise ValueError("scale operation requires 'factors' or ('variables' and 'factor').")
        factors = {str(var): float(factor) for var in variables}

    updates: Dict[str, xr.DataArray] = {}
    for name, factor in factors.items():
        if name not in ds:
            raise ValueError(f"Variable '{name}' not found for scaling.")
        updates[name] = ds[name] * float(factor)
    return ds.assign(**updates)


@_register_operation("assign_attrs")
def _op_assign_attrs(ds: xr.Dataset, params: Mapping[str, Any]) -> xr.Dataset:
    attrs = params.get("attrs")
    if not isinstance(attrs, Mapping):
        raise ValueError("assign_attrs operation requires an 'attrs' mapping.")
    updated = dict(ds.attrs)
    for key, value in attrs.items():
        updated[str(key)] = value
    ds.attrs = updated
    return ds


TIME_DIM_CANDIDATES: tuple[str, ...] = ("time", "valid_time", "forecast_time", "lead_time", "step")


def _apply_operations(ds: xr.Dataset, operations: Sequence[Mapping[str, Any]]) -> xr.Dataset:
    result = ds
    for idx, op in enumerate(operations):
        if not isinstance(op, Mapping):
            raise ValueError(f"Operation at index {idx} must be a mapping.")
        name = op.get("name")
        if not isinstance(name, str):
            raise ValueError(f"Operation at index {idx} is missing a 'name'.")
        handler = _OPERATION_REGISTRY.get(name)
        if handler is None:
            known = ", ".join(sorted(_OPERATION_REGISTRY))
            raise ValueError(f"Unknown operation '{name}'. Known operations: {known}")
        params = {k: v for k, v in op.items() if k != "name"}
        result = handler(result, params)
    return result


def _default_output_path(sources: Sequence[Path], output_dir: Path) -> Path:
    digest = hashlib.sha1("::".join(sorted(str(p) for p in sources)).encode("utf-8")).hexdigest()[:12]
    stem = sources[0].stem if sources else "era5"
    return output_dir / f"{stem}_preprocessed_{digest}.nc"


def _ensure_float32(ds: xr.Dataset) -> xr.Dataset:
    updated: Dict[str, xr.DataArray] = {}
    for name, data in ds.data_vars.items():
        if np.issubdtype(data.dtype, np.floating) and data.dtype != np.float32:
            updated[name] = data.astype(np.float32)
    if not updated:
        return ds
    return ds.assign(**updated)


def _standardize_time_dim(ds: xr.Dataset, target: str = "time") -> xr.Dataset:
    """Rename common ERA5 time dimensions (e.g., valid_time) to the target name."""
    if target in ds.dims:
        return ds
    for candidate in TIME_DIM_CANDIDATES:
        if candidate in ds.dims:
            if candidate == target:
                return ds
            return ds.rename({candidate: target})
    return ds


def preprocess_era5(
    input_files: Sequence[str],
    *,
    output_path: Optional[str] = None,
    combine_mode: str = "merge",
    combine_dim: Optional[str] = None,
    variables: Optional[Sequence[str]] = None,
    rename_map: Optional[Mapping[str, str]] = None,
    operations: Optional[Sequence[Mapping[str, Any]]] = None,
    enforce_dims: Optional[Sequence[str]] = ("time", "latitude", "longitude"),
    ensure_float32: bool = True,
    compression_level: Optional[int] = 4,
    output_dir: str = "./data/preprocessed",
    engine: str = "netcdf4",
    **legacy_kwargs: Any,
) -> ToolResponse:
    """
    Merge ERA5 NetCDF files into a single dataset and write it back to disk.

    Args:
        input_files: Sequence of NetCDF paths to merge. Useful when variables were
            downloaded separately but need to be packaged for model ingestion.
        output_path: Optional destination NetCDF path. Defaults to a hashed name in
            ``output_dir`` so the same combination is re-usable.
        combine_mode: Either "merge" (default) to align by coordinates or "concat"
            to concatenate along ``combine_dim``.
        combine_dim: Required when ``combine_mode`` is "concat".
        variables: Optional ordered subset of variables to keep after merging.
        rename_map: Optional mapping of old->new variable names applied before saving.
        operations: Optional list of operation dicts (see registry) for extensible
            preprocessing steps.
        enforce_dims: Dimensions that must be present with equal lengths across all
            inputs. Set to ``None`` to disable the check.
        ensure_float32: Down-cast float variables to float32 to match most model
            checkpoints and reduce disk footprint.
        compression_level: If not ``None``, enable zlib compression with the given
            level (0-9) during NetCDF export.
        output_dir: Directory where the auto-generated outputs reside.
        engine: NetCDF backend passed to xarray.
        legacy_kwargs: Optional legacy parameters. Currently supports
            ``compress_level`` as an alias for ``compression_level`` so older
            tool calls keep working.

    Returns:
        ToolResponse describing the merged dataset and saved location.
    """
    legacy_compress = legacy_kwargs.pop("compress_level", None)
    if legacy_compress is not None:
        compression_level = legacy_compress
    if legacy_kwargs:
        return _error(f"Unsupported arguments: {', '.join(sorted(legacy_kwargs))}")

    try:
        if not input_files:
            return _error("At least one NetCDF path must be provided.")

        source_paths = [Path(path).expanduser() for path in input_files]
        for path in source_paths:
            if not path.exists():
                return _error(f"NetCDF file not found: {path}")

        out_path = Path(output_path).expanduser() if output_path else None
        output_root = Path(output_dir).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        if out_path is None:
            out_path = _default_output_path(source_paths, output_root)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)

        loaded: list[xr.Dataset] = []
        for path in source_paths:
            with xr.open_dataset(path, engine=engine) as ds:
                normalized = _normalize_era5(ds)
                loaded.append(normalized.load())

        if combine_mode == "merge":
            combined = xr.merge(
                loaded,
                compat="override",
                combine_attrs="override",
                join="outer",
            )
        elif combine_mode == "concat":
            if not combine_dim:
                return _error("combine_dim must be specified when combine_mode='concat'.")
            combined = xr.concat(
                loaded,
                dim=combine_dim,
                data_vars="all",
                coords="minimal",
                compat="override",
                combine_attrs="override",
            )
        else:
            return _error(f"Unsupported combine_mode '{combine_mode}'. Use 'merge' or 'concat'.")

        combined = _standardize_time_dim(combined, target="time")

        if enforce_dims:
            for dim in enforce_dims:
                if dim not in combined.dims:
                    return _error(f"Dimension '{dim}' not present after merge; found {list(combined.dims)}.")

        if variables:
            missing = [var for var in variables if var not in combined.data_vars]
            if missing:
                return _error(f"Variables not present after merge: {missing}")
            combined = combined[variables]

        if rename_map:
            combined = combined.rename(rename_map)

        if operations:
            combined = _apply_operations(combined, operations)

        if ensure_float32:
            combined = _ensure_float32(combined)

        encoding: Optional[Dict[str, Dict[str, Any]]] = None
        if compression_level is not None:
            encoding = {
                var: {"zlib": True, "complevel": int(compression_level)}
                for var in combined.data_vars
            }

        combined.to_netcdf(out_path, engine=engine, encoding=encoding)
        file_size = out_path.stat().st_size

        metadata = {
            "output_path": str(out_path),
            "input_files": [str(p) for p in source_paths],
            "combine_mode": combine_mode,
            "combine_dim": combine_dim,
            "variables": list(map(str, combined.data_vars)),
            "dims": {name: int(size) for name, size in combined.dims.items()},
            "operations": operations,
            "rename_map": rename_map,
            "enforce_dims": list(enforce_dims) if enforce_dims else None,
            "ensure_float32": ensure_float32,
            "compression_level": compression_level,
            "file_size_bytes": int(file_size),
        }
        return _ok(metadata)
    except Exception as exc:  # pragma: no cover - safety net
        return _error(f"Unexpected failure during preprocess_era5: {exc}")


def _coerce_paths(netcdf_path: Union[str, Sequence[str]]) -> list[Path]:
    if isinstance(netcdf_path, (list, tuple, set)):
        paths = list(netcdf_path)
    else:
        paths = [netcdf_path]
    expanded: list[Path] = []
    for p in paths:
        path = Path(p).expanduser()
        if path.is_dir():
            nc_files = sorted(path.glob("*.nc"))
            if not nc_files:
                raise FileNotFoundError(f"No NetCDF files found in directory: {path}")
            expanded.extend(nc_files)
        else:
            expanded.append(path)
    return expanded


def _format_attr_subset(attrs: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    subset: Dict[str, Any] = {}
    for key in keys:
        value = attrs.get(key)
        if value is not None:
            subset[str(key)] = value
    return subset


def _preview_coord(values: np.ndarray, limit: int) -> list[Any]:
    if values.size == 0 or limit <= 0:
        return []
    flat = values.ravel()
    preview = flat[: limit].tolist()
    return [_sanitize(item) for item in preview]


def describe_netcdf_variables(
    netcdf_path: Union[str, Sequence[str]],
    *,
    variables: Optional[Sequence[str]] = None,
    include_coords: bool = True,
    coord_preview: int = 3,
    include_dataset_attrs: bool = True,
    attr_keys: Sequence[str] = ("long_name", "standard_name", "units", "short_name", "description"),
    engine: str = "netcdf4",
) -> ToolResponse:
    """
    Provide lightweight metadata about the variables stored inside a NetCDF file.

    This helper is meant to summarize model outputs prior to calling the math agent,
    giving it the dimensional context (dims, units, coordinate ranges) it needs to
    perform calculations without re-opening large binary files.
    """

    try:
        paths = _coerce_paths(netcdf_path)
        for path in paths:
            if not path.exists():
                return _error(f"NetCDF file not found: {path}")

        if len(paths) == 1:
            ds_context = xr.open_dataset(paths[0], engine=engine)
        else:
            ds_context = xr.open_mfdataset([str(p) for p in paths], combine="by_coords", engine=engine)

        summaries: list[Dict[str, Any]] = []
        dataset_attrs: Dict[str, Any] | None = None

        with ds_context as ds:
            ds = _normalize_era5(ds)
            selected_vars = list(ds.data_vars)
            if variables:
                missing = [var for var in variables if var not in ds.data_vars]
                if missing:
                    return _error(f"Variables not found in dataset: {missing}")
                selected_vars = list(variables)

            for name in selected_vars:
                da = ds[name]
                dims = list(da.dims)
                shape = [int(da.sizes[dim]) for dim in dims]
                attrs_subset = _format_attr_subset(da.attrs, attr_keys)
                coord_info: list[Dict[str, Any]] = []
                if include_coords:
                    for dim in dims:
                        if dim not in ds.coords:
                            continue
                        coord = ds[dim]
                        coord_summary = {
                            "name": dim,
                            "size": int(coord.sizes.get(dim, coord.size)),
                            "dtype": str(coord.dtype),
                            "attrs": _format_attr_subset(coord.attrs, ("units", "long_name")),
                        }
                        if coord_preview > 0:
                            coord_summary["preview"] = _preview_coord(np.asarray(coord.values), coord_preview)
                        coord_info.append(coord_summary)
                summaries.append(
                    {
                        "name": name,
                        "dims": dims,
                        "shape": shape,
                        "dtype": str(da.dtype),
                        "attrs": attrs_subset or None,
                        "coords": coord_info or None,
                    }
                )

            if include_dataset_attrs:
                dataset_attrs = _format_attr_subset(ds.attrs, ds.attrs.keys())  # type: ignore[arg-type]

        description_lines = []
        for summary in summaries:
            dims_desc = ", ".join(f"{dim}={size}" for dim, size in zip(summary["dims"], summary["shape"]))
            attr_desc = summary.get("attrs") or {}
            attr_text = ", ".join(f"{k}={v}" for k, v in attr_desc.items()) if attr_desc else ""
            if attr_text:
                attr_text = f" ({attr_text})"
            description_lines.append(f"{summary['name']}: dtype={summary['dtype']} dims[{dims_desc}]{attr_text}")

        metadata = {
            "source_paths": [str(p) for p in paths],
            "variables": summaries,
            "dataset_attrs": dataset_attrs,
            "text_summary": "\n".join(description_lines),
        }
        return _ok(metadata)
    except Exception as exc:  # pragma: no cover - safety net
        return _error(f"Unexpected failure during describe_netcdf_variables: {exc}")
