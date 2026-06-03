"""
Generic regridding helper to align arbitrary datasets to a target grid.
"""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

try:  # optional dependency for conservative regridding
    import xesmf as xe
except ImportError:  # pragma: no cover
    xe = None  # type: ignore

try:  # optional GeoTIFF support
    import rioxarray as rxr
except ImportError:  # pragma: no cover
    rxr = None  # type: ignore

try:  # optional raster fallback for GeoTIFF
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None  # type: ignore

__all__ = ["regrid_to_target_grid"]


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


def _detect_lat_lon(ds: xr.Dataset) -> Tuple[str, str]:
    candidates = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("y", "x"),
    ]
    for lat_name, lon_name in candidates:
        if lat_name in ds.coords and lon_name in ds.coords:
            return lat_name, lon_name
        if lat_name in ds.dims and lon_name in ds.dims:
            return lat_name, lon_name
    raise KeyError("Could not find latitude/longitude coordinates in dataset.")


def _open_dataset_with_engine(path: Path, engine: Optional[str]) -> xr.Dataset:
    if engine:
        if engine == "zarr":
            return xr.open_zarr(path)
        return xr.open_dataset(path, engine=engine)
    try:
        return xr.open_dataset(path)
    except Exception as exc:
        last_exc = exc
        for candidate in ("netcdf4", "scipy", "cfgrib", "earthkit", "pydap"):
            try:
                return xr.open_dataset(path, engine=candidate)
            except Exception as cand_exc:
                last_exc = cand_exc
        raise last_exc


def _load_raster(path: Path) -> Tuple[xr.Dataset, str, str]:
    if rxr is not None:
        da = rxr.open_rasterio(path)
        band_dim = "band" if "band" in da.dims else da.dims[0]
        ds = da.to_dataset(name="band_data").squeeze(dim=band_dim, drop=True)
        lat_name, lon_name = _detect_lat_lon(ds)
        return ds, lat_name, lon_name
    if rasterio is None:
        raise ImportError("rasterio or rioxarray is required to open GeoTIFF inputs.")
    with rasterio.open(path) as src:
        transform = src.transform
        if transform.b != 0 or transform.d != 0:
            raise ValueError("Rotated rasters are not supported; reproject to a north-up grid first.")
        data = src.read()
        x_coords = transform.c + (np.arange(src.width) + 0.5) * transform.a
        y_coords = transform.f + (np.arange(src.height) + 0.5) * transform.e
        coords = {
            "x": x_coords,
            "y": y_coords,
            "band": np.arange(1, src.count + 1),
        }
        ds = xr.Dataset({"band_data": (("band", "y", "x"), data)}, coords=coords)
        lat_name, lon_name = _detect_lat_lon(ds)
        return ds, lat_name, lon_name


def _load_netcdf(path: Path, engine: Optional[str] = None) -> Tuple[xr.Dataset, str, str, Optional[Path]]:
    cleanup_dir: Optional[Path] = None
    open_path = path
    if path.suffix == ".zip":
        cleanup_dir = Path(tempfile.mkdtemp(prefix="regrid_nc_"))
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(cleanup_dir)
        nc_files = sorted(cleanup_dir.rglob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found inside zip: {path}")
        open_path = nc_files[0]
    ds = _open_dataset_with_engine(open_path, engine)
    lat_name, lon_name = _detect_lat_lon(ds)
    return ds, lat_name, lon_name, cleanup_dir


def _load_dataset(data: str, engine: Optional[str] = None) -> Tuple[xr.Dataset, str, str, str, Optional[Path]]:
    src = Path(data).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"data path not found: {src}")

    if src.is_dir():
        if src.suffix == ".zarr" or (src / ".zgroup").exists():
            ds = xr.open_zarr(src)
            lat_name, lon_name = _detect_lat_lon(ds)
            return ds, lat_name, lon_name, str(src), None
        nc_files = sorted(src.glob("*.nc"))
        if nc_files:
            ds = xr.open_mfdataset([str(p) for p in nc_files], combine="by_coords")
            lat_name, lon_name = _detect_lat_lon(ds)
            return ds, lat_name, lon_name, str(src), None
        raise FileNotFoundError(f"No NetCDF files found in directory: {src}")

    if src.suffix.lower() == ".zip":
        cleanup_dir = Path(tempfile.mkdtemp(prefix="regrid_zip_"))
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(cleanup_dir)
        tif_files = sorted(cleanup_dir.rglob("*.tif")) + sorted(cleanup_dir.rglob("*.tiff"))
        if tif_files:
            ds, lat_name, lon_name = _load_raster(tif_files[0])
            return ds, lat_name, lon_name, str(src), cleanup_dir
        nc_files = sorted(cleanup_dir.rglob("*.nc"))
        if nc_files:
            ds = _open_dataset_with_engine(nc_files[0], engine)
            lat_name, lon_name = _detect_lat_lon(ds)
            return ds, lat_name, lon_name, str(src), cleanup_dir
        raise FileNotFoundError(f"No supported files found inside zip: {src}")

    if src.suffix.lower() in {".tif", ".tiff", ".geotiff"}:
        ds, lat_name, lon_name = _load_raster(src)
        return ds, lat_name, lon_name, str(src), None

    if src.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Non-georeferenced images (png/jpg) are not supported; use GeoTIFF.")

    if src.suffix.lower() in {".grib", ".grb", ".grib2"}:
        ds = _open_dataset_with_engine(src, engine or "cfgrib")
        lat_name, lon_name = _detect_lat_lon(ds)
        return ds, lat_name, lon_name, str(src), None

    if src.suffix.lower() == ".zarr":
        ds = xr.open_zarr(src)
        lat_name, lon_name = _detect_lat_lon(ds)
        return ds, lat_name, lon_name, str(src), None

    ds, lat_name, lon_name, cleanup_dir = _load_netcdf(src, engine=engine)
    return ds, lat_name, lon_name, str(src), cleanup_dir


def _prepare_grid(values: Sequence[float]) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Grid coordinates must be 1D arrays.")
    if arr.size < 2:
        raise ValueError("Grid coordinates must have at least two points.")
    ascending = np.all(np.diff(arr) >= 0)
    if ascending:
        return arr, False
    return arr[::-1], True


def _extract_target_grid(
    target_grid: Any,
    fallback_lat_name: str,
    fallback_lon_name: str,
) -> Tuple[np.ndarray, np.ndarray, str, str, str]:
    source = "inline"
    if isinstance(target_grid, str):
        path = Path(target_grid).expanduser()
        if path.exists():
            cleanup_dir: Optional[Path] = None
            if path.suffix.lower() in {".tif", ".tiff", ".geotiff"}:
                ds, lat_name, lon_name = _load_raster(path)
            else:
                ds, lat_name, lon_name, cleanup_dir = _load_netcdf(path)
            lat_vals = np.asarray(ds[lat_name].values, dtype=float)
            lon_vals = np.asarray(ds[lon_name].values, dtype=float)
            source = str(path)
            ds.close()
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            return lat_vals, lon_vals, lat_name, lon_name, source
    if isinstance(target_grid, dict):
        if "grid_path" in target_grid:
            return _extract_target_grid(target_grid["grid_path"], fallback_lat_name, fallback_lon_name)
        lat_key = "latitudes" if "latitudes" in target_grid else "lats" if "lats" in target_grid else None
        lon_key = "longitudes" if "longitudes" in target_grid else "lons" if "lons" in target_grid else None
        if lat_key and lon_key:
            lat_vals = np.asarray(target_grid[lat_key], dtype=float)
            lon_vals = np.asarray(target_grid[lon_key], dtype=float)
            lat_name = target_grid.get("lat_name", fallback_lat_name)
            lon_name = target_grid.get("lon_name", fallback_lon_name)
            return lat_vals, lon_vals, str(lat_name), str(lon_name), source
    raise ValueError("target_grid must be a NetCDF path or dict with latitudes/longitudes.")


def _regrid_conservative(
    ds: xr.Dataset,
    lat_name: str,
    lon_name: str,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    target_lat_name: str,
    target_lon_name: str,
) -> xr.Dataset:
    if xe is None:
        raise ImportError("xesmf is required for conservative regridding. Install xesmf and retry.")
    # rename to standard names for xesmf
    src = ds.rename({lat_name: "lat", lon_name: "lon"})
    tgt = xr.Dataset(coords={"lat": target_lats, "lon": target_lons})
    regridder = xe.Regridder(src, tgt, "conservative", reuse_weights=True)
    out = regridder(src)
    try:
        regridder.clean_weight_file()
    except Exception:
        pass
    return out.rename({"lat": target_lat_name, "lon": target_lon_name})


def regrid_to_target_grid(
    data: str,
    target_grid: Any,
    method: str = "bilinear",
    variables: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
    allow_extrapolation: bool = False,
    engine: Optional[str] = None,
) -> ToolResponse:
    """
    Reproject and resample a gridded dataset onto a target grid definition.

    Args:
        data: NetCDF/GeoTIFF path to regrid.
        target_grid: NetCDF path or dict with {"latitudes": [...], "longitudes": [...]}.
        method: bilinear|cubic|nearest|conservative (conservative requires xesmf; cubic for smooth continuous fields).
        variables: Optional subset of data variables to regrid.
        output_path: Optional NetCDF output path (defaults to ./data/regridded/<stem>_<method>.nc).
        allow_extrapolation: If True, allow extrapolation beyond source domain for interp methods.
        engine: Optional xarray engine override (e.g., "netcdf4", "cfgrib", "zarr").
    """
    ds: Optional[xr.Dataset] = None
    cleanup_dir: Optional[Path] = None
    try:
        method_norm = method.lower()
        if method_norm not in {"bilinear", "nearest", "conservative", "linear", "cubic"}:
            return _error("method must be one of bilinear|nearest|conservative|cubic.")

        ds, src_lat_name, src_lon_name, data_source, cleanup_dir = _load_dataset(data, engine=engine)
        if variables:
            missing = [v for v in variables if v not in ds.data_vars]
            if missing:
                ds.close()
                return _error(f"Variables not found in dataset: {missing}")
            ds = ds[variables]

        tgt_lats_raw, tgt_lons_raw, tgt_lat_name, tgt_lon_name, tgt_source = _extract_target_grid(
            target_grid, src_lat_name, src_lon_name
        )
        tgt_lats_sorted, tgt_lat_reversed = _prepare_grid(tgt_lats_raw)
        tgt_lons_sorted, tgt_lon_reversed = _prepare_grid(tgt_lons_raw)

        # Align dataset coordinate names to target names for consistent output
        ds_aligned = ds.rename({src_lat_name: tgt_lat_name, src_lon_name: tgt_lon_name})
        ds_aligned = ds_aligned.sortby(tgt_lat_name).sortby(tgt_lon_name)

        if method_norm == "conservative":
            regridded = _regrid_conservative(
                ds_aligned, tgt_lat_name, tgt_lon_name, tgt_lats_sorted, tgt_lons_sorted, tgt_lat_name, tgt_lon_name
            )
        else:
            if method_norm == "bilinear":
                interp_method = "linear"
            elif method_norm == "linear":
                interp_method = "linear"
            else:
                interp_method = method_norm
            kwargs = {"fill_value": "extrapolate"} if allow_extrapolation else {}
            regridded = ds_aligned.interp(
                {
                    tgt_lat_name: xr.DataArray(tgt_lats_sorted, dims=(tgt_lat_name,)),
                    tgt_lon_name: xr.DataArray(tgt_lons_sorted, dims=(tgt_lon_name,)),
                },
                method=interp_method,
                kwargs=kwargs,
            )

        # Restore target grid ordering if the input was descending
        if tgt_lat_reversed:
            regridded = regridded.sel({tgt_lat_name: tgt_lats_raw})
        if tgt_lon_reversed:
            regridded = regridded.sel({tgt_lon_name: tgt_lons_raw})

        out_path = (
            Path(output_path).expanduser()
            if output_path
            else Path("./data/regridded") / f"{Path(data).stem}_regrid_{method_norm}.nc"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        regridded.to_netcdf(out_path)

        payload = _sanitize(
            {
                "output_path": str(out_path),
                "method": method_norm,
                "data_source": data_source,
                "target_grid_source": tgt_source,
                "variables": list(regridded.data_vars),
                "dims": {k: int(v) for k, v in regridded.dims.items()},
                "lat_name": tgt_lat_name,
                "lon_name": tgt_lon_name,
            }
        )
        return _ok(payload)
    except Exception as exc:  # pragma: no cover - runtime safety
        return _error(f"regrid_to_target_grid failed: {exc}")
    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
