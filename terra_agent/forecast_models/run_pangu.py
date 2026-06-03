"""Download ERA5 data and run inference with the Pangu-Weather ONNX model."""

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import onnxruntime as ort
import xarray as xr
from cdsapi import Client as CDSClient

# from dotenv import load_dotenv
# load_dotenv()

GRAVITY = 9.80665  # m s^-2
PRESSURE_LEVELS = (
    1000,
    925,
    850,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    50,
)

UPPER_VARS = {
    "z": ("geopotential", "z"),
    "t": ("temperature", "t"),
    "u": ("u_component_of_wind", "u"),
    "v": ("v_component_of_wind", "v"),
    "r": ("relative_humidity", "r"),
}

SURFACE_VARS = {
    "t2m": ("2m_temperature", "t2m"),
    "u10": ("10m_u_component_of_wind", "u10"),
    "v10": ("10m_v_component_of_wind", "v10"),
    "msl": ("mean_sea_level_pressure", "msl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ERA5 data and run a Pangu-Weather ONNX checkpoint."
    )
    parser.add_argument(
        "--valid-time",
        required=True,
        help="Analysis time in ISO format (e.g. 2023-01-01T06:00).",
    )
    parser.add_argument(
        "--download-dir",
        default="./downloads/era5_pangu",
        help="Directory used to cache downloaded ERA5 files.",
    )
    parser.add_argument(
        "--model-path",
        default="Pangu_weather/pangu_weight/pangu_weather_6.onnx",
        help="Path to the Pangu-Weather ONNX model.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output_pangu",
        help="Directory where inference outputs will be written (.npy).",
    )
    parser.add_argument(
        "--upper-input-name",
        default="input",
        help="Name of the upper-air input tensor in the ONNX graph.",
    )
    parser.add_argument(
        "--surface-input-name",
        default="input_surface",
        help="Name of the surface input tensor in the ONNX graph.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU execution provider instead of CUDA.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA device index to use when CUDAExecutionProvider is enabled.",
    )
    parser.add_argument(
        "--keep-inputs",
        action="store_true",
        help="Save the generated input tensors alongside the outputs.",
    )
    parser.add_argument(
        "--cds-url",
        default="https://cds.climate.copernicus.eu/api",
        help="Copernicus CDS API endpoint.",
    )
    parser.add_argument(
        "--cds-uid",
        type=str,
        default=None,
        help="Copernicus CDS user ID (optional if CDSAPI_KEY already encodes uid:key).",
    )
    parser.add_argument(
        "--cds-api-key",
        type=str,
        default=None,
        help="Copernicus CDS API key (optional if CDSAPI_KEY already encodes uid:key).",
    )
    parser.add_argument(
        "--cds-insecure",
        action="store_true",
        help="Disable TLS certificate verification when talking to the CDS API.",
    )
    return parser.parse_args()


class ERA5Downloader:
    """Minimal wrapper around the CDS API client with consistent config."""

    def __init__(self, url: str, key: str, verify: bool = True) -> None:
        self._client = CDSClient(url=url, key=key, verify=verify, progress=True, timeout=600)

    def retrieve(self, dataset: str, request: dict, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        self._client.retrieve(dataset, request, str(target))
        return target


def resolve_cds_key(args: argparse.Namespace) -> str:
    env_key = os.environ.get("CDSAPI_KEY")


    key = args.cds_api_key or env_key

    if key is None:
        raise RuntimeError(
            "CDS API key not provided. Supply --cds-api-key/--cds-uid or set CDSAPI_KEY."
        )

    if ":" in key:
        return key


    return f"{key}"


def ensure_download(
    downloader: ERA5Downloader,
    dataset: str,
    request: dict,
    target: Path,
) -> Path:
    if target.exists():
        return target
    return downloader.retrieve(dataset, request, target)


def format_range_suffix(times: Sequence[dt.datetime]) -> str:
    if not times:
        raise ValueError("times must contain at least one datetime.")
    start = times[0].strftime("%Y%m%dT%H")
    end = times[-1].strftime("%Y%m%dT%H")
    return f"{start}_{end}"


def time_components(times: Iterable[dt.datetime]) -> tuple[list[str], list[str], list[str], list[str]]:
    years = sorted({t.strftime("%Y") for t in times})
    months = sorted({t.strftime("%m") for t in times})
    days = sorted({t.strftime("%d") for t in times})
    hours = sorted({t.strftime("%H:00") for t in times})
    return years, months, days, hours


def ensure_surface_dataset(
    downloader: ERA5Downloader,
    root: Path,
    times: Sequence[dt.datetime],
) -> Path:
    times = list(times)
    if not times:
        raise ValueError("times must contain at least one datetime.")
    suffix = format_range_suffix(times)
    target = root / f"surface_{suffix}.nc"
    if target.exists():
        return target
    years, months, days, hours = time_components(times)
    request = {
        "product_type": "reanalysis",
        "variable": [request_name for request_name, _ in SURFACE_VARS.values()],
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "format": "netcdf",
    }
    return ensure_download(
        downloader,
        "reanalysis-era5-single-levels",
        request,
        target,
    )


def ensure_pressure_dataset(
    downloader: ERA5Downloader,
    root: Path,
    times: Sequence[dt.datetime],
    pressure_levels: Sequence[int],
) -> Path:
    times = list(times)
    if not times:
        raise ValueError("times must contain at least one datetime.")
    suffix = format_range_suffix(times)
    target = root / f"pressure_{suffix}.nc"
    if target.exists():
        return target
    years, months, days, hours = time_components(times)
    request = {
        "product_type": "reanalysis",
        "variable": [request_name for request_name, _ in UPPER_VARS.values()],
        "pressure_level": [str(level) for level in pressure_levels],
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "format": "netcdf",
    }
    return ensure_download(
        downloader,
        "reanalysis-era5-pressure-levels",
        request,
        target,
    )


def time_coord_name(ds: xr.Dataset) -> str:
    for candidate in ("valid_time", "time"):
        if candidate in ds.coords:
            return candidate
    raise KeyError("Dataset does not contain a recognised time coordinate.")


def lon_coord_name(ds: xr.Dataset) -> str:
    for candidate in ("longitude", "lon"):
        if candidate in ds.coords:
            return candidate
    raise KeyError("Dataset does not contain a recognised longitude coordinate.")


def lat_coord_name(ds: xr.Dataset) -> str:
    for candidate in ("latitude", "lat"):
        if candidate in ds.coords:
            return candidate
    raise KeyError("Dataset does not contain a recognised latitude coordinate.")


def wrap_longitudes(ds: xr.Dataset, coord_name: Optional[str] = None) -> xr.Dataset:
    if coord_name is None:
        coord_name = lon_coord_name(ds)
    lon_values = ds[coord_name].values
    if np.all((lon_values >= 0) & (lon_values < 360)):
        return ds.sortby(coord_name)
    wrapped = np.mod(lon_values, 360.0)
    ds = ds.assign_coords({coord_name: wrapped})
    return ds.sortby(coord_name)


def ensure_descending_latitudes(ds: xr.Dataset, coord_name: Optional[str] = None) -> xr.Dataset:
    if coord_name is None:
        coord_name = lat_coord_name(ds)
    values = ds[coord_name].values
    if values.ndim != 1:
        raise ValueError("Latitude coordinate must be 1-D for downstream processing.")
    diffs = np.diff(values)
    if np.all(diffs < 0):
        return ds
    if np.all(diffs > 0):
        return ds.sortby(coord_name, ascending=False)
    raise ValueError("Latitude coordinate must be monotonic.")


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    if "expver" in ds.dims:
        try:
            ds = ds.sel(expver=ds["expver"].max())
        except Exception:
            ds = ds.isel(expver=-1)
    return ensure_descending_latitudes(ds)


def select_time(ds: xr.Dataset, valid_time: dt.datetime) -> xr.Dataset:
    coord = time_coord_name(ds)
    target = np.datetime64(valid_time.replace(tzinfo=None))
    selected = ds.sel({coord: target})
    if coord in selected.dims:
        selected = selected.isel({coord: 0})
    return selected


def load_surface_tensor(
    surface_path: Path,
    valid_time: dt.datetime,
) -> np.ndarray:
    tensors: list[np.ndarray] = []
    with xr.open_dataset(surface_path, engine="netcdf4") as ds:
        ds = normalize_dataset(ds)
        ds = select_time(ds, valid_time)
        lon_name = lon_coord_name(ds)
        lat_name = lat_coord_name(ds)
        ds = wrap_longitudes(ds, lon_name)
        for _, (_, data_name) in SURFACE_VARS.items():
            if data_name not in ds.variables:
                raise KeyError(f"Variable {data_name} not found in surface dataset.")
            da = ds[data_name]
            for dim in tuple(da.dims):
                if dim not in (lat_name, lon_name):
                    da = da.isel({dim: 0})
            data = da.transpose(lat_name, lon_name).values.astype(np.float32)
            tensors.append(data)
    stacked = np.stack(tensors, axis=0)  # (vars, lat, lon)
    return stacked  # (vars, lat, lon)


def load_upper_tensor(
    pressure_path: Path,
    valid_time: dt.datetime,
) -> np.ndarray:
    tensors: list[np.ndarray] = []
    with xr.open_dataset(pressure_path, engine="netcdf4") as ds:
        ds = normalize_dataset(ds)
        ds = select_time(ds, valid_time)
        ds = ds.sel(pressure_level=list(PRESSURE_LEVELS))
        lon_name = lon_coord_name(ds)
        lat_name = lat_coord_name(ds)
        ds = wrap_longitudes(ds, lon_name)
        for key, (_, data_name) in UPPER_VARS.items():
            if data_name not in ds.variables:
                raise KeyError(f"Variable {data_name} not found in pressure dataset.")
            da = ds[data_name]
            for dim in tuple(da.dims):
                if dim not in ("pressure_level", lat_name, lon_name):
                    da = da.isel({dim: 0})
            data = (
                da.transpose("pressure_level", lat_name, lon_name)
                .values.astype(np.float32)
            )
            if key == "z":
                data = data / GRAVITY
            tensors.append(data)
    stacked = np.stack(tensors, axis=0).astype(np.float32)  # (vars, levels, lat, lon)
    return stacked


def create_session(model_path: str, use_cpu: bool, gpu_id: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 1

    if use_cpu:
        providers: list[tuple[str, dict]] = [("CPUExecutionProvider", {})]
    else:
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": gpu_id,
                    "arena_extend_strategy": "kSameAsRequested",
                },
            )
        ]
    return ort.InferenceSession(model_path, sess_options=options, providers=providers)


def save_arrays(output_dir: Path, names: Iterable[str], arrays: Sequence[np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in zip(names, arrays):
        np.save(output_dir / f"{name}.npy", array)


def main() -> None:
    args = parse_args()
    valid_time = dt.datetime.fromisoformat(args.valid_time)

    download_root = Path(args.download_dir).expanduser()
    cds_key = resolve_cds_key(args)
    downloader = ERA5Downloader(
        url=args.cds_url,
        key=cds_key,
        verify=not args.cds_insecure,
    )
    download_root.mkdir(parents=True, exist_ok=True)
    times = [valid_time]
    surface_path = ensure_surface_dataset(downloader, download_root, times)
    pressure_path = ensure_pressure_dataset(
        downloader,
        download_root,
        times,
        PRESSURE_LEVELS,
    )

    surface_tensor = load_surface_tensor(surface_path, valid_time)
    upper_tensor = load_upper_tensor(pressure_path, valid_time)

    if args.keep_inputs:
        input_dir = Path(args.output_dir).expanduser() / "inputs"
        save_arrays(input_dir, ("upper", "surface"), (upper_tensor, surface_tensor))

    if args.cpu:
        gpu_id = None
    else:
        gpu_id = args.gpu_id

    session = create_session(
        args.model_path,
        use_cpu=args.cpu,
        gpu_id=0 if gpu_id is None else gpu_id,
    )
    provider = "CPUExecutionProvider" if args.cpu else f"CUDAExecutionProvider(device_id={gpu_id})"
    print(f"ONNX Runtime initialised with {provider}.")
    inputs = {
        args.upper_input_name: upper_tensor,
        args.surface_input_name: surface_tensor,
    }
    outputs = session.run(None, inputs)
    output_names = [out.name for out in session.get_outputs()]

    output_dir = Path(args.output_dir).expanduser()
    save_arrays(output_dir, output_names, outputs)

    print("Inference complete.")
    for name, array in zip(output_names, outputs):
        print(f"{name}: shape={array.shape}, dtype={array.dtype}")


if __name__ == "__main__":
    main()
