"""Aurora inference script with configurable ERA5 downloads."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cdsapi
import numpy as np
import torch
import xarray as xr

from aurora import Aurora, Batch, Metadata, rollout
from aurora import AuroraSmallPretrained  # e.g., a small model for quick tests



SURFACE_REQUEST_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
SURFACE_TO_BATCH = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
    "msl": "msl",
}

ATMOS_REQUEST_VARIABLES = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]
ATMOS_TO_BATCH = {
    "t": "t",
    "u": "u",
    "v": "v",
    "q": "q",
    "z": "z",
}

STATIC_REQUEST_VARIABLES = [
    "geopotential",
    "land_sea_mask",
    "soil_type",
]
STATIC_TO_BATCH = {
    "z": "z",
    "lsm": "lsm",
    "slt": "slt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Aurora inference for a configurable ERA5 time."
    )
    parser.add_argument(
        "--valid-time",
        default="2023-01-01T06:00",
        help="ISO timestamp (UTC) representing the analysis time for Aurora.",
    )
    parser.add_argument(
        "--history-steps",
        type=int,
        default=2,
        help="Number of consecutive ERA5 times to include (oldest to newest).",
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=6,
        help="Hour spacing between consecutive history steps.",
    )
    parser.add_argument(
        "--time-block",
        type=int,
        default=None,
        help="Number of consecutive times to feed Aurora (uses most recent block). Defaults to history steps.",
    )
    parser.add_argument(
        "--pressure-levels",
        nargs="+",
        default=[
            "50",
            "100",
            "150",
            "200",
            "250",
            "300",
            "400",
            "500",
            "600",
            "700",
            "850",
            "925",
            "1000",
        ],
        help="Pressure levels (in hPa) to request from ERA5.",
    )
    parser.add_argument(
        "--download-dir",
        default="./downloads/era5",
        help="Directory where ERA5 files are cached.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=2,
        help="Number of rollout steps to generate with Aurora.",
    )
    parser.add_argument(
        "--model-device",
        default="cuda:0",
        help="Device for running the Aurora model (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--input-device",
        default="cpu",
        help="Device for storing ERA5 tensors (typically cpu).",
    )
    return parser.parse_args()


def compute_history_times(
    valid_time: dt.datetime, history_steps: int, interval_hours: int
) -> List[dt.datetime]:
    if history_steps <= 0:
        raise ValueError("history_steps must be positive.")
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive.")
    start = valid_time - dt.timedelta(hours=interval_hours * (history_steps - 1))
    return [start + dt.timedelta(hours=interval_hours * i) for i in range(history_steps)]


def format_range_suffix(times: Sequence[dt.datetime]) -> str:
    start = times[0].strftime("%Y%m%dT%H")
    end = times[-1].strftime("%Y%m%dT%H")
    return f"{start}_{end}"


def time_components(times: Iterable[dt.datetime]) -> Tuple[List[str], List[str], List[str], List[str]]:
    years = sorted({t.strftime("%Y") for t in times})
    months = sorted({t.strftime("%m") for t in times})
    days = sorted({t.strftime("%d") for t in times})
    hours = sorted({t.strftime("%H:00") for t in times})
    return years, months, days, hours


def as_np_datetimes(times: Sequence[dt.datetime]) -> np.ndarray:
    return np.array([np.datetime64(t.replace(tzinfo=None), "ns") for t in times])


def ensure_static_dataset(client: cdsapi.Client, download_dir: Path) -> Path:
    path = download_dir / "static.nc"
    if path.exists():
        return path
    request = {
        "product_type": "reanalysis",
        "variable": STATIC_REQUEST_VARIABLES,
        "year": "2023",
        "month": "01",
        "day": "01",
        "time": "00:00",
        "format": "netcdf",
    }
    client.retrieve("reanalysis-era5-single-levels", request, str(path))
    return path


def ensure_surface_dataset(
    client: cdsapi.Client, download_dir: Path, times: Sequence[dt.datetime]
) -> Path:
    suffix = format_range_suffix(times)
    path = download_dir / f"surface_{suffix}.nc"
    if path.exists():
        return path
    years, months, days, hours = time_components(times)
    request = {
        "product_type": "reanalysis",
        "variable": SURFACE_REQUEST_VARIABLES,
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "format": "netcdf",
    }
    client.retrieve("reanalysis-era5-single-levels", request, str(path))
    return path


def ensure_pressure_dataset(
    client: cdsapi.Client,
    download_dir: Path,
    times: Sequence[dt.datetime],
    pressure_levels: Sequence[int],
) -> Path:
    suffix = format_range_suffix(times)
    path = download_dir / f"pressure_{suffix}.nc"
    if path.exists():
        return path
    years, months, days, hours = time_components(times)
    request = {
        "product_type": "reanalysis",
        "variable": ATMOS_REQUEST_VARIABLES,
        "pressure_level": [str(level) for level in pressure_levels],
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "format": "netcdf",
    }
    client.retrieve("reanalysis-era5-pressure-levels", request, str(path))
    return path


def time_coord_name(ds: xr.Dataset) -> str:
    for candidate in ("valid_time", "time"):
        if candidate in ds.coords:
            return candidate
    raise KeyError("No suitable time coordinate (valid_time or time) found.")


def lon_coord_name(ds: xr.Dataset) -> str:
    for candidate in ("longitude", "lon"):
        if candidate in ds.coords:
            return candidate
    raise KeyError("No longitude coordinate found.")


def lat_coord_name(ds: xr.Dataset) -> str:
    for candidate in ("latitude", "lat"):
        if candidate in ds.coords:
            return candidate
    raise KeyError("No latitude coordinate found.")


def wrap_longitudes(ds: xr.Dataset, coord_name: str) -> xr.Dataset:
    lon_values = ds[coord_name].values
    if np.all((lon_values >= 0) & (lon_values < 360)):
        return ds
    wrapped = np.mod(lon_values, 360.0)
    ds = ds.assign_coords({coord_name: wrapped})
    return ds.sortby(coord_name)


def ensure_descending_latitudes(ds: xr.Dataset, lat_name: str) -> xr.Dataset:
    values = ds[lat_name].values
    diffs = np.diff(values)
    if np.all(diffs < 0):
        return ds
    if np.all(diffs > 0):
        return ds.sortby(lat_name, ascending=False)
    raise ValueError("Latitude coordinate must be monotonic for Aurora ingest.")


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    if "expver" in ds.dims:
        try:
            ds = ds.sel(expver=ds["expver"].max())
        except Exception:
            ds = ds.isel(expver=-1)
    lat_name = lat_coord_name(ds)
    ds = ensure_descending_latitudes(ds, lat_name)
    return ds


def select_times(ds: xr.Dataset, times: Sequence[dt.datetime]) -> xr.Dataset:
    coord = time_coord_name(ds)
    target = as_np_datetimes(times)
    return ds.sel({coord: target})


def build_batch(
    surface_path: Path,
    pressure_path: Path,
    static_path: Path,
    times: Sequence[dt.datetime],
    device: str | torch.device,
) -> Batch:
    times = list(times)
    with xr.open_dataset(surface_path, engine="netcdf4") as surf_ds:
        surf_ds = normalize_dataset(surf_ds)
        surf_ds = select_times(surf_ds, times)
        lon_name = lon_coord_name(surf_ds)
        lat_name = lat_coord_name(surf_ds)
        surf_ds = wrap_longitudes(surf_ds, lon_name)
        latitudes = surf_ds[lat_name].values
        longitudes = surf_ds[lon_name].values
        print(
            f"[aurora] surface lat/lon sizes: {latitudes.shape} / {longitudes.shape}; "
            f"time steps: {len(times)}"
        )
        surface_tensors = {}
        for batch_key, var_name in SURFACE_TO_BATCH.items():
            data = surf_ds[var_name].values.astype(np.float32, copy=False)
            tensor = torch.from_numpy(data)[None].to(device=device, dtype=torch.float32)
            surface_tensors[batch_key] = tensor
            print(f"[aurora] surface '{var_name}' tensor shape: {tensor.shape}")

    with xr.open_dataset(pressure_path, engine="netcdf4") as atmos_ds:
        atmos_ds = normalize_dataset(atmos_ds)
        atmos_ds = select_times(atmos_ds, times)
        lon_name = lon_coord_name(atmos_ds)
        lat_name = lat_coord_name(atmos_ds)
        atmos_ds = wrap_longitudes(atmos_ds, lon_name)
        time_name = time_coord_name(atmos_ds)
        atmos_ds = atmos_ds.transpose(time_name, "pressure_level", lat_name, lon_name)
        if "expver" in atmos_ds.dims:
            atmos_ds = atmos_ds.isel(expver=-1)
        atmos_tensors = {}
        for batch_key, var_name in ATMOS_TO_BATCH.items():
            data = atmos_ds[var_name].values.astype(np.float32, copy=False)
            tensor = torch.from_numpy(data)[None].to(device=device, dtype=torch.float32)
            atmos_tensors[batch_key] = tensor
            print(f"[aurora] atmos '{var_name}' tensor shape: {tensor.shape}")
        pressure_levels = tuple(int(level) for level in atmos_ds.pressure_level.values)
        print(f"[aurora] pressure levels: {pressure_levels}")

    with xr.open_dataset(static_path, engine="netcdf4") as static_ds:
        static_ds = normalize_dataset(static_ds)
        lon_name = lon_coord_name(static_ds)
        lat_name = lat_coord_name(static_ds)
        static_ds = wrap_longitudes(static_ds, lon_name)
        static_ds = static_ds.sel({lat_name: latitudes, lon_name: longitudes})
        static_tensors = {}
        for batch_key, var_name in STATIC_TO_BATCH.items():
            da = static_ds[var_name]
            for dim in tuple(da.dims):
                if dim not in (lat_name, lon_name):
                    da = da.isel({dim: 0})
            data = da.values.astype(np.float32, copy=False)
            static_tensors[batch_key] = torch.from_numpy(data).to(device=device, dtype=torch.float32)
            print(f"[aurora] static '{var_name}' tensor shape: {static_tensors[batch_key].shape}")

    metadata = Metadata(
        lat=torch.from_numpy(latitudes.astype(np.float32)).to(device),
        lon=torch.from_numpy(longitudes.astype(np.float32)).to(device),
        time=tuple(times),
        atmos_levels=pressure_levels,
    )
    print("[aurora] batch metadata:", {"lat": latitudes.shape, "lon": longitudes.shape, "time_steps": len(times)})
    return Batch(
        surf_vars=surface_tensors,
        static_vars=static_tensors,
        atmos_vars=atmos_tensors,
        metadata=metadata,
    )


def main() -> None:
    args = parse_args()
    valid_time = dt.datetime.fromisoformat(args.valid_time)
    history_times = compute_history_times(
        valid_time,
        history_steps=args.history_steps,
        interval_hours=args.interval_hours,
    )
    block_size = args.time_block or len(history_times)
    if block_size <= 0:
        raise ValueError("time_block must be positive.")
    if block_size > len(history_times):
        raise ValueError("time_block cannot exceed history_steps.")
    selected_times = history_times[-block_size:]
    print(
        f"Using {block_size} ERA5 snapshots from {selected_times[0].isoformat()} "
        f"to {selected_times[-1].isoformat()}."
    )

    download_dir = Path(args.download_dir).expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    static_path = ensure_static_dataset(client, download_dir)
    surface_path = ensure_surface_dataset(client, download_dir, history_times)
    pressure_levels = [int(level) for level in args.pressure_levels]
    pressure_path = ensure_pressure_dataset(
        client,
        download_dir,
        history_times,
        pressure_levels,
    )

    batch = build_batch(
        surface_path=surface_path,
        pressure_path=pressure_path,
        static_path=static_path,
        times=selected_times,
        device=args.input_device,
    )


    # model = AuroraSmallPretrained()
    # model.load_checkpoint() 

    model = Aurora(use_lora=False)
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
    model.eval()
    model = model.to(args.model_device)

    output_count = 0
    with torch.inference_mode():
        for pred in rollout(model, batch, steps=args.rollout_steps):
            pred.to("cpu")
            output_count += 1

    print(f"Generated {output_count} rollout steps.")
    print("ERA5 files used:")
    print(f"  Static  : {static_path}")
    print(f"  Surface : {surface_path}")
    print(f"  Pressure: {pressure_path}")


if __name__ == "__main__":
    main()
