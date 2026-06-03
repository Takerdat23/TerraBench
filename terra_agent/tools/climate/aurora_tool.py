"""
Aurora model inference tool for the climate agent.

This wraps the forecast helper in a callable agentscope tool
that accepts pre-downloaded ERA5 NetCDF files, runs the Aurora rollout on the
requested GPU, and emits forecast fields as a NetCDF file ready for the plotting
utilities.
"""

import datetime as dt
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import xarray as xr
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from .model_inference_base import (
    ForecastGridSpec,
    ForecastStep,
    summarise_steps,
    write_forecast_dataset,
)

try:
    from terra_agent.forecast_models.aurora import (  # type: ignore[import]
        build_batch,
        time_coord_name,
    )
except ImportError as exc:  # pragma: no cover - safety net for missing local module
    raise RuntimeError(
        "Aurora helper functions could not be imported. Ensure terra_agent.forecast_models.aurora is available."
    ) from exc

try:  # pragma: no cover - optional dependency guarded at runtime
    import torch
except ImportError:  # pragma: no cover - allow graceful error return later
    torch = None  # type: ignore[assignment]

REQUIRED_SURFACE_VARIABLES = ("t2m", "u10", "v10", "msl")
REQUIRED_PRESSURE_VARIABLES = ("t", "u", "v", "q", "z")
REQUIRED_STATIC_VARIABLES = ("z", "lsm", "slt")


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value


def _ok(payload: Dict[str, Any]) -> ToolResponse:
    sanitized = _sanitize(payload)
    return ToolResponse(
        content=[TextBlock(type="text", text=json.dumps(sanitized))],
        metadata=sanitized,
    )


def _error(msg: str, *, trace: Optional[str] = None) -> ToolResponse:
    metadata: Dict[str, Any] = {"error": True, "message": msg}
    if trace:
        metadata["trace"] = trace
        metadata["trace_lines"] = trace.rstrip("\n").splitlines()
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {msg}")],
        metadata=metadata,
    )


def _ensure_required_variables(path: Path, *, required: Sequence[str], context: str) -> None:
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        missing = [name for name in required if name not in dataset.data_vars]
    if missing:
        req_list = ", ".join(required)
        missing_list = ", ".join(missing)
        raise ValueError(
            f"{context} is missing required variables ({missing_list}). "
            f"Ensure the download request includes: {req_list}"
        )


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _squeeze_to_nd(arr: np.ndarray, target_ndim: int, context: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim < target_ndim:
        raise ValueError(f"{context} has fewer than {target_ndim} dimensions (got shape {arr.shape}).")
    # Drop leading singleton dimensions until we reach the desired ndim
    while arr.ndim > target_ndim:
        if arr.shape[0] == 1:
            arr = arr.reshape(arr.shape[1:])
        else:
            squeeze_axes = [i for i, size in enumerate(arr.shape) if size == 1]
            if not squeeze_axes:
                break
            arr = np.squeeze(arr, axis=tuple(squeeze_axes))
    if arr.ndim != target_ndim:
        raise ValueError(
            f"{context} expected {target_ndim} dimensions after squeezing; got shape {arr.shape}"
        )
    return arr


def _to_datetime(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value.replace(tzinfo=None)
    if hasattr(value, "to_datetime"):
        return value.to_datetime().replace(tzinfo=None)
    if isinstance(value, np.datetime64):
        seconds = value.astype("datetime64[s]").astype(np.int64)
        return dt.datetime.fromtimestamp(int(seconds), tz=dt.timezone.utc).replace(tzinfo=None)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc).replace(tzinfo=None)
    try:
        return dt.datetime.fromisoformat(str(value)).replace(tzinfo=None)
    except Exception as exc:  # pragma: no cover - robustness
        raise ValueError(f"Unable to interpret datetime value: {value}") from exc


def _read_available_times(path: Path) -> List[dt.datetime]:
    with xr.open_dataset(path, engine="netcdf4") as ds:
        coord = time_coord_name(ds)
        values = ds[coord].values
    return sorted(_to_datetime(value) for value in np.atleast_1d(values))


def _resolve_grid(batch: Any) -> ForecastGridSpec:
    metadata = getattr(batch, "metadata", None)
    lat = None
    lon = None
    levels = None
    if metadata is not None:
        lat = getattr(metadata, "lat", None)
        lon = getattr(metadata, "lon", None)
        levels = getattr(metadata, "atmos_levels", None)
    if lat is None or lon is None:
        raise ValueError("Aurora batch metadata does not include lat/lon coordinates.")
    lat_values = _tensor_to_numpy(lat)
    lon_values = _tensor_to_numpy(lon)
    pressure_levels = tuple(int(lvl) for lvl in levels) if levels is not None else None
    return ForecastGridSpec(latitude=lat_values, longitude=lon_values, pressure_levels=pressure_levels)


def _batch_to_forecast_step(
    batch: Any,
    *,
    analysis_time: dt.datetime,
    lead_hours: Optional[float],
) -> ForecastStep:
    surface_vars = getattr(batch, "surf_vars", {}) or {}
    atmos_vars = getattr(batch, "atmos_vars", {}) or {}
    surface_data: Dict[str, np.ndarray] = {}
    for name, tensor in surface_vars.items():
        arr = _tensor_to_numpy(tensor)
        arr = _squeeze_to_nd(arr.astype(np.float32, copy=False), 2, f"Surface '{name}'")
        surface_data[name] = arr
    atmos_data: Dict[str, np.ndarray] = {}
    for name, tensor in atmos_vars.items():
        arr = _tensor_to_numpy(tensor)
        arr = _squeeze_to_nd(arr.astype(np.float32, copy=False), 3, f"Atmos '{name}'")
        atmos_data[name] = arr
    valid_time = analysis_time
    if lead_hours is not None:
        valid_time = analysis_time + dt.timedelta(hours=lead_hours)
    metadata = {}
    if lead_hours is not None:
        metadata["lead_hours"] = lead_hours
    return ForecastStep(valid_time=valid_time, surface=surface_data, atmosphere=atmos_data, metadata=metadata)


def run_aurora_forecast(
    surface_path: str,
    pressure_path: str,
    static_path: str,
    *,
    selected_times: Optional[Sequence[str]] = None,
    history_steps: Optional[int] = None,
    time_block: Optional[int] = None,
    rollout_steps: int = 4,
    forecast_interval_hours: Optional[float] = 6.0,
    analysis_time: Optional[str] = None,
    output_dir: str = "./data/model_forecasts",
    model_repo: str = "microsoft/aurora",
    checkpoint_name: str = "aurora-0.25-pretrained.ckpt",
    model_kwargs: Optional[Mapping[str, Any]] = None,
    device: str = "cuda:0",
    input_device: str = "cpu",
) -> ToolResponse:
    """
    Run Aurora rollout and export predictions to NetCDF for downstream tools.

    Args:
        surface_path: Path to a NetCDF file containing the ERA5 surface history.
        pressure_path: Path to a NetCDF file containing ERA5 pressure-level history.
        static_path: Path to a NetCDF file containing ERA5 static fields (geopotential, etc.).
        selected_times: Optional ordered list of ISO timestamps (UTC) to use as Aurora inputs.
        history_steps: If `selected_times` is omitted, number of most-recent timesteps to use.
        time_block: Optional number of most recent steps (from the history) to feed the model.
        rollout_steps: Number of rollout steps to generate.
        forecast_interval_hours: Lead increment between successive outputs (None to keep metadata time).
        analysis_time: Optional ISO timestamp used for metadata; defaults to the most recent input time.
        output_dir: Directory where the forecast NetCDF will be written.
        model_repo: HuggingFace repository or local directory for checkpoints.
        checkpoint_name: Checkpoint filename within `model_repo`.
        model_kwargs: Extra keyword arguments forwarded to `Aurora(...)`.
        device: Torch device for model execution (e.g., "cuda:0" or "cpu").
        input_device: Torch device for the ERA5 tensors (usually "cpu").

    Required ERA5 content:
        - Surface file must include `t2m`, `u10`, `v10`, `msl`.
        - Pressure-level file must include `t`, `u`, `v`, `q`, `z` on the desired levels.
        - Static file must include `z`, `lsm`, `slt`.
      The snippet in the Aurora README uses CDS requests equivalent to the sample
      commands shown in the user instructions (static + surface + pressure-level downloads).
    """
    if torch is None:
        return _error("PyTorch is required for Aurora inference. Install torch and retry.")
    try:
        from aurora import Aurora, rollout  # type: ignore[import]
    except ImportError:
        return _error(
            "Aurora python package is not available. Install the official Aurora "
            "implementation before using this tool."
        )

    try:
        surface_file = Path(surface_path).expanduser()
        pressure_file = Path(pressure_path).expanduser()
        static_file = Path(static_path).expanduser()
        for path in (surface_file, pressure_file, static_file):
            if not path.exists():
                return _error(f"Required ERA5 file not found: {path}")

        _ensure_required_variables(
            surface_file,
            required=REQUIRED_SURFACE_VARIABLES,
            context="Surface ERA5 dataset",
        )
        _ensure_required_variables(
            pressure_file,
            required=REQUIRED_PRESSURE_VARIABLES,
            context="Pressure-level ERA5 dataset",
        )
        _ensure_required_variables(
            static_file,
            required=REQUIRED_STATIC_VARIABLES,
            context="Static ERA5 dataset",
        )

        available_times = _read_available_times(surface_file)
        if not available_times:
            return _error("Surface dataset does not contain any time steps.")

        if selected_times:
            parsed_times = [_to_datetime(ts) for ts in selected_times]
        else:
            total_history = history_steps or len(available_times)
            if total_history > len(available_times):
                return _error(
                    f"history_steps={total_history} exceeds available timesteps ({len(available_times)})."
                )
            history_window = available_times[-total_history:]
            block = time_block or len(history_window)
            if block <= 0:
                return _error("time_block must be positive.")
            if block > len(history_window):
                return _error("time_block cannot exceed the number of selected history steps.")
            parsed_times = history_window[-block:]

        if len(parsed_times) == 0:
            return _error("No timesteps selected for Aurora inference.")

        analysis_dt = (
            _to_datetime(analysis_time) if analysis_time is not None else parsed_times[-1]
        )

        batch = build_batch(
            surface_path=surface_file,
            pressure_path=pressure_file,
            static_path=static_file,
            times=parsed_times,
            device=input_device,
        )

        grid = _resolve_grid(batch)
        print(
            f"[aurora_tool] grid lat/lon sizes: {grid.latitude.shape} / {grid.longitude.shape}; "
            f"pressure_levels: {grid.pressure_levels}"
        )

        resolved_model_kwargs = dict(model_kwargs or {})
        resolved_model_kwargs.setdefault("use_lora", False)

        model = Aurora(**resolved_model_kwargs)
        model.load_checkpoint(model_repo, checkpoint_name)
        model.eval()
        model = model.to(device)

        steps: List[ForecastStep] = []
        with torch.inference_mode():
            for idx, pred in enumerate(rollout(model, batch, steps=rollout_steps), start=1):
                # Move each predicted batch off GPU immediately to avoid keeping the
                # entire rollout resident in VRAM at once.
                pred_cpu = pred.to("cpu")
                lead_hours = forecast_interval_hours * idx if forecast_interval_hours else None
                step = _batch_to_forecast_step(
                    pred_cpu,
                    analysis_time=analysis_dt,
                    lead_hours=lead_hours,
                )
                print(
                    f"[aurora_tool] step {idx} lead_hours={lead_hours}: "
                    f"surface shape {step.surface['10u'].shape} vs grid lat {grid.latitude.size}, lon {grid.longitude.size}"
                )
                steps.append(step)

        output_dir_path = Path(output_dir).expanduser()
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir_path
            / f"aurora_forecast_{analysis_dt:%Y%m%dT%H}_{rollout_steps:02d}steps.nc"
        )

        extra_attrs: Dict[str, Any] = {
            "model": "aurora",
            "checkpoint": checkpoint_name,
            "model_repo": model_repo,
            "analysis_time": analysis_dt.isoformat(),
            "input_timesteps": [ts.isoformat() for ts in parsed_times],
            "rollout_steps": rollout_steps,
        }
        export = write_forecast_dataset(
            output_path,
            grid,
            steps,
            global_attrs=extra_attrs,
            base_time=analysis_dt,
        )

        metadata = {
            "model": "aurora",
            "output_path": str(export.output_path),
            "forecast_summary": summarise_steps(steps, base_time=analysis_dt),
            "surface_variables": export.variables,
            "pressure_variables": export.pressure_variables,
            "analysis_time": analysis_dt.isoformat(),
            "input_files": {
                "static": str(static_file),
                "surface": str(surface_file),
                "pressure": str(pressure_file),
            },
            "device": device,
        }
        return _ok(metadata)

    except Exception as exc:  # pragma: no cover - safety net
        return _error(f"Aurora inference failed: {exc}", trace=traceback.format_exc())
