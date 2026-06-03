"""
Pangu-Weather inference tool that consumes pre-downloaded ERA5 analysis data,
runs the ONNX checkpoint, and exports forecasts to NetCDF for downstream plotting.
"""

import datetime as dt
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    from terra_agent.forecast_models.run_pangu import (  # type: ignore[import]
        PRESSURE_LEVELS,
        SURFACE_VARS,
        UPPER_VARS,
        create_session,
        load_surface_tensor,
        load_upper_tensor,
        normalize_dataset,
        wrap_longitudes,
        lon_coord_name,
        lat_coord_name,
    )
except ImportError as exc:  # pragma: no cover - ensure source file present
    raise RuntimeError(
        "Pangu helper functions could not be imported. Ensure terra_agent.forecast_models.run_pangu exists."
    ) from exc


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
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


def _extract_lat_lon(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path, engine="netcdf4") as ds:
        ds = normalize_dataset(ds)
        lon_name = lon_coord_name(ds)
        lat_name = lat_coord_name(ds)
        ds = wrap_longitudes(ds, lon_name)
        lats = ds[lat_name].values.astype(np.float32, copy=True)
        lons = ds[lon_name].values.astype(np.float32, copy=True)
    return lats, lons


def _parse_device(device: str) -> Tuple[bool, int]:
    device = device.lower()
    if device == "cpu":
        return True, 0
    if device.startswith("cuda"):
        parts = device.split(":")
        if len(parts) == 1 or not parts[1]:
            return False, 0
        try:
            return False, int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Unable to parse CUDA device index from '{device}'.") from exc
    raise ValueError(f"Unsupported device string '{device}'. Use 'cpu' or 'cuda:<index>'.")


def _normalise_surface_output(
    array: np.ndarray,
    *,
    surface_keys: Sequence[str],
    lat_size: int,
    lon_size: int,
) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != len(surface_keys):
            raise ValueError(
                f"Surface output expected leading dimension {len(surface_keys)}, found {arr.shape[0]}."
            )
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 4:
        if arr.shape[1] != len(surface_keys) and arr.shape[0] == len(surface_keys):
            arr = arr.transpose(1, 0, 2, 3)
        if arr.shape[1] != len(surface_keys):
            raise ValueError(
                f"Surface output expected channel dimension {len(surface_keys)}, found {arr.shape[1]}."
            )
    else:
        raise ValueError("Surface output must be 3-D or 4-D (lead, variables, lat, lon).")
    if arr.shape[-2:] != (lat_size, lon_size):
        raise ValueError(
            f"Surface output spatial shape mismatch; expected ({lat_size}, {lon_size}), got {arr.shape[-2:]}."
        )
    return arr


def _normalise_upper_output(
    array: np.ndarray,
    *,
    upper_keys: Sequence[str],
    levels: Sequence[int],
    lat_size: int,
    lon_size: int,
) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[0] != len(upper_keys) or arr.shape[1] != len(levels):
            raise ValueError(
                "Upper-air output expected shape (variables, levels, lat, lon) when no lead dimension is present."
            )
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 5:
        if arr.shape[1] != len(upper_keys) or arr.shape[2] != len(levels):
            raise ValueError(
                "Upper-air output expected shape (lead, variables, levels, lat, lon)."
            )
    else:
        raise ValueError("Upper-air output must be 4-D or 5-D (lead, variables, levels, lat, lon).")
    if arr.shape[-2:] != (lat_size, lon_size):
        raise ValueError(
            f"Upper-air output spatial shape mismatch; expected ({lat_size}, {lon_size}), got {arr.shape[-2:]}."
        )
    return arr


def _build_forecast_steps(
    *,
    valid_dt: dt.datetime,
    lead_hours: Sequence[float],
    surface_block: Optional[np.ndarray],
    upper_block: Optional[np.ndarray],
    surface_keys: Sequence[str],
    upper_keys: Sequence[str],
) -> List[ForecastStep]:
    lead_count = 0
    if surface_block is not None:
        lead_count = max(lead_count, surface_block.shape[0])
    if upper_block is not None:
        lead_count = max(lead_count, upper_block.shape[0])
    if lead_count == 0:
        raise ValueError("No forecast outputs detected.")
    if len(lead_hours) != lead_count:
        raise ValueError(
            f"lead_hours length ({len(lead_hours)}) must match number of forecast steps ({lead_count})."
        )
    steps: List[ForecastStep] = []
    for idx in range(lead_count):
        step_time = valid_dt + dt.timedelta(hours=float(lead_hours[idx]))
        surface_fields: Dict[str, np.ndarray] = {}
        if surface_block is not None:
            for var_idx, key in enumerate(surface_keys):
                surface_fields[key] = surface_block[idx, var_idx]
        upper_fields: Dict[str, np.ndarray] = {}
        if upper_block is not None:
            for var_idx, key in enumerate(upper_keys):
                upper_fields[key] = upper_block[idx, var_idx]
        steps.append(
            ForecastStep(
                valid_time=step_time,
                surface=surface_fields,
                atmosphere=upper_fields,
                metadata={"lead_hours": float(lead_hours[idx])},
            )
        )
    return steps


def run_pangu_forecast(
    surface_path: str,
    pressure_path: str,
    *,
    valid_time: str,
    max_forecast_hours: Optional[float] = None,
    output_dir: str = "./data/model_forecasts",
    model_path: Optional[str] = None,
    device: str = "cuda:0",
    upper_input_name: str = "input",
    surface_input_name: str = "input_surface",
    lead_hours: Optional[Sequence[float]] = None,
) -> ToolResponse:
    """
    Execute the Pangu-Weather ONNX model and export forecasts to NetCDF.

    Args:
        surface_path: Path to the ERA5 surface-level NetCDF file (includes the analysis time).
        pressure_path: Path to the ERA5 pressure-level NetCDF file aligned with `surface_path`.
        valid_time: ISO-8601 timestamp representing the ERA5 analysis time.
        output_dir: Directory where the NetCDF forecast file will be written.
        model_path: Path to the Pangu-Weather ONNX checkpoint. Defaults to
            PANGU_MODEL_PATH or ./Pangu_weight/pangu_weather_6.onnx.
        device: Execution device string, e.g., "cuda:0" or "cpu".
        upper_input_name: Name of the upper-air input tensor in the ONNX graph.
        surface_input_name: Name of the surface input tensor in the ONNX graph.
        lead_hours: Sequence of lead times (hours) corresponding to successive outputs.
            Defaults to [6, 12, 18, 24] if not provided. When `max_forecast_hours`
            is set and the model outputs fewer hours, the tool will loop,
            feeding the last predicted step back into the model until the target
            horizon is covered (or outputs stop carrying both surface and upper fields).
        max_forecast_hours: Optional target horizon (in hours) for iterative rollouts.
            If omitted, the tool stops at whatever leads the ONNX model returns.
    """
    try:
        import onnxruntime as ort  # type: ignore[import]
    except ImportError:
        return _error("onnxruntime is required for Pangu inference. Install onnxruntime and retry.")

    try:
        valid_dt = dt.datetime.fromisoformat(valid_time)
    except ValueError:
        return _error(
            f"valid_time must be ISO-8601 (e.g., 2023-01-01T06:00). Received: {valid_time}"
        )

    surface_file = Path(surface_path).expanduser()
    pressure_file = Path(pressure_path).expanduser()
    for path in (surface_file, pressure_file):
        if not path.exists():
            return _error(f"ERA5 input file not found: {path}")

    resolved_lead_hours = list(lead_hours or (6.0, 12.0, 18.0, 24.0))
    resolved_levels = list(PRESSURE_LEVELS)
    target_horizon = float(max_forecast_hours) if max_forecast_hours is not None else float(resolved_lead_hours[-1])

    try:
        use_cpu, gpu_id = _parse_device(device)
    except ValueError as exc:
        return _error(str(exc))

    resolved_model_path = model_path or os.getenv("PANGU_MODEL_PATH") or "./Pangu_weight/pangu_weather_6.onnx"

    try:
        session = create_session(
            resolved_model_path,
            use_cpu=use_cpu,
            gpu_id=gpu_id,
        )
        provider = "CPUExecutionProvider" if use_cpu else f"CUDAExecutionProvider(device_id={gpu_id})"
        lats, lons = _extract_lat_lon(surface_file)
        surface_tensor = load_surface_tensor(surface_file, valid_dt)
        upper_tensor = load_upper_tensor(pressure_file, valid_dt)

        steps: List[ForecastStep] = []
        unknown_outputs: Dict[str, Tuple[int, ...]] = {}
        lat_size, lon_size = len(lats), len(lons)
        surface_keys = list(SURFACE_VARS.keys())
        upper_keys = list(UPPER_VARS.keys())

        cumulative_hours = 0.0
        iteration = 0
        while cumulative_hours < target_horizon + 1e-6:
            iteration += 1
            inputs = {
                upper_input_name: upper_tensor,
                surface_input_name: surface_tensor,
            }
            outputs = session.run(None, inputs)
            output_names = [out.name for out in session.get_outputs()]

            surface_block: Optional[np.ndarray] = None
            upper_block: Optional[np.ndarray] = None

            for name, array in zip(output_names, outputs):
                try:
                    surface_candidate = _normalise_surface_output(
                        array,
                        surface_keys=surface_keys,
                        lat_size=lat_size,
                        lon_size=lon_size,
                    )
                    if surface_block is None:
                        surface_block = surface_candidate
                    else:
                        raise ValueError(
                            "Multiple surface-like outputs detected; specify the correct ONNX output names."
                        )
                    continue
                except ValueError:
                    pass
                try:
                    upper_candidate = _normalise_upper_output(
                        array,
                        upper_keys=upper_keys,
                        levels=resolved_levels,
                        lat_size=lat_size,
                        lon_size=lon_size,
                    )
                    if upper_block is None:
                        upper_block = upper_candidate
                    else:
                        raise ValueError(
                            "Multiple upper-air-like outputs detected; specify the correct ONNX output names."
                        )
                    continue
                except ValueError:
                    unknown_outputs[name] = tuple(np.asarray(array).shape)

            lead_count = 0
            if surface_block is not None:
                lead_count = max(lead_count, surface_block.shape[0])
            if upper_block is not None:
                lead_count = max(lead_count, upper_block.shape[0])
            if lead_count == 0:
                raise ValueError("Unable to identify Pangu outputs. Inspect the ONNX model outputs.")

            if len(resolved_lead_hours) < lead_count:
                if lead_hours is None:
                    last = resolved_lead_hours[-1] if resolved_lead_hours else 0.0
                    while len(resolved_lead_hours) < lead_count:
                        last = last + 6.0 if last else 6.0
                        resolved_lead_hours.append(float(last))
                else:
                    raise ValueError(
                        f"lead_hours length ({len(resolved_lead_hours)}) is smaller than detected leads ({lead_count})."
                    )
            elif len(resolved_lead_hours) > lead_count:
                resolved_lead_hours = resolved_lead_hours[:lead_count]

            for idx, lh in enumerate(resolved_lead_hours):
                abs_lead = cumulative_hours + float(lh)
                if abs_lead - target_horizon > 1e-6:
                    break
                step_time = valid_dt + dt.timedelta(hours=abs_lead)
                surface_fields: Dict[str, np.ndarray] = {}
                upper_fields: Dict[str, np.ndarray] = {}
                if surface_block is not None:
                    for var_idx, key in enumerate(surface_keys):
                        surface_fields[key] = surface_block[idx, var_idx]
                if upper_block is not None:
                    for var_idx, key in enumerate(upper_keys):
                        upper_fields[key] = upper_block[idx, var_idx]
                steps.append(
                    ForecastStep(
                        valid_time=step_time,
                        surface=surface_fields,
                        atmosphere=upper_fields,
                        metadata={"lead_hours": abs_lead, "iteration": iteration},
                    )
                )

            cumulative_hours += float(resolved_lead_hours[-1])
            if cumulative_hours >= target_horizon - 1e-6:
                break
            if surface_block is None or upper_block is None:
                # Cannot roll forward without both fields
                break
            # Use last-step outputs as next inputs
            surface_tensor = surface_block[-1]
            upper_tensor = upper_block[-1]

        grid = ForecastGridSpec(latitude=lats, longitude=lons, pressure_levels=resolved_levels)
        output_dir_path = Path(output_dir).expanduser()
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / f"pangu_forecast_{valid_dt:%Y%m%dT%H}.nc"
        extra_attrs: Dict[str, Any] = {
            "model": "pangu_weather",
            "checkpoint": Path(resolved_model_path).name,
            "analysis_time": valid_dt.isoformat(),
            "lead_hours": list(map(float, resolved_lead_hours)),
            "target_horizon": target_horizon,
            "device": device,
        }
        export = write_forecast_dataset(
            output_path,
            grid,
            steps,
            global_attrs=extra_attrs,
            base_time=valid_dt,
        )

        metadata = {
            "model": "pangu_weather",
            "output_path": str(export.output_path),
            "forecast_summary": summarise_steps(steps, base_time=valid_dt),
            "surface_variables": export.variables,
            "pressure_variables": export.pressure_variables,
            "analysis_time": valid_dt.isoformat(),
            "provider": provider,
            "input_files": {
                "surface": str(surface_file),
                "pressure": str(pressure_file),
            },
        }
        if unknown_outputs:
            metadata["unknown_outputs"] = {
                name: list(shape) for name, shape in unknown_outputs.items()
            }
        return _ok(metadata)

    except Exception as exc:  # pragma: no cover - safety net
        return _error(f"Pangu inference failed: {exc}", trace=traceback.format_exc())
