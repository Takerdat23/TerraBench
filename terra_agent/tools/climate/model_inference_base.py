"""
Shared helpers for wrapping machine-learning weather model inference as agent tools.

This module provides light-weight data structures that normalise forecast outputs
onto a common latitude/longitude grid and convenience utilities to persist the
result as NetCDF so downstream plotting tools (e.g., ``plot_era5_map``) can use it.

It also includes a minimal template (`ForecastModelTemplate`) that can be copied
when wiring up additional models in the future.
"""


import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import xarray as xr


@dataclass
class ForecastGridSpec:
    """
    Description of the spatial grid shared by all forecast steps.

    Args:
        latitude: 1-D array of latitude values (north-to-south descending preferred).
        longitude: 1-D array of longitude values in degrees east (0–360 or -180–180).
        pressure_levels: Optional sequence of pressure levels (hPa) for 3-D outputs.
        lat_name: Name of the latitude coordinate used when materialising datasets.
        lon_name: Name of the longitude coordinate used when materialising datasets.
        level_name: Name of the pressure-level coordinate.
    """

    latitude: np.ndarray
    longitude: np.ndarray
    pressure_levels: Optional[Sequence[int]] = None
    lat_name: str = "latitude"
    lon_name: str = "longitude"
    level_name: str = "pressure_level"

    def normalised(self) -> "ForecastGridSpec":
        """
        Ensure numpy arrays are 1-D float32/float64 and coerce pressure levels to tuple.
        """
        lat = np.asarray(self.latitude).astype(np.float32, copy=False).reshape(-1)
        lon = np.asarray(self.longitude).astype(np.float32, copy=False).reshape(-1)
        if self.pressure_levels is not None:
            levels = tuple(int(lvl) for lvl in self.pressure_levels)
        else:
            levels = None
        return ForecastGridSpec(
            latitude=lat,
            longitude=lon,
            pressure_levels=levels,
            lat_name=self.lat_name,
            lon_name=self.lon_name,
            level_name=self.level_name,
        )


@dataclass
class ForecastStep:
    """
    Container holding model outputs for a single lead time.

    Args:
        valid_time: Datetime representing the forecast valid time for this step.
        surface: Mapping of variable key -> array shaped (lat, lon).
        atmosphere: Mapping of variable key -> array shaped (levels, lat, lon).
        metadata: Free-form dictionary for provenance (e.g., lead_hours, checkpoints).
    """

    valid_time: dt.datetime
    surface: Mapping[str, np.ndarray] = field(default_factory=dict)
    atmosphere: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass
class ForecastExportResult:
    """
    Summary describing the persisted forecast dataset.
    """

    output_path: Path
    variables: list[str]
    pressure_variables: list[str]
    valid_times: list[str]
    lead_hours: list[float]


class ForecastModelTemplate:
    """
    Template illustrating the common structure for future model tool wrappers.

    Copy this skeleton when adding a new tool:

    ```python
    class MyModelAdapter(ForecastModelTemplate):
        model_name = \"my_future_model\"

        def prepare_inputs(self, request: dict[str, Any]) -> tuple[Any, ForecastGridSpec]:
            # 1. Download or load the analysis data needed by the model.
            # 2. Build and return the model-specific input tensors along with the grid spec.
            ...

        def run_model(self, inputs: Any, device: str) -> Iterable[ForecastStep]:
            # 3. Execute the model on the selected device and yield ForecastStep objects.
            ...
    ```

    Supplying the output of `run_model` to `write_forecast_dataset` will yield a
    NetCDF file that is immediately usable by the plotting tools.
    """

    model_name: str = "model_template"

    def prepare_inputs(self, request: Mapping[str, Any]) -> tuple[Any, ForecastGridSpec]:
        raise NotImplementedError

    def run_model(
        self,
        inputs: Any,
        *,
        device: str,
    ) -> Iterable[ForecastStep]:
        raise NotImplementedError


def _infer_spatial_shape_from_steps(
    steps: Sequence[ForecastStep],
) -> Optional[tuple[int, int]]:
    """
    Inspect the provided steps and return the (lat, lon) shape seen in the data.

    Prefers surface variables (2-D) and falls back to atmospheric variables (3-D)
    if no surface data exist.
    """
    for step in steps:
        for array in step.surface.values():
            data = np.asarray(array)
            if data.ndim >= 2:
                return int(data.shape[-2]), int(data.shape[-1])
    for step in steps:
        for array in step.atmosphere.values():
            data = np.asarray(array)
            if data.ndim >= 3:
                return int(data.shape[-2]), int(data.shape[-1])
    return None


def _align_grid_with_steps(grid: ForecastGridSpec, steps: Sequence[ForecastStep]) -> ForecastGridSpec:
    """
    Ensure the grid latitude/longitude arrays match the data arrays.

    Some model pipelines can emit tensors with one fewer latitude/longitude cell
    than reported in the metadata (e.g., 720x1440 instead of 721x1440). When that
    happens we trim the coordinate arrays so NetCDF export succeeds while keeping
    the grid definition internally consistent.
    """
    target = _infer_spatial_shape_from_steps(steps)
    if target is None:
        return grid
    lat_target, lon_target = target
    latitudes = grid.latitude
    longitudes = grid.longitude
    changed = False
    if lat_target != latitudes.size:
        if lat_target > latitudes.size:
            raise ValueError(
                f"Forecast outputs have {lat_target} latitude points but the grid only defines "
                f"{latitudes.size}. Ensure the input data share the same resolution."
            )
        latitudes = latitudes[:lat_target]
        changed = True
    if lon_target != longitudes.size:
        if lon_target > longitudes.size:
            raise ValueError(
                f"Forecast outputs have {lon_target} longitude points but the grid only defines "
                f"{longitudes.size}. Ensure the input data share the same resolution."
            )
        longitudes = longitudes[:lon_target]
        changed = True
    if not changed:
        return grid
    return ForecastGridSpec(
        latitude=latitudes,
        longitude=longitudes,
        pressure_levels=grid.pressure_levels,
        lat_name=grid.lat_name,
        lon_name=grid.lon_name,
        level_name=grid.level_name,
    )


def _stack_surface_data(
    grid: ForecastGridSpec,
    steps: Sequence[ForecastStep],
    *,
    var_aliases: Optional[Mapping[str, str]] = None,
) -> dict[str, xr.DataArray]:
    if not steps:
        return {}
    var_names = sorted({key for step in steps for key in step.surface})
    dataarrays: dict[str, xr.DataArray] = {}
    times = np.array([np.datetime64(step.valid_time) for step in steps])
    for name in var_names:
        records = []
        for step in steps:
            arr = step.surface.get(name)
            if arr is None:
                raise ValueError(f"Surface variable '{name}' missing for one of the forecast steps.")
            data = np.asarray(arr, dtype=np.float32)
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]
            if data.shape != (grid.latitude.size, grid.longitude.size):
                raise ValueError(
                    f"Surface '{name}' expected shape {(grid.latitude.size, grid.longitude.size)} "
                    f"but received {data.shape}."
                )
            records.append(data)
        stacked = np.stack(records, axis=0)
        alias = var_aliases.get(name, name) if var_aliases else name
        dataarrays[alias] = xr.DataArray(
            stacked,
            dims=("time", grid.lat_name, grid.lon_name),
            coords={
                "time": times,
                grid.lat_name: grid.latitude,
                grid.lon_name: grid.longitude,
            },
            attrs={"source_variable": name},
        )
    return dataarrays


def _stack_atmospheric_data(
    grid: ForecastGridSpec,
    steps: Sequence[ForecastStep],
    *,
    var_aliases: Optional[Mapping[str, str]] = None,
) -> dict[str, xr.DataArray]:
    if not steps:
        return {}
    if grid.pressure_levels is None:
        raise ValueError("pressure_levels must be provided to export atmospheric variables.")
    var_names = sorted({key for step in steps for key in step.atmosphere})
    if not var_names:
        return {}
    times = np.array([np.datetime64(step.valid_time) for step in steps])
    level_count = len(grid.pressure_levels)
    dataarrays: dict[str, xr.DataArray] = {}
    for name in var_names:
        records = []
        for step in steps:
            arr = step.atmosphere.get(name)
            if arr is None:
                raise ValueError(
                    f"Atmospheric variable '{name}' missing for one of the forecast steps."
                )
            data = np.asarray(arr, dtype=np.float32)
            if data.ndim == 4 and data.shape[0] == 1:
                data = data[0]
            if data.shape != (level_count, grid.latitude.size, grid.longitude.size):
                raise ValueError(
                    f"Atmospheric '{name}' expected shape "
                    f"({level_count}, {grid.latitude.size}, {grid.longitude.size}) "
                    f"but received {data.shape}."
                )
            records.append(data)
        stacked = np.stack(records, axis=0)
        alias = var_aliases.get(name, name) if var_aliases else name
        dataarrays[alias] = xr.DataArray(
            stacked,
            dims=("time", grid.level_name, grid.lat_name, grid.lon_name),
            coords={
                "time": times,
                grid.level_name: np.array(grid.pressure_levels, dtype=np.int32),
                grid.lat_name: grid.latitude,
                grid.lon_name: grid.longitude,
            },
            attrs={"source_variable": name},
        )
    return dataarrays


def write_forecast_dataset(
    output_path: Path | str,
    grid: ForecastGridSpec,
    steps: Sequence[ForecastStep],
    *,
    surface_aliases: Optional[Mapping[str, str]] = None,
    atmosphere_aliases: Optional[Mapping[str, str]] = None,
    global_attrs: Optional[Mapping[str, Any]] = None,
    base_time: Optional[dt.datetime] = None,
) -> ForecastExportResult:
    """
    Materialise a NetCDF file containing the provided forecast steps.

    Args:
        output_path: Destination for the NetCDF dataset.
        grid: Shared grid definition across all forecast steps.
        steps: Ordered forecast steps (earliest lead first).
        surface_aliases: Optional mapping from raw surface keys to dataset names.
        atmosphere_aliases: Optional mapping for atmospheric keys.
        global_attrs: Attributes attached to the resulting xarray dataset.
    """
    if not steps:
        raise ValueError("At least one forecast step is required to export a dataset.")
    grid = grid.normalised()
    grid = _align_grid_with_steps(grid, steps)
    surface_arrays = _stack_surface_data(grid, steps, var_aliases=surface_aliases)
    atmosphere_arrays = _stack_atmospheric_data(grid, steps, var_aliases=atmosphere_aliases)

    dataset = xr.Dataset()
    dataset = dataset.assign_coords(
        {
            grid.lat_name: (grid.lat_name, grid.latitude),
            grid.lon_name: (grid.lon_name, grid.longitude),
        }
    )

    for name, da in surface_arrays.items():
        dataset[name] = da
    for name, da in atmosphere_arrays.items():
        dataset[name] = da

    if global_attrs:
        dataset = dataset.assign_attrs({key: value for key, value in global_attrs.items()})

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)

    lead_hours: list[float] = []
    lead_origin = base_time or steps[0].valid_time
    for step in steps:
        delta = step.valid_time - lead_origin
        lead_hours.append(delta.total_seconds() / 3600.0)

    return ForecastExportResult(
        output_path=output_path,
        variables=sorted(surface_arrays),
        pressure_variables=sorted(atmosphere_arrays),
        valid_times=[step.valid_time.isoformat() for step in steps],
        lead_hours=lead_hours,
    )


def summarise_steps(
    steps: Sequence[ForecastStep],
    *,
    base_time: Optional[dt.datetime] = None,
) -> dict[str, Any]:
    """
    Convenience helper returning JSON-safe metadata describing the steps.
    """
    if not steps:
        return {"count": 0}
    lead_origin = base_time or steps[0].valid_time
    summary = {
        "count": len(steps),
        "first_valid_time": steps[0].valid_time.isoformat(),
        "lead_hours": [],
    }
    for step in steps:
        delta = step.valid_time - lead_origin
        summary["lead_hours"].append(delta.total_seconds() / 3600.0)
    return summary
