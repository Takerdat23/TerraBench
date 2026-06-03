"""
Derive tropical-cyclone-like tracks from ERA5 surface/pressure fields.

Inputs
------
--surface PATH  ERA5 surface NetCDF with at least `msl` (Pa); optional `u10`,`v10`.
--pressure PATH ERA5 pressure-level NetCDF with `u`,`v` including 850 hPa.

Outputs
-------
--png PATH  Figure with global tracks on a Mollweide projection.
--csv PATH  CSV of fixes: storm_id,time,lat,lon,msl_hPa,zeta850_s-1,ws10_ms

Options
-------
--min-points N   Drop tracks shorter than N fixes (default 5).
--max-link-km R  Max great-circle distance (km) to link fixes between steps (default 500 km).
"""

import argparse
import csv
import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

EARTH_RADIUS_M = 6_371_000.0
MIN_VORT = 5e-5  # s^-1 threshold for candidate centers
MSL_MAX_HPA = 1015.0  # ignore weak highs


@dataclass
class Fix:
    time: np.datetime64
    lat: float
    lon: float  # degrees in [-180, 180]
    msl_hpa: float
    zeta: float  # 850-hPa relative vorticity (s^-1)
    ws10: Optional[float] = None


@dataclass
class Track:
    storm_id: int
    fixes: list[Fix] = field(default_factory=list)
    last_time_idx: int = -1

    def add(self, fix: Fix, time_idx: int) -> None:
        self.fixes.append(fix)
        self.last_time_idx = time_idx


def _great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km using haversine formula."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return (EARTH_RADIUS_M * c) / 1000.0


def _compute_vorticity(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Relative vorticity (s^-1) on a regular lat/lon grid using centered differences.
    u, v: 2D arrays (lat, lon) in m/s
    lats, lons: 1D arrays in degrees
    """
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)

    # np.gradient returns derivatives w.r.t. coordinate units; convert to meters.
    dv_dlat, dv_dlon = np.gradient(v, lat_rad, lon_rad, edge_order=2)
    du_dlat, du_dlon = np.gradient(u, lat_rad, lon_rad, edge_order=2)

    cos_lat = np.cos(lat_rad)[:, None]
    # Avoid division by zero at poles by clipping cos_lat.
    cos_lat = np.clip(cos_lat, 1e-6, None)

    dv_dx = dv_dlon / (EARTH_RADIUS_M * cos_lat)
    du_dy = du_dlat / EARTH_RADIUS_M

    return dv_dx - du_dy


def _local_minima(mask_field: np.ndarray) -> np.ndarray:
    """
    Boolean mask of local minima using a 3x3 neighborhood, wrapping in longitude.
    """
    lat_pad = np.pad(mask_field, ((1, 1), (0, 0)), mode="edge")
    padded = np.pad(lat_pad, ((0, 0), (1, 1)), mode="wrap")

    # Compute 3x3 rolling minimum via reductions over shifted views.
    neighbors = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            neighbors.append(padded[1 + di : 1 + di + mask_field.shape[0], 1 + dj : 1 + dj + mask_field.shape[1]])
    window_min = np.minimum.reduce(neighbors)
    return mask_field <= window_min


def _wrap_lon(lon: np.ndarray) -> np.ndarray:
    lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
    return lon_wrapped


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth in degrees from point 1 to point 2."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0


def _detect_candidates(
    msl_hpa: np.ndarray,
    zeta: np.ndarray,
    u10: Optional[np.ndarray],
    v10: Optional[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    max_candidates: int = 200,
) -> list[tuple[int, int, float, float, float, Optional[float]]]:
    """
    Return list of candidate centers sorted by lowest MSLP.
    Each entry: (ilat, ilon, msl_hpa, zeta, ws10)
    """
    minima_mask = _local_minima(msl_hpa)
    vort_mask = zeta >= MIN_VORT
    pressure_mask = msl_hpa <= MSL_MAX_HPA
    mask = minima_mask & vort_mask & pressure_mask
    idx = np.argwhere(mask)
    if idx.size == 0:
        return []

    ws10_field = None
    if u10 is not None and v10 is not None:
        ws10_field = np.hypot(u10, v10)

    candidates = []
    for ilat, ilon in idx:
        msl_val = float(msl_hpa[ilat, ilon])
        zeta_val = float(zeta[ilat, ilon])
        ws_val = float(ws10_field[ilat, ilon]) if ws10_field is not None else None
        candidates.append((ilat, ilon, msl_val, zeta_val, ws_val))

    candidates.sort(key=lambda tup: tup[2])  # lowest pressure first
    return candidates[:max_candidates]


def _link_tracks_greedy(
    tracks: list[Track],
    candidates: list[tuple[int, int, float, float, Optional[float]]],
    lats: np.ndarray,
    lons: np.ndarray,
    time_value: np.datetime64,
    time_idx: int,
    next_id: int,
    max_link_km: float,
) -> int:
    """
    Greedy nearest-neighbor linking to extend existing storms by <= max_link_km.
    Returns next_id to use for new tracks.
    """
    # Consider only storms updated at the previous timestep.
    active = [t for t in tracks if t.last_time_idx == time_idx - 1]

    for ilat, ilon, msl_val, zeta_val, ws_val in candidates:
        lat = float(lats[ilat])
        lon = float(lons[ilon])
        lon = _wrap_lon(lon)

        best_track: Optional[Track] = None
        best_dist = None
        for track in active:
            last_fix = track.fixes[-1]
            dist = _great_circle_km(lat, lon, last_fix.lat, last_fix.lon)
            if dist <= max_link_km and (best_dist is None or dist < best_dist):
                best_track = track
                best_dist = dist

        fix = Fix(time=time_value, lat=lat, lon=lon, msl_hpa=msl_val, zeta=zeta_val, ws10=ws_val)
        if best_track is None:
            new_track = Track(storm_id=next_id)
            new_track.add(fix, time_idx)
            tracks.append(new_track)
            next_id += 1
        else:
            best_track.add(fix, time_idx)

    return next_id


def _track_motion(track: Track) -> tuple[float, Optional[float]]:
    """Return (speed_kmh, heading_deg) from the last two fixes; 0/None if not available."""
    if len(track.fixes) < 2:
        return 0.0, None
    f_prev, f_last = track.fixes[-2], track.fixes[-1]
    dt_hours = float((f_last.time - f_prev.time) / np.timedelta64(1, "h"))
    if dt_hours <= 0:
        return 0.0, None
    dist = _great_circle_km(f_prev.lat, f_prev.lon, f_last.lat, f_last.lon)
    speed = dist / dt_hours
    heading = _bearing_deg(f_prev.lat, f_prev.lon, f_last.lat, f_last.lon)
    return speed, heading


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between headings in degrees."""
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _link_tracks_cost_min(
    tracks: list[Track],
    candidates: list[tuple[int, int, float, float, Optional[float]]],
    lats: np.ndarray,
    lons: np.ndarray,
    time_value: np.datetime64,
    time_idx: int,
    next_id: int,
    gate_km: float,
    cost_cutoff: float,
    w_d: float,
    w_u: float,
    w_alpha: float,
    w_zeta: float,
    w_p: float,
    u_ref: float,
    zeta_ref: float,
    p_ref: float,
) -> int:
    """
    Hungarian assignment with a Hodges-style cost:
    J = w_d*(Δs/R_gate)^2 + w_u*(Δu/u_ref)^2 + w_alpha*(Δalpha/π)^2
        + w_zeta*(Δzeta/zeta_ref)^2 + w_p*(Δmsl/p_ref)^2
    """
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "scipy is required for cost-minimizing linking. Install scipy to use linking_mode='hungarian'."
        ) from exc

    active = [t for t in tracks if t.last_time_idx == time_idx - 1]
    if not active or not candidates:
        for ilat, ilon, msl_val, zeta_val, ws_val in candidates:
            fix = Fix(
                time=time_value,
                lat=float(lats[ilat]),
                lon=float(_wrap_lon(np.array([lons[ilon]]))[0]),
                msl_hpa=msl_val,
                zeta=zeta_val,
                ws10=ws_val,
            )
            new_track = Track(storm_id=next_id)
            new_track.add(fix, time_idx)
            tracks.append(new_track)
            next_id += 1
        return next_id

    n_t, n_c = len(active), len(candidates)
    big_penalty = 1e12
    cost = np.full((n_t, n_c), fill_value=big_penalty, dtype=float)
    deltas = np.full((n_t, n_c), fill_value=np.inf, dtype=float)

    ref_time = active[0].fixes[-1].time
    dt_hours = float((time_value - ref_time) / np.timedelta64(1, "h"))
    if dt_hours <= 0:
        dt_hours = 1.0

    for i, track in enumerate(active):
        last_fix = track.fixes[-1]
        speed_prev, heading_prev = _track_motion(track)
        for j, (ilat, ilon, msl_val, zeta_val, ws_val) in enumerate(candidates):
            lat = float(lats[ilat])
            lon = float(_wrap_lon(np.array([lons[ilon]]))[0])
            dist = _great_circle_km(lat, lon, last_fix.lat, last_fix.lon)
            deltas[i, j] = dist
            if dist > gate_km:
                cost[i, j] = big_penalty
                continue
          
            speed_new = dist / dt_hours if dt_hours > 0 else 0.0
            delta_u = abs(speed_new - speed_prev) if speed_prev is not None else speed_new

            heading_new = _bearing_deg(last_fix.lat, last_fix.lon, lat, lon) if dist > 0 else heading_prev or 0.0
            if heading_prev is None:
                delta_alpha = 0.0
            else:
                delta_alpha = math.radians(_angle_diff_deg(heading_new, heading_prev))

            delta_zeta = abs(zeta_val - last_fix.zeta)
            delta_msl = abs(msl_val - last_fix.msl_hpa)

            j_cost = (
                w_d * (dist / gate_km) ** 2
                + w_u * (delta_u / max(u_ref, 1e-6)) ** 2
                + w_alpha * (delta_alpha / math.pi) ** 2
                + w_zeta * (delta_zeta / max(zeta_ref, 1e-9)) ** 2
                + w_p * (delta_msl / max(p_ref, 1e-3)) ** 2
            )
            cost[i, j] = j_cost

    if not np.isfinite(cost).any():
        row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)
    else:
        # print("[track linking] Solving assignment problem...")
        # print("[track linking] Cost matrix stats: min =", np.nanmin(cost), "max =", np.nanmax(cost))
        # print(f"[track linking] Cost: {cost}")
        row_ind, col_ind = linear_sum_assignment(cost)
        # print("[track linking] Assignment complete., found", len(row_ind), "assignments.")

    assigned_candidates: set[int] = set()
    for i, j in zip(row_ind, col_ind, strict=False):
        if np.isinf(cost[i, j]) or cost[i, j] > cost_cutoff or deltas[i, j] > gate_km:
            continue
        lat = float(lats[candidates[j][0]])
        lon = float(_wrap_lon(np.array([lons[candidates[j][1]]]))[0])
        msl_val, zeta_val, ws_val = candidates[j][2], candidates[j][3], candidates[j][4]
        fix = Fix(time=time_value, lat=lat, lon=lon, msl_hpa=msl_val, zeta=zeta_val, ws10=ws_val)
        active[i].add(fix, time_idx)
        assigned_candidates.add(j)

    for j, cand in enumerate(candidates):
        if j in assigned_candidates:
            continue
        lat = float(lats[cand[0]])
        lon = float(_wrap_lon(np.array([lons[cand[1]]]))[0])
        msl_val, zeta_val, ws_val = cand[2], cand[3], cand[4]
        fix = Fix(time=time_value, lat=lat, lon=lon, msl_hpa=msl_val, zeta=zeta_val, ws10=ws_val)
        new_track = Track(storm_id=next_id)
        new_track.add(fix, time_idx)
        tracks.append(new_track)
        next_id += 1

    return next_id


def _plot_tracks(tracks: Iterable[Track], png_path: Path, color_mode: str = "multi") -> None:
    """
    Plot candidate tracks on a Mollweide map.

    color_mode: "mono" for all-blue lines, "multi" for per-track colors.
    """
    tracks_list = list(tracks)
    plt.figure(figsize=(13, 6.5))

    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
        from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Cartopy is required to render the world map background. "
            "Install it (e.g., pip install cartopy) and retry."
        ) from exc

    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.set_facecolor("#e9f0f7")

    land = cfeature.LAND.with_scale("110m")
    ocean = cfeature.OCEAN.with_scale("110m")
    borders = cfeature.BORDERS.with_scale("110m")
    coastline = cfeature.COASTLINE.with_scale("110m")
    lakes = cfeature.LAKES.with_scale("110m")

    ax.add_feature(ocean, facecolor="#e9f0f7", edgecolor="none")
    ax.add_feature(land, facecolor="#f5f5f3", edgecolor="none")
    ax.add_feature(lakes, facecolor="#dfe9f2", edgecolor="0.6", linewidth=0.3)
    ax.add_feature(coastline, linewidth=0.8, color="#3d3d3d")
    ax.add_feature(borders, linewidth=0.4, color="0.35")

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="0.7",
        linestyle=":",
        alpha=0.8,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    if color_mode not in {"mono", "multi"}:
        raise ValueError("color_mode must be 'mono' or 'multi'.")

    mono_color = "#2f6c9f"
    colors = (
        plt.cm.tab20(np.linspace(0, 1, max(1, len(tracks_list))))
        if color_mode == "multi"
        else [mono_color] * max(1, len(tracks_list))
    )

    for idx, track in enumerate(tracks_list):
        lats = np.array([f.lat for f in track.fixes])
        lons = np.array([f.lon for f in track.fixes])
        lons_deg = _wrap_lon(lons)
        ax.plot(
            lons_deg,
            lats,
            linewidth=0.9,
            color=colors[idx % len(colors)],
            alpha=0.55 if color_mode == "mono" else 0.8,
            solid_capstyle="round",
            transform=ccrs.PlateCarree(),
        )

    ax.set_title("Tropical-Cyclone Track Candidates (ERA5)", fontsize=12)
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=200)
    plt.close()


def _write_csv(tracks: Iterable[Track], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["storm_id", "time", "lat", "lon", "msl_hPa", "zeta850_s-1", "ws10_ms"])
        for track in tracks:
            for fix in track.fixes:
                writer.writerow(
                    [
                        track.storm_id,
                        np.datetime_as_string(fix.time, unit="s"),
                        f"{fix.lat:.2f}",
                        f"{fix.lon:.2f}",
                        f"{fix.msl_hpa:.2f}",
                        f"{fix.zeta:.6f}",
                        "" if fix.ws10 is None else f"{fix.ws10:.2f}",
                    ]
                )


def _get_level_name(ds: xr.Dataset) -> str:
    for name in ("level", "pressure_level", "isobaricInhPa"):
        if name in ds.coords:
            return name
    raise ValueError("Could not find a pressure level coordinate (expected 'level' or 'pressure_level').")


def _get_coord_name(ds: xr.Dataset, candidates: tuple[str, ...], label: str) -> str:
    """Return the first matching coordinate name in `candidates`."""
    for cand in candidates:
        if cand in ds.coords:
            return cand
        if cand in ds.variables and ds[cand].dims:  # sometimes stored as variable
            return cand
    raise ValueError(f"Could not find a {label} coordinate (tried {candidates}).")


def _drop_optional_dims(da: xr.DataArray, dims: tuple[str, ...]) -> xr.DataArray:
    """Drop leading optional dims like ensemble/expver by taking the first entry."""
    for dim in dims:
        if dim in da.dims:
            da = da.isel({dim: 0})
    return da


def _sanitize(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (np.datetime64,)):
        return np.datetime_as_string(obj, unit="s")
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _ok(payload: dict) -> ToolResponse:
    sanitized = _sanitize(payload)
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(sanitized))], metadata=sanitized)  # type: ignore[name-defined]


def _error(message: str) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {message}")],
        metadata={"error": True, "message": message},
    )


def derive_tracks(
    surface: str,
    pressure: str,
    png: str,
    csv: str,
    *,
    min_points: int = 5,
    max_link_km: float = 500.0,
    linking_mode: str = "greedy",
    cost_cutoff: float = 5.0,
    cost_weights: Optional[dict] = None,
    gate_km: Optional[float] = None,
    color_mode: str = "multi",
) -> dict:
    surface_ds = xr.open_dataset(surface)
    pressure_ds = xr.open_dataset(pressure)

    if "msl" not in surface_ds:
        raise ValueError("Surface file must contain 'msl' (mean sea-level pressure).")

    # Handle varied coordinate naming across ERA5/ECMWF exports.
    time_name = _get_coord_name(surface_ds, ("time", "valid_time"), "time")
    lat_name = _get_coord_name(surface_ds, ("latitude", "lat"), "latitude")
    lon_name = _get_coord_name(surface_ds, ("longitude", "lon"), "longitude")

    level_name = _get_level_name(pressure_ds)
    levels = pressure_ds[level_name].values
    if 850 not in levels:
        raise ValueError(f"Pressure file must include 850 hPa; found {levels}.")

    lats = surface_ds[lat_name].values
    lons = surface_ds[lon_name].values
    times = surface_ds[time_name].values

    msl_da = _drop_optional_dims(surface_ds["msl"], ("number", "expver"))
    u10_da = _drop_optional_dims(surface_ds["u10"], ("number", "expver")) if "u10" in surface_ds else None
    v10_da = _drop_optional_dims(surface_ds["v10"], ("number", "expver")) if "v10" in surface_ds else None

    u850_da = _drop_optional_dims(pressure_ds["u"], ("number", "expver")).sel({level_name: 850}, method="nearest")
    v850_da = _drop_optional_dims(pressure_ds["v"], ("number", "expver")).sel({level_name: 850}, method="nearest")

    msl_all = msl_da.values  # Pa
    u850_all = u850_da.values
    v850_all = v850_da.values

    tracks: list[Track] = []
    next_id = 1

    cost_cfg = {
        "w_d": 1.0,
        "w_u": 0.3,
        "w_alpha": 0.2,
        "w_zeta": 0.5,
        "w_p": 0.2,
        "u_ref": 50.0,  # km/h
        "zeta_ref": 5e-5,
        "p_ref": 15.0,  # hPa
    }
    if cost_weights:
        cost_cfg.update({k: v for k, v in cost_weights.items() if k in cost_cfg})

    gate_distance = gate_km or max_link_km
    linking_mode_norm = (linking_mode or "greedy").lower()

    for t_idx, t_val in enumerate(times):
        msl_hpa = (msl_all[t_idx] / 100.0).astype(np.float32)
        u850 = u850_all[t_idx]
        v850 = v850_all[t_idx]
        zeta = _compute_vorticity(u850, v850, lats, lons)

        u10_t = u10_da.values[t_idx] if u10_da is not None else None
        v10_t = v10_da.values[t_idx] if v10_da is not None else None

        candidates = _detect_candidates(msl_hpa, zeta, u10_t, v10_t, lats, lons)
        if candidates:
            if linking_mode_norm == "greedy":
                next_id = _link_tracks_greedy(
                    tracks,
                    candidates,
                    lats,
                    lons,
                    time_value=t_val,
                    time_idx=t_idx,
                    next_id=next_id,
                    max_link_km=gate_distance,
                )
            elif linking_mode_norm == "hungarian":
                next_id = _link_tracks_cost_min(
                    tracks,
                    candidates,
                    lats,
                    lons,
                    time_value=t_val,
                    time_idx=t_idx,
                    next_id=next_id,
                    gate_km=gate_distance,
                    cost_cutoff=cost_cutoff,
                    w_d=cost_cfg["w_d"],
                    w_u=cost_cfg["w_u"],
                    w_alpha=cost_cfg["w_alpha"],
                    w_zeta=cost_cfg["w_zeta"],
                    w_p=cost_cfg["w_p"],
                    u_ref=cost_cfg["u_ref"],
                    zeta_ref=cost_cfg["zeta_ref"],
                    p_ref=cost_cfg["p_ref"],
                )
            else:
                raise ValueError("linking_mode must be 'greedy' or 'hungarian'.")

    filtered = [trk for trk in tracks if len(trk.fixes) >= min_points]

    csv_path = Path(csv)
    png_path = Path(png)
    _write_csv(filtered, csv_path)
    _plot_tracks(filtered, png_path, color_mode=color_mode)

    fix_count = sum(len(t.fixes) for t in filtered)
    return {
        "tracks": len(filtered),
        "fixes": fix_count,
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "min_points": min_points,
        "max_link_km": max_link_km,
        "linking_mode": linking_mode_norm,
    }


def run_tc_tracks_from_era5(
    surface_path: str,
    pressure_path: str,
    png_path: str = "./plots/tc_tracks.png",
    csv_path: str = "./data/tc_tracks.csv",
    min_points: int = 5,
    max_link_km: float = 500.0,
    linking_mode: str = "greedy",
    cost_cutoff: float = 5.0,
    cost_weights: Optional[dict] = None,
    gate_km: Optional[float] = None,
    color_mode: str = "multi",
) -> ToolResponse:
    """
    Derive TC-like tracks from ERA5 and emit CSV + PNG artifacts.
    color_mode: "mono" (all blue lines) or "multi" (per-track colors).
    linking_mode: "greedy" (nearest-neighbor) or "hungarian" (cost-minimizing with motion/intensity continuity).
    """
    try:
        result = derive_tracks(
            surface=surface_path,
            pressure=pressure_path,
            png=png_path,
            csv=csv_path,
            min_points=min_points,
            max_link_km=max_link_km,
            linking_mode=linking_mode,
            cost_cutoff=cost_cutoff,
            cost_weights=cost_weights,
            gate_km=gate_km,
            color_mode=color_mode,
        )
        return _ok(
            {
                "ok": True,
                **result,
            }
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        return _error(str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive TC-like tracks from ERA5 msl + 850-hPa winds.")
    parser.add_argument("--surface", required=True, help="ERA5 surface NetCDF with msl (Pa) and optional u10/v10.")
    parser.add_argument("--pressure", required=True, help="ERA5 pressure-level NetCDF with u/v including 850 hPa.")
    parser.add_argument("--png", required=True, help="Output PNG path for global track plot (Mollweide).")
    parser.add_argument("--csv", required=True, help="Output CSV path for track fixes.")
    parser.add_argument("--min-points", type=int, default=5, help="Minimum fixes per track to keep (default 5).")
    parser.add_argument("--max-link-km", type=float, default=500.0, help="Max link distance between steps (km).")
    parser.add_argument(
        "--linking-mode",
        choices=("greedy", "hungarian"),
        default="greedy",
        help="Linking algorithm: 'greedy' nearest neighbor or 'hungarian' cost-minimizing assignment.",
    )
    parser.add_argument(
        "--cost-cutoff",
        type=float,
        default=5.0,
        help="Reject Hungarian assignments with cost above this threshold.",
    )
    parser.add_argument(
        "--color-mode",
        choices=("mono", "multi"),
        default="multi",
        help="Plot tracks as single blue color ('mono') or per-track colors ('multi').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    derive_tracks(
        surface=ns.surface,
        pressure=ns.pressure,
        png=ns.png,
        csv=ns.csv,
        min_points=ns.min_points,
        max_link_km=ns.max_link_km,
        linking_mode=ns.linking_mode,
        cost_cutoff=ns.cost_cutoff,
        color_mode=ns.color_mode,
    )
