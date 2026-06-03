"""
Google Earth Engine-backed Sentinel-2 tools for TerraBench/TerraAgent.

Functions:
- gee_fetch_sentinel2_indices: compute NDVI/NBR/NDBI/NDWI (or a subset), return
  AOI stats, and download colorized thumbnails.
- gee_index_area_stats: derive simple land/water/built/burn area fractions from
  the spectral indices.
- gee_fetch_truecolor_thumbnail: download a natural-color composite thumbnail.

The Earth Engine project ID is read from the environment (GEE_PROJECT_ID,
EARTHENGINE_PROJECT, EE_PROJECT, or EE_PROJECT_ID) unless explicitly passed.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import certifi
import numpy as np
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    def load_dotenv(*_: Any, **__: Any) -> bool:
        return False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency guard
    matplotlib = None  # type: ignore
    plt = None  # type: ignore

try:
    import rasterio
    from rasterio import warp
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
except ImportError:  # pragma: no cover - optional dependency guard
    rasterio = None  # type: ignore
    warp = None  # type: ignore
    Resampling = None  # type: ignore
    from_bounds = None  # type: ignore

# Load .env so service account + project IDs set there are available to the tool.
load_dotenv()

try:
    import ee
except ImportError:  # pragma: no cover - optional dependency guard
    ee = None  # type: ignore

__all__ = [
    "gee_fetch_sentinel2_indices",
    "gee_index_area_stats",
    "gee_fetch_truecolor_thumbnail",
    "composite_satellite_environmental_image",
]

DEFAULT_INDICES = ("NDVI", "NBR", "NDBI", "NDWI")
NORTH_UP_TOL = 1e-9

CLASS_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "vegetation": {"index": "NDVI", "min": 0.25},
    "water": {"index": "NDWI", "min": 0.1},
    "built_up": {"index": "NDBI", "min": 0.0},
    "burn_scars": {"index": "NBR", "max": 0.1},
}

INDEX_VIZ_PARAMS: Dict[str, Dict[str, Any]] = {
    "NDVI": {
        "min": -0.5,
        "max": 1.0,
        "palette": [
            "#7f3b08",
            "#b35806",
            "#f1a340",
            "#fee0b6",
            "#f7f7f7",
            "#d8f0d3",
            "#7fbf7b",
            "#1b7837",
            "#00441b",
        ],
    },
    "NBR": {
        "min": -1.0,
        "max": 1.0,
        "palette": [
            "#7a0403",
            "#d73027",
            "#f46d43",
            "#fdae61",
            "#fee08b",
            "#d9ef8b",
            "#66bd63",
            "#1a9850",
            "#006837",
        ],
    },
    "NDBI": {
        "min": -0.5,
        "max": 0.5,
        "palette": [
            "#081d58",
            "#253494",
            "#225ea8",
            "#1d91c0",
            "#41b6c4",
            "#7fcdbb",
            "#c7e9b4",
            "#edf8b1",
            "#ffffd9",
        ],
    },
    "NDWI": {
        "min": -0.5,
        "max": 1.0,
        "palette": [
            "#8c510a",
            "#d8b365",
            "#f6e8c3",
            "#c7eae5",
            "#5ab4ac",
            "#01665e",
            "#003c30",
        ],
    },
}


@dataclass
class TextBlock:
    type: str
    text: str


@dataclass
class ToolResponse:
    content: list[TextBlock]
    metadata: Dict[str, Any]


BASE_DIR = Path(__file__).resolve().parents[3]
_COMBINED_BUNDLE_PATH = Path("/tmp/gee_combined.pem")
_CA_ENV_KEYS: tuple[str, ...] = ("GEE_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")
_VERIFY_FLAG_ENV = "GEE_VERIFY"


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return value


def _ok(payload: Dict[str, Any]) -> ToolResponse:
    cleaned = _sanitize(payload)
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(cleaned))], metadata=cleaned)


def _error(msg: str) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=f"Error: {msg}")], metadata={"error": True, "message": msg})


def _select_project_id(project_id: Optional[str]) -> Optional[str]:
    if project_id:
        return project_id
    for key in ("GEE_PROJECT_ID", "EARTHENGINE_PROJECT", "EE_PROJECT", "EE_PROJECT_ID"):
        val = os.getenv(key)
        if val:
            return val
    return None


def _init_earth_engine(project_id: Optional[str]) -> Tuple[Optional[str], Optional[ToolResponse]]:
    if ee is None:
        return None, _error("earthengine-api is not installed. Install via 'pip install earthengine-api' and retry.")

    pid = _select_project_id(project_id)

    # 1) Service account flow (non-interactive) if env vars are present.
    service_account = os.getenv("GEE_SERVICE_ACCOUNT")
    key_path = os.getenv("GEE_PRIVATE_KEY_FILE")
    if service_account and key_path:
        try:
            credentials = ee.ServiceAccountCredentials(service_account, key_path)
            ee.Initialize(credentials=credentials, project=pid)
            return pid, None
        except Exception as exc:  # pragma: no cover
            return None, _error(f"Service account initialization failed: {exc}")

    # 2) Reuse existing user credentials if available.
    try:
        ee.Initialize(project=pid)
        return pid, None
    except Exception as exc:  # pragma: no cover - runtime dependent
        # 3) App-default / no-browser auth hint for headless use.
        try:
            ee.Authenticate(auth_mode="appdefault")
            ee.Initialize(project=pid)
            return pid, None
        except Exception as exc2:  # pragma: no cover
            hint = (
                "Run `earthengine authenticate --quiet --no-browser --code <AUTH_CODE>` once on this host "
                "or set GEE_SERVICE_ACCOUNT + GEE_PRIVATE_KEY_FILE in .env."
            )
            return None, _error(f"Failed to initialize Earth Engine: {exc2}. {hint}")


def _resolve_verify_setting() -> Any:
    flag = os.getenv(_VERIFY_FLAG_ENV)
    if isinstance(flag, str) and flag.lower() in {"0", "false", "no", "off"}:
        return False

    env_values: List[str] = []
    ca_paths: List[Path] = []
    for env_key in _CA_ENV_KEYS:
        value = os.getenv(env_key)
        if not value:
            continue
        env_values.append(value)
        expanded = Path(value).expanduser()
        if not expanded.is_absolute():
            candidate = BASE_DIR / expanded
            if candidate.exists():
                expanded = candidate
        if expanded.exists():
            ca_paths.append(expanded)

    if ca_paths:
        try:
            combined_parts = [Path(certifi.where()).read_text()]
            for path in ca_paths:
                try:
                    combined_parts.append(path.read_text())
                except FileNotFoundError:
                    continue
            _COMBINED_BUNDLE_PATH.write_text("\n".join(combined_parts))
            return str(_COMBINED_BUNDLE_PATH)
        except Exception:
            return str(ca_paths[0])

    if env_values:
        return env_values[0]

    return True


def _make_aoi(bbox: Sequence[float]) -> ee.Geometry:
    if len(bbox) != 4:
        raise ValueError("bbox must be [north, west, south, east].")
    north, west, south, east = map(float, bbox)
    return ee.Geometry.Rectangle([west, south, east, north])


def _mask_sentinel2_clouds(image: Any) -> Any:
    qa = image.select("QA60")
    cloud_mask = 1 << 10
    cirrus_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_mask).eq(0).And(qa.bitwiseAnd(cirrus_mask).eq(0))
    masked = image.updateMask(mask)
    optical = masked.select(
        ["B2", "B3", "B4", "B8", "B11", "B12"],
        ["blue", "green", "red", "nir", "swir1", "swir2"],
    )
    return optical.divide(10000.0)


def _sentinel_collection(aoi: Any, start_date: str, end_date: str, max_cloud: float) -> Any:
    sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    sr_count = sr.size()
    sr_ok = ee.Number(sr_count).gt(0)

    def use_toa():
        return (
            ee.ImageCollection("COPERNICUS/S2")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
        )

    base_collection = ee.ImageCollection(ee.Algorithms.If(sr_ok, sr, use_toa()))
    return (
        base_collection.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud)).map(_mask_sentinel2_clouds)
    )


def _add_indices(image: Any, indices: Iterable[str]) -> Any:
    desired = [idx.strip().upper() for idx in indices]
    bands = []
    for idx in desired:
        if idx == "NDVI":
            bands.append(image.normalizedDifference(["nir", "red"]).rename("NDVI"))
        elif idx == "NBR":
            bands.append(image.normalizedDifference(["nir", "swir2"]).rename("NBR"))
        elif idx == "NDBI":
            bands.append(image.normalizedDifference(["swir1", "nir"]).rename("NDBI"))
        elif idx == "NDWI":
            bands.append(image.normalizedDifference(["green", "nir"]).rename("NDWI"))
    if bands:
        return image.addBands(bands)
    return image


def _geometry_area_sq_km(geom: Any) -> float:
    return float(ee.Number(geom.area(maxError=1)).divide(1_000_000).getInfo())


def _prepare_index_image(
    bbox: Sequence[float],
    start_date: str,
    end_date: str,
    max_cloud: float,
    indices: Iterable[str],
    project_id: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[ToolResponse]]:
    pid, err = _init_earth_engine(project_id)
    if err:
        return None, err

    try:
        aoi = _make_aoi(bbox)
        collection = _sentinel_collection(aoi, start_date, end_date, max_cloud)
        size = int(collection.size().getInfo())
        if size == 0:
            return None, _error("No Sentinel-2 images found for the requested AOI/time range after cloud filtering.")
        composite = collection.median().clip(aoi)
        with_indices = _add_indices(composite, indices)
        area_sq_km = _geometry_area_sq_km(aoi)
    except Exception as exc:  # pragma: no cover - EE runtime dependent
        return None, _error(f"Failed to prepare Sentinel-2 composite: {exc}")

    return (
        {
            "project_id": pid,
            "aoi": aoi,
            "collection_size": size,
            "image": with_indices,
            "area_sq_km": area_sq_km,
        },
        None,
    )


def _index_stats(image: Any, region: Any, indices: Iterable[str], scale: float) -> Dict[str, Dict[str, Any]]:
    reducer = (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
    )
    results: Dict[str, Dict[str, Any]] = {}
    for idx in indices:
        band = idx.upper()
        stats = image.select(band).reduceRegion(
            reducer=reducer,
            geometry=region,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
        )
        try:
            info = stats.getInfo() or {}
        except Exception:  # pragma: no cover
            info = {}
        results[band] = info
    return results


def _download_index_png(
    image: Any,
    index_name: str,
    region: Any,
    dimensions: int,
    out_dir: Path,
    start_date: str,
    end_date: str,
) -> Optional[str]:
    viz = INDEX_VIZ_PARAMS.get(index_name.upper(), {"min": -1.0, "max": 1.0, "palette": ["#000000", "#ffffff"]})
    rgb_image = image.select(index_name.upper()).visualize(**viz)
    thumb_params = {"region": region, "dimensions": dimensions, "format": "png"}
    try:
        url = rgb_image.getThumbURL(thumb_params)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to get thumbnail URL for {index_name}: {exc}") from exc

    safe_start = start_date.replace("-", "")
    safe_end = end_date.replace("-", "")
    filename = f"{index_name.upper()}_{safe_start}_{safe_end}.png"
    filepath = out_dir / filename

    response = requests.get(url, stream=True, timeout=120, verify=_resolve_verify_setting())
    response.raise_for_status()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return str(filepath)


def _download_truecolor(
    image: Any,
    region: Any,
    dimensions: int,
    out_dir: Path,
    start_date: str,
    end_date: str,
    overlay_image: Any = None,
) -> Optional[str]:
    rgb_image = image.select(["red", "green", "blue"]).visualize(min=0.0, max=0.3)
    if overlay_image is not None:
        rgb_image = rgb_image.blend(overlay_image)
    thumb_params = {"region": region, "dimensions": dimensions, "format": "png"}
    try:
        url = rgb_image.getThumbURL(thumb_params)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to get truecolor thumbnail URL: {exc}") from exc

    safe_start = start_date.replace("-", "")
    safe_end = end_date.replace("-", "")
    filename = f"TRUECOLOR_{safe_start}_{safe_end}.png"
    filepath = out_dir / filename

    response = requests.get(url, stream=True, timeout=120, verify=_resolve_verify_setting())
    response.raise_for_status()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return str(filepath)


def _coerce_overlay_geometry(item: Any) -> Any:
    """Convert common geometry payloads to ee.Geometry."""
    if item is None:
        return None
    # bbox as [north, west, south, east]
    if isinstance(item, (list, tuple)) and len(item) == 4:
        try:
            north, west, south, east = map(float, item)
            return ee.Geometry.Rectangle([west, south, east, north])
        except Exception:
            return None
    try:
        return ee.Geometry(item)
    except Exception:
        return None


def _build_overlay_image(
    aoi: Any,
    overlay_geometries: Optional[Sequence[Any]],
    draw_aoi_outline: bool,
    overlay_color: str,
    overlay_width: int,
    overlay_opacity: float,
    overlay_fill: bool,
) -> Optional[Any]:
    geometries: List[Any] = []
    if draw_aoi_outline and aoi is not None:
        geometries.append(aoi)
    for item in overlay_geometries or []:
        geom = _coerce_overlay_geometry(item)
        if geom is not None:
            geometries.append(geom)
    if not geometries:
        return None

    fc = ee.FeatureCollection([ee.Feature(g) for g in geometries]).geometry()
    overlay = ee.Image().paint(fc, 1, overlay_width)
    if overlay_fill:
        overlay = overlay.paint(fc, 1)
    return overlay.visualize(palette=[overlay_color], min=0, max=1, opacity=overlay_opacity)


def _require_rasterio() -> None:
    if rasterio is None or warp is None or Resampling is None or from_bounds is None:
        raise ImportError("rasterio is required for raster validation and normalization.")


def _is_north_up(transform: Any, tol: float = NORTH_UP_TOL) -> bool:
    try:
        return abs(float(transform.b)) <= tol and abs(float(transform.d)) <= tol
    except Exception:
        return False


def _transform_sane(transform: Any) -> bool:
    try:
        values = np.array([transform.a, transform.b, transform.c, transform.d, transform.e, transform.f], dtype=float)
    except Exception:
        return False
    if not np.isfinite(values).all():
        return False
    if abs(values[0]) <= 0 or abs(values[4]) <= 0:
        return False
    return True


def _bounds_sane(bounds: Any) -> bool:
    try:
        values = np.array([bounds.left, bounds.bottom, bounds.right, bounds.top], dtype=float)
    except Exception:
        return False
    if not np.isfinite(values).all():
        return False
    return float(bounds.left) < float(bounds.right) and float(bounds.bottom) < float(bounds.top)


def _normalize_north_up(path: Path, resampling: Any = None) -> Tuple[Path, Dict[str, Any]]:
    _require_rasterio()
    with rasterio.open(path) as src:
        crs = src.crs
        if crs is None:
            raise ValueError("Raster CRS is missing.")
        transform = src.transform
        if not _transform_sane(transform):
            raise ValueError("Raster transform is invalid or has zero pixel size.")
        bounds = src.bounds
        if not _bounds_sane(bounds):
            raise ValueError("Raster bounds are invalid.")

        north_up = _is_north_up(transform)
        normalized = False
        out_path = path
        if not north_up:
            resampling = resampling or Resampling.bilinear
            dst_transform, width, height = warp.calculate_default_transform(
                crs,
                crs,
                src.width,
                src.height,
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
            )
            profile = src.profile.copy()
            profile.update(transform=dst_transform, width=width, height=height, crs=crs)
            out_path = path.with_name(f"{path.stem}_northup{path.suffix}")
            with rasterio.open(out_path, "w", **profile) as dst:
                for band in range(1, src.count + 1):
                    warp.reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        src_transform=transform,
                        src_crs=crs,
                        dst_transform=dst_transform,
                        dst_crs=crs,
                        resampling=resampling,
                    )
            normalized = True
            with rasterio.open(out_path) as check:
                transform = check.transform
                bounds = check.bounds
                north_up = _is_north_up(transform)

        metadata = {
            "path": str(out_path),
            "crs": str(crs),
            "epsg": crs.to_epsg(),
            "transform": {
                "a": float(transform.a),
                "b": float(transform.b),
                "c": float(transform.c),
                "d": float(transform.d),
                "e": float(transform.e),
                "f": float(transform.f),
            },
            "pixel_size": {"x": float(transform.a), "y": float(transform.e)},
            "bounds": {
                "left": float(bounds.left),
                "bottom": float(bounds.bottom),
                "right": float(bounds.right),
                "top": float(bounds.top),
            },
            "north_up": bool(north_up),
            "normalized": bool(normalized),
        }
        if not north_up:
            raise ValueError("Raster is not north-up and normalization failed.")
        return out_path, metadata


def _thumbnail_geotiff(
    png_path: Path,
    bbox: Sequence[float],
    out_dir: Path,
    filename_prefix: str,
    start_date: str,
    end_date: str,
) -> Path:
    _require_rasterio()
    north, west, south, east = map(float, bbox)
    with rasterio.open(png_path) as src:
        data = src.read()
        height = src.height
        width = src.width
        count = src.count
        dtype = src.dtypes[0]
    transform = from_bounds(west, south, east, north, width, height)
    safe_start = start_date.replace("-", "")
    safe_end = end_date.replace("-", "")
    out_path = out_dir / f"{filename_prefix}_THUMB_{safe_start}_{safe_end}.tif"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": "EPSG:4326",
        "transform": transform,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
    return out_path


def _bounds_overlap(bounds_a: Dict[str, float], bounds_b: Dict[str, float]) -> bool:
    lon_min = max(bounds_a["lon_min"], bounds_b["lon_min"])
    lon_max = min(bounds_a["lon_max"], bounds_b["lon_max"])
    lat_min = max(bounds_a["lat_min"], bounds_b["lat_min"])
    lat_max = min(bounds_a["lat_max"], bounds_b["lat_max"])
    return lon_min < lon_max and lat_min < lat_max


def _download_geotiff(
    image: Any,
    region: Any,
    scale: float,
    out_dir: Path,
    start_date: str,
    end_date: str,
    filename_prefix: str,
    crs: str = "EPSG:4326",
    file_per_band: bool = False,
) -> Dict[str, Optional[str]]:
    """Download an ee.Image as a GeoTIFF (returns a .tif path; zip is removed on success)."""
    params = {"region": region, "scale": scale, "crs": crs, "filePerBand": file_per_band, "format": "GEO_TIFF"}
    try:
        url = image.getDownloadURL(params)
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError(f"Failed to get GeoTIFF download URL: {exc}") from exc

    safe_start = start_date.replace("-", "")
    safe_end = end_date.replace("-", "")
    zip_name = f"{filename_prefix}_{safe_start}_{safe_end}.zip"
    zip_path = out_dir / zip_name
    out_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=300, verify=_resolve_verify_setting())
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    tif_path: Optional[Path] = None
    cleanup_zip = False
    try:
        with open(zip_path, "rb") as fp:
            header = fp.read(4)
        is_zip = header.startswith(b"PK\x03\x04") or header.startswith(b"PK\x05\x06") or header.startswith(b"PK\x07\x08")
        is_tiff = header.startswith(b"II*\x00") or header.startswith(b"MM\x00*")

        target = out_dir / f"{filename_prefix}_{safe_start}_{safe_end}.tif"
        target.parent.mkdir(parents=True, exist_ok=True)

        if is_zip:
            with zipfile.ZipFile(zip_path, "r") as zf:
                tif_members = [m for m in zf.namelist() if m.lower().endswith(".tif")]
                if not tif_members:
                    raise RuntimeError("Downloaded archive contained no GeoTIFF; request may have been too large.")
                member = tif_members[0]
                extracted = Path(zf.extract(member, path=out_dir))
                extracted_path = extracted if extracted.is_absolute() else out_dir / extracted
                shutil.move(str(extracted_path), target)
                tif_path = target
                cleanup_zip = True
        elif is_tiff:
            shutil.move(str(zip_path), target)
            tif_path = target
        else:
            with open(zip_path, "rb") as fp:
                snippet = fp.read(512)
            raise RuntimeError(f"Unexpected download payload (not zip/tiff): {snippet[:200]!r}")
    finally:
        if cleanup_zip:
            try:
                zip_path.unlink(missing_ok=True)
            except Exception:
                pass

    return {"zip_path": None, "tif_path": str(tif_path) if tif_path else None}


def _class_area_breakdown(
    image: Any,
    region: Any,
    area_sq_km: float,
    scale: float,
    thresholds: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    pixel_area = ee.Image.pixelArea()
    results: Dict[str, Dict[str, Any]] = {}
    for class_name, spec in thresholds.items():
        idx = spec.get("index", "").upper()
        band = image.select(idx)
        mask = None
        if "min" in spec and "max" in spec:
            mask = band.gte(float(spec["min"])).And(band.lte(float(spec["max"])))
        elif "min" in spec:
            mask = band.gte(float(spec["min"]))
        elif "max" in spec:
            mask = band.lte(float(spec["max"]))
        if mask is None:
            continue
        area_img = pixel_area.updateMask(mask)
        area_info = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
        )
        try:
            area_m2 = area_info.get("area").getInfo() if area_info else None
        except Exception:  # pragma: no cover
            area_m2 = None
        if area_m2 is None:
            continue
        area_km2 = float(area_m2) / 1_000_000.0
        results[class_name] = {
            "index": idx,
            "area_sq_km": area_km2,
            "fraction_of_aoi": area_km2 / area_sq_km if area_sq_km > 0 else None,
            "threshold": {k: v for k, v in spec.items() if k in {"min", "max"}},
        }
    return results


def gee_fetch_sentinel2_indices(
    *,
    bbox: Sequence[float],
    start_date: str,
    end_date: str,
    max_cloud: float = 40.0,
    indices: Optional[Sequence[str]] = None,
    scale: float = 20.0,
    dimensions: int = 1024,
    output_dir: str = "./outputs/gee_indices",
    project_id: Optional[str] = None,
    include_truecolor: bool = True,
    include_geotiff: bool = False,
    geotiff_scale: Optional[float] = None,
) -> ToolResponse:
    """
    Build a Sentinel-2 median composite for the AOI/time range, add spectral indices,
    compute summary stats, and download PNG thumbnails. Optionally download the composite
    (with all computed bands) as a GeoTIFF.

    Args:
        bbox: [north, west, south, east] degrees.
        start_date, end_date: YYYY-MM-DD range.
        max_cloud: Max CLOUDY_PIXEL_PERCENTAGE (0–100).
        indices: List of indices to compute (default NDVI/NBR/NDBI/NDWI).
        scale: Resolution in meters for stats and area calculations.
        dimensions: Max width/height in pixels for thumbnails.
        output_dir: Where to save PNGs/GeoTIFFs.
        project_id: Optional Earth Engine project ID (defaults to env vars).
        include_truecolor: If True, also download a truecolor composite PNG.
        include_geotiff: If True, also download a GeoTIFF of the composite (includes indices).
        geotiff_scale: Optional override for GeoTIFF export resolution in meters (defaults to `scale`).
    """
    idx = indices or DEFAULT_INDICES
    prepared, err = _prepare_index_image(bbox, start_date, end_date, max_cloud, idx, project_id)
    if err or prepared is None:
        return err  # type: ignore[return-value]

    image = prepared["image"]
    aoi = prepared["aoi"]
    out_dir = Path(output_dir)

    try:
        stats = _index_stats(image, aoi, [i.upper() for i in idx], scale)
        thumbs: Dict[str, str] = {}
        thumb_geotiffs: Dict[str, str] = {}
        thumb_guardrails: Dict[str, Any] = {}
        for name in idx:
            key = name.upper()
            png_path = _download_index_png(
                image, name, aoi, dimensions, out_dir, start_date, end_date
            )
            thumbs[key] = png_path
            tif_path = _thumbnail_geotiff(
                Path(png_path),
                bbox,
                out_dir,
                key,
                start_date,
                end_date,
            )
            normalized_path, guard = _normalize_north_up(tif_path)
            if normalized_path != tif_path:
                try:
                    Path(tif_path).unlink(missing_ok=True)
                except Exception:
                    pass
            thumb_geotiffs[key] = str(normalized_path)
            thumb_guardrails[key] = guard
        truecolor_path = None
        truecolor_thumb_geotiff = None
        truecolor_guardrail = None
        if include_truecolor:
            truecolor_path = _download_truecolor(image, aoi, dimensions, out_dir, start_date, end_date)
            thumb_tif = _thumbnail_geotiff(
                Path(truecolor_path),
                bbox,
                out_dir,
                "TRUECOLOR",
                start_date,
                end_date,
            )
            normalized_path, guard = _normalize_north_up(thumb_tif)
            if normalized_path != thumb_tif:
                try:
                    Path(thumb_tif).unlink(missing_ok=True)
                except Exception:
                    pass
            truecolor_thumb_geotiff = str(normalized_path)
            truecolor_guardrail = guard
        geotiff_info: Optional[Dict[str, Optional[str]]] = None
        geotiff_guardrail = None
        if include_geotiff:
            geotiff_info = _download_geotiff(
                image,
                aoi,
                geotiff_scale if geotiff_scale is not None else scale,
                out_dir,
                start_date,
                end_date,
                "SENTINEL2_COMPOSITE",
            )
            tif_path = geotiff_info.get("tif_path") if geotiff_info else None
            if tif_path:
                normalized_path, guard = _normalize_north_up(Path(tif_path))
                if normalized_path != Path(tif_path):
                    try:
                        Path(tif_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                geotiff_info["tif_path"] = str(normalized_path)
                geotiff_guardrail = guard
    except Exception as exc:  # pragma: no cover
        return _error(f"Failed to compute indices or download thumbnails: {exc}")

    payload = {
        "project_id": prepared["project_id"],
        "bbox": [float(b) for b in bbox],
        "start_date": start_date,
        "end_date": end_date,
        "cloud_filter_percent": max_cloud,
        "collection_size": prepared["collection_size"],
        "aoi_area_sq_km": prepared["area_sq_km"],
        "indices": stats,
        "thumbnails": thumbs,
        "thumbnail_geotiffs": thumb_geotiffs,
        "thumbnail_guardrails": thumb_guardrails,
    }
    if include_truecolor:
        payload["truecolor_png"] = truecolor_path
        payload["truecolor_thumbnail_geotiff"] = truecolor_thumb_geotiff
        payload["truecolor_guardrail"] = truecolor_guardrail
    if include_geotiff:
        payload["geotiff"] = geotiff_info
        payload["geotiff_guardrail"] = geotiff_guardrail
    return _ok(payload)


def gee_index_area_stats(
    *,
    bbox: Sequence[float],
    start_date: str,
    end_date: str,
    max_cloud: float = 40.0,
    scale: float = 20.0,
    class_thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
    project_id: Optional[str] = None,
) -> ToolResponse:
    """
    Compute simple land-surface area fractions (vegetation, water, built-up, burn scars)
    using Sentinel-2 indices over the AOI/time window.

    Args:
        bbox: [north, west, south, east] degrees.
        start_date, end_date: YYYY-MM-DD range.
        max_cloud: Max CLOUDY_PIXEL_PERCENTAGE (0–100).
        scale: Resolution in meters for the area reducer.
        class_thresholds: Optional override for class thresholds; values follow
            {"index": "NDVI", "min": 0.25} or {"index": "NBR", "max": 0.1}.
        project_id: Optional Earth Engine project ID (defaults to env vars).
    """
    idx = DEFAULT_INDICES
    prepared, err = _prepare_index_image(bbox, start_date, end_date, max_cloud, idx, project_id)
    if err or prepared is None:
        return err  # type: ignore[return-value]

    image = prepared["image"]
    aoi = prepared["aoi"]
    thresholds = class_thresholds or CLASS_THRESHOLDS
    try:
        stats = _index_stats(image, aoi, [i.upper() for i in idx], scale)
        areas = _class_area_breakdown(image, aoi, prepared["area_sq_km"], scale, thresholds)
    except Exception as exc:  # pragma: no cover
        return _error(f"Failed to compute area breakdown: {exc}")

    payload = {
        "project_id": prepared["project_id"],
        "bbox": [float(b) for b in bbox],
        "start_date": start_date,
        "end_date": end_date,
        "cloud_filter_percent": max_cloud,
        "collection_size": prepared["collection_size"],
        "aoi_area_sq_km": prepared["area_sq_km"],
        "index_stats": stats,
        "class_areas": areas,
        "class_thresholds": thresholds,
    }
    return _ok(payload)


def gee_fetch_truecolor_thumbnail(
    *,
    bbox: Sequence[float],
    start_date: str,
    end_date: str,
    max_cloud: float = 40.0,
    dimensions: int = 1024,
    output_dir: str = "./outputs/gee_indices",
    overlay_geometries: Optional[Sequence[Any]] = None,
    draw_aoi_outline: bool = True,
    overlay_color: str = "#ff7f0e",
    overlay_width: int = 2,
    overlay_opacity: float = 0.8,
    overlay_fill: bool = False,
    project_id: Optional[str] = None,
    include_geotiff: bool = False,
    geotiff_scale: Optional[float] = None,
) -> ToolResponse:
    """
    Download a truecolor (RGB) Sentinel-2 median composite thumbnail for the AOI/time range.
    Optionally overlay the AOI outline and/or additional GeoJSON geometries (e.g., OSM masks).
    Optionally export the composite as a GeoTIFF.
    """
    prepared, err = _prepare_index_image(bbox, start_date, end_date, max_cloud, [], project_id)
    if err or prepared is None:
        return err  # type: ignore[return-value]

    aoi = prepared["aoi"]
    overlay_img = _build_overlay_image(
        aoi=aoi,
        overlay_geometries=overlay_geometries,
        draw_aoi_outline=draw_aoi_outline,
        overlay_color=overlay_color,
        overlay_width=overlay_width,
        overlay_opacity=overlay_opacity,
        overlay_fill=overlay_fill,
    )
    out_dir = Path(output_dir)
    truecolor_image = prepared["image"].select(["red", "green", "blue"])
    try:
        png_path = _download_truecolor(
            prepared["image"],
            aoi,
            dimensions,
            out_dir,
            start_date,
            end_date,
            overlay_image=overlay_img,
        )
        thumb_tif = _thumbnail_geotiff(
            Path(png_path),
            bbox,
            out_dir,
            "TRUECOLOR",
            start_date,
            end_date,
        )
        normalized_path, guard = _normalize_north_up(thumb_tif)
        if normalized_path != thumb_tif:
            try:
                Path(thumb_tif).unlink(missing_ok=True)
            except Exception:
                pass
        thumb_geotiff = str(normalized_path)
        thumb_guardrail = guard
        geotiff_info: Optional[Dict[str, Optional[str]]] = None
        geotiff_guardrail = None
        if include_geotiff:
            geotiff_info = _download_geotiff(
                truecolor_image,
                aoi,
                geotiff_scale if geotiff_scale is not None else 10.0,
                out_dir,
                start_date,
                end_date,
                "TRUECOLOR",
                file_per_band=False,
            )
            tif_path = geotiff_info.get("tif_path") if geotiff_info else None
            if tif_path:
                normalized_path, guard = _normalize_north_up(Path(tif_path))
                if normalized_path != Path(tif_path):
                    try:
                        Path(tif_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                geotiff_info["tif_path"] = str(normalized_path)
                geotiff_guardrail = guard
    except Exception as exc:  # pragma: no cover
        return _error(f"Failed to download truecolor thumbnail: {exc}")

    payload = {
        "project_id": prepared["project_id"],
        "bbox": [float(b) for b in bbox],
        "start_date": start_date,
        "end_date": end_date,
        "cloud_filter_percent": max_cloud,
        "collection_size": prepared["collection_size"],
        "aoi_area_sq_km": prepared["area_sq_km"],
        "truecolor_png": png_path,
        "truecolor_thumbnail_geotiff": thumb_geotiff,
        "truecolor_guardrail": thumb_guardrail,
        "overlay_applied": overlay_img is not None,
        "overlay_geometries_count": len(overlay_geometries) if overlay_geometries else 0,
        "draw_aoi_outline": draw_aoi_outline,
        "overlay_color": overlay_color,
        "overlay_width": overlay_width,
        "overlay_opacity": overlay_opacity,
        "overlay_fill": overlay_fill,
    }
    if include_geotiff:
        payload["geotiff"] = geotiff_info
        payload["geotiff_guardrail"] = geotiff_guardrail
    return _ok(payload)


# ----- Composite overlay using GEE GeoTIFF ----------------------------------
def _extract_geotiff_path(source: Any) -> Path:
    """
    Resolve a GeoTIFF path from common GEE tool outputs.

    Accepts:
    - str/Path: path to a GeoTIFF file.
    - dict: expects a "geotiff" or "tif_path"/"path" entry from gee_fetch_* metadata.
    - ToolResponse: metadata is inspected for the same keys.
    """
    if isinstance(source, ToolResponse):
        source = source.metadata or {}

    if isinstance(source, (str, Path)):
        candidate = Path(source).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"GeoTIFF not found at {candidate}")

    if isinstance(source, dict):
        pools: list[dict[str, Any]] = []
        pools.append(source)
        geotiff_section = source.get("geotiff")
        if isinstance(geotiff_section, dict):
            pools.append(geotiff_section)
        candidates: list[Path] = []
        for item in pools:
            for key in ("tif_path", "path", "geo_tiff"):
                val = item.get(key)
                if val:
                    candidates.append(Path(str(val)).expanduser())
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if candidates:
            raise FileNotFoundError(f"GeoTIFF path missing on disk: {candidates[0]}")
        raise ValueError("No GeoTIFF path found in the provided metadata. Call the GEE tools with include_geotiff=True.")

    raise ValueError("Satellite image must be a GeoTIFF path or a GEE tool output containing one.")


def _load_satellite(path: Path) -> tuple[np.ndarray, Any, Any, Any, int, int]:
    if rasterio is None:
        raise ImportError("rasterio is required for composite overlays; install rasterio and retry.")

    sat_path = Path(path).expanduser()
    if not sat_path.exists():
        raise FileNotFoundError(f"Satellite image not found: {sat_path}")

    with rasterio.open(sat_path) as src:
        data = src.read(out_dtype=np.float32)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        height, width = src.height, src.width

    if crs is None:
        raise ValueError("Satellite image is missing a CRS; cannot reproject environmental data onto it.")

    return data, transform, crs, bounds, height, width


def _prepare_rgb(data: np.ndarray, stretch: Sequence[float] = (2, 98)) -> np.ndarray:
    """Convert rasterio band-major array -> HWC RGB with percentile stretch."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if data.shape[0] >= 3:
        rgb = data[:3]
    else:
        rgb = np.repeat(data[0:1], 3, axis=0)

    finite = np.isfinite(rgb)
    if not finite.any():
        return np.zeros((data.shape[1], data.shape[2], 3), dtype=np.float32)

    p_low, p_high = np.nanpercentile(rgb[finite], stretch)
    scale = p_high - p_low if p_high > p_low else 1.0
    norm = np.clip((rgb - p_low) / scale, 0, 1)
    return np.moveaxis(norm, 0, -1)


def _satellite_bounds_wgs84(sat_bounds, sat_crs) -> Optional[Dict[str, float]]:
    if warp is None or sat_crs is None:
        return None
    try:
        left, bottom, right, top = warp.transform_bounds(
            sat_crs, "EPSG:4326", sat_bounds.left, sat_bounds.bottom, sat_bounds.right, sat_bounds.top, densify_pts=21
        )
        return {"lon_min": float(left), "lon_max": float(right), "lat_min": float(bottom), "lat_max": float(top)}
    except Exception:
        return None


def _anomaly_bounds(field) -> Dict[str, float]:
    return {
        "lon_min": float(np.nanmin(field.lon)),
        "lon_max": float(np.nanmax(field.lon)),
        "lat_min": float(np.nanmin(field.lat)),
        "lat_max": float(np.nanmax(field.lat)),
    }


def _source_grid(field) -> tuple[np.ndarray, Any]:
    lat_name = getattr(field, "lat_name", None) or "lat"
    lon_name = getattr(field, "lon_name", None) or "lon"
    lat = np.asarray(getattr(field, "lat"))
    lon = np.asarray(getattr(field, "lon"))
    transform = from_bounds(float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()), len(lon), len(lat))
    return np.asarray(field.data), transform


def _reproject_to_satellite(field, target_transform, target_crs, height: int, width: int) -> np.ndarray:
    if warp is None or Resampling is None:
        raise ImportError("rasterio is required for reprojection.")
    data, src_transform = _source_grid(field)
    dst = np.zeros((height, width), dtype=np.float32)
    warp.reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
    )
    return dst


def _auto_limits(array: np.ndarray, center_zero: bool) -> tuple[Optional[float], Optional[float]]:
    if not np.isfinite(array).any():
        return None, None
    if center_zero:
        max_abs = float(np.nanmax(np.abs(array)))
        return -max_abs, max_abs
    return None, None


def composite_satellite_environmental_image(
    satellite_image: Any,
    anomaly_path: str,
    *,
    anomaly_variable: Optional[str] = None,
    anomaly_time_index: Optional[int] = None,
    anomaly_aggregate: Optional[str] = "auto",
    mode: str = "overlay",
    right_field_path: Optional[str] = None,
    right_field_variable: Optional[str] = None,
    right_field_time_index: Optional[int] = None,
    right_field_aggregate: Optional[str] = "auto",
    right_label: str = "event mask / ensemble",
    output_path: str = "./plots/composite.png",
    overlay_alpha: float = 0.55,
    overlay_cmap: str = "RdBu_r",
    overlay_center_zero: bool = True,
    overlay_vmin: Optional[float] = None,
    overlay_vmax: Optional[float] = None,
    right_cmap: str = "viridis",
    right_vmin: Optional[float] = None,
    right_vmax: Optional[float] = None,
) -> ToolResponse:
    """
    Blend a GEE-exported satellite GeoTIFF with an environmental NetCDF field.

    Pass either:
    - the GeoTIFF path returned by `gee_fetch_sentinel2_indices(..., include_geotiff=True)`
      or `gee_fetch_truecolor_thumbnail(..., include_geotiff=True)`, or
    - the metadata dictionary/ToolResponse containing that GeoTIFF path.
    """
    try:
        if plt is None:
            return _error("matplotlib is required for composite satellite/environmental images.")
        from .latlon_field import _prepare_latlon_field

        sat_path = _extract_geotiff_path(satellite_image)
        sat_data, sat_transform, sat_crs, sat_bounds, sat_height, sat_width = _load_satellite(sat_path)
        base_rgb = _prepare_rgb(sat_data)
        extent = (sat_bounds.left, sat_bounds.right, sat_bounds.bottom, sat_bounds.top)

        anomaly_field = _prepare_latlon_field(
            netcdf_path=anomaly_path,
            variable=anomaly_variable,
            time_index=anomaly_time_index,
            aggregate=anomaly_aggregate,
            auto_time_mean_threshold=12,
        )
        sat_ll_bounds = _satellite_bounds_wgs84(sat_bounds, sat_crs)
        if sat_ll_bounds is not None:
            env_bounds = _anomaly_bounds(anomaly_field)
            if not _bounds_overlap(sat_ll_bounds, env_bounds):
                return _error(
                    "Satellite and anomaly bounds do not overlap. "
                    f"Satellite (WGS84) bounds: {sat_ll_bounds}, "
                    f"anomaly lat/lon bounds: {env_bounds}."
                )
        anomaly_regridded = _reproject_to_satellite(
            anomaly_field, sat_transform, sat_crs, height=sat_height, width=sat_width
        )
        if not np.isfinite(anomaly_regridded).any():
            env_bounds = _anomaly_bounds(anomaly_field)
            return _error(
                "Reprojected anomaly field is empty over the satellite footprint. "
                f"Satellite (WGS84) bounds: {sat_ll_bounds or 'unavailable'}, "
                f"anomaly lat/lon bounds: {env_bounds}. "
                "Ensure the ERA5/NetCDF region fully covers the satellite scene."
            )

        mode_normalized = mode.lower()
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if mode_normalized == "overlay":
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(base_rgb, extent=extent, origin="upper")

            vmin = overlay_vmin
            vmax = overlay_vmax
            if vmin is None or vmax is None:
                auto_vmin, auto_vmax = _auto_limits(anomaly_regridded, overlay_center_zero)
                vmin = vmin if vmin is not None else auto_vmin
                vmax = vmax if vmax is not None else auto_vmax

            mesh = ax.imshow(
                anomaly_regridded,
                extent=extent,
                origin="upper",
                cmap=overlay_cmap,
                alpha=overlay_alpha,
                vmin=vmin,
                vmax=vmax,
            )
            cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"{anomaly_field.var_name} ({anomaly_field.units})")

            ax.set_title(f"{anomaly_field.var_name} over satellite scene ({anomaly_field.time_label})")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close(fig)

            return _ok({"output_path": str(out_path), "mode": "overlay", "satellite_path": str(sat_path)})

        elif mode_normalized == "triptych":
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(base_rgb, extent=extent, origin="upper")
            axes[0].set_title("Satellite")
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            vmin = overlay_vmin
            vmax = overlay_vmax
            if vmin is None or vmax is None:
                auto_vmin, auto_vmax = _auto_limits(anomaly_regridded, overlay_center_zero)
                vmin = vmin if vmin is not None else auto_vmin
                vmax = vmax if vmax is not None else auto_vmax

            mesh = axes[1].imshow(
                anomaly_regridded,
                extent=extent,
                origin="upper",
                cmap=overlay_cmap,
                vmin=vmin,
                vmax=vmax,
            )
            cbar = plt.colorbar(mesh, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.set_label(f"{anomaly_field.var_name} ({anomaly_field.units})")
            axes[1].set_title(f"Anomaly ({anomaly_field.time_label})")
            axes[1].set_xticks([])
            axes[1].set_yticks([])

            if right_field_path:
                right_field = _prepare_latlon_field(
                    netcdf_path=right_field_path,
                    variable=right_field_variable,
                    time_index=right_field_time_index,
                    aggregate=right_field_aggregate,
                    auto_time_mean_threshold=12,
                )
                if sat_ll_bounds is not None:
                    right_bounds = _anomaly_bounds(right_field)
                    if not _bounds_overlap(sat_ll_bounds, right_bounds):
                        return _error(
                            "Satellite and right-field bounds do not overlap. "
                            f"Satellite (WGS84) bounds: {sat_ll_bounds}, "
                            f"right-field lat/lon bounds: {right_bounds}."
                        )
                right_regridded = _reproject_to_satellite(
                    right_field, sat_transform, sat_crs, height=sat_height, width=sat_width
                )
                mesh = axes[2].imshow(
                    right_regridded,
                    extent=extent,
                    origin="upper",
                    cmap=right_cmap,
                    vmin=right_vmin,
                    vmax=right_vmax,
                )
                cbar = plt.colorbar(mesh, ax=axes[2], fraction=0.046, pad=0.04)
                cbar.set_label(f"{right_field.var_name} ({right_field.units})")
            else:
                axes[2].imshow(base_rgb, extent=extent, origin="upper")
            axes[2].set_title(right_label)
            axes[2].set_xticks([])
            axes[2].set_yticks([])

            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close(fig)
            return _ok({"output_path": str(out_path), "mode": "triptych", "satellite_path": str(sat_path)})
        else:
            return _error("mode must be 'overlay' or 'triptych'")
    except Exception as exc:
        return _error(f"Composite failed: {exc}")
