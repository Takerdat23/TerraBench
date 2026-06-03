"""
Generic OSM/Nominatim utilities exposed as agent tools.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import certifi
import requests
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
from dotenv import load_dotenv
from pyproj import Geod
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
    shape,
)

__all__ = [
    "gis_resolve_region",
    "gis_feature_query",
    "gis_feature_stats",
    "gis_distance_matrix",
    "gis_overlay",
    "gis_source_metadata",
    "overpass_poi_search",
]

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "climate-agent-osm-tools/0.1"
GLOBAL_GEOD = Geod(ellps="WGS84")

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
_COMBINED_BUNDLE_PATH = Path("/tmp/osm_combined.pem")
_CA_ENV_KEYS: tuple[str, ...] = ("OSM_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")
_VERIFY_FLAG_ENV = "OSM_VERIFY"


class OSMError(Exception):
    """Raised when OSM/Nominatim inputs or responses are invalid."""


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


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


def _ok(payload: Dict[str, Any]) -> ToolResponse:
    cleaned = _sanitize(payload)
    return ToolResponse(
        content=[TextBlock(type="text", text=json.dumps(cleaned))],
        metadata=cleaned,
    )


def _error(msg: str) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {msg}")],
        metadata={"error": True, "message": msg},
    )


def _normalize_bbox(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise OSMError("bbox must be [south, west, north, east].")
    south, west, north, east = map(float, bbox)
    if south >= north or west >= east:
        raise OSMError(
            f"bbox order is invalid (south < north, west < east expected): {bbox}"
        )
    return south, west, north, east


def _bbox_polygon(bbox: Sequence[float]) -> Polygon:
    south, west, north, east = _normalize_bbox(bbox)
    return Polygon([(west, south), (east, south), (east, north), (west, north)])


def _http_get(url: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    resp = requests.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=45,
        verify=_resolve_verify_setting(),
    )
    if not resp.ok:
        raise OSMError(f"HTTP {resp.status_code} from {url}: {resp.text[:500]}")
    return resp.json()


def _http_post(url: str, data: str) -> Dict[str, Any]:
    resp = requests.post(
        url,
        data=data.encode("utf-8"),
        headers={"User-Agent": USER_AGENT},
        timeout=60,
        verify=_resolve_verify_setting(),
    )
    if not resp.ok:
        raise OSMError(f"HTTP {resp.status_code} from {url}: {resp.text[:500]}")
    return resp.json()


def _polygon_to_overpass(geom: Polygon) -> str:
    coords = geom.exterior.coords
    return " ".join(f"{lat} {lon}" for lon, lat in coords)


def _geometry_clause(
    geometry: Mapping[str, Any],
) -> Tuple[str, Optional[Polygon]]:
    if not geometry:
        raise OSMError("geometry must be provided.")
    if "bbox" in geometry:
        bbox_poly = _bbox_polygon(geometry["bbox"])
        s, w, n, e = _normalize_bbox(geometry["bbox"])
        return f"({s},{w},{n},{e})", bbox_poly
    if "polygon" in geometry:
        geom = shape(geometry["polygon"])
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)
        if not isinstance(geom, Polygon):
            geom = geom.convex_hull
        return f'(poly:"{_polygon_to_overpass(geom)}")', geom
    if "center" in geometry and "radius_km" in geometry:
        center = geometry["center"]
        lat = float(center["lat"])
        lon = float(center["lon"])
        radius_m = float(geometry["radius_km"]) * 1000.0
        approx_deg = float(geometry["radius_km"]) / 111.0
        return (
            f"(around:{radius_m:.0f},{lat},{lon})",
            Point(lon, lat).buffer(approx_deg),
        )
    if "type" in geometry and ("coordinates" in geometry or "geometries" in geometry):
        raise OSMError(
            "raw GeoJSON is not valid for 'geometry'; pass it as "
            "geometry={'polygon': <GeoJSON>} or use bbox / center+radius_km."
        )
    raise OSMError("geometry must include bbox, polygon, or center+radius_km.")


def _geometry_types_to_osm(types: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not types:
        return ("node", "way", "relation")
    osm_types = set()
    for typ in types:
        t = typ.lower()
        if t == "point":
            osm_types.add("node")
        elif t == "line":
            osm_types.update({"way", "relation"})
        elif t == "polygon":
            osm_types.update({"way", "relation"})
        else:
            raise OSMError(f"Unsupported geometry type '{typ}'.")
    return tuple(sorted(osm_types))


def _build_filter_clauses(filters: Optional[Sequence[Mapping[str, Any]]]) -> List[str]:
    if not filters:
        return [""]
    clauses: List[str] = []
    for item in filters:
        if not isinstance(item, Mapping):
            raise OSMError(
                "Each filter must be an object with 'key' and optional 'values'."
            )
        key = item.get("key")
        if not key:
            if len(item) == 1:
                bad_key, _bad_value = next(iter(item.items()))
                raise OSMError(
                    f"Malformed filter {item!r}; use "
                    f"{{'key': '{bad_key}'}} for tag existence or "
                    f"{{'key': '{bad_key}', 'values': [...]}} for specific values."
                )
            raise OSMError(
                "Each filter must include 'key' and optional 'values'."
            )
        raw_values = item.get("values")
        if raw_values is None:
            values = []
        else:
            if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
                raise OSMError(
                    f"Filter values for key '{key}' must be a list, not {type(raw_values).__name__}."
                )
            values = list(raw_values)
        if values:
            regex = "|".join(sorted({str(v) for v in values}))
            clauses.append(f'["{key}"~"^({regex})$"]')
        else:
            clauses.append(f'["{key}"]')
    return clauses or [""]


def _geometry_from_element(el: Mapping[str, Any]) -> Optional[Any]:
    if el.get("type") == "node" and "lat" in el and "lon" in el:
        return Point(el["lon"], el["lat"])
    coords = None
    if "geometry" in el and el["geometry"]:
        coords = [(g["lon"], g["lat"]) for g in el["geometry"]]
    if coords:
        if len(coords) >= 4 and coords[0] == coords[-1]:
            try:
                return Polygon(coords)
            except Exception:
                return LineString(coords)
        return LineString(coords)
    if "center" in el:
        c = el["center"]
        return Point(c["lon"], c["lat"])
    return None


def _geometry_area_km2(geom: Any) -> Optional[float]:
    try:
        area, _ = GLOBAL_GEOD.geometry_area_perimeter(geom)
        return abs(area) / 1e6
    except Exception:
        return None


def _geometry_length_km(geom: Any) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    if geom.geom_type == "Point":
        return 0.0
    if isinstance(geom, (MultiLineString, MultiPolygon)):
        return sum(_geometry_length_km(g) for g in geom.geoms)
    if isinstance(geom, Polygon):
        exterior = LineString(geom.exterior.coords)
        interior = sum(_geometry_length_km(LineString(r.coords)) for r in geom.interiors)
        return _geometry_length_km(exterior) + interior
    if isinstance(geom, (LineString,)):
        lons, lats = geom.xy
        return GLOBAL_GEOD.line_length(lons, lats) / 1000.0
    return 0.0


def _feature_from_element(el: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Any]]:
    geom = _geometry_from_element(el)
    feature = {
        "id": f"{el['type']}/{el['id']}",
        "osm_type": el["type"],
        "geometry": mapping(geom) if geom else None,
        "geometry_type": geom.geom_type if geom else None,
        "properties": {
            "tags": el.get("tags", {}),
        },
    }
    if "center" in el:
        feature["properties"]["center"] = {"lat": el["center"]["lat"], "lon": el["center"]["lon"]}
    elif geom:
        centroid = geom.centroid
        feature["properties"]["center"] = {"lat": centroid.y, "lon": centroid.x}
    return feature, geom


def _collect_bounds(geoms: Iterable[Any]) -> Optional[List[float]]:
    bounds = None
    for g in geoms:
        if g is None or g.is_empty:
            continue
        minx, miny, maxx, maxy = g.bounds
        if bounds is None:
            bounds = [miny, minx, maxy, maxx]
        else:
            bounds = [
                min(bounds[0], miny),
                min(bounds[1], minx),
                max(bounds[2], maxy),
                max(bounds[3], maxx),
            ]
    return bounds


def _group_key(props: Mapping[str, Any], tags: Mapping[str, Any], group_by: Sequence[str]) -> Tuple[Tuple[str, Any], ...]:
    if not group_by:
        return (("group", "all"),)
    key_parts = []
    for field in group_by:
        if field.startswith("tag:"):
            tag_name = field.split(":", 1)[1]
            key_parts.append((field, tags.get(tag_name)))
        else:
            key_parts.append((field, props.get(field)))
    return tuple(key_parts)


def _coerce_features(items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    coerced: List[Dict[str, Any]] = []
    for item in items:
        if "features" in item and isinstance(item["features"], list):
            coerced.extend(_coerce_features(item["features"]))
            continue
        if "geometry" in item:
            coerced.append(item)
        elif "type" in item and "coordinates" in item:
            coerced.append({"geometry": item, "properties": {}})
        else:
            raise OSMError("targets/zones items must contain GeoJSON geometry or features.")
    return coerced


def _intersection_metrics(geom: Any, summary_metrics: Sequence[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if "count" in summary_metrics:
        metrics["count"] = 1
    if "length_km" in summary_metrics:
        metrics["length_km"] = _geometry_length_km(geom)
    if "area_km2" in summary_metrics or "sum:property=area" in summary_metrics:
        metrics["area_km2"] = _geometry_area_km2(geom) or 0.0
    return metrics


def _feature_density(area: Optional[float], total: int) -> Optional[float]:
    if area and area > 0:
        return total / area
    return None


def gis_resolve_region(
    *,
    place_query: Optional[str] = None,
    admin_code: Optional[str] = None,
    return_geometry_type: str = "bbox",
    nominatim_url: str = NOMINATIM_URL,
) -> ToolResponse:
    """Resolve a place name/admin code into bbox/polygon/centroid using Nominatim."""
    try:
        query = (place_query or admin_code or "").strip()
        if not query:
            return _error("place_query or admin_code must be provided.")

        params = {
            "q": query,
            "format": "json",
            "limit": 1,
            "polygon_geojson": 1,
        }
        results = _http_get(nominatim_url, params=params)
        if not results:
            return _error(f"No matches found for '{query}'.")

        hit = results[0]
        bbox_raw = hit.get("boundingbox")
        bbox = None
        if bbox_raw:
            south, north, west, east = map(float, bbox_raw)
            bbox = [south, west, north, east]

        polygon = hit.get("geojson")
        centroid = None
        if polygon:
            geom = shape(polygon)
            centroid = {"lat": geom.centroid.y, "lon": geom.centroid.x}
        elif "lat" in hit and "lon" in hit:
            centroid = {"lat": float(hit["lat"]), "lon": float(hit["lon"])}

        payload = {"query": query}
        if return_geometry_type in ("bbox", "all"):
            payload["bbox"] = bbox
        if return_geometry_type in ("polygon", "all"):
            payload["polygon"] = polygon
        if return_geometry_type in ("centroid", "all"):
            payload["centroid"] = centroid
        return _ok(payload)
    except Exception as exc:  # pragma: no cover
        return _error(f"gis_resolve_region failed: {exc}")


def gis_feature_query(
    *,
    source: str,
    geometry: Mapping[str, Any],
    filters: Optional[Sequence[Mapping[str, Any]]] = None,
    geometry_types: Optional[Sequence[str]] = None,
    max_features: int = 2000,
    overpass_url: str = OVERPASS_URL,
    as_of_date: Optional[str] = None,
) -> ToolResponse:
    """Query vector features from OSM Overpass with generic filters."""
    try:
        if source.lower() != "osm":
            return _error("Only 'osm' source is supported currently.")

        geom_clause, geom_shape = _geometry_clause(geometry)
        osm_types = _geometry_types_to_osm(geometry_types)
        selectors = _build_filter_clauses(filters)

        settings = "[out:json][timeout:55];"
        if as_of_date:
            settings = f'[out:json][timeout:55][date:"{as_of_date}"];'

        body = "".join(f"{osm}{selector}{geom_clause};" for selector in selectors for osm in osm_types)
        query = f"{settings}({body});out geom {max_features};"

        data = _http_post(overpass_url, query)
        features: Dict[str, Dict[str, Any]] = {}
        geoms: List[Any] = []

        for el in data.get("elements", []):
            feature, geom = _feature_from_element(el)
            if feature is None:
                continue
            features[feature["id"]] = feature
            if geom:
                geoms.append(geom)

        feature_list = list(features.values())
        bounds = _collect_bounds(geoms)
        tag_counts: Dict[str, Dict[str, int]] = {}
        for f in feature_list:
            tags = f.get("properties", {}).get("tags", {})
            for k, v in tags.items():
                tag_counts.setdefault(k, {})
                tag_counts[k][v] = tag_counts[k].get(v, 0) + 1

        summary: Dict[str, Any] = {
            "total_features": len(feature_list),
            "result_bbox": bounds,
            "tag_value_counts": tag_counts,
        }
        if geom_shape is not None:
            area = _geometry_area_km2(geom_shape)
            summary["area_sq_km"] = area
            summary["feature_density_per_sq_km"] = _feature_density(area, len(feature_list))

        data_status = {
            "coverage_ok": len(feature_list) < max_features,
            "notes": "" if len(feature_list) < max_features else f"Reached max_features ({max_features}); results may be truncated.",
        }
        return _ok({"features": feature_list, "summary": summary, "data_status": data_status})
    except Exception as exc:  # pragma: no cover
        return _error(f"gis_feature_query failed: {exc}")


def gis_feature_stats(
    *,
    features: Sequence[Mapping[str, Any]],
    group_by: Optional[Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
    normalizers: Optional[Mapping[str, Any]] = None,
) -> ToolResponse:
    """Compute generic stats (count/length/area) over a feature collection."""
    try:
        metrics = metrics or ["count"]
        group_by = group_by or []
        normalizers = normalizers or {}
        if isinstance(features, Mapping) and "features" in features:
            features = features["features"]  # type: ignore[assignment]

        aggregates: Dict[Tuple[Tuple[str, Any], ...], Dict[str, float]] = {}
        for feature in features:
            geom = shape(feature["geometry"]) if feature.get("geometry") else None
            area_km2 = _geometry_area_km2(geom) if geom else 0.0
            length_km = _geometry_length_km(geom) if geom else 0.0
            props = feature.get("properties", {})
            tags = props.get("tags", {})
            key = _group_key(props, tags, group_by)
            agg = aggregates.setdefault(key, {"count": 0, "length_km": 0.0, "area_km2": 0.0})
            agg["count"] += 1
            agg["length_km"] += length_km or 0.0
            agg["area_km2"] += area_km2 or 0.0

        rows: List[Dict[str, Any]] = []
        for key, agg in aggregates.items():
            group_entry = {k: v for k, v in key}
            row: Dict[str, Any] = {"group": group_entry}
            if "count" in metrics:
                row["count"] = agg["count"]
            if "length_km" in metrics:
                row["length_km"] = agg["length_km"]
            if "area_km2" in metrics:
                row["area_km2"] = agg["area_km2"]

            area_norm = None
            if "area_km2" in normalizers:
                norm_val = normalizers["area_km2"]
                if isinstance(norm_val, (int, float)) and norm_val > 0:
                    area_norm = float(norm_val)
                elif norm_val and agg["area_km2"] > 0:
                    area_norm = agg["area_km2"]
            if area_norm:
                if "count" in metrics:
                    row["count_per_km2"] = agg["count"] / area_norm
                if "length_km" in metrics:
                    row["length_per_km2"] = agg["length_km"] / area_norm
            rows.append(row)
        return _ok(rows)
    except Exception as exc:  # pragma: no cover
        return _error(f"gis_feature_stats failed: {exc}")


def gis_distance_matrix(
    *,
    origins: Sequence[Mapping[str, Any]],
    destinations: Sequence[Mapping[str, Any]],
    metric: str = "geodesic_km",
) -> ToolResponse:
    """Compute pairwise geodesic distances between origins and destinations."""
    try:
        if metric != "geodesic_km":
            return _error("Only geodesic_km metric is supported at the moment.")
        matrix = []
        for origin in origins:
            for dest in destinations:
                lon1 = float(origin["lon"])
                lat1 = float(origin["lat"])
                lon2 = float(dest["lon"])
                lat2 = float(dest["lat"])
                _, _, distance_m = GLOBAL_GEOD.inv(lon1, lat1, lon2, lat2)
                matrix.append(
                    {
                        "origin_id": origin.get("id"),
                        "dest_id": dest.get("id"),
                        "distance_km": distance_m / 1000.0,
                    }
                )

        nearest = []
        for origin in origins:
            matches = [m for m in matrix if m["origin_id"] == origin.get("id")]
            if matches:
                nearest.append(min(matches, key=lambda m: m["distance_km"]))
        return _ok({"matrix": matrix, "nearest": nearest})
    except Exception as exc:  # pragma: no cover
        return _error(f"gis_distance_matrix failed: {exc}")


def gis_overlay(
    *,
    targets: Sequence[Mapping[str, Any]],
    zones: Sequence[Mapping[str, Any]],
    operation: str = "intersect",
    summary_metrics: Optional[Sequence[str]] = None,
) -> ToolResponse:
    """Overlay targets against zones and return exposure stats plus optional feature list."""
    try:
        summary_metrics = summary_metrics or ["count"]
        target_features = _coerce_features(targets)
        zone_features = _coerce_features(zones)

        exposed: List[Dict[str, Any]] = []
        stats = {"total_targets": len(target_features), "targets_in_zones": 0, "metrics": {}}

        zone_geoms = [shape(z["geometry"]) for z in zone_features if z.get("geometry")]
        for target in target_features:
            geom = shape(target["geometry"]) if target.get("geometry") else None
            if geom is None or geom.is_empty:
                continue
            for idx, zone_geom in enumerate(zone_geoms):
                if operation == "intersect":
                    intersects = geom.intersects(zone_geom)
                    overlap = geom.intersection(zone_geom) if intersects else None
                elif operation == "within":
                    intersects = geom.within(zone_geom)
                    overlap = geom if intersects else None
                elif operation == "touches":
                    intersects = geom.touches(zone_geom)
                    overlap = geom if intersects else None
                else:
                    return _error(f"Unsupported overlay operation '{operation}'.")

                if not intersects or overlap is None or overlap.is_empty:
                    continue

                stats["targets_in_zones"] += 1
                exposed.append(
                    {
                        "target": target,
                        "zone_index": idx,
                        "intersection_geometry": mapping(overlap),
                    }
                )
                metrics = _intersection_metrics(overlap, summary_metrics)
                for k, v in metrics.items():
                    stats["metrics"][k] = stats["metrics"].get(k, 0.0) + v
                break
        return _ok({"exposed_features": exposed, "stats": stats})
    except Exception as exc:  # pragma: no cover
        return _error(f"gis_overlay failed: {exc}")


def gis_source_metadata(
    *,
    source: str,
    geometry: Mapping[str, Any],
    feature_types: Sequence[str],
    overpass_url: str = OVERPASS_URL,
) -> ToolResponse:
    """Provide a coarse completeness signal for a source/geometry/feature_type combo."""
    try:
        if source.lower() != "osm":
            return _error("Only 'osm' source is supported currently.")

        _geom_clause, region_geom = _geometry_clause(geometry)
        area = _geometry_area_km2(region_geom) if region_geom else None

        completeness: Dict[str, str] = {}
        notes: List[str] = []
        feature_map = {
            "building": {"filters": [{"key": "building", "values": []}], "geometry_types": ["polygon"]},
            "road": {"filters": [{"key": "highway", "values": []}], "geometry_types": ["line"]},
            "amenity": {"filters": [{"key": "amenity", "values": []}], "geometry_types": ["point", "polygon"]},
        }

        for ftype in feature_types:
            config = feature_map.get(ftype, {"filters": [{"key": ftype, "values": []}], "geometry_types": None})
            result = gis_feature_query(
                source="osm",
                geometry=geometry,
                filters=config["filters"],
                geometry_types=config["geometry_types"],
                max_features=500,
                overpass_url=overpass_url,
            )
            if result.metadata.get("error"):
                completeness[ftype] = "unknown"
                notes.append(f"{ftype} query failed: {result.metadata.get('message')}")
                continue
            count = result.metadata["summary"]["total_features"]
            density = _feature_density(area, count)
            if density is None:
                level = "unknown"
            elif density < 0.5:
                level = "low"
            elif density < 5:
                level = "medium"
            else:
                level = "high"
            if not result.metadata["data_status"]["coverage_ok"]:
                notes.append(f"{ftype} may be truncated at 500 features.")
            completeness[ftype] = level

        return _ok({"completeness": completeness, "notes": "; ".join(notes)})
    except Exception as exc:  # pragma: no cover
        return _error(f"gis_source_metadata failed: {exc}")


def overpass_poi_search(
    bbox: Sequence[float],
    tag_key: str = "amenity",
    tag_value: str = "hospital",
    max_results: int = 200,
    as_of_date: Optional[str] = None,
) -> ToolResponse:
    """Helper to fetch POIs by tag in a bbox (legacy compatibility wrapper)."""
    try:
        return gis_feature_query(
            source="osm",
            geometry={"bbox": bbox},
            filters=[{"key": tag_key, "values": [tag_value]}],
            geometry_types=["point", "line", "polygon"],
            max_features=max_results,
            as_of_date=as_of_date,
        )
    except Exception as exc:  # pragma: no cover
        return _error(f"overpass_poi_search failed: {exc}")
