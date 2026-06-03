"""
OSM-based routing and travel-time helper for accessibility/EMS use cases.

Requires `osmnx` (and its dependencies). If unavailable, the tool returns
an informative error. Can load a local GraphML or fetch a bbox from OSM.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

try:
    import networkx as nx
    import osmnx as ox
except ImportError:  # pragma: no cover - optional dependency guard
    nx = None  # type: ignore
    ox = None  # type: ignore

__all__ = ["osm_travel_time_matrix"]


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _ok(payload: Dict[str, Any]) -> ToolResponse:
    cleaned = _sanitize(payload)
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(cleaned))], metadata=cleaned)


def _error(msg: str) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=f"Error: {msg}")], metadata={"error": True, "message": msg})


def _load_graph(
    *,
    bbox: Optional[Sequence[float]],
    graphml_path: Optional[str],
    mode: str,
) -> Tuple[Optional[Any], Optional[str]]:
    if ox is None:
        return None, "osmnx is not installed. Install osmnx to enable routing."

    if graphml_path:
        path = Path(graphml_path).expanduser()
        if not path.is_file():
            return None, f"GraphML not found: {path}"
        try:
            return ox.load_graphml(path), "graphml"
        except Exception as exc:  # pragma: no cover
            return None, f"Failed to load GraphML: {exc}"

    if bbox:
        if len(bbox) != 4:
            return None, "bbox must be [north, south, east, west]."
        north, south, east, west = map(float, bbox)
        try:
            G = ox.graph_from_bbox(north, south, east, west, network_type=mode)
            return G, "bbox"
        except Exception as exc:  # pragma: no cover
            return None, f"Failed to fetch OSM graph: {exc}"

    return None, "Provide either bbox or graphml_path."


def _prepare_speeds(G: Any, speed_kmh_overrides: Optional[Mapping[str, float]]) -> Any:
    """Ensure travel_time edge weight exists; apply optional speed overrides."""
    if ox is None:
        return G
    try:
        if speed_kmh_overrides:
            for u, v, k, data in G.edges(keys=True, data=True):
                hwy = None
                tags = data.get("highway")
                if isinstance(tags, list) and tags:
                    hwy = tags[0]
                elif isinstance(tags, str):
                    hwy = tags
                if hwy and hwy in speed_kmh_overrides:
                    data["speed_kph"] = float(speed_kmh_overrides[hwy])
        if not all("speed_kph" in d for _, _, d in G.edges(data=True)):
            G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    except Exception:  # pragma: no cover
        pass
    return G


def _nearest_nodes(G: Any, points: Sequence[Mapping[str, Any]]) -> List[Any]:
    nodes = []
    for pt in points:
        lat = float(pt["lat"])
        lon = float(pt["lon"])
        node = ox.distance.nearest_nodes(G, lon, lat) if ox else None
        nodes.append(node)
    return nodes


def osm_travel_time_matrix(
    *,
    origins: Sequence[Mapping[str, Any]],
    destinations: Sequence[Mapping[str, Any]],
    mode: str = "drive",
    bbox: Optional[Sequence[float]] = None,
    graphml_path: Optional[str] = None,
    speed_kmh_overrides: Optional[Mapping[str, float]] = None,
) -> ToolResponse:
    """
    Compute travel-time matrix between origins and destinations using an OSM road graph.

    Args:
        origins: list of {lat, lon, id}
        destinations: list of {lat, lon, id}
        mode: OSMnx network_type (e.g., 'drive', 'walk', 'bike')
        bbox: [north, south, east, west]; used if graphml_path is not provided
        graphml_path: optional path to pre-downloaded GraphML
        speed_kmh_overrides: optional dict of highway tag -> speed_kph
    """
    if ox is None or nx is None:
        return _error("osmnx is required for routing; install it to use osm_travel_time_matrix.")

    G, status = _load_graph(bbox=bbox, graphml_path=graphml_path, mode=mode)
    if G is None:
        return _error(status or "Failed to build graph.")

    G = _prepare_speeds(G, speed_kmh_overrides)

    origin_nodes = _nearest_nodes(G, origins)
    dest_nodes = _nearest_nodes(G, destinations)

    results: List[Dict[str, Any]] = []
    for o_idx, o_node in enumerate(origin_nodes):
        if o_node is None:
            continue
        try:
            dist_map = nx.shortest_path_length(G, source=o_node, weight="length")
            time_map = nx.shortest_path_length(G, source=o_node, weight="travel_time")
        except Exception as exc:  # pragma: no cover
            return _error(f"Routing failed: {exc}")
        for d_idx, d_node in enumerate(dest_nodes):
            if d_node is None:
                continue
            dist_m = dist_map.get(d_node)
            time_s = time_map.get(d_node)
            results.append(
                {
                    "origin_id": origins[o_idx].get("id", o_idx),
                    "destination_id": destinations[d_idx].get("id", d_idx),
                    "travel_time_minutes": time_s / 60.0 if time_s is not None else None,
                    "travel_distance_km": dist_m / 1000.0 if dist_m is not None else None,
                }
            )

    return _ok(
        {
            "mode": mode,
            "graph_source": status,
            "bbox": list(bbox) if bbox else None,
            "graphml_path": str(graphml_path) if graphml_path else None,
            "speed_overrides": speed_kmh_overrides or {},
            "matrix": results,
        }
    )
