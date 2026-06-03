"""
Post-processing utility to draw bounding boxes or circles onto an image using pixel
projections from lat/lon geometries and a known image bounding box.
"""

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
from PIL import Image, ImageDraw

__all__ = ["draw_geometries_on_image"]


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _ok(payload: Mapping[str, Any]) -> ToolResponse:
    cleaned = _sanitize(payload)
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(cleaned))], metadata=cleaned)


def _error(msg: str) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=f"Error: {msg}")], metadata={"error": True, "message": msg})


def _parse_bbox(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("image_bbox must be [north, west, south, east].")
    north, west, south, east = map(float, bbox)
    if east == west or north == south:
        raise ValueError("Invalid image_bbox; north/south or east/west are equal.")
    return north, west, south, east


def _hex_to_rgba(color: str, opacity: float = 1.0) -> Tuple[int, int, int, int]:
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise ValueError(f"Invalid color '{color}'. Expected hex like #ff0000.")
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    a = max(0, min(255, int(opacity * 255)))
    return r, g, b, a


def _latlon_to_px(lat: float, lon: float, *, north: float, west: float, south: float, east: float, width: int, height: int) -> Tuple[float, float]:
    x = (lon - west) / (east - west) * width
    y = (north - lat) / (north - south) * height
    return x, y


def _geometry_to_pixel_coords(geom: Mapping[str, Any], *, bbox: Tuple[float, float, float, float], width: int, height: int, point_radius_px: int) -> Tuple[str, list]:
    north, west, south, east = bbox
    gtype = geom.get("type", "").lower()
    if gtype == "feature":
        return _geometry_to_pixel_coords(geom.get("geometry", {}), bbox=bbox, width=width, height=height, point_radius_px=point_radius_px)
    if gtype == "point":
        lon, lat = geom.get("coordinates", [None, None])[:2]
        x, y = _latlon_to_px(float(lat), float(lon), north=north, west=west, south=south, east=east, width=width, height=height)
        r = point_radius_px
        return "point", [(x - r, y - r, x + r, y + r)]
    if gtype == "polygon":
        rings = []
        for ring in geom.get("coordinates", []):
            pts = [
                _latlon_to_px(float(lat), float(lon), north=north, west=west, south=south, east=east, width=width, height=height)
                for lon, lat in ring
            ]
            rings.append(pts)
        return "polygon", rings
    if gtype == "multipolygon":
        polygons = []
        for poly in geom.get("coordinates", []):
            for ring in poly:
                pts = [
                    _latlon_to_px(float(lat), float(lon), north=north, west=west, south=south, east=east, width=width, height=height)
                    for lon, lat in ring
                ]
                polygons.append(pts)
        return "polygon", polygons
    # Allow bbox array [south, west, north, east] or dict
    if isinstance(geom, dict) and {"south", "west", "north", "east"} <= set(geom.keys()):
        south_g, west_g, north_g, east_g = float(geom["south"]), float(geom["west"]), float(geom["north"]), float(geom["east"])
        coords = [
            (west_g, south_g),
            (east_g, south_g),
            (east_g, north_g),
            (west_g, north_g),
            (west_g, south_g),
        ]
        pts = [
            _latlon_to_px(float(lat), float(lon), north=north, west=west, south=south, east=east, width=width, height=height)
            for lon, lat in coords
        ]
        return "polygon", [pts]
    if isinstance(geom, (list, tuple)) and len(geom) == 4:
        south_g, west_g, north_g, east_g = map(float, geom)
        coords = [
            (west_g, south_g),
            (east_g, south_g),
            (east_g, north_g),
            (west_g, north_g),
            (west_g, south_g),
        ]
        pts = [
            _latlon_to_px(lat, lon, north=north, west=west, south=south, east=east, width=width, height=height)
            for lon, lat in coords
        ]
        return "polygon", [pts]
    raise ValueError(f"Unsupported geometry type: {gtype or type(geom)}")


def draw_geometries_on_image(
    *,
    image_path: str,
    image_bbox: Sequence[float],
    geometries: Sequence[Mapping[str, Any]],
    stroke_color: str = "#ff0000",
    stroke_width_px: int = 4,
    fill_color: Optional[str] = None,
    fill_opacity: float = 0.4,
    point_radius_px: int = 10,
    output_path: Optional[str] = None,
) -> ToolResponse:
    """
    Draw bounding boxes or circles onto an image using lat/lon geometries and a known image bbox.

    Args:
        image_path: Input image file (PNG/JPEG).
        image_bbox: [north, west, south, east] lat/lon of the image extent.
        geometries: List of GeoJSON geometries or Features (Point, Polygon, MultiPolygon).
        stroke_color: Outline color (hex).
        stroke_width_px: Outline width in pixels.
        fill_color: Optional fill color (hex). If None, shapes are hollow.
        fill_opacity: 0-1 opacity for fills.
        point_radius_px: Radius for drawing Point geometries as circles.
        output_path: Optional output path; defaults to `<image>_overlay.png`.
    """
    try:
        img_path = Path(image_path)
        if not img_path.is_file():
            return _error(f"Image not found: {img_path}")
        north, west, south, east = _parse_bbox(image_bbox)
        image = Image.open(img_path).convert("RGBA")
        width, height = image.size
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        stroke_rgba = _hex_to_rgba(stroke_color, 1.0)
        fill_rgba = _hex_to_rgba(fill_color, fill_opacity) if fill_color else None

        drawn = 0
        for geom in geometries:
            try:
                g_type, coords = _geometry_to_pixel_coords(
                    geom, bbox=(north, west, south, east), width=width, height=height, point_radius_px=point_radius_px
                )
            except Exception:
                continue
            if g_type == "point":
                for ellipse in coords:
                    draw.ellipse(ellipse, outline=stroke_rgba, width=stroke_width_px, fill=fill_rgba)
                    drawn += 1
            elif g_type == "polygon":
                for ring in coords:
                    draw.polygon(ring, outline=stroke_rgba, width=stroke_width_px, fill=fill_rgba)
                    drawn += 1

        composite = Image.alpha_composite(image, overlay)
        out_path = Path(output_path) if output_path else img_path.with_name(f"{img_path.stem}_overlay.png")
        composite.save(out_path)

        return _ok(
            {
                "input_image": str(img_path),
                "output_image": str(out_path),
                "image_bbox": [north, west, south, east],
                "geometries_drawn": drawn,
                "stroke_color": stroke_color,
                "stroke_width_px": stroke_width_px,
                "fill_color": fill_color,
                "fill_opacity": fill_opacity if fill_color else None,
                "point_radius_px": point_radius_px,
            }
        )
    except Exception as exc:  # pragma: no cover
        return _error(f"draw_geometries_on_image failed: {exc}")
