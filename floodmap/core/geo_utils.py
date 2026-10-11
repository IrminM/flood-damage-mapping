"""Small geographic helpers that need no GDAL/PROJ.

Exact reprojection is done with pyproj/rasterio in the modules that need it; these
helpers only decide *which* CRS to use and compute approximate fetch extents.
"""
from __future__ import annotations

import math
from typing import Any, Iterator

from .contracts import AOI

Bounds = tuple[float, float, float, float]

KM_PER_DEG_LAT = 111.32


def _iter_positions(geometry: dict[str, Any]) -> Iterator[tuple[float, float]]:
    gtype = geometry["type"]
    polys = [geometry["coordinates"]] if gtype == "Polygon" else geometry["coordinates"]
    for poly in polys:
        for ring in poly:
            for pt in ring:
                yield float(pt[0]), float(pt[1])


def aoi_bounds(aoi: AOI) -> Bounds:
    """(min_lon, min_lat, max_lon, max_lat) of the AOI."""
    xs, ys = zip(*_iter_positions(aoi.geometry))
    return min(xs), min(ys), max(xs), max(ys)


def bounds_center(b: Bounds) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """EPSG code of the WGS84/UTM zone containing (lon, lat).

    Standard 6° zones (the Norway/Svalbard exceptions are irrelevant for the Himalaya).
    Nepal falls in zones 44N–45N (EPSG:32644/32645).
    """
    if not (-80.0 <= lat <= 84.0):
        raise ValueError("UTM is not defined beyond 80°S/84°N")
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def analysis_crs_for_aoi(aoi: AOI, rule: str = "auto_utm") -> str:
    """CRS string used for metric analysis of this AOI ("auto_utm" or an explicit 'EPSG:xxxx')."""
    if rule == "auto_utm":
        lon, lat = bounds_center(aoi_bounds(aoi))
        return f"EPSG:{utm_epsg_for_lonlat(lon, lat)}"
    return rule


def buffer_lonlat_bounds_km(b: Bounds, km: float) -> Bounds:
    """Expand lon/lat bounds by ~`km` on every side (approximate; for data *fetch* extents only)."""
    min_lon, min_lat, max_lon, max_lat = b
    dlat = km / KM_PER_DEG_LAT
    ref_lat = max(abs(min_lat), abs(max_lat))
    dlon = km / (KM_PER_DEG_LAT * max(math.cos(math.radians(ref_lat)), 1e-6))
    return (
        max(-180.0, min_lon - dlon),
        max(-90.0, min_lat - dlat),
        min(180.0, max_lon + dlon),
        min(90.0, max_lat + dlat),
    )


def bounds_to_polygon(b: Bounds) -> dict[str, Any]:
    """GeoJSON Polygon geometry for a lon/lat bounding box."""
    x0, y0, x1, y1 = b
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def aoi_from_bounds(b: Bounds, name: str | None = None) -> AOI:
    return AOI(geometry=bounds_to_polygon(b), name=name)


def aoi_from_geojson(obj: dict[str, Any], name: str | None = None) -> AOI:
    """Build an AOI from a GeoJSON geometry, Feature or FeatureCollection (polygons are merged
    into one MultiPolygon if there are several)."""
    t = obj.get("type")
    if t == "FeatureCollection":
        geoms = [f["geometry"] for f in obj.get("features", []) if f.get("geometry")]
    elif t == "Feature":
        geoms = [obj["geometry"]]
        name = name or (obj.get("properties") or {}).get("name")
    else:
        geoms = [obj]
    polys: list[Any] = []
    for g in geoms:
        if g["type"] == "Polygon":
            polys.append(g["coordinates"])
        elif g["type"] == "MultiPolygon":
            polys.extend(g["coordinates"])
        else:
            raise ValueError(f"AOI must contain polygons, found {g['type']}")
    if not polys:
        raise ValueError("No polygon geometry found for AOI")
    geometry = {"type": "Polygon", "coordinates": polys[0]} if len(polys) == 1 else \
        {"type": "MultiPolygon", "coordinates": polys}
    return AOI(geometry=geometry, name=name)


def _parse_bbox(text: str) -> Bounds | None:
    """'min_lon,min_lat,max_lon,max_lat' -> bounds, or None if `text` is not exactly four numbers."""
    parts = [v.strip() for v in text.split(",")]
    if len(parts) != 4:
        return None
    try:
        b = tuple(float(v) for v in parts)
    except ValueError:
        return None
    if not (b[0] < b[2] and b[1] < b[3]):
        raise ValueError(f"AOI bbox must be min_lon,min_lat,max_lon,max_lat with min < max, got {text!r}")
    return b  # type: ignore[return-value]


def load_aoi(path_or_bbox: str) -> AOI:
    """AOI from one of:
      * a GeoJSON file (geometry, Feature or FeatureCollection of polygons),
      * a JSON file with an "aoi" entry, e.g. data/raw/<aoi>/s1_candidates.json written by
        search_s1_scenes.py (so the exact AOI used for the scene search is reused),
      * a bbox string 'min_lon,min_lat,max_lon,max_lat'.
    Relative paths are resolved against the current working directory. A path that does not
    exist is reported as such (never re-interpreted as coordinates).
    """
    import json
    from pathlib import Path

    text = str(path_or_bbox).strip().strip('"').strip("'")
    p = Path(text)
    if p.is_file():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"AOI file {p.resolve()} is not valid JSON/GeoJSON: {e}") from e
        if isinstance(obj, dict) and isinstance(obj.get("aoi"), dict) and "geometry" in obj["aoi"]:
            return AOI.model_validate(obj["aoi"])          # e.g. s1_candidates.json
        return aoi_from_geojson(obj, name=p.stem)
    bbox = _parse_bbox(text)
    if bbox is not None:
        return aoi_from_bounds(bbox)
    raise ValueError(
        f"AOI file not found: {p.resolve()} (working directory: {Path.cwd()}). "
        "Pass an existing GeoJSON/JSON file or 'min_lon,min_lat,max_lon,max_lat'."
    )
