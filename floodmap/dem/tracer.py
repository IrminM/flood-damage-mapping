"""Deterministic DEM-based downstream flood path tracing engine for Stage 6."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyproj
import rasterio.transform
from shapely.geometry import LineString, Point

from ..core.config import DownstreamConfig
from ..core.georaster import GeoRaster
from ..satellite.terrain import priority_flood_routing


@dataclass
class DownstreamPathResult:
    """Detailed container for traced downstream terrain path."""
    coords_lonlat: list[tuple[float, float]]
    coords_metric: list[tuple[float, float]]
    elevations_m: list[float]
    distances_m: list[float]
    total_length_m: float
    start_elevation_m: float
    end_elevation_m: float
    total_drop_m: float
    termination_reason: str
    metric_crs: str
    snapped_coords_lonlat: tuple[float, float]
    original_coords_lonlat: tuple[float, float]
    snapping_distance_m: float

    @property
    def line_geom_lonlat(self) -> LineString:
        if len(self.coords_lonlat) >= 2:
            return LineString(self.coords_lonlat)
        elif len(self.coords_lonlat) == 1:
            pt = self.coords_lonlat[0]
            return LineString([pt, pt])
        return LineString()

    @property
    def line_geom_metric(self) -> LineString:
        if len(self.coords_metric) >= 2:
            return LineString(self.coords_metric)
        elif len(self.coords_metric) == 1:
            pt = self.coords_metric[0]
            return LineString([pt, pt])
        return LineString()


def parse_upstream_point(point_input: Any) -> tuple[float, float]:
    """Parse various upstream point formats into (lon, lat)."""
    if isinstance(point_input, (tuple, list)) and len(point_input) >= 2:
        return float(point_input[0]), float(point_input[1])
    if isinstance(point_input, Point):
        return float(point_input.x), float(point_input.y)
    if isinstance(point_input, dict):
        if "coordinates" in point_input:
            coords = point_input["coordinates"]
            return float(coords[0]), float(coords[1])
        if "geometry" in point_input and "coordinates" in point_input["geometry"]:
            coords = point_input["geometry"]["coordinates"]
            return float(coords[0]), float(coords[1])
        if "lon" in point_input and "lat" in point_input:
            return float(point_input["lon"]), float(point_input["lat"])
    if isinstance(point_input, (str, Path)):
        p = Path(point_input)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("type") == "Feature":
                coords = data["geometry"]["coordinates"]
                return float(coords[0]), float(coords[1])
            elif data.get("type") == "FeatureCollection" and data.get("features"):
                coords = data["features"][0]["geometry"]["coordinates"]
                return float(coords[0]), float(coords[1])
            elif data.get("type") == "Point":
                coords = data["coordinates"]
                return float(coords[0]), float(coords[1])
        s = str(point_input).strip()
        parts = s.replace(",", " ").split()
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    raise ValueError(f"Cannot parse upstream point: {point_input!r}")


def _resolve_metric_crs(dem: GeoRaster, target_metric_crs: Optional[str] = None) -> str:
    """Determine projected metric CRS for distance and slope calculations."""
    if target_metric_crs:
        try:
            crs_obj = pyproj.CRS(target_metric_crs)
            if not crs_obj.is_geographic:
                return target_metric_crs
        except Exception:
            pass

    crs_obj = pyproj.CRS(dem.crs)
    if not crs_obj.is_geographic:
        return str(dem.crs)

    # If geographic EPSG:4326, estimate UTM based on raster center
    bounds = dem.bounds
    mid_lon = (bounds[0] + bounds[2]) / 2.0
    mid_lat = (bounds[1] + bounds[3]) / 2.0
    utm_zone = int(math.floor((mid_lon + 180.0) / 6.0)) + 1
    hemisphere = "326" if mid_lat >= 0 else "327"
    return f"EPSG:{hemisphere}{utm_zone:02d}"


def trace_downstream_path(
    dem: GeoRaster,
    upstream_point: Any,
    cfg: DownstreamConfig,
    target_metric_crs: Optional[str] = None,
) -> DownstreamPathResult:
    """Trace a deterministic downstream flow path from an upstream point.

    Algorithm:
      1. Validates upstream point within DEM bounding box.
      2. Snaps to nearest valid finite elevation cell if NoData is encountered.
      3. Follows steepest downhill slope in D8 neighborhood.
      4. If a flat or sink is reached:
         - When sink_handling == "fill": utilizes Barnes et al. (2014) depression filling
           to determine the spillway/overflow neighbor towards the catchment outlet.
         - When sink_handling == "stop": terminates at the sink with a clear status.
      5. Prevents cycles via visited cell tracking.
      6. Terminates when maximum length, step limit, raster boundary, or outlet is reached.
    """
    orig_lon, orig_lat = parse_upstream_point(upstream_point)
    metric_crs = _resolve_metric_crs(dem, target_metric_crs)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    rev_transformer = pyproj.Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)

    arr = dem.data[0].astype(np.float64)
    rows, cols = arr.shape
    transform = dem.transform

    # Check raster bounds (transform upstream point to DEM CRS if projected)
    dem_crs_obj = pyproj.CRS(dem.crs)
    if not dem_crs_obj.is_geographic:
        wgs_to_dem = pyproj.Transformer.from_crs("EPSG:4326", dem.crs, always_xy=True)
        dem_to_wgs = pyproj.Transformer.from_crs(dem.crs, "EPSG:4326", always_xy=True)
        pt_x, pt_y = wgs_to_dem.transform(orig_lon, orig_lat)
    else:
        wgs_to_dem = None
        dem_to_wgs = None
        pt_x, pt_y = orig_lon, orig_lat

    left, bottom, right, top = dem.bounds
    if not (left <= pt_x <= right and bottom <= pt_y <= top):
        raise ValueError(
            f"Upstream point ({orig_lon:.5f}, {orig_lat:.5f}) lies outside DEM bounds "
            f"[{left:.5f}, {bottom:.5f}, {right:.5f}, {top:.5f}]"
        )

    # Pixel coordinate of upstream point
    col_f, row_f = ~rasterio.transform.Affine(*transform) @ (pt_x, pt_y)
    c0 = int(round(col_f))
    r0 = int(round(row_f))

    c0 = max(0, min(cols - 1, c0))
    r0 = max(0, min(rows - 1, r0))

    # Validate / snap if cell is NoData or NaN
    snapped_r, snapped_c = r0, c0
    if not np.isfinite(arr[r0, c0]):
        # Search outward in expanding rings up to radius 15
        found = False
        for rad in range(1, 16):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    nr, nc = r0 + dr, c0 + dc
                    if 0 <= nr < rows and 0 <= nc < cols and np.isfinite(arr[nr, nc]):
                        snapped_r, snapped_c = nr, nc
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            raise ValueError(
                f"Upstream point ({orig_lon:.5f}, {orig_lat:.5f}) falls on NoData "
                "and could not be snapped to a valid elevation cell"
            )

    snap_x, snap_y = rasterio.transform.xy(rasterio.transform.Affine(*transform), snapped_r, snapped_c)
    if dem_to_wgs is not None:
        snap_lon, snap_lat = dem_to_wgs.transform(snap_x, snap_y)
    else:
        snap_lon, snap_lat = snap_x, snap_y
    orig_mx, orig_my = transformer.transform(orig_lon, orig_lat)
    snap_mx, snap_my = transformer.transform(snap_lon, snap_lat)
    snapping_dist_m = float(math.hypot(snap_mx - orig_mx, snap_my - orig_my))

    # Precompute cell resolutions in metric meters
    mid_lat = (bottom + top) / 2.0
    crs_obj = pyproj.CRS(dem.crs)
    if crs_obj.is_geographic:
        res_x_m = abs(transform[0]) * 111320.0 * math.cos(math.radians(mid_lat))
        res_y_m = abs(transform[4]) * 111320.0
    else:
        res_x_m = abs(transform[0])
        res_y_m = abs(transform[4])

    # If sink_handling == "fill", run depression-filled routing once
    parent_map: Optional[np.ndarray] = None
    if cfg.sink_handling == "fill":
        _, parent_map, _ = priority_flood_routing(arr, eps=1e-4)

    # Downstream stepping
    D8_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    curr_r, curr_c = snapped_r, snapped_c
    path_cells: list[tuple[int, int]] = [(curr_r, curr_c)]
    visited_cells: set[tuple[int, int]] = {(curr_r, curr_c)}
    termination_reason = "max_steps_reached"

    cum_dist_m = 0.0
    distances_m: list[float] = [0.0]
    elevations_m: list[float] = [float(arr[curr_r, curr_c])]
    coords_lonlat: list[tuple[float, float]] = [(snap_lon, snap_lat)]
    coords_metric: list[tuple[float, float]] = [(snap_mx, snap_my)]

    while len(path_cells) < cfg.max_steps:
        # Check raster boundaries
        if curr_r <= 0 or curr_r >= rows - 1 or curr_c <= 0 or curr_c >= cols - 1:
            termination_reason = "Reached DEM raster boundary"
            break

        curr_z = arr[curr_r, curr_c]

        next_cell: Optional[tuple[int, int]] = None

        if cfg.sink_handling == "fill" and parent_map is not None:
            curr_idx = curr_r * cols + curr_c
            p = parent_map[curr_idx]
            if p >= 0:
                pr, pc = divmod(int(p), cols)
                if (pr, pc) not in visited_cells and 0 <= pr < rows and 0 <= pc < cols:
                    next_cell = (pr, pc)
                else:
                    termination_reason = "Loop or cycle encountered during depression traversal"
                    break
            else:
                termination_reason = "Reached drainage catchment outlet / DEM terminus"
                break
        else:
            # Evaluate D8 downhill neighbors on raw DEM
            best_neighbor: Optional[tuple[int, int]] = None
            best_gradient = -float("inf")
            best_z = float("inf")

            for dr, dc in D8_OFFS:
                nr, nc = curr_r + dr, curr_c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if (nr, nc) in visited_cells:
                    continue
                nz = arr[nr, nc]
                if not np.isfinite(nz):
                    continue

                drop = curr_z - nz
                if drop > cfg.min_elevation_drop_m:
                    step_len = math.hypot(dc * res_x_m, dr * res_y_m)
                    gradient = drop / step_len
                    if gradient > best_gradient or (math.isclose(gradient, best_gradient) and nz < best_z):
                        best_gradient = gradient
                        best_z = nz
                        best_neighbor = (nr, nc)

            if best_neighbor is not None:
                next_cell = best_neighbor
            else:
                termination_reason = "Terminated at local depression/sink (no lower neighbor)"
                break

        if next_cell is None:
            break

        next_r, next_c = next_cell
        step_dist = math.hypot((next_c - curr_c) * res_x_m, (next_r - curr_r) * res_y_m)
        cum_dist_m += step_dist

        next_x, next_y = rasterio.transform.xy(rasterio.transform.Affine(*transform), next_r, next_c)
        if dem_to_wgs is not None:
            next_lon, next_lat = dem_to_wgs.transform(next_x, next_y)
        else:
            next_lon, next_lat = next_x, next_y
        next_mx, next_my = transformer.transform(next_lon, next_lat)
        next_z = float(arr[next_r, next_c])

        path_cells.append((next_r, next_c))
        visited_cells.add((next_r, next_c))
        distances_m.append(float(cum_dist_m))
        elevations_m.append(next_z)
        coords_lonlat.append((float(next_lon), float(next_lat)))
        coords_metric.append((float(next_mx), float(next_my)))

        curr_r, curr_c = next_r, next_c

        if cum_dist_m >= cfg.max_length_m:
            termination_reason = f"Maximum path length reached ({cfg.max_length_m / 1000.0:.1f} km)"
            break

    total_len_m = distances_m[-1] if distances_m else 0.0
    start_z = elevations_m[0] if elevations_m else 0.0
    end_z = elevations_m[-1] if elevations_m else 0.0
    total_drop_m = float(start_z - end_z)

    return DownstreamPathResult(
        coords_lonlat=coords_lonlat,
        coords_metric=coords_metric,
        elevations_m=elevations_m,
        distances_m=distances_m,
        total_length_m=total_len_m,
        start_elevation_m=start_z,
        end_elevation_m=end_z,
        total_drop_m=total_drop_m,
        termination_reason=termination_reason,
        metric_crs=metric_crs,
        snapped_coords_lonlat=(snap_lon, snap_lat),
        original_coords_lonlat=(orig_lon, orig_lat),
        snapping_distance_m=snapping_dist_m,
    )
