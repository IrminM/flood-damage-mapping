"""Terrain preprocessing for SAR flood mapping in steep (Himalayan) terrain.

Why: in mountain valleys SAR backscatter is dominated by geometry. Slopes facing
away from the radar fall into SHADOW (very dark, easily mistaken for water); slopes
facing the radar steeper than the incidence angle suffer LAYOVER. Water also cannot
pond on steep slopes or high above the drainage network. So we derive, from the DEM:

  * slope (deg) and aspect (downslope azimuth, deg clockwise from north)
  * approximate radar shadow / layover masks from the local range-direction slope
  * HAND — Height Above Nearest Drainage (m)

APPROXIMATIONS (documented, prototype level):
  * shadow/layover use a LOCAL test on each pixel's slope with a nominal incidence angle
    and nominal satellite heading; no ray tracing, so shadows cast by distant ridges
    are not modelled.
  * HAND uses priority-flood depression filling + D8-style flow routing on the DEM
    extent only. Upstream area outside the DEM is unknown, so the drainage network
    is approximate near the DEM edges.

All computations run on a coarse terrain grid (default 30 m, the GLO-30 posting) and
are then resampled to the analysis grid.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.config import TerrainConfig
from ..core.contracts import TerrainFlag
from ..core.georaster import GeoRaster
from .grid import AnalysisGrid, GridSpec
from .reproject import reproject_to_grid


# ------------------------------------------------------------------ slope / aspect

def slope_aspect(dem: np.ndarray, res_x: float, res_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (deg) and aspect = azimuth of steepest DESCENT (deg clockwise from north).

    Assumes a north-up grid (row index increases southwards). NaN propagates.
    """
    dz_drow, dz_dcol = np.gradient(dem.astype(np.float64))
    dz_dx = dz_dcol / res_x            # eastward gradient
    dz_dy = -dz_drow / res_y           # northward gradient (rows go south)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect = (np.degrees(np.arctan2(-dz_dx, -dz_dy)) + 360.0) % 360.0   # downslope direction
    return slope.astype(np.float32), aspect.astype(np.float32)


def range_slope_deg(slope_deg: np.ndarray, aspect_deg: np.ndarray, look_azimuth_deg: float) -> np.ndarray:
    """Terrain slope in the radar range plane; POSITIVE when the slope faces the radar.

    A slope faces the radar when its downslope direction points back towards the sensor,
    i.e. aspect ≈ look_azimuth + 180°.
    """
    rel = np.radians(aspect_deg - look_azimuth_deg)
    return np.degrees(np.arctan(np.tan(np.radians(slope_deg)) * -np.cos(rel))).astype(np.float32)


def radar_shadow_layover(slope_deg: np.ndarray, aspect_deg: np.ndarray, look_azimuth_deg: float,
                         incidence_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Local approximation: layover if range-slope > incidence; shadow if the slope faces
    away from the radar steeper than (90° - incidence)."""
    rs = range_slope_deg(slope_deg, aspect_deg, look_azimuth_deg)
    layover = rs > incidence_deg
    shadow = -rs >= (90.0 - incidence_deg)
    return shadow & np.isfinite(rs), layover & np.isfinite(rs)


# ------------------------------------------------------------------ flow routing / HAND

_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def priority_flood_routing(dem: np.ndarray, eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Priority-flood (Barnes et al. 2014) with epsilon filling.

    Returns
      order  : flat indices in processing order (outlets first; every cell after its receiver)
      parent : flat index of the downstream (receiving) cell, -1 for outlets / NaN cells
      filled : depression-filled elevation (every cell drains to an outlet)
    Outlets are valid cells on the raster edge or next to NaN (water leaves the DEM there).
    """
    rows, cols = dem.shape
    W = cols + 2
    pad = np.full((rows + 2, W), np.nan)
    pad[1:-1, 1:-1] = dem
    flat = pad.ravel()
    valid = np.isfinite(flat)
    offs = [dr * W + dc for dr, dc in _OFFS]

    # seeds: valid cells with at least one NaN neighbour (includes the raster border)
    v2 = valid.reshape(pad.shape)
    nan_nb = np.zeros_like(v2)
    for dr, dc in _OFFS:
        nan_nb[1:-1, 1:-1] |= ~v2[1 + dr:rows + 1 + dr, 1 + dc:cols + 1 + dc]
    seeds = np.flatnonzero(v2 & nan_nb)

    filled = flat.copy()
    parent = np.full(flat.size, -1, dtype=np.int64)
    visited = ~valid                      # NaN cells are never processed
    heap: list[tuple[float, int, int]] = []
    counter = 0
    for s in seeds.tolist():
        heap.append((float(flat[s]), counter, s))
        visited[s] = True
        counter += 1
    heapq.heapify(heap)

    order: list[int] = []
    filled_l = filled.tolist()
    visited_l = visited.tolist()
    parent_l = parent.tolist()
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        z, _, c = pop(heap)
        order.append(c)
        for o in offs:
            n = c + o
            if visited_l[n]:
                continue
            visited_l[n] = True
            zn = filled_l[n]
            if zn <= z:
                zn = z + eps
                filled_l[n] = zn
            parent_l[n] = c
            counter += 1
            push(heap, (zn, counter, n))

    # map padded indices back to unpadded flat indices
    pr, pc = np.divmod(np.asarray(order, dtype=np.int64), W)
    order_u = (pr - 1) * cols + (pc - 1)
    par = np.asarray(parent_l, dtype=np.int64)
    parent_u = np.full(rows * cols, -1, dtype=np.int64)
    src_r, src_c = np.divmod(np.arange(flat.size), W)
    inside = (src_r >= 1) & (src_r <= rows) & (src_c >= 1) & (src_c <= cols)
    has_par = inside & (par >= 0)
    p_r, p_c = np.divmod(par[has_par], W)
    parent_u[(src_r[has_par] - 1) * cols + (src_c[has_par] - 1)] = (p_r - 1) * cols + (p_c - 1)
    filled_u = np.asarray(filled_l).reshape(pad.shape)[1:-1, 1:-1]
    return order_u, parent_u, filled_u


def flow_accumulation(order: np.ndarray, parent: np.ndarray, size: int, cell_area: float) -> np.ndarray:
    """Upstream contributing area (same units as cell_area) for each cell (flat)."""
    acc = np.zeros(size, dtype=np.float64)
    acc[order] = cell_area
    acc_l = acc.tolist()
    par_l = parent.tolist()
    for c in reversed(order.tolist()):
        p = par_l[c]
        if p >= 0:
            acc_l[p] += acc_l[c]
    return np.asarray(acc_l)


def height_above_nearest_drainage(dem: np.ndarray, order: np.ndarray, parent: np.ndarray,
                                  stream: np.ndarray) -> np.ndarray:
    """HAND (m): elevation minus elevation of the drainage cell each cell flows to.

    Cells whose flow path leaves the DEM without meeting a drainage cell get NaN (unknown).
    """
    z = dem.ravel()
    ref = np.full(z.size, -1, dtype=np.int64)
    stream_f = stream.ravel()
    ref_l = ref.tolist()
    par_l = parent.tolist()
    st_l = stream_f.tolist()
    for c in order.tolist():               # receivers are always processed before donors
        if st_l[c]:
            ref_l[c] = c
        else:
            p = par_l[c]
            ref_l[c] = ref_l[p] if p >= 0 else -1
    ref = np.asarray(ref_l, dtype=np.int64)
    hand = np.full(z.size, np.nan, dtype=np.float32)
    ok = ref >= 0
    hand[ok] = np.maximum(z[ok] - z[ref[ok]], 0.0)
    return hand.reshape(dem.shape)


# ------------------------------------------------------------------ products

@dataclass
class LookGeometry:
    orbit_direction: Optional[str]
    look_azimuth_deg: Optional[float]
    incidence_deg: float
    source: str                      # "sidecar" | "nominal"


def look_geometry(orbit_direction: Optional[str], incidence_deg: Optional[float], cfg: TerrainConfig) -> LookGeometry:
    inc = float(incidence_deg) if incidence_deg is not None else cfg.nominal_incidence_deg
    src = "sidecar" if incidence_deg is not None else "nominal"
    if orbit_direction not in cfg.nominal_heading_deg:
        return LookGeometry(orbit_direction, None, inc, src)
    heading = cfg.nominal_heading_deg[orbit_direction]
    return LookGeometry(orbit_direction, (heading + 90.0) % 360.0, inc, src)   # S1 is right-looking


@dataclass
class TerrainProducts:
    """Terrain layers on the ANALYSIS grid."""
    slope_deg: GeoRaster
    hand_m: GeoRaster
    shadow: np.ndarray               # bool
    layover: np.ndarray              # bool
    no_dem: np.ndarray               # bool
    geometry: LookGeometry
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _terrain_spec(grid: AnalysisGrid, cfg: TerrainConfig) -> GridSpec:
    s = grid.spec
    r, m = cfg.dem_resolution_m, cfg.margin_m
    x0 = s.transform[2] - m
    y1 = s.transform[5] + m
    w = int(math.ceil((s.width * s.resolution + 2 * m) / r))
    h = int(math.ceil((s.height * s.resolution + 2 * m) / r))
    return GridSpec(s.crs, (r, 0.0, x0, 0.0, -r, y1), w, h)


def compute_terrain(dem: GeoRaster, grid: AnalysisGrid, cfg: TerrainConfig, geom: LookGeometry) -> TerrainProducts:
    warnings: list[str] = []
    tspec = _terrain_spec(grid, cfg)
    z = reproject_to_grid(dem.select_bands(["elevation"]) if "elevation" in dem.band_names else dem,
                          tspec, "bilinear").data[0].astype(np.float64)
    if not np.isfinite(z).any():
        raise ValueError("DEM does not cover the analysis area")
    cover = float(np.isfinite(z).mean())
    if cover < 0.99:
        warnings.append(f"DEM covers only {cover:.0%} of the terrain analysis extent (AOI + {cfg.margin_m:.0f} m)")

    r = tspec.resolution
    slope, aspect = slope_aspect(z, r, r)
    if geom.look_azimuth_deg is not None:
        shadow, layover = radar_shadow_layover(slope, aspect, geom.look_azimuth_deg, geom.incidence_deg)
    else:
        shadow = layover = np.zeros(z.shape, bool)
        warnings.append("Orbit direction unknown: radar shadow/layover NOT masked")

    order, parent, _ = priority_flood_routing(np.where(np.isfinite(z), z, np.nan), cfg.fill_epsilon_m)
    acc_km2 = flow_accumulation(order, parent, z.size, r * r).reshape(z.shape) / 1e6
    stream = (acc_km2 >= cfg.hand_stream_threshold_km2) & np.isfinite(z)
    hand = height_above_nearest_drainage(z, order, parent, stream)

    def to_grid(arr: np.ndarray, name: str, resampling: str) -> GeoRaster:
        src = GeoRaster(arr[None], tspec.transform, tspec.crs, nodata=None if arr.dtype == np.uint8 else np.nan,
                        band_names=[name])
        return reproject_to_grid(src, grid.spec, resampling, dst_nodata=255 if arr.dtype == np.uint8 else None)

    slope_g = to_grid(slope.astype(np.float32), "slope_deg", "bilinear")
    hand_g = to_grid(hand.astype(np.float32), "hand_m", "bilinear")
    geo_codes = (shadow.astype(np.uint8) * 1 + layover.astype(np.uint8) * 2).astype(np.uint8)
    geo_g = to_grid(geo_codes, "geo", "nearest").data[0]
    z_g = to_grid(z.astype(np.float32), "z", "bilinear").data[0]
    no_dem = ~np.isfinite(z_g)

    in_aoi = grid.aoi_mask
    stats = {
        "terrain_grid": {"crs": tspec.crs, "resolution_m": r, "shape": [tspec.height, tspec.width]},
        "dem_coverage_fraction": round(cover, 4),
        "look_azimuth_deg": geom.look_azimuth_deg,
        "incidence_deg": geom.incidence_deg,
        "geometry_source": geom.source,
        "stream_threshold_km2": cfg.hand_stream_threshold_km2,
        "stream_cells": int(stream.sum()),
        "aoi_shadow_fraction": round(float(((geo_g & 1) > 0)[in_aoi].mean()), 4),
        "aoi_layover_fraction": round(float(((geo_g & 2) > 0)[in_aoi].mean()), 4),
        "aoi_median_slope_deg": float(np.nanmedian(slope_g.data[0][in_aoi])),
        "aoi_hand_unknown_fraction": round(float(np.isnan(hand_g.data[0][in_aoi]).mean()), 4),
    }
    return TerrainProducts(
        slope_deg=slope_g, hand_m=hand_g,
        shadow=((geo_g & 1) > 0) & (geo_g != 255), layover=((geo_g & 2) > 0) & (geo_g != 255),
        no_dem=no_dem, geometry=geom, stats=stats, warnings=warnings,
    )


def terrain_flag_raster(tp: TerrainProducts, water_max_slope: float, water_max_hand: float,
                        change_max_slope: float) -> np.ndarray:
    """uint8 TerrainFlag raster for output/debugging (thresholds come from change-detection config)."""
    s, h = tp.slope_deg.data[0], tp.hand_m.data[0]
    f = np.zeros(s.shape, np.uint8)
    f[tp.shadow] |= np.uint8(TerrainFlag.RADAR_SHADOW)
    f[tp.layover] |= np.uint8(TerrainFlag.RADAR_LAYOVER)
    with np.errstate(invalid="ignore"):
        f[s > water_max_slope] |= np.uint8(TerrainFlag.STEEP_FOR_WATER)
        f[h > water_max_hand] |= np.uint8(TerrainFlag.HIGH_HAND)
        f[s > change_max_slope] |= np.uint8(TerrainFlag.STEEP_FOR_CHANGE)
    f[tp.no_dem] |= np.uint8(TerrainFlag.NO_DEM)
    return f
