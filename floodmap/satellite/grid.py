"""Analysis grid: the single pixel grid every raster is aligned to.

The grid is derived from the AOI, not from the input files, so the same AOI always
gives the same grid regardless of which scenes/DEM are supplied (reproducible,
and before/after images become pixel-aligned by construction).

    AOI (EPSG:4326 polygon)
      -> project to analysis CRS (auto UTM)       [pyproj]
      -> bounds + margin, snapped to resolution
      -> GridSpec(crs, transform, width, height)
      -> AOI mask (True inside the polygon)        [rasterio.features]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..core.contracts import AOI
from ..core.geo_utils import analysis_crs_for_aoi
from ..core.georaster import GeoRaster, Transform, normalize_crs


@dataclass(frozen=True)
class GridSpec:
    crs: str
    transform: Transform
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def resolution(self) -> float:
        return abs(self.transform[0])

    def template(self, bands: int = 1, dtype: Any = np.float32, fill: float = np.nan,
                 band_names: Optional[list[str]] = None, nodata: Optional[float] = np.nan) -> GeoRaster:
        data = np.full((bands, self.height, self.width), fill, dtype=dtype)
        return GeoRaster(data, self.transform, self.crs, nodata=nodata, band_names=band_names or [])

    @classmethod
    def from_raster(cls, r: GeoRaster) -> "GridSpec":
        return cls(r.crs, r.transform, r.width, r.height)

    def scaled(self, resolution: float) -> "GridSpec":
        """Same extent (rounded outward) at another resolution, e.g. 30 m for DEM analysis."""
        a, _, c, _, e, f = self.transform
        width_m, height_m = self.width * abs(a), self.height * abs(e)
        w, h = int(math.ceil(width_m / resolution)), int(math.ceil(height_m / resolution))
        return GridSpec(self.crs, (resolution, 0.0, c, 0.0, -resolution, f), w, h)


@dataclass(frozen=True)
class AnalysisGrid:
    spec: GridSpec
    aoi_mask: np.ndarray            # bool (rows, cols): True inside the AOI polygon
    aoi_geometry: dict[str, Any]    # AOI polygon in the grid CRS (GeoJSON-like)

    @property
    def aoi_area_m2(self) -> float:
        return float(self.aoi_mask.sum()) * self.spec.resolution ** 2


def _densify_ring(ring: list[list[float]], n: int = 16) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:]):
        for i in range(n):
            t = i / n
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    out.append(tuple(ring[-1]))  # type: ignore[arg-type]
    return out


def project_aoi(aoi: AOI, dst_crs: str) -> dict[str, Any]:
    """Project the AOI polygon to `dst_crs`, densifying edges so curved projected edges are respected."""
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    g = aoi.geometry
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    out_polys = []
    for poly in polys:
        rings = []
        for ring in poly:
            pts = _densify_ring([list(map(float, p[:2])) for p in ring])
            xs, ys = tr.transform([p[0] for p in pts], [p[1] for p in pts])
            rings.append([[x, y] for x, y in zip(xs, ys)])
        out_polys.append(rings)
    if g["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": out_polys[0]}
    return {"type": "MultiPolygon", "coordinates": out_polys}


def _geom_bounds(geom: dict[str, Any]) -> tuple[float, float, float, float]:
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    xs = [p[0] for poly in polys for ring in poly for p in ring]
    ys = [p[1] for poly in polys for ring in poly for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def rasterize_mask(geom: dict[str, Any], spec: GridSpec, all_touched: bool = False) -> np.ndarray:
    """Bool mask (True inside geometry) on the grid."""
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine

    return geometry_mask([geom], out_shape=spec.shape, transform=Affine(*spec.transform),
                         invert=True, all_touched=all_touched)


def make_analysis_grid(aoi: AOI, resolution_m: float, margin_m: float, crs_rule: str = "auto_utm") -> AnalysisGrid:
    crs = normalize_crs(analysis_crs_for_aoi(aoi, crs_rule))
    geom = project_aoi(aoi, crs)
    minx, miny, maxx, maxy = _geom_bounds(geom)
    r = float(resolution_m)
    minx = math.floor((minx - margin_m) / r) * r
    miny = math.floor((miny - margin_m) / r) * r
    maxx = math.ceil((maxx + margin_m) / r) * r
    maxy = math.ceil((maxy + margin_m) / r) * r
    width, height = int(round((maxx - minx) / r)), int(round((maxy - miny) / r))
    spec = GridSpec(crs, (r, 0.0, minx, 0.0, -r, maxy), width, height)
    return AnalysisGrid(spec=spec, aoi_mask=rasterize_mask(geom, spec), aoi_geometry=geom)
