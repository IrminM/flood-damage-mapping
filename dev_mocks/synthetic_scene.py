"""TEST/DEVELOPMENT ONLY — synthetic Sentinel-1-like before/after scenes + valley DEM.

Purpose: exercise the software interfaces (loading, alignment, preprocessing, terrain,
change detection) against a KNOWN ground truth. These rasters are NOT real observations
and must never be used to report results. Every output is stamped
source_type=synthetic-test and data_mode=SYNTHETIC_TEST, which propagates into
provenance and summaries.

Synthetic world (10 m S1 grid, EPSG:32645, valley axis running north-south):
  * DEM (30 m): river floor sloping 1 % to the south, a flat floodplain (±400 m from the
    axis, ~0.5 % cross-slope), then 30° valley sides.
  * Backscatter (linear sigma0, gamma speckle): land VV 0.08 (≈-11 dB), VH 0.02 (≈-17 dB);
    open water VV 0.005 (≈-23 dB), VH 0.001 (≈-30 dB).
  * permanent_water : 60 m wide river along the axis, water in both dates
  * flood_truth     : floodplain patch east of the river, land before -> water after
  * change_truth    : floodplain patch west of the river, backscatter x change_gain after
                      (default x4 = +6 dB)
                      ("surface change", NOT water)
  * slope_false_water: patch on the steep eastern valley side that turns dark after
                      (must be rejected by the slope/HAND gate)
  * after_nodata    : last columns of the after image are NaN (swath edge)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from floodmap.core.contracts import DataMode, SceneMetadata, SourceType
from floodmap.core.georaster import GeoRaster

SYNTHETIC_CRS = "EPSG:32645"
# Arbitrary origin in UTM 45N; not tied to any real place's data.
SYNTHETIC_ORIGIN = (330000.0, 3110000.0)

LAND_VV, LAND_VH = 0.08, 0.02
WATER_VV, WATER_VH = 0.005, 0.001


@dataclass
class SyntheticPair:
    before: GeoRaster
    after: GeoRaster
    flood_truth: np.ndarray        # bool: newly flooded
    permanent_water: np.ndarray    # bool: water in both dates
    before_meta: SceneMetadata
    after_meta: SceneMetadata
    change_truth: np.ndarray = None        # bool: non-water backscatter change
    slope_false_water: np.ndarray = None   # bool: dark-after patch on steep slope
    after_nodata: np.ndarray = None        # bool: NaN in after image
    dem: GeoRaster = None                  # 30 m valley DEM covering the same extent
    truth_notes: dict = field(default_factory=dict)


def _meta(when: datetime) -> SceneMetadata:
    return SceneMetadata(
        sensor="S1", platform="synthetic", product_type="GRD", acquisition_time=when,
        bands=["VV", "VH"], units="linear",
        processing="SYNTHETIC — dev_mocks.synthetic_scene (not a real observation)",
        source_type=SourceType.SYNTHETIC_TEST, relative_orbit=1, orbit_direction="ascending",
        product_ids=["SYNTHETIC"], data_mode=DataMode.SYNTHETIC_TEST,
    )


def valley_elevation(x_from_axis_m: np.ndarray, y_from_top_m: np.ndarray,
                     floodplain_half_width_m: float = 400.0, side_slope_deg: float = 30.0,
                     floor_top_m: float = 1000.0, along_slope: float = 0.01, cross_slope: float = 0.005) -> np.ndarray:
    d = np.abs(x_from_axis_m)
    z = floor_top_m - along_slope * y_from_top_m + cross_slope * np.minimum(d, floodplain_half_width_m)
    return z + np.maximum(d - floodplain_half_width_m, 0.0) * math.tan(math.radians(side_slope_deg))


def make_synthetic_dem(rows: int, cols: int, pixel: float, dem_pixel: float = 30.0,
                       origin: tuple[float, float] = SYNTHETIC_ORIGIN) -> GeoRaster:
    w_m, h_m = cols * pixel, rows * pixel
    dc, dr = int(math.ceil(w_m / dem_pixel)), int(math.ceil(h_m / dem_pixel))
    xs = (np.arange(dc) + 0.5) * dem_pixel - w_m / 2.0          # metres from valley axis
    ys = (np.arange(dr) + 0.5) * dem_pixel                       # metres from top
    X, Y = np.meshgrid(xs, ys)
    z = valley_elevation(X, Y).astype(np.float32)
    return GeoRaster(z, (dem_pixel, 0.0, origin[0], 0.0, -dem_pixel, origin[1]), SYNTHETIC_CRS,
                     nodata=None, band_names=["elevation"], metadata={"data_mode": DataMode.SYNTHETIC_TEST.value})


def make_synthetic_s1_pair(rows: int = 150, cols: int = 200, pixel: float = 10.0,
                           looks: int = 5, seed: int = 0, nodata_cols: int = 6,
                           change_gain: float = 4.0) -> SyntheticPair:
    rng = np.random.default_rng(seed)
    transform = (pixel, 0.0, SYNTHETIC_ORIGIN[0], 0.0, -pixel, SYNTHETIC_ORIGIN[1])
    axis = cols // 2
    rr, cc = np.mgrid[0:rows, 0:cols]

    perm = np.abs(cc - axis + 0.5) <= 3                              # 60 m river
    flood = (rr >= rows * 0.27) & (rr < rows * 0.53) & (cc >= axis + 5) & (cc < axis + 35)
    change = (rr >= rows * 0.6) & (rr < rows * 0.8) & (cc >= axis - 30) & (cc < axis - 10)
    x_from_axis = (cc + 0.5) * pixel - cols * pixel / 2.0
    steep_east = x_from_axis > 520                                    # > 120 m into the 30° side
    slope_dark = (rr >= rows * 0.13) & (rr < rows * 0.3) & steep_east & (cc < cols - nodata_cols - 2)

    def scene(water: np.ndarray, gain: np.ndarray | None = None) -> np.ndarray:
        vv = np.where(water, WATER_VV, LAND_VV)
        vh = np.where(water, WATER_VH, LAND_VH)
        if gain is not None:
            vv, vh = vv * gain, vh * gain
        speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=(2, rows, cols))
        return (np.stack([vv, vh]) * speckle).astype(np.float32)

    before_data = scene(perm)
    after_data = scene(perm | flood | slope_dark, gain=np.where(change, change_gain, 1.0))
    nod = np.zeros((rows, cols), bool)
    if nodata_cols:
        nod[:, cols - nodata_cols:] = True
        after_data[:, nod] = np.nan

    before_t = datetime(2026, 8, 20, 0, 15, tzinfo=timezone.utc)
    after_t = datetime(2026, 9, 1, 0, 15, tzinfo=timezone.utc)
    md = {"data_mode": DataMode.SYNTHETIC_TEST.value}
    before = GeoRaster(before_data, transform, SYNTHETIC_CRS, nodata=None, band_names=["VV", "VH"], metadata=dict(md))
    after = GeoRaster(after_data, transform, SYNTHETIC_CRS, nodata=None, band_names=["VV", "VH"], metadata=dict(md))
    return SyntheticPair(
        before=before, after=after, flood_truth=flood, permanent_water=perm,
        before_meta=_meta(before_t), after_meta=_meta(after_t),
        change_truth=change, slope_false_water=slope_dark, after_nodata=nod,
        dem=make_synthetic_dem(rows, cols, pixel),
        truth_notes={"valley_axis_col": axis, "floodplain_half_width_m": 400, "side_slope_deg": 30},
    )


def synthetic_loaded_pair(pair: SyntheticPair):
    """Wrap the in-memory rasters as LoadedScene objects (no files needed)."""
    from floodmap.core.contracts import SceneRef
    from floodmap.data_sources.base import LoadedScene

    return (LoadedScene(SceneRef(path=Path("SYNTHETIC_before.tif"), metadata=pair.before_meta), pair.before),
            LoadedScene(SceneRef(path=Path("SYNTHETIC_after.tif"), metadata=pair.after_meta), pair.after))


def synthetic_aoi(pair: SyntheticPair, inset_m: float = 0.0):
    """AOI (EPSG:4326) covering the synthetic raster extent minus `inset_m` (needs pyproj)."""
    from pyproj import Transformer

    from floodmap.core.contracts import AOI

    x0, y0, x1, y1 = pair.before.bounds
    x0, y0, x1, y1 = x0 + inset_m, y0 + inset_m, x1 - inset_m, y1 - inset_m
    tr = Transformer.from_crs(SYNTHETIC_CRS, "EPSG:4326", always_xy=True)
    ring = [tr.transform(x, y) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]
    return AOI(geometry={"type": "Polygon", "coordinates": [[list(p) for p in ring]]}, name="SYNTHETIC_AOI")


def write_synthetic_pair(out_dir: Path | str, with_dem: bool = True, **kw) -> tuple[Path, Path, SyntheticPair]:
    """Write the pair (and DEM) as GeoTIFF + sidecars (requires rasterio)."""
    from floodmap.data_sources.satellite_local import write_scene_metadata
    from floodmap.io.geotiff import write_geotiff

    out_dir = Path(out_dir)
    pair = make_synthetic_s1_pair(**kw)
    bpath = write_geotiff(pair.before, out_dir / "SYNTHETIC_s1_before.tif")
    apath = write_geotiff(pair.after, out_dir / "SYNTHETIC_s1_after.tif")
    write_scene_metadata(bpath, pair.before_meta)
    write_scene_metadata(apath, pair.after_meta)
    if with_dem:
        dpath = write_geotiff(pair.dem, out_dir / "SYNTHETIC_dem.tif")
        dpath.with_suffix(".meta.json").write_text(
            '{"source_type": "synthetic-test", "product": "SYNTHETIC valley DEM (dev_mocks)", '
            '"data_mode": "SYNTHETIC_TEST"}', encoding="utf-8")
    return bpath, apath, pair
