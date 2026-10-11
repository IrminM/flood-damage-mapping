"""Sentinel-1 before/after preprocessing.

Input : two LoadedScene objects (terrain-corrected backscatter GeoTIFFs + sidecars,
        linear power or dB as declared), the AnalysisGrid, S1PreprocessConfig.
Output: PreprocessedPair — VV/VH in dB on the analysis grid for both dates, NaN
        wherever either date is invalid or the pixel is outside the AOI, plus the
        shared valid mask and diagnostics.

Order of operations (each choice matters):
  1. units -> linear power (dB inputs are converted back; averaging dB is biased)
  2. non-positive / non-finite values -> nodata (border zeros, no-data fill)
  3. resample to the analysis grid in linear power (bilinear)
  4. Lee speckle filter in linear power (nodata-aware), optional
  5. linear -> dB with a floor; values outside valid_db_range -> nodata
  6. shared valid mask = valid(before) & valid(after) & inside AOI

This module knows nothing about change detection or AI models: it only produces
analysis-ready arrays. Any detector (baseline change detection, the AI model) can
consume a PreprocessedPair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from ..core.config import S1PreprocessConfig
from ..core.errors import InputDataError
from ..core.georaster import GeoRaster
from ..data_sources.base import LoadedScene
from .grid import AnalysisGrid
from .reproject import reproject_to_grid


# ------------------------------------------------------------------ units

def linear_to_db(x: np.ndarray, floor: float = 1e-5) -> np.ndarray:
    """10*log10(x) with a floor (NaN stays NaN)."""
    with np.errstate(invalid="ignore"):
        return (10.0 * np.log10(np.maximum(x, floor))).astype(np.float32)


def db_to_linear(x: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(x, dtype=np.float64) / 10.0).astype(np.float32)


# ------------------------------------------------------------------ speckle

def _local_mean(x: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    """Mean over a size x size window using only valid pixels (normalised convolution)."""
    xv = np.where(valid, x, 0.0).astype(np.float64)
    num = ndimage.uniform_filter(xv, size=size, mode="nearest")
    den = ndimage.uniform_filter(valid.astype(np.float64), size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-9, num / den, np.nan)


def lee_filter(img: np.ndarray, window: int = 5, enl: float = 4.4) -> np.ndarray:
    """Classic Lee filter for intensity (linear power) images; NaN = nodata (kept NaN).

    out = m + W (x - m),  W = clip(1 - Cu²/Ci², 0, 1),  Cu² = 1/ENL,  Ci² = var/m²
    Homogeneous areas (Ci ≈ Cu) are smoothed to the local mean; edges/targets
    (Ci >> Cu) are preserved.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("Lee window must be an odd integer >= 3")
    valid = np.isfinite(img)
    m = _local_mean(img, valid, window)
    m2 = _local_mean(np.where(valid, img, 0.0) ** 2, valid, window)
    var = np.maximum(m2 - m ** 2, 0.0)
    cu2 = 1.0 / enl
    with np.errstate(invalid="ignore", divide="ignore"):
        ci2 = np.where(m > 0, var / (m ** 2), 0.0)
        w = np.clip(1.0 - cu2 / np.where(ci2 > 0, ci2, np.inf), 0.0, 1.0)
    out = m + w * (img - m)
    out[~valid] = np.nan
    return out.astype(np.float32)


# ------------------------------------------------------------------ pair

@dataclass
class PreprocessedPair:
    before_db: GeoRaster           # bands e.g. ["VV", "VH"], dB, NaN = invalid
    after_db: GeoRaster
    valid: np.ndarray              # bool: valid in both dates and inside AOI
    aoi_mask: np.ndarray           # bool: inside AOI
    nodata_in_aoi: np.ndarray      # bool: inside AOI but invalid in either date
    bands: list[str]
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _scene_to_linear(scene: LoadedScene, bands: Sequence[str]) -> GeoRaster:
    r = scene.raster.select_bands(bands)
    data = r.data.astype(np.float32)
    if r.nodata is not None and not np.isnan(r.nodata):
        data = np.where(data == r.nodata, np.nan, data)
    if scene.ref.metadata.units == "dB":
        data = db_to_linear(data)
    elif scene.ref.metadata.units != "linear":
        raise InputDataError(f"{scene.ref.path}: S1 units must be 'linear' or 'dB', got {scene.ref.metadata.units}")
    data = np.where(np.isfinite(data) & (data > 0), data, np.nan).astype(np.float32)
    return r.with_data(data, band_names=list(bands), nodata=np.nan)


def _prepare_one(scene: LoadedScene, grid: AnalysisGrid, bands: Sequence[str],
                 cfg: S1PreprocessConfig) -> GeoRaster:
    lin = _scene_to_linear(scene, bands)
    lin = reproject_to_grid(lin, grid.spec, "bilinear")
    sf = cfg.speckle_filter
    out = np.empty_like(lin.data)
    for i in range(lin.count):
        band = lin.data[i]
        if sf.enabled:
            if sf.method != "lee":
                raise ValueError(f"Unknown speckle filter {sf.method!r}")
            band = lee_filter(band, sf.window, sf.enl)
        db = linear_to_db(band, cfg.linear_floor)
        lo, hi = cfg.valid_db_range
        db[~np.isfinite(band) | (db < lo) | (db > hi)] = np.nan
        out[i] = db
    return lin.with_data(out, band_names=list(bands), nodata=np.nan,
                         metadata={**lin.metadata, "units": "dB"})


def select_bands(before: LoadedScene, after: LoadedScene, wanted: Sequence[str]) -> list[str]:
    common = [b for b in wanted if b in before.raster.band_names and b in after.raster.band_names]
    if "VV" not in common:
        raise InputDataError(f"VV band required in both scenes; before={before.raster.band_names}, "
                             f"after={after.raster.band_names}")
    return common


def preprocess_s1_pair(before: LoadedScene, after: LoadedScene, grid: AnalysisGrid,
                       cfg: S1PreprocessConfig, bands: Optional[Sequence[str]] = None) -> PreprocessedPair:
    bands = select_bands(before, after, bands or ["VV", "VH"])
    warnings: list[str] = []
    if len(bands) < 2:
        warnings.append(f"Only {bands} available in both scenes; VH-based checks are disabled")
    b = _prepare_one(before, grid, bands, cfg)
    a = _prepare_one(after, grid, bands, cfg)

    vb = np.all(np.isfinite(b.data), axis=0)
    va = np.all(np.isfinite(a.data), axis=0)
    aoi = grid.aoi_mask
    valid = vb & va & aoi
    n_aoi = int(aoi.sum())
    frac_b = float((vb & aoi).sum()) / max(n_aoi, 1)
    frac_a = float((va & aoi).sum()) / max(n_aoi, 1)
    frac = float(valid.sum()) / max(n_aoi, 1)
    if frac == 0:
        raise InputDataError("Before and after images have no valid overlapping pixels inside the AOI")
    if min(frac_b, frac_a) < cfg.min_valid_fraction:
        warnings.append(f"Low valid coverage inside AOI: before {frac_b:.0%}, after {frac_a:.0%} "
                        f"(min_valid_fraction={cfg.min_valid_fraction:.0%}); results cover only part of the AOI")

    b.data[:, ~valid] = np.nan
    a.data[:, ~valid] = np.nan
    stats = {
        "bands": bands,
        "aoi_pixels": n_aoi,
        "valid_fraction_before": round(frac_b, 4),
        "valid_fraction_after": round(frac_a, 4),
        "valid_fraction_both": round(frac, 4),
        "median_db_before": {bn: float(np.nanmedian(b.data[i])) for i, bn in enumerate(bands)},
        "median_db_after": {bn: float(np.nanmedian(a.data[i])) for i, bn in enumerate(bands)},
        "speckle_filter": cfg.speckle_filter.model_dump(),
    }
    return PreprocessedPair(before_db=b, after_db=a, valid=valid, aoi_mask=aoi,
                            nodata_in_aoi=aoi & ~valid, bands=bands, stats=stats, warnings=warnings)
