"""Warp any GeoRaster onto a GridSpec (rasterio.warp).

Rules that matter scientifically:
  * SAR backscatter must be resampled in LINEAR power (averaging dB values biases
    the result). `preprocess.py` converts dB->linear before calling this.
  * Categorical rasters (masks, classes, flags) use nearest neighbour.
  * Nodata stays nodata: output is NaN (float) / `dst_nodata` (integer) where the
    source has no data or does not cover the grid.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..core.georaster import GeoRaster
from .grid import GridSpec

CONTINUOUS = ("bilinear", "cubic", "average")
CATEGORICAL = ("nearest", "mode")


def reproject_to_grid(
    src: GeoRaster,
    grid: GridSpec,
    resampling: str = "bilinear",
    dst_nodata: Optional[float] = None,
) -> GeoRaster:
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    if resampling not in CONTINUOUS + CATEGORICAL:
        raise ValueError(f"Unsupported resampling {resampling!r}")
    floating = np.issubdtype(src.data.dtype, np.floating) or resampling in CONTINUOUS
    if floating:
        dtype = np.float32 if src.data.dtype != np.float64 else np.float64
        dst_nodata = np.nan
        src_data = src.data.astype(dtype, copy=False)
        src_nodata = src.nodata if src.nodata is not None else np.nan
        if src_nodata is not None and not (isinstance(src_nodata, float) and math.isnan(src_nodata)):
            # normalise source nodata to NaN so GDAL ignores it during interpolation
            src_data = np.where(src_data == src_nodata, np.nan, src_data).astype(dtype)
            src_nodata = np.nan
    else:
        dtype = src.data.dtype
        if dst_nodata is None:
            dst_nodata = src.nodata if src.nodata is not None else 255
        src_data = src.data
        src_nodata = src.nodata

    # GDAL treats dst_nodata as a source mask when the source declares none, which would
    # silently erase valid pixels equal to dst_nodata (e.g. warping a 0/1 mask with
    # dst_nodata=0). For categorical data, warp with a sentinel absent from the valid
    # data, then map the sentinel to the requested nodata value.
    warp_nodata = dst_nodata
    if not floating:
        present = set(np.unique(src_data).tolist())
        if src_nodata is not None:
            present.discard(src_nodata)
        if dst_nodata in present:
            info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
            candidates = range(int(info.max), int(info.min) - 1, -1) if info else []
            warp_nodata = next((v for v in candidates if v not in present), None)
            if warp_nodata is None:
                raise ValueError("No free value in dtype to use as warp nodata")
        if src_nodata is None:
            src_nodata = warp_nodata

    out = np.full((src.count, grid.height, grid.width), warp_nodata, dtype=dtype)
    reproject(
        source=src_data,
        destination=out,
        src_transform=Affine(*src.transform),
        src_crs=src.crs,
        src_nodata=src_nodata,
        dst_transform=Affine(*grid.transform),
        dst_crs=grid.crs,
        dst_nodata=warp_nodata,
        resampling=getattr(Resampling, resampling),
    )
    if warp_nodata != dst_nodata:
        out[out == warp_nodata] = dst_nodata
    return GeoRaster(out, grid.transform, grid.crs, nodata=dst_nodata,
                     band_names=list(src.band_names), metadata=dict(src.metadata))
