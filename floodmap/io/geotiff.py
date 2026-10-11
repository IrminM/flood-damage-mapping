"""GeoTIFF <-> GeoRaster I/O (rasterio).

This is the only module in the core that talks to rasterio/GDAL for raster files,
so the rest of the pipeline stays testable on plain numpy GeoRasters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..core.errors import InputDataError
from ..core.georaster import GeoRaster


def _rasterio():
    try:
        import rasterio  # noqa: WPS433
        return rasterio
    except ImportError as e:  # pragma: no cover - environment issue
        raise ImportError("rasterio is required for GeoTIFF I/O: pip install -r requirements.txt") from e


def read_geotiff(path: Path | str, bands: Optional[Sequence[int]] = None) -> GeoRaster:
    """Read a GeoTIFF into a GeoRaster. `bands` are 1-based indexes (default: all)."""
    rasterio = _rasterio()
    path = Path(path)
    if not path.exists():
        raise InputDataError(f"Raster not found: {path}")
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise InputDataError(f"{path} has no CRS; georeferenced input is required")
            idx = list(bands) if bands else list(range(1, src.count + 1))
            data = src.read(idx)
            descs = [src.descriptions[i - 1] for i in idx]
            names = [d if d else f"band_{i}" for d, i in zip(descs, idx)]
            t = src.transform
            return GeoRaster(
                data=data,
                transform=(t.a, t.b, t.c, t.d, t.e, t.f),
                crs=src.crs.to_string(),
                nodata=src.nodata,
                band_names=names,
                metadata={"source_path": str(path), "driver": src.driver, "tags": src.tags()},
            )
    except rasterio.errors.RasterioError as e:
        raise InputDataError(f"Cannot read raster {path}: {e}") from e


def write_geotiff(raster: GeoRaster, path: Path | str, compress: str = "deflate") -> Path:
    """Write a GeoRaster to a tiled, compressed GeoTIFF (band names stored as descriptions)."""
    rasterio = _rasterio()
    from rasterio.transform import Affine

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = raster.data
    if data.dtype == np.bool_:
        data = data.astype(np.uint8)
    profile = {
        "driver": "GTiff",
        "height": raster.height,
        "width": raster.width,
        "count": raster.count,
        "dtype": data.dtype.name,
        "crs": raster.crs,
        "transform": Affine(*raster.transform),
        "compress": compress,
    }
    if raster.height >= 256 and raster.width >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    if raster.nodata is not None:
        profile["nodata"] = raster.nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, name in enumerate(raster.band_names, start=1):
            dst.set_band_description(i, name)
    return path
