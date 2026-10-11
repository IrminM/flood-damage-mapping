"""GeoRaster: the single in-memory raster type used across the pipeline.

A GeoRaster is a numpy array (bands, rows, cols) plus the georeferencing needed to
place it on Earth: an affine transform, a CRS identifier, a nodata value and band names.

It is deliberately pure numpy (no rasterio/GDAL) so that preprocessing, change
detection, fusion and metrics can be unit-tested anywhere. File I/O lives in
floodmap/io/geotiff.py.

Transform convention (same as rasterio/affine, GDAL-ordered as a,b,c,d,e,f):
    x = a * col + b * row + c
    y = d * col + e * row + f
Only north-up rasters (b == d == 0) are supported; Sentinel products delivered
on a map grid (UTM/EPSG:4326) satisfy this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .errors import GridMismatchError

Transform = tuple[float, float, float, float, float, float]
Bounds = tuple[float, float, float, float]  # (minx, miny, maxx, maxy)


def normalize_crs(crs: str) -> str:
    """Canonical text form for simple CRS comparison ("epsg:32645" -> "EPSG:32645")."""
    c = str(crs).strip()
    if c.lower().startswith("epsg:"):
        return "EPSG:" + c.split(":", 1)[1].strip()
    return c


@dataclass
class GeoRaster:
    data: np.ndarray
    transform: Transform
    crs: str
    nodata: Optional[float] = None
    band_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.data)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3:
            raise ValueError(f"GeoRaster data must be 2-D or 3-D (bands, rows, cols); got shape {arr.shape}")
        self.data = arr
        if len(self.transform) != 6:
            raise ValueError("transform must have 6 elements (a, b, c, d, e, f)")
        self.transform = tuple(float(v) for v in self.transform)  # type: ignore[assignment]
        a, b, _, d, e, _ = self.transform
        if b != 0 or d != 0:
            raise ValueError("Rotated rasters are not supported (transform b and d must be 0)")
        if a == 0 or e == 0:
            raise ValueError("Pixel size cannot be zero")
        self.crs = normalize_crs(self.crs)
        if not self.band_names:
            self.band_names = [f"band_{i + 1}" for i in range(arr.shape[0])]
        if len(self.band_names) != arr.shape[0]:
            raise ValueError(f"{len(self.band_names)} band names for {arr.shape[0]} bands")

    # ------------------------------------------------------------ geometry
    @property
    def count(self) -> int:
        return self.data.shape[0]

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def res(self) -> tuple[float, float]:
        """Pixel size (x, y) as positive numbers in CRS units."""
        return abs(self.transform[0]), abs(self.transform[4])

    @property
    def pixel_area(self) -> float:
        """Area of one pixel in CRS units² (m² for a projected CRS such as UTM)."""
        return self.res[0] * self.res[1]

    @property
    def bounds(self) -> Bounds:
        a, _, c, _, e, f = self.transform
        x0, x1 = c, c + a * self.width
        y0, y1 = f, f + e * self.height
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def xy(self, row: float, col: float) -> tuple[float, float]:
        """CRS coordinates of a pixel *centre*."""
        a, _, c, _, e, f = self.transform
        return c + a * (col + 0.5), f + e * (row + 0.5)

    def rowcol(self, x: float, y: float) -> tuple[int, int]:
        """Pixel (row, col) containing CRS coordinate (x, y). May be outside the raster."""
        a, _, c, _, e, f = self.transform
        return int(math.floor((y - f) / e)), int(math.floor((x - c) / a))

    # ------------------------------------------------------------ data access
    def band(self, name: str) -> np.ndarray:
        try:
            return self.data[self.band_names.index(name)]
        except ValueError:
            raise KeyError(f"Band {name!r} not in {self.band_names}") from None

    def valid_mask(self) -> np.ndarray:
        """(rows, cols) bool: True where *all* bands are finite and not nodata."""
        d = self.data
        valid = np.ones(self.shape, dtype=bool)
        if np.issubdtype(d.dtype, np.floating):
            valid &= np.all(np.isfinite(d), axis=0)
        if self.nodata is not None and not (isinstance(self.nodata, float) and math.isnan(self.nodata)):
            valid &= np.all(d != self.nodata, axis=0)
        return valid

    def with_data(
        self,
        data: np.ndarray,
        band_names: Optional[Sequence[str]] = None,
        nodata: Optional[float] = ...,  # type: ignore[assignment]
        metadata: Optional[dict[str, Any]] = None,
    ) -> "GeoRaster":
        """New raster on the *same grid* with different pixel values (derived products)."""
        arr = np.asarray(data)
        if arr.shape[-2:] != self.shape:
            raise GridMismatchError(f"Data shape {arr.shape[-2:]} does not match grid {self.shape}")
        nbands = 1 if arr.ndim == 2 else arr.shape[0]
        if band_names:
            names = list(band_names)
        elif nbands == self.count:
            names = list(self.band_names)   # same band layout -> keep names (e.g. VV/VH)
        else:
            names = [f"band_{i + 1}" for i in range(nbands)]
        return GeoRaster(
            data=arr,
            transform=self.transform,
            crs=self.crs,
            nodata=self.nodata if nodata is ... else nodata,
            band_names=names,
            metadata=dict(self.metadata if metadata is None else metadata),
        )

    def select_bands(self, names: Iterable[str]) -> "GeoRaster":
        names = list(names)
        idx = [self.band_names.index(n) for n in names]
        return self.with_data(self.data[idx], band_names=names)

    # ------------------------------------------------------------ grids
    def same_grid(self, other: "GeoRaster", rel_tol: float = 1e-6) -> bool:
        """True if both rasters share CRS, shape and transform (within tolerance)."""
        if self.crs != other.crs or self.shape != other.shape:
            return False
        scale = max(self.res)
        return all(
            abs(p - q) <= rel_tol * max(1.0, abs(p), abs(q), scale)
            for p, q in zip(self.transform, other.transform)
        )

    def require_same_grid(self, other: "GeoRaster", what: str = "rasters") -> None:
        if not self.same_grid(other):
            raise GridMismatchError(
                f"{what} are not on the same grid: "
                f"{self.crs} {self.shape} {self.transform} vs {other.crs} {other.shape} {other.transform}"
            )

    def crop_to_bounds(self, bounds: Bounds) -> "GeoRaster":
        """Crop to the pixel window covering `bounds` (same CRS), snapped outward to whole pixels."""
        minx, miny, maxx, maxy = bounds
        a, _, c, _, e, f = self.transform
        cols = sorted(((minx - c) / a, (maxx - c) / a))
        rows = sorted(((maxy - f) / e, (miny - f) / e))
        c0 = max(0, int(math.floor(cols[0] + 1e-9)))
        c1 = min(self.width, int(math.ceil(cols[1] - 1e-9)))
        r0 = max(0, int(math.floor(rows[0] + 1e-9)))
        r1 = min(self.height, int(math.ceil(rows[1] - 1e-9)))
        if c1 <= c0 or r1 <= r0:
            raise ValueError(f"Bounds {bounds} do not overlap raster bounds {self.bounds}")
        new_t: Transform = (a, 0.0, c + a * c0, 0.0, e, f + e * r0)
        return GeoRaster(
            data=self.data[:, r0:r1, c0:c1].copy(),
            transform=new_t,
            crs=self.crs,
            nodata=self.nodata,
            band_names=list(self.band_names),
            metadata=dict(self.metadata),
        )

    def __repr__(self) -> str:
        return (f"GeoRaster(bands={self.band_names}, shape={self.shape}, crs={self.crs}, "
                f"res={self.res}, dtype={self.data.dtype}, nodata={self.nodata})")
