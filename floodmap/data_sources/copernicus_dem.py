"""Copernicus DEM GLO-30 tile access (public AWS Open Data bucket, no login).

Bucket layout (1°x1° Cloud-Optimised GeoTIFF tiles in EPSG:4326, named by lower-left corner):
    https://copernicus-dem-30m.s3.amazonaws.com/
        Copernicus_DSM_COG_10_N27_00_E085_00_DEM/Copernicus_DSM_COG_10_N27_00_E085_00_DEM.tif

`fetch_dem` downloads the tiles covering AOI + buffer, mosaics and crops them, and writes
one GeoTIFF (EPSG:4326, float32, explicit nodata) plus a provenance sidecar. The pipeline
reprojects it to the analysis grid itself.

Robustness:
  * tiles are streamed to <tile>.partial.tif, fully decoded and checked (1 band, EPSG:4326,
    plausible elevations), then renamed; a corrupt cached tile is detected and re-downloaded;
  * transient network errors are retried with bounded exponential backoff;
  * HTTP 404 = no tile for that 1° cell (open ocean); recorded, not fatal unless nothing remains;
  * gaps are written as nodata (never as 0 m); AOI coverage is checked and recorded;
  * the output DEM is written atomically; the sidecar records each tile's SHA-256.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..core.contracts import AOI, SourceType
from ..core.errors import InputDataError
from ..core.geo_utils import aoi_bounds, buffer_lonlat_bounds_km
from ..core.provenance import sha256_file
from ..core.retry import with_retries

BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
ATTRIBUTION = ("Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space "
               "GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.")
DEM_NODATA = -32767.0
_MIN_ELEV, _MAX_ELEV = -500.0, 9000.0
_CHUNK = 1 << 20


class TileDownloadError(ConnectionError):
    """Tile file truncated/unreadable after download. ConnectionError subclass => retried."""


class TileNotAvailable(Exception):
    """HTTP 404: no GLO-30 tile exists for this 1° cell."""


def tile_name(lat_floor: int, lon_floor: int) -> str:
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def tile_url(name: str, base: str = BASE_URL) -> str:
    return f"{base}/{name}/{name}.tif"


def tiles_for_bounds(b: tuple[float, float, float, float]) -> list[str]:
    min_lon, min_lat, max_lon, max_lat = b
    names = []
    for lat in range(math.floor(min_lat), max(math.ceil(max_lat), math.floor(min_lat) + 1)):
        for lon in range(math.floor(min_lon), max(math.ceil(max_lon), math.floor(min_lon) + 1)):
            names.append(tile_name(lat, lon))
    return names


def snap_bounds(b: tuple[float, float, float, float], res: tuple[float, float],
                origin: tuple[float, float]) -> tuple[float, float, float, float]:
    """Expand lon/lat bounds outward to whole pixels of a grid with resolution `res` anchored at
    `origin` (left, top). Tolerates float noise so exact pixel edges are not expanded."""
    rx, ry = res
    ox, oy = origin
    tol = 1e-6
    left = ox + math.floor((b[0] - ox) / rx + tol) * rx
    right = ox + math.ceil((b[2] - ox) / rx - tol) * rx
    top = oy - math.floor((oy - b[3]) / ry + tol) * ry
    bottom = oy - math.ceil((oy - b[1]) / ry - tol) * ry
    return left, bottom, right, top


def validate_tile(path: Path) -> None:
    """Fully decode a tile; raise TileDownloadError if it is not a usable GLO-30 tile."""
    import rasterio

    try:
        with rasterio.open(path) as src:
            ok_meta = src.count == 1 and src.crs is not None and src.crs.to_epsg() == 4326
            data = src.read(1, masked=True)
    except Exception as e:  # noqa: BLE001 - any read failure means unusable
        raise TileDownloadError(f"{path.name} is not a readable GeoTIFF: {e}") from e
    if not ok_meta:
        raise TileDownloadError(f"{path.name}: expected 1 band in EPSG:4326")
    vals = data.compressed()
    if vals.size == 0 or not np.isfinite(vals).any():
        raise TileDownloadError(f"{path.name} has no valid elevation values")
    finite = vals[np.isfinite(vals)]
    if finite.min() < _MIN_ELEV or finite.max() > _MAX_ELEV:
        raise TileDownloadError(f"{path.name}: elevations outside [{_MIN_ELEV}, {_MAX_ELEV}] m")


def _download_tile(name: str, dest: Path, get: Callable, timeout: int) -> None:
    partial = dest.with_name(dest.stem + ".partial" + dest.suffix)
    partial.unlink(missing_ok=True)
    try:
        resp = get(tile_url(name), timeout=timeout, stream=True)
        if resp.status_code == 404:
            raise TileNotAvailable(name)
        if resp.status_code == 403:
            raise InputDataError(
                f"HTTP 403 for {tile_url(name)}: access denied (tile may not exist, or the network blocks AWS S3)")
        resp.raise_for_status()
        with open(partial, "wb") as f:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if chunk:
                    f.write(chunk)
        validate_tile(partial)
        os.replace(partial, dest)
    finally:
        partial.unlink(missing_ok=True)


def _ensure_tile(name: str, tile_dir: Path, get: Callable, attempts: int, base_delay: float, timeout: int,
                 log: Callable[[str], None]) -> Optional[Path]:
    dest = tile_dir / f"{name}.tif"
    if dest.exists():
        try:
            validate_tile(dest)
            log(f"  cached  {name}")
            return dest
        except TileDownloadError as e:
            log(f"  cached tile invalid ({e}); re-downloading")
            dest.unlink()
    log(f"  download {name}")
    try:
        with_retries(lambda: _download_tile(name, dest, get, timeout), f"tile {name}", attempts, base_delay, log)
    except TileNotAvailable:
        log(f"  no tile {name} (HTTP 404: no land data for this 1° cell)")
        return None
    return dest


def _coverage(mosaic: np.ndarray, transform, aoi_box: tuple[float, float, float, float]) -> float:
    """Fraction of DEM pixels with valid elevation inside the AOI bounding box."""
    from rasterio.transform import rowcol

    r0, c0 = rowcol(transform, aoi_box[0], aoi_box[3])
    r1, c1 = rowcol(transform, aoi_box[2], aoi_box[1])
    r0, c0 = max(r0, 0), max(c0, 0)
    sub = mosaic[0, r0:r1 + 1, c0:c1 + 1]
    if sub.size == 0:
        return 0.0
    return float((np.isfinite(sub) & (sub != DEM_NODATA)).mean())


def fetch_dem(aoi: AOI, out_path: Path, buffer_km: float, get: Optional[Callable] = None,
              log: Callable[[str], None] = print, attempts: int = 4, base_delay: float = 10.0,
              timeout: int = 300) -> Path:
    """Download tiles covering AOI + buffer, mosaic, crop and write `out_path` + sidecar."""
    import rasterio
    from rasterio.merge import merge

    if get is None:
        import requests
        get = requests.get
    out_path = Path(out_path)
    bounds = buffer_lonlat_bounds_km(aoi_bounds(aoi), buffer_km)
    names = tiles_for_bounds(bounds)
    tile_dir = out_path.parent / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    log(f"Copernicus DEM GLO-30: {len(names)} tile(s) for bounds {[round(v, 4) for v in bounds]}")

    paths, missing = [], []
    for n in names:
        p = _ensure_tile(n, tile_dir, get, attempts, base_delay, timeout, log)
        (paths if p is not None else missing).append(p if p is not None else n)
    if not paths:
        raise InputDataError(f"No Copernicus DEM tiles available for bounds {bounds} (all returned 404)")

    srcs = [rasterio.open(p) for p in paths]
    try:
        # Snap the crop outward to the tiles' native pixel grid: the mosaic is then an exact,
        # pixel-aligned copy of the source (no sub-pixel resampling, no dropped edge rows).
        merge_bounds = snap_bounds(bounds, srcs[0].res, (srcs[0].bounds.left, srcs[0].bounds.top))
        mosaic, transform = merge(srcs, bounds=merge_bounds, nodata=DEM_NODATA, dtype="float32")
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()
    mosaic = mosaic.astype(np.float32)
    mosaic[~np.isfinite(mosaic)] = DEM_NODATA
    cover = _coverage(mosaic, transform, aoi_bounds(aoi))
    if cover < 0.99:
        log(f"  WARNING: DEM covers only {cover:.1%} of the AOI bounding box (missing tiles: {missing})")

    profile = {"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "count": 1,
               "dtype": "float32", "crs": crs, "transform": transform, "nodata": DEM_NODATA,
               "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.stem + ".partial" + out_path.suffix)
    try:
        with rasterio.open(partial, "w", **profile) as dst:
            dst.write(mosaic)
            dst.set_band_description(1, "elevation")
        os.replace(partial, out_path)
    finally:
        partial.unlink(missing_ok=True)

    meta: dict[str, Any] = {
        "source_type": SourceType.COPERNICUS_DEM.value,
        "product": "Copernicus DEM GLO-30 (DSM)",
        "tiles": [{"name": p.stem, "url": tile_url(p.stem), "sha256": sha256_file(p)} for p in paths],
        "missing_tiles": missing,
        "bounds_lonlat": list(bounds),
        "aoi_bounds_lonlat": list(aoi_bounds(aoi)),
        "buffer_km": buffer_km,
        "aoi_coverage_fraction": round(cover, 4),
        "crs": str(crs), "nodata": DEM_NODATA, "units": "m",
        "source_url": BASE_URL,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attribution": ATTRIBUTION,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"  wrote {out_path} ({mosaic.shape[2]} x {mosaic.shape[1]} px, AOI coverage {cover:.1%})")
    return out_path


def existing_dem_covers(out_path: Path, aoi: AOI, buffer_km: float) -> bool:
    """True if `out_path` + sidecar exist, the DEM is readable and its recorded bounds cover AOI + buffer."""
    side = Path(out_path).with_suffix(".meta.json")
    if not (Path(out_path).exists() and side.exists()):
        return False
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
        b = meta["bounds_lonlat"]
        import rasterio
        with rasterio.open(out_path) as src:
            src.read(1, out_shape=(1, max(1, src.height // 16), max(1, src.width // 16)))
    except Exception:  # noqa: BLE001
        return False
    need = buffer_lonlat_bounds_km(aoi_bounds(aoi), buffer_km)
    eps = 1e-9
    return (meta.get("source_type") == SourceType.COPERNICUS_DEM.value and b[0] <= need[0] + eps
            and b[1] <= need[1] + eps and b[2] >= need[2] - eps and b[3] >= need[3] - eps)
