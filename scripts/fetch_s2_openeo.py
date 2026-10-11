#!/usr/bin/env python
"""Download AOI-clipped Sentinel-2 Level-1C 13-band reflectance via Copernicus Data Space
openEO service, resample to 10m nearest-neighbour, and align to the reference Sentinel-1 grid.

Outputs:
    <out>/s2_after_20260827_l1c.tif
    <out>/s2_after_20260827_l1c.meta.json
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

import _bootstrap  # noqa: F401

from floodmap.core.contracts import AOI, DataMode, SceneMetadata, SourceType
from floodmap.core.geo_utils import aoi_bounds, buffer_lonlat_bounds_km, load_aoi
from floodmap.data_sources.satellite_local import write_scene_metadata

OPENEO_URL = "openeo.dataspace.copernicus.eu"
TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
JOB_DONE, JOB_FAILED = {"finished"}, {"error", "canceled"}

S2_L1C_COLLECTION_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B10", "B11", "B12"
]
S2_TARGET_BAND_NAMES = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
]
NATIVE_RESOLUTIONS = {
    "B1": 60, "B2": 10, "B3": 10, "B4": 10, "B5": 20, "B6": 20,
    "B7": 20, "B8": 10, "B8A": 20, "B9": 60, "B10": 60, "B11": 20, "B12": 20
}

_sleep: Callable[[float], None] = time.sleep
_monotonic: Callable[[], float] = time.monotonic


class IncompleteDownload(ConnectionError):
    pass


class EmptyDownload(RuntimeError):
    pass


class JobFailed(RuntimeError):
    pass


class JobTimeout(RuntimeError):
    pass


def connect():
    import openeo
    conn = openeo.connect(OPENEO_URL)
    cid, secret = os.environ.get("CDSE_OPENEO_CLIENT_ID"), os.environ.get("CDSE_OPENEO_CLIENT_SECRET")
    if cid and secret:
        conn.authenticate_oidc_client_credentials(client_id=cid, client_secret=secret)
    else:
        conn.authenticate_oidc()
    return conn


def is_transient(exc: BaseException) -> bool:
    retry_types: tuple = (ConnectionError, TimeoutError)
    try:
        import requests
        retry_types += (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.Timeout)
    except ImportError:
        pass
    if isinstance(exc, retry_types):
        return True
    code = getattr(exc, "http_status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code in TRANSIENT_HTTP_CODES


def with_retries(fn: Callable[[], Any], what: str, attempts: int, base_delay: float,
                 log: Callable[[str], None] = print) -> Any:
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:
            if not is_transient(e) or k == attempts - 1:
                raise
            delay = base_delay * (2 ** k)
            log(f"  {what}: {type(e).__name__}: {e} -- retry {k + 1}/{attempts - 1} in {delay:.0f} s")
            _sleep(delay)
    raise AssertionError("unreachable")


def partial_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + ".partial" + out_path.suffix)


def job_state_path(out_path: Path) -> Path:
    return out_path.with_suffix(".job.json")


def build_s2_cube(conn, aoi: AOI, target_date: date, relative_orbit: int = 119, buffer_km: float = 2.0):
    w, s, e, n = buffer_lonlat_bounds_km(aoi_bounds(aoi), buffer_km)
    cube = conn.load_collection(
        "SENTINEL2_L1C",
        spatial_extent={"west": w, "south": s, "east": e, "north": n},
        temporal_extent=[target_date.isoformat(), (target_date + timedelta(days=1)).isoformat()],
        bands=S2_L1C_COLLECTION_BANDS,
        properties={"sat:relative_orbit": lambda v, r=relative_orbit: v == r} if relative_orbit else None
    )
    cube = cube.resample_spatial(resolution=10.0, projection=32645, method="near")
    cube = cube.reduce_dimension(dimension="t", reducer="mean")
    return cube


def _get_or_create_job(conn, out_path: Path, make_cube: Callable[[], Any], title: str,
                       attempts: int, base_delay: float, log: Callable[[str], None]):
    state = job_state_path(out_path)
    if state.exists():
        try:
            saved = json.loads(state.read_text(encoding="utf-8"))
        except ValueError:
            saved = {}
        if saved.get("job_id"):
            job = conn.job(saved["job_id"])
            status = with_retries(job.status, "job status", attempts, base_delay, log)
            if status not in JOB_FAILED:
                log(f"  reusing openEO batch job {saved['job_id']} (status: {status})")
                return job
            log(f"  previous job {saved['job_id']} ended '{status}'; creating new job")
    cube = make_cube()
    job = with_retries(lambda: cube.create_job(title=title, out_format="GTiff"), "create job",
                       attempts, base_delay, log)
    state.write_text(json.dumps({"job_id": job.job_id, "title": title}, indent=2), encoding="utf-8")
    log(f"  created openEO batch job {job.job_id}")
    return job


def _wait_for_job(job, attempts: int, base_delay: float, poll_interval: float, timeout_s: float,
                  log: Callable[[str], None]) -> None:
    status = with_retries(job.status, "job status", attempts, base_delay, log)
    if status == "created":
        with_retries(job.start, "start job", attempts, base_delay, log)
    deadline = _monotonic() + timeout_s
    last = None
    while True:
        status = with_retries(job.status, "job status", attempts, base_delay, log)
        if status != last:
            log(f"  job {job.job_id}: {status}")
            last = status
        if status in JOB_DONE:
            return
        if status in JOB_FAILED:
            raise JobFailed(f"openEO batch job {job.job_id} ended '{status}'")
        if _monotonic() > deadline:
            raise JobTimeout(f"openEO batch job {job.job_id} still '{status}' after {timeout_s / 60:.0f} min")
        _sleep(poll_interval)


def _tif_asset(job, attempts: int, base_delay: float, log: Callable[[str], None]):
    results = with_retries(job.get_results, "get results", attempts, base_delay, log)
    assets = with_retries(results.get_assets, "list result assets", attempts, base_delay, log)
    tifs = [a for a in assets
            if str(a.name).lower().endswith((".tif", ".tiff")) or "tiff" in str((a.metadata or {}).get("type", ""))]
    if len(tifs) != 1:
        raise RuntimeError(f"expected exactly one GeoTIFF result from job {job.job_id}, got {[a.name for a in assets]}")
    return tifs[0]


def align_to_reference_grid(raw_s2_path: Path, ref_s1_path: Path, final_out_path: Path) -> dict[str, Any]:
    """Reproject/crop downloaded S2 GeoTIFF to the exact pixel grid and affine transform of S1."""
    with rasterio.open(ref_s1_path) as s1:
        ref_crs = s1.crs
        ref_transform = s1.transform
        ref_height = s1.height
        ref_width = s1.width
        ref_bounds = s1.bounds

    with rasterio.open(raw_s2_path) as src:
        src_count = src.count
        if src_count != 13:
            raise ValueError(f"Expected 13 bands from openEO S2 download, got {src_count}")
        
        aligned_data = np.zeros((13, ref_height, ref_width), dtype=np.uint16)
        
        for b in range(1, 14):
            src_band = src.read(b)
            # Clip negative/NaNs if any and cast to float for reprojection
            src_band_clean = np.nan_to_num(src_band, nan=0.0, posinf=65535.0, neginf=0.0)
            dst_band = np.zeros((ref_height, ref_width), dtype=np.float32)
            
            reproject(
                source=src_band_clean,
                destination=dst_band,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
                src_nodata=0,
                dst_nodata=0,
            )
            # Clip to valid uint16 reflectance range [0, 65535]
            aligned_data[b - 1] = np.clip(np.round(dst_band), 0, 65535).astype(np.uint16)

    # Write aligned final GeoTIFF
    profile = {
        "driver": "GTiff",
        "height": ref_height,
        "width": ref_width,
        "count": 13,
        "dtype": "uint16",
        "crs": ref_crs,
        "transform": ref_transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    final_out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = final_out_path.with_suffix(".tmp.tif")
    with rasterio.open(temp_target, "w", **profile) as dst:
        dst.write(aligned_data)
        for i, name in enumerate(S2_TARGET_BAND_NAMES, start=1):
            dst.set_band_description(i, name)
            dst.update_tags(i, native_resolution_m=str(NATIVE_RESOLUTIONS[name]))
        dst.update_tags(
            AREA_OR_POINT="Area",
            RESAMPLING_METHOD="nearest-neighbour",
            SOURCE_PRODUCT="S2B_MSIL1C_20260827T045659_N0512_R119_T45RUM_20260827T083139",
            CO_REGISTRATION="exact_grid_match_to_s1",
        )
    os.replace(temp_target, final_out_path)

    return {
        "height": ref_height,
        "width": ref_width,
        "crs": str(ref_crs),
        "bounds": [ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top],
        "transform": list(ref_transform)[:6],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aoi", default="config/aoi_examples/trishuli_corridor_dev.geojson")
    ap.add_argument("--date", default="2026-08-27")
    ap.add_argument("--orbit", type=int, default=119)
    ap.add_argument("--ref-s1", default="data/raw/dev/s1_after_20260828.tif")
    ap.add_argument("--out", default="data/raw/dev/s2_after_20260827_l1c.tif")
    ap.add_argument("--job-id", default="j-2610071957014bdb8c4f2eb833f9794e", help="Optional existing job ID")
    args = ap.parse_args()

    aoi = load_aoi(args.aoi)
    target_date = date.fromisoformat(args.date)
    out_path = Path(args.out)
    ref_path = Path(args.ref_s1)

    print(f"Connecting to CDSE openEO ({OPENEO_URL})...")
    conn = connect()

    # Pre-seed existing job state if given
    state_file = job_state_path(out_path)
    if args.job_id and not state_file.exists():
        state_file.write_text(json.dumps({"job_id": args.job_id, "title": "s2_l1c_trishuli"}, indent=2), encoding="utf-8")

    make_cube = lambda: build_s2_cube(conn, aoi, target_date, relative_orbit=args.orbit)
    job = _get_or_create_job(conn, out_path, make_cube, "s2_l1c_trishuli", attempts=4, base_delay=15.0, log=print)

    print(f"Waiting for openEO job {job.job_id} to finish...")
    _wait_for_job(job, attempts=4, base_delay=15.0, poll_interval=20.0, timeout_s=3600, log=print)

    print("Fetching finished GeoTIFF asset...")
    asset = _tif_asset(job, attempts=4, base_delay=15.0, log=print)
    raw_download_path = partial_path(out_path)
    asset.download(raw_download_path)
    print(f"Downloaded raw asset to {raw_download_path} ({raw_download_path.stat().st_size} bytes)")

    print(f"Aligning to reference S1 grid ({ref_path})...")
    grid_info = align_to_reference_grid(raw_download_path, ref_path, out_path)
    raw_download_path.unlink(missing_ok=True)
    state_file.unlink(missing_ok=True)
    print(f"Successfully generated aligned GeoTIFF: {out_path}")

    # Write accompanying SceneMetadata sidecar
    meta = SceneMetadata(
        sensor="S2",
        platform="sentinel-2b",
        product_type="L1C",
        acquisition_time=datetime(2026, 8, 27, 4, 56, 59, 24000, tzinfo=timezone.utc),
        bands=S2_TARGET_BAND_NAMES,
        units="reflectance",
        processing=(
            "Copernicus Data Space Ecosystem openEO SENTINEL2_L1C resampled 10.0 m "
            "nearest-neighbour to EPSG:32645, co-registered and aligned to s1_after_20260828.tif"
        ),
        source_type=SourceType.SENTINEL_2,
        relative_orbit=args.orbit,
        orbit_direction="descending",
        product_ids=["S2B_MSIL1C_20260827T045659_N0512_R119_T45RUM_20260827T083139"],
        cloud_cover_pct=65.6,
        data_mode=DataMode.REAL,
        extra={
            "cdse_product_name": "S2B_MSIL1C_20260827T045659_N0512_R119_T45RUM_20260827T083139",
            "cdse_id": "b39da178-566b-4b7b-bda6-103f61662188",
            "tile": "T45RUM",
            "product_level": "Level-1C",
            "source_crs": "EPSG:32645",
            "target_crs": grid_info["crs"],
            "source_native_resolutions": NATIVE_RESOLUTIONS,
            "target_resolution_m": 10.0,
            "resampling_method": "nearest-neighbour",
            "target_raster_dimensions": [grid_info["height"], grid_info["width"]],
            "target_bounds": grid_info["bounds"],
            "target_affine_transform": grid_info["transform"],
            "band_order": S2_TARGET_BAND_NAMES,
            "qi_data_cloud_mask_availability": "available_in_product",
            "openeo_job_id": job.job_id,
            "temporal_pairing_note": "Post-event optical scene (1 day prior to post-event S1 on 2026-08-28)",
        }
    )
    sidecar = write_scene_metadata(out_path, meta)
    print(f"Wrote metadata sidecar: {sidecar}")


if __name__ == "__main__":
    main()
