#!/usr/bin/env python
"""Download AOI-clipped Sentinel-1 VV/VH backscatter for a before/after pair via the
Copernicus Data Space openEO service, and write the metadata sidecars the pipeline requires.

Steps:
    1. python scripts/search_s1_scenes.py ... --out data/raw/my_aoi/s1_candidates.json
    2. python scripts/fetch_s1_openeo.py --candidates data/raw/my_aoi/s1_candidates.json \
           --pair 0 --out data/raw/my_aoi
       (first run opens a browser/device login for your free CDSE account)

Outputs:
    <out>/s1_before_<date>.tif + .meta.json
    <out>/s1_after_<date>.tif  + .meta.json

Backscatter is LINEAR power, coefficient sigma0-ellipsoid (the only value the CDSE
backend accepts). sigma0-ellipsoid is NOT radiometrically terrain-flattened: slopes
facing the radar look brighter and slopes facing away look darker. Stage 2 handles this
with its DEM-based shadow/layover and slope/HAND gates, and the coefficient is recorded
in each scene sidecar (-> provenance). Both scenes of a pair always use the same
coefficient.

Download modes:
  batch (default) — an openEO batch job computes the scene server-side; the script polls
         its status with short requests and then downloads the finished GeoTIFF as a file.
         A dropped connection only costs one poll or one file-download attempt, never the
         computation. The job id is saved in <name>.job.json, so a rerun after a failed
         download reuses the finished job instead of recomputing it.
  sync  — one long HTTP request that computes and streams the result (openEO
         `download`). A connection reset loses the whole computation; kept as a fallback.

Robustness (both modes):
  * transient network errors (connection reset, timeouts, HTTP 429/5xx) are retried a
    limited number of times with exponential backoff; other errors fail immediately;
  * each scene is downloaded to <name>.partial.tif, fully decoded and checked (band count,
    not empty), and only then renamed to <name>.tif — an interrupted or truncated download
    can never look like a finished scene;
  * scenes already complete with the same coefficient are skipped on rerun
    (use --overwrite to force a fresh download).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import _bootstrap  # noqa: F401

from floodmap.core.contracts import AOI, SceneMetadata
from floodmap.core.geo_utils import analysis_crs_for_aoi, aoi_bounds, buffer_lonlat_bounds_km
from floodmap.data_sources.satellite_local import write_scene_metadata

OPENEO_URL = "openeo.dataspace.copernicus.eu"
# Only coefficient supported by CDSE openEO sar_backscatter (live backend response, Oct 2026).
DEFAULT_COEFFICIENT = "sigma0-ellipsoid"

TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
JOB_DONE, JOB_FAILED = {"finished"}, {"error", "canceled"}

# Indirection so tests can replace time without real waiting.
_sleep: Callable[[float], None] = time.sleep
_monotonic: Callable[[], float] = time.monotonic


class IncompleteDownload(ConnectionError):
    """Downloaded file is truncated/unreadable. Subclass of ConnectionError => retried."""


class EmptyDownload(RuntimeError):
    """Download is complete but contains no valid backscatter. Not retried."""


class JobFailed(RuntimeError):
    """The openEO batch job ended in 'error' or 'canceled'."""


class JobTimeout(RuntimeError):
    """The openEO batch job did not finish within --job-timeout-min."""


def connect():
    import openeo
    conn = openeo.connect(OPENEO_URL)
    cid, secret = os.environ.get("CDSE_OPENEO_CLIENT_ID"), os.environ.get("CDSE_OPENEO_CLIENT_SECRET")
    if cid and secret:
        conn.authenticate_oidc_client_credentials(client_id=cid, client_secret=secret)
    else:
        conn.authenticate_oidc()  # interactive device/browser login; token is cached locally
    return conn


# ------------------------------------------------------------------ retry policy

def is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying: dropped/reset connections, timeouts, HTTP 429/5xx."""
    retry_types: tuple = (ConnectionError, TimeoutError)        # includes ConnectionResetError (10054)
    try:
        import requests
        retry_types += (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.Timeout)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(exc, retry_types):
        return True
    code = getattr(exc, "http_status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code in TRANSIENT_HTTP_CODES


def with_retries(fn: Callable[[], Any], what: str, attempts: int, base_delay: float,
                 log: Callable[[str], None] = print) -> Any:
    """Call fn(); on a transient error wait base_delay * 2**k and retry, at most `attempts` calls."""
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - classified below
            if not is_transient(e) or k == attempts - 1:
                raise
            delay = base_delay * (2 ** k)
            log(f"  {what}: {type(e).__name__}: {e} -- retry {k + 1}/{attempts - 1} in {delay:.0f} s")
            _sleep(delay)
    raise AssertionError("unreachable")


# ------------------------------------------------------------------ files

def partial_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + ".partial" + out_path.suffix)


def job_state_path(out_path: Path) -> Path:
    return out_path.with_suffix(".job.json")


def processing_text(coefficient: str, resolution: float, crs: str) -> str:
    return (f"CDSE openEO SENTINEL1_GRD sar_backscatter(coefficient={coefficient}), "
            f"resampled {resolution} m to {crs}")


def is_complete(out_path: Path, coefficient: str) -> bool:
    """A scene counts as complete if the GeoTIFF and its sidecar exist and the sidecar
    records the same backscatter coefficient (never mix coefficients within a pair)."""
    side = out_path.with_suffix(".meta.json")
    if not (out_path.exists() and side.exists()):
        return False
    try:
        processing = json.loads(side.read_text(encoding="utf-8")).get("processing", "")
    except (OSError, ValueError):
        return False
    return f"coefficient={coefficient})" in processing


def validate_geotiff(path: Path, expected_bands: int) -> None:
    """Fully decode the file: catches truncated transfers (retried) and empty results (not retried)."""
    import numpy as np
    import rasterio

    try:
        with rasterio.open(path) as src:
            count = src.count
            data = src.read()
            nodata = src.nodata
    except Exception as e:  # noqa: BLE001 - any read failure means the file is not usable
        raise IncompleteDownload(f"downloaded file {path.name} is not a readable GeoTIFF: {e}") from e
    if count != expected_bands:
        raise IncompleteDownload(f"downloaded file has {count} bands, expected {expected_bands}")
    valid = np.isfinite(data) & (data > 0)
    if nodata is not None and np.isfinite(nodata):
        valid &= data != nodata
    if not valid.any():
        raise EmptyDownload(
            f"downloaded {path.name} contains no valid backscatter. The orbit/date filter may not "
            "match any acquisition over the AOI; check the pair in s1_candidates.json")


# ------------------------------------------------------------------ openEO

def requested_bands(scene: SceneMetadata) -> list[str]:
    return [b for b in scene.bands if b in ("VV", "VH")] or ["VV", "VH"]


def build_cube(conn, scene: SceneMetadata, aoi: AOI, coefficient: str, resolution: float, buffer_km: float):
    """openEO process graph for one scene (unchanged logic: collection, orbit filters,
    sar_backscatter coefficient, resampling to the AOI's UTM zone, single-date reduction)."""
    w, s, e, n = buffer_lonlat_bounds_km(aoi_bounds(aoi), buffer_km)
    day = scene.acquisition_time.date()
    epsg = int(analysis_crs_for_aoi(aoi).split(":")[1])
    properties = {}
    if scene.relative_orbit is not None:
        properties["sat:relative_orbit"] = lambda v, r=scene.relative_orbit: v == r
    if scene.orbit_direction:
        properties["sat:orbit_state"] = lambda v, d=scene.orbit_direction: v == d
    cube = conn.load_collection(
        "SENTINEL1_GRD",
        spatial_extent={"west": w, "south": s, "east": e, "north": n},
        temporal_extent=[day.isoformat(), (day + timedelta(days=1)).isoformat()],
        bands=requested_bands(scene),
        properties=properties or None,
    )
    cube = cube.sar_backscatter(coefficient=coefficient)
    cube = cube.resample_spatial(resolution=resolution, projection=epsg)
    cube = cube.reduce_dimension(dimension="t", reducer="mean")  # one acquisition in the window
    return cube


def _job_logs_tail(job, n: int = 5) -> str:
    try:
        logs = list(job.logs())[-n:]
        return " | ".join(str(getattr(l, "message", None) or (l.get("message") if hasattr(l, "get") else l))
                          for l in logs)
    except Exception:  # noqa: BLE001 - logs are best-effort diagnostics
        return "(job logs unavailable)"


def _get_or_create_job(conn, out_path: Path, coefficient: str, make_cube: Callable[[], Any], title: str,
                       attempts: int, base_delay: float, log: Callable[[str], None]):
    state = job_state_path(out_path)
    if state.exists():
        try:
            saved = json.loads(state.read_text(encoding="utf-8"))
        except ValueError:
            saved = {}
        if saved.get("job_id") and saved.get("coefficient") == coefficient:
            job = conn.job(saved["job_id"])
            status = with_retries(job.status, "job status", attempts, base_delay, log)
            if status not in JOB_FAILED:
                log(f"  reusing openEO batch job {saved['job_id']} (status: {status})")
                return job
            log(f"  previous job {saved['job_id']} ended '{status}'; creating a new job")
    cube = make_cube()
    job = with_retries(lambda: cube.create_job(title=title, out_format="GTiff"), "create job",
                       attempts, base_delay, log)
    state.write_text(json.dumps({"job_id": job.job_id, "coefficient": coefficient, "title": title}, indent=2),
                     encoding="utf-8")
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
            raise JobFailed(f"openEO batch job {job.job_id} ended '{status}': {_job_logs_tail(job)}")
        if _monotonic() > deadline:
            raise JobTimeout(f"openEO batch job {job.job_id} still '{status}' after {timeout_s / 60:.0f} min; "
                             "rerun the same command later to resume waiting on this job")
        _sleep(poll_interval)


def _tif_asset(job, attempts: int, base_delay: float, log: Callable[[str], None]):
    results = with_retries(job.get_results, "get results", attempts, base_delay, log)
    assets = with_retries(results.get_assets, "list result assets", attempts, base_delay, log)
    tifs = [a for a in assets
            if str(a.name).lower().endswith((".tif", ".tiff")) or "tiff" in str((a.metadata or {}).get("type", ""))]
    if len(tifs) != 1:
        raise RuntimeError(f"expected exactly one GeoTIFF result from job {job.job_id}, "
                           f"got {[a.name for a in assets]}")
    return tifs[0]


def download_scene(conn, scene: SceneMetadata, aoi: AOI, out_path: Path, coefficient: str,
                   resolution: float, buffer_km: float, mode: str = "batch", attempts: int = 4,
                   base_delay: float = 15.0, poll_interval: float = 30.0, job_timeout_min: float = 90.0,
                   log: Callable[[str], None] = print) -> dict[str, Any]:
    """Download one scene to `out_path` (atomically, validated). Returns info for the sidecar."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path(out_path)
    n_bands = len(requested_bands(scene))
    make_cube = lambda: build_cube(conn, scene, aoi, coefficient, resolution, buffer_km)  # noqa: E731
    info: dict[str, Any] = {"openeo_download_mode": mode}
    try:
        if mode == "batch":
            title = f"floodmap {out_path.stem} {coefficient}"
            job = _get_or_create_job(conn, out_path, coefficient, make_cube, title, attempts, base_delay, log)
            info["openeo_job_id"] = job.job_id
            _wait_for_job(job, attempts, base_delay, poll_interval, job_timeout_min * 60, log)
            asset = _tif_asset(job, attempts, base_delay, log)

            def fetch_file() -> None:
                partial.unlink(missing_ok=True)
                asset.download(partial)
                validate_geotiff(partial, n_bands)
            with_retries(fetch_file, "download result", attempts, base_delay, log)
        elif mode == "sync":
            cube = make_cube()

            def fetch_sync() -> None:
                partial.unlink(missing_ok=True)
                cube.download(str(partial), format="GTiff")
                validate_geotiff(partial, n_bands)
            with_retries(fetch_sync, "synchronous download", attempts, base_delay, log)
        else:
            raise ValueError(f"unknown download mode {mode!r}")
        os.replace(partial, out_path)           # atomic: the final name only appears when complete
    finally:
        partial.unlink(missing_ok=True)
    return info


# ------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, type=Path, help="JSON from search_s1_scenes.py")
    ap.add_argument("--pair", type=int, default=0, help="Index into the 'pairs' list")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--coefficient", default=DEFAULT_COEFFICIENT,
                    help=f"sar_backscatter coefficient (default: {DEFAULT_COEFFICIENT}, the only one CDSE accepts)")
    ap.add_argument("--resolution", type=float, default=10.0)
    ap.add_argument("--buffer-km", type=float, default=1.0, help="Margin around the AOI")
    ap.add_argument("--overwrite", action="store_true", help="Re-download scenes even if already complete")
    ap.add_argument("--mode", choices=("batch", "sync"), default="batch",
                    help="batch (default, robust to dropped connections) or sync (single long request)")
    ap.add_argument("--retries", type=int, default=4, help="Attempts per network operation (default 4)")
    ap.add_argument("--retry-delay", type=float, default=15.0, help="First retry wait in s, doubles each time")
    ap.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between batch-job status checks")
    ap.add_argument("--job-timeout-min", type=float, default=90.0, help="Max minutes to wait for one batch job")
    args = ap.parse_args(argv)
    if args.retries < 1:
        ap.error("--retries must be >= 1")

    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    if not payload.get("pairs"):
        raise SystemExit("No valid pairs in candidates file")
    pair = payload["pairs"][args.pair]
    aoi = AOI.model_validate(payload["aoi"])
    conn = None
    for role in ("before", "after"):
        meta = SceneMetadata.model_validate(pair[role])
        out = args.out / f"s1_{role}_{meta.acquisition_time:%Y%m%d}.tif"
        if not args.overwrite and is_complete(out, args.coefficient):
            print(f"Skipping {role}: already complete ({out})")
            continue
        if args.overwrite:
            job_state_path(out).unlink(missing_ok=True)
        if conn is None:
            conn = connect()
        print(f"Downloading {role} (coefficient={args.coefficient}, mode={args.mode}): {meta.product_ids} -> {out}")
        out.with_suffix(".meta.json").unlink(missing_ok=True)   # never leave a sidecar for an old/partial file
        info = download_scene(conn, meta, aoi, out, args.coefficient, args.resolution, args.buffer_km,
                              mode=args.mode, attempts=args.retries, base_delay=args.retry_delay,
                              poll_interval=args.poll_interval, job_timeout_min=args.job_timeout_min)
        meta = meta.model_copy(update={
            "units": "linear",
            "processing": processing_text(args.coefficient, args.resolution, analysis_crs_for_aoi(aoi)),
            "extra": {**meta.extra, **info},
        })
        write_scene_metadata(out, meta)
        job_state_path(out).unlink(missing_ok=True)              # job no longer needed once the scene is complete
        print(f"  done: {out}")
    print("Done.")


if __name__ == "__main__":
    main()
