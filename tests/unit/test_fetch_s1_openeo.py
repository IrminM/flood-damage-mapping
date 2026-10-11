"""Offline tests for scripts/fetch_s1_openeo.py with a fake openEO backend.

No network or login: `connect()` is replaced by a fake connection whose cubes, batch jobs
and result assets record calls and write small GeoTIFFs. Connection resets (WinError
10054 / ChunkedEncodingError) are injected to test retries, resume and partial-file
handling. The real authentication code is not exercised (it works against the live service).
"""
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
requests = pytest.importorskip("requests")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))   # for the script's `import _bootstrap`
spec = importlib.util.spec_from_file_location("fetch_s1_openeo", ROOT / "scripts" / "fetch_s1_openeo.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)

from floodmap.core.geo_utils import aoi_from_bounds  # noqa: E402
from floodmap.data_sources.satellite_local import LocalSceneProvider  # noqa: E402
from floodmap.satellite.pairing import check_pair  # noqa: E402
from tests.conftest import make_s1_meta  # noqa: E402

RESET = lambda: requests.exceptions.ChunkedEncodingError(  # noqa: E731
    "Connection broken: ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host')")


def write_tif(path, value=0.05, bands=2):
    data = np.full((bands, 20, 20), value, np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=20, width=20, count=bands, dtype="float32",
                       crs="EPSG:32645", transform=rasterio.transform.from_origin(330000, 3100000, 10, 10)) as dst:
        dst.write(data)


class Backend:
    """Shared fake server state + scripted failures."""
    def __init__(self):
        self.log, self.jobs, self.next_id = [], {}, 1
        self.fail = {}            # operation name -> list of exceptions to raise on successive calls
        self.job_statuses = None  # optional scripted status sequence for new jobs
        self.result_value = 0.05
        self.truncate_downloads = 0

    def maybe_fail(self, op):
        q = self.fail.get(op)
        if q:
            exc = q.pop(0)
            if exc is not None:
                raise exc


class FakeAsset:
    def __init__(self, be):
        self.be, self.name, self.metadata = be, "openEO.tif", {"type": "image/tiff; application=geotiff"}

    def download(self, target):
        self.be.log.append(("asset_download", Path(target).name))
        self.be.maybe_fail("asset_download")
        if self.be.truncate_downloads:
            self.be.truncate_downloads -= 1
            Path(target).write_bytes(b"II*\x00 truncated")     # looks like a TIFF header, then cut off
            return Path(target)
        write_tif(target, self.be.result_value)
        return Path(target)


class FakeResults:
    def __init__(self, be):
        self.be = be

    def get_assets(self):
        self.be.maybe_fail("get_assets")
        return [FakeAsset(self.be), type("A", (), {"name": "job-results.json", "metadata": {"type": "application/json"}})()]


class FakeJob:
    def __init__(self, be, job_id, statuses):
        self.be, self.job_id, self.statuses, self.started = be, job_id, list(statuses), False

    def status(self):
        self.be.maybe_fail("status")
        if not self.started:
            return "created"
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def start(self):
        self.be.log.append(("start", self.job_id))
        self.started = True

    def get_results(self):
        return FakeResults(self.be)

    def logs(self):
        return [{"level": "error", "message": "simulated backend failure"}]


class FakeCube:
    def __init__(self, be):
        self.be = be

    def sar_backscatter(self, coefficient):
        self.be.log.append(("sar_backscatter", coefficient))
        return self

    def resample_spatial(self, resolution, projection):
        return self

    def reduce_dimension(self, dimension, reducer):
        self.be.log.append(("reduce_dimension", dimension, reducer))
        return self

    def create_job(self, title, out_format):
        self.be.log.append(("create_job", out_format))
        self.be.maybe_fail("create_job")
        jid = f"j-{self.be.next_id}"
        self.be.next_id += 1
        self.be.jobs[jid] = FakeJob(self.be, jid, self.be.job_statuses or ["queued", "running", "finished"])
        return self.be.jobs[jid]

    def download(self, path, format):                      # sync mode
        self.be.log.append(("sync_download", Path(path).name, format))
        self.be.maybe_fail("sync_download")
        write_tif(path, self.be.result_value)


class FakeConn:
    def __init__(self, be):
        self.be = be

    def load_collection(self, cid, **kw):
        self.be.log.append(("load_collection", cid, kw["temporal_extent"][0]))
        return FakeCube(self.be)

    def job(self, job_id):
        self.be.log.append(("reuse_job", job_id))
        return self.be.jobs[job_id]


@pytest.fixture
def be(monkeypatch):
    b = Backend()
    sleeps = []
    monkeypatch.setattr(fetch, "_sleep", lambda s: sleeps.append(s))
    b.sleeps = sleeps
    monkeypatch.setattr(fetch, "connect", lambda: FakeConn(b))
    return b


@pytest.fixture
def candidates(tmp_path):
    aoi = aoi_from_bounds((85.20, 28.00, 85.25, 28.04), name="test")
    before = make_s1_meta("2026-08-16", product_ids=["TEST_FIXTURE_BEFORE"])
    after = make_s1_meta("2026-08-28", product_ids=["TEST_FIXTURE_AFTER"])
    p = tmp_path / "s1_candidates.json"
    p.write_text(json.dumps({"aoi": aoi.model_dump(), "event_date": "2026-08-26", "scenes": [],
                             "pairs": [{"before": json.loads(before.model_dump_json()),
                                        "after": json.loads(after.model_dump_json()), "check": {}}]}))
    return p


def run(candidates, out, *extra):
    fetch.main(["--candidates", str(candidates), "--pair", "0", "--out", str(out), "--poll-interval", "1", *extra])


def retry_waits(be):
    """Backoff waits only (status polling uses --poll-interval 1 in these tests)."""
    return [w for w in be.sleeps if w != 1.0]


def names(d):
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


BEFORE, AFTER = "s1_before_20260816", "s1_after_20260828"
COMPLETE = [f"{AFTER}.meta.json", f"{AFTER}.tif", f"{BEFORE}.meta.json", f"{BEFORE}.tif"]


# ------------------------------------------------------------------ basics

def test_default_coefficient_and_mode():
    assert fetch.DEFAULT_COEFFICIENT == "sigma0-ellipsoid"
    src = (ROOT / "scripts" / "fetch_s1_openeo.py").read_text(encoding="utf-8")
    assert "gamma0-terrain" not in src and "gamma-terrain" not in src
    assert 'default="batch"' in src


@pytest.mark.parametrize("exc,transient", [
    (ConnectionResetError(10054, "forcibly closed"), True),
    (requests.exceptions.ChunkedEncodingError("Connection broken"), True),
    (requests.exceptions.ConnectionError("reset"), True),
    (requests.exceptions.ReadTimeout("slow"), True),
    (type("ApiErr", (Exception,), {"http_status_code": 503})(), True),
    (type("ApiErr", (Exception,), {"http_status_code": 429})(), True),
    (type("ApiErr", (Exception,), {"http_status_code": 400})(), False),   # bad request: retrying won't help
    (ValueError("bad"), False),
    (fetch.EmptyDownload("no data"), False),
    (fetch.IncompleteDownload("truncated"), True),
])
def test_transient_classification(exc, transient):
    assert fetch.is_transient(exc) is transient


# ------------------------------------------------------------------ batch mode

def test_batch_download_happy_path_loads_in_stage2(be, candidates, tmp_path, guard, settings):
    out = tmp_path / "raw"
    run(candidates, out)
    assert [c[1] for c in be.log if c[0] == "sar_backscatter"] == ["sigma0-ellipsoid"] * 2
    assert [c for c in be.log if c[0] == "create_job"] == [("create_job", "GTiff")] * 2
    assert all(c[1].endswith(".partial.tif") for c in be.log if c[0] == "asset_download")
    assert names(out) == COMPLETE                     # no .partial.tif, no .job.json left behind
    side = json.loads((out / f"{BEFORE}.meta.json").read_text())
    assert side["extra"]["openeo_job_id"] == "j-1" and side["extra"]["openeo_download_mode"] == "batch"
    prov = LocalSceneProvider(guard)
    b, a = prov.load_scene(out / f"{BEFORE}.tif"), prov.load_scene(out / f"{AFTER}.tif")
    assert "coefficient=sigma0-ellipsoid" in b.ref.metadata.processing
    assert check_pair(b.ref.metadata, a.ref.metadata, date(2026, 8, 26), settings.satellite.s1).valid


def test_connection_reset_during_file_download_is_retried_without_recomputing(be, candidates, tmp_path):
    be.fail["asset_download"] = [RESET(), ConnectionResetError(10054, "forcibly closed")]
    out = tmp_path / "raw"
    run(candidates, out)
    assert names(out) == COMPLETE
    assert sum(c[0] == "create_job" for c in be.log) == 2       # one job per scene, never recreated
    assert retry_waits(be)[:2] == [15.0, 30.0]                  # exponential backoff


def test_reset_during_status_polling_is_retried(be, candidates, tmp_path):
    be.fail["status"] = [None, RESET(), None, RESET()]
    run(candidates, tmp_path / "raw")
    assert names(tmp_path / "raw") == COMPLETE


def test_persistent_reset_fails_cleanly_then_rerun_reuses_finished_job(be, candidates, tmp_path):
    out = tmp_path / "raw"
    be.fail["asset_download"] = [RESET()] * 4                   # all 4 attempts for 'before' fail
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        run(candidates, out)
    # nothing that looks like a finished scene; only the job id is kept for resuming
    assert names(out) == [f"{BEFORE}.job.json"]
    assert retry_waits(be) == [15.0, 30.0, 60.0]                 # limited: 4 attempts, 3 waits

    be.log.clear()
    run(candidates, out)                                         # network is back
    assert ("reuse_job", "j-1") in be.log
    assert sum(c[0] == "create_job" for c in be.log) == 1        # only 'after' needed a new job
    assert names(out) == COMPLETE


def test_truncated_download_is_detected_and_retried(be, candidates, tmp_path):
    be.truncate_downloads = 1
    out = tmp_path / "raw"
    run(candidates, out)
    assert names(out) == COMPLETE
    with rasterio.open(out / f"{BEFORE}.tif") as src:            # the accepted file is the full one
        assert src.count == 2 and np.allclose(src.read(), 0.05)


def test_truncated_every_time_never_produces_a_tif(be, candidates, tmp_path):
    be.truncate_downloads = 99
    out = tmp_path / "raw"
    with pytest.raises(fetch.IncompleteDownload):
        run(candidates, out, "--retries", "2")
    assert not list(out.glob("*.tif"))


def test_empty_result_is_a_clear_error_and_not_retried(be, candidates, tmp_path):
    be.result_value = np.nan
    with pytest.raises(fetch.EmptyDownload, match="no valid backscatter"):
        run(candidates, tmp_path / "raw")
    assert sum(c[0] == "asset_download" for c in be.log) == 1
    assert not list((tmp_path / "raw").glob("*.tif"))


def test_failed_job_reports_id_and_backend_log(be, candidates, tmp_path):
    be.job_statuses = ["queued", "error"]
    with pytest.raises(fetch.JobFailed, match=r"j-1.*simulated backend failure"):
        run(candidates, tmp_path / "raw")
    assert not list((tmp_path / "raw").glob("*.tif"))


def test_rerun_after_failed_job_creates_a_new_job(be, candidates, tmp_path):
    out = tmp_path / "raw"
    be.job_statuses = ["error"]
    with pytest.raises(fetch.JobFailed):
        run(candidates, out)
    be.job_statuses = None
    run(candidates, out)
    assert names(out) == COMPLETE and "j-2" in json.loads((out / f"{BEFORE}.meta.json").read_text())["extra"]["openeo_job_id"]


def test_job_timeout_is_bounded(be, candidates, tmp_path, monkeypatch):
    be.job_statuses = ["running"]                                # never finishes
    clock = iter(range(0, 100000, 600))                          # +10 min per call
    monkeypatch.setattr(fetch, "_monotonic", lambda: next(clock))
    with pytest.raises(fetch.JobTimeout, match="rerun the same command"):
        run(candidates, tmp_path / "raw", "--job-timeout-min", "30")
    assert names(tmp_path / "raw") == [f"{BEFORE}.job.json"]     # job kept so a rerun resumes waiting


def test_non_transient_error_is_not_retried(be, candidates, tmp_path):
    be.fail["create_job"] = [type("ApiErr", (Exception,), {"http_status_code": 400})("bad process graph")]
    with pytest.raises(Exception, match="bad process graph"):
        run(candidates, tmp_path / "raw")
    assert sum(c[0] == "create_job" for c in be.log) == 1 and retry_waits(be) == []


# ------------------------------------------------------------------ sync mode, resume, CLI

def test_sync_mode_retries_reset(be, candidates, tmp_path):
    be.fail["sync_download"] = [RESET()]
    out = tmp_path / "raw"
    run(candidates, out, "--mode", "sync")
    assert names(out) == COMPLETE
    assert sum(c[0] == "sync_download" for c in be.log) == 3     # 1 failed + 2 scenes
    assert not any(c[0] == "create_job" for c in be.log)


def test_completed_scene_skipped_overwrite_forces(be, candidates, tmp_path):
    out = tmp_path / "raw"
    run(candidates, out)
    be.log.clear()
    run(candidates, out)
    assert not any(c[0] == "load_collection" for c in be.log)    # both skipped, no login needed either
    run(candidates, out, "--overwrite")
    assert sum(c[0] == "create_job" for c in be.log) == 2
    assert not fetch.is_complete(out / f"{BEFORE}.tif", "some-other-coefficient")


def test_cli_signature_unchanged():
    import subprocess
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_s1_openeo.py"), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    for flag in ("--candidates", "--pair", "--out", "--coefficient", "--resolution", "--buffer-km",
                 "--overwrite", "--mode", "--retries", "--retry-delay", "--poll-interval", "--job-timeout-min"):
        assert flag in p.stdout
    assert "sigma0-ellipsoid" in p.stdout
