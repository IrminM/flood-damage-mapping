"""Offline tests for the Copernicus DEM GLO-30 downloader (fake tile server, no network).

Fake tiles are real GeoTIFFs in EPSG:4326 covering exact 1° cells, with a known terrain
function, so mosaicking, cropping, nodata handling and the downstream lon/lat -> UTM
terrain analysis can be checked against expected values.
"""
import io
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
requests = pytest.importorskip("requests")
pytest.importorskip("pyproj")

import floodmap.core.retry as retry  # noqa: E402
from floodmap.core.errors import InputDataError  # noqa: E402
from floodmap.core.geo_utils import aoi_from_bounds  # noqa: E402
from floodmap.data_sources import copernicus_dem as cdem  # noqa: E402
from floodmap.data_sources.dem_local import LocalDEMProvider  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TILE_PX = 1200                      # 3 arc-seconds (real GLO-30 tiles are 3600 px, 1 arc-second)
SLOPE_DEG = 3.0


def terrain(lon, lat):
    """Plane rising eastwards with SLOPE_DEG (metres)."""
    x_m = (np.asarray(lon) - 85.0) * 111320.0 * np.cos(np.radians(lat))
    return 500.0 + math.tan(math.radians(SLOPE_DEG)) * x_m


_TILE_CACHE: dict[str, bytes] = {}


def tile_bytes(name: str) -> bytes:
    if name not in _TILE_CACHE:
        lat0 = int(name[23:25]) * (1 if name[22] == "N" else -1)
        lon0 = int(name[30:33]) * (1 if name[29] == "E" else -1)
        step = 1.0 / TILE_PX
        lons = lon0 + (np.arange(TILE_PX) + 0.5) * step
        lats = lat0 + 1 - (np.arange(TILE_PX) + 0.5) * step
        z = terrain(lons[None, :], lats[:, None]).astype(np.float32)
        buf = io.BytesIO()
        with rasterio.MemoryFile() as mem:
            with mem.open(driver="GTiff", height=TILE_PX, width=TILE_PX, count=1, dtype="float32", crs="EPSG:4326",
                          transform=rasterio.transform.from_origin(lon0, lat0 + 1, step, step)) as dst:
                dst.write(z[None])
            buf.write(mem.read())
        _TILE_CACHE[name] = buf.getvalue()
    return _TILE_CACHE[name]


class FakeResp:
    def __init__(self, status, body=b"", reset_after=None):
        self.status_code, self.body, self.reset_after = status, body, reset_after

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def iter_content(self, chunk_size):
        sent = 0
        for i in range(0, len(self.body), 65536):
            if self.reset_after is not None and sent >= self.reset_after:
                raise requests.exceptions.ChunkedEncodingError(
                    "Connection broken: ConnectionResetError(10054, 'forcibly closed by the remote host')")
            part = self.body[i:i + 65536]
            sent += len(part)
            yield part


class FakeServer:
    """behaviour[name] = list of per-call actions: 'ok', 'reset', 'truncate', 404, 403, 503."""
    def __init__(self, behaviour=None):
        self.behaviour = behaviour or {}
        self.calls = []

    def get(self, url, timeout, stream):
        name = url.rsplit("/", 1)[-1][:-4]
        self.calls.append(name)
        actions = self.behaviour.get(name, [])
        action = actions.pop(0) if actions else "ok"
        body = tile_bytes(name)
        if action == "ok":
            return FakeResp(200, body)
        if action == "reset":
            return FakeResp(200, body, reset_after=len(body) // 3)
        if action == "truncate":
            return FakeResp(200, body[: len(body) // 4])
        return FakeResp(int(action))


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(retry, "sleep", lambda s: waits.append(s))
    return waits


SEAM_AOI = aoi_from_bounds((85.30, 27.97, 85.40, 28.03), name="seam")   # crosses the 28°N tile edge
N27, N28 = cdem.tile_name(27, 85), cdem.tile_name(28, 85)


def fetch(tmp_path, server, aoi=SEAM_AOI, buffer_km=1.0, **kw):
    return cdem.fetch_dem(aoi, tmp_path / "dem" / "dem_glo30.tif", buffer_km, get=server.get,
                          log=lambda m: None, **kw)


def test_mosaic_across_tile_seam_matches_terrain(tmp_path, guard):
    srv = FakeServer()
    out = fetch(tmp_path, srv)
    assert sorted(srv.calls) == sorted([N27, N28])
    with rasterio.open(out) as src:
        assert src.crs.to_epsg() == 4326 and src.nodata == cdem.DEM_NODATA and src.dtypes[0] == "float32"
        b = src.bounds
        need = cdem.buffer_lonlat_bounds_km(cdem.aoi_bounds(SEAM_AOI), 1.0)
        assert b.left <= need[0] + 1e-3 and b.top >= need[3] - 1e-3          # cropped to AOI + buffer
        assert b.right - b.left < 0.2                                         # not the whole 1° tile
        for lon, lat in [(85.33, 27.98), (85.35, 28.0004), (85.35, 27.9996), (85.39, 28.02)]:   # incl. seam
            (v,) = next(src.sample([(lon, lat)]))
            assert v == pytest.approx(float(terrain(lon, lat)), abs=6.0)
        assert (src.read(1) != cdem.DEM_NODATA).all()
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["source_type"] == "copernicus-dem" and meta["aoi_coverage_fraction"] == 1.0
    assert {t["name"] for t in meta["tiles"]} == {N27, N28} and meta["missing_tiles"] == []
    for t in meta["tiles"]:
        assert t["sha256"] == cdem.sha256_file(out.parent / "tiles" / f"{t['name']}.tif")
    dem = LocalDEMProvider(guard).load_dem(out)                              # unchanged Stage-2 loader
    assert dem.metadata["source_type"] == "copernicus-dem" and dem.band_names == ["elevation"]
    assert not list((out.parent / "tiles").glob("*.partial.tif"))


def test_mosaic_is_pixel_aligned_with_tiles(tmp_path):
    out = fetch(tmp_path, FakeServer())
    with rasterio.open(out) as src:
        step = 1.0 / TILE_PX
        for edge in (src.bounds.left, src.bounds.top):
            assert abs(edge / step - round(edge / step)) < 1e-6        # on the tile pixel grid
        # pixel centres hold exactly the tile's values (no resampling)
        (v,) = next(src.sample([(src.bounds.left + step / 2, src.bounds.top - step / 2)]))
        assert v == pytest.approx(float(terrain(src.bounds.left + step / 2, src.bounds.top - step / 2)), abs=1e-3)


def test_snap_bounds():
    r = (0.25, 0.25)
    assert cdem.snap_bounds((85.1, 27.9, 85.6, 28.3), r, (85.0, 29.0)) == (85.0, 27.75, 85.75, 28.5)
    assert cdem.snap_bounds((85.25, 28.0, 85.5, 28.25), r, (85.0, 29.0)) == (85.25, 28.0, 85.5, 28.25)


def test_connection_reset_mid_tile_is_retried(tmp_path, no_wait):
    srv = FakeServer({N27: ["reset", "reset"]})
    fetch(tmp_path, srv)
    assert srv.calls.count(N27) == 3 and no_wait == [10.0, 20.0]


def test_transient_http_503_is_retried(tmp_path):
    srv = FakeServer({N28: [503]})
    fetch(tmp_path, srv)
    assert srv.calls.count(N28) == 2


def test_persistent_truncation_fails_without_caching_or_output(tmp_path):
    srv = FakeServer({N27: ["truncate"] * 4})
    with pytest.raises(cdem.TileDownloadError):
        fetch(tmp_path, srv)
    tiles = tmp_path / "dem" / "tiles"
    assert not (tiles / f"{N27}.tif").exists() and not list(tiles.glob("*.partial.tif"))
    assert not (tmp_path / "dem" / "dem_glo30.tif").exists()


def test_corrupt_cached_tile_is_redownloaded(tmp_path):
    tiles = tmp_path / "dem" / "tiles"
    tiles.mkdir(parents=True)
    (tiles / f"{N27}.tif").write_bytes(b"II*\x00 truncated by an older run")
    srv = FakeServer()
    fetch(tmp_path, srv)
    assert N27 in srv.calls
    cdem.validate_tile(tiles / f"{N27}.tif")


def test_rerun_uses_validated_cache(tmp_path):
    fetch(tmp_path, FakeServer())
    srv = FakeServer()
    fetch(tmp_path, srv)
    assert srv.calls == []


def test_missing_tile_is_nodata_not_zero(tmp_path):
    srv = FakeServer({N27: [404]})
    out = fetch(tmp_path, srv)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["missing_tiles"] == [N27] and 0.3 < meta["aoi_coverage_fraction"] < 0.8
    with rasterio.open(out) as src:
        (south,) = next(src.sample([(85.35, 27.98)]))
        (north,) = next(src.sample([(85.35, 28.02)]))
    assert south == cdem.DEM_NODATA            # gap marked as nodata, never 0 m
    assert north == pytest.approx(float(terrain(85.35, 28.02)), abs=6.0)


def test_all_tiles_missing_and_403_are_clear_errors(tmp_path):
    with pytest.raises(InputDataError, match="No Copernicus DEM tiles"):
        fetch(tmp_path / "a", FakeServer({N27: [404], N28: [404]}))
    srv = FakeServer({N27: [403]})
    with pytest.raises(InputDataError, match="403"):
        fetch(tmp_path / "b", srv)
    assert srv.calls.count(N27) == 1           # access denied is not retried


def test_existing_dem_covers_and_cli_skip(tmp_path, monkeypatch, capsys):
    out = fetch(tmp_path, FakeServer(), buffer_km=3.2)
    assert cdem.existing_dem_covers(out, SEAM_AOI, 3.2)
    assert not cdem.existing_dem_covers(out, SEAM_AOI, 10.0)             # needs a larger extent
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("fetch_dem_glo30", ROOT / "scripts" / "fetch_dem_glo30.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "fetch_dem", lambda *a, **k: pytest.fail("must not download when DEM is current"))
    aoi_file = tmp_path / "aoi.geojson"
    aoi_file.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": SEAM_AOI.geometry}))
    assert mod.main(["--aoi", str(aoi_file), "--out", str(out), "--buffer-km", "3.2"]) == 0
    assert "already present" in capsys.readouterr().out


def test_lonlat_dem_drives_terrain_in_change_pipeline(tmp_path):
    """Real-data path: an EPSG:4326 GLO-30 mosaic reprojected to the UTM analysis grid."""
    from dev_mocks.synthetic_scene import synthetic_aoi, write_synthetic_pair
    from floodmap.core.config import load_settings
    from floodmap.core.contracts import AnalysisRequest
    from floodmap.pipeline.change_pipeline import run_change_pipeline

    bpath, apath, pair = write_synthetic_pair(tmp_path / "s1", with_dem=False)
    aoi = synthetic_aoi(pair, inset_m=30)
    dem = cdem.fetch_dem(aoi, tmp_path / "dem" / "dem_glo30.tif", 3.2, get=FakeServer().get, log=lambda m: None)
    settings = load_settings(environ={"FLOODMAP__TERRAIN__HAND_STREAM_THRESHOLD_KM2": "0.05",
                                      "FLOODMAP__GRID__MARGIN_M": "0"})
    res = run_change_pipeline(AnalysisRequest(aoi=aoi, event_date=date(2026, 8, 26), before_scene=bpath,
                                              after_scene=apath, dem=dem),
                              settings, tmp_path / "run", write_quicklooks=False)
    assert res.summary.terrain_applied and res.terrain.stats["dem_coverage_fraction"] > 0.99
    slope = res.terrain.slope_deg.data[0][res.grid.aoi_mask]
    assert np.nanmedian(slope) == pytest.approx(SLOPE_DEG, abs=0.5)      # known terrain recovered after reprojection
    prov = json.loads(res.provenance_path.read_text())
    dem_in = next(i for i in prov["inputs"] if i["role"] == "dem")
    assert dem_in["source_type"] == "copernicus-dem" and len(dem_in["sha256"]) == 64
    assert dem_in["metadata"]["tiles"][0]["name"] == N28
