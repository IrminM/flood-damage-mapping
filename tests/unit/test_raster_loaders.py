"""GeoTIFF read/write + satellite/DEM loaders. Needs rasterio (skipped if not installed)."""
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from dev_mocks.synthetic_scene import write_synthetic_pair  # noqa: E402
from floodmap.core.contracts import DataMode  # noqa: E402
from floodmap.core.errors import DataRuleViolation, InputDataError  # noqa: E402
from floodmap.core.georaster import GeoRaster  # noqa: E402
from floodmap.data_sources.dem_local import LocalDEMProvider  # noqa: E402
from floodmap.data_sources.satellite_local import LocalSceneProvider  # noqa: E402
from floodmap.io.geotiff import read_geotiff, write_geotiff  # noqa: E402

T = (30.0, 0.0, 330000.0, 0.0, -30.0, 3110000.0)


def test_geotiff_roundtrip(tmp_path):
    r = GeoRaster(np.random.rand(2, 20, 30).astype(np.float32), T, "EPSG:32645", nodata=-9999.0,
                  band_names=["VV", "VH"])
    p = write_geotiff(r, tmp_path / "r.tif")
    back = read_geotiff(p)
    assert back.same_grid(r) and back.band_names == ["VV", "VH"] and back.nodata == -9999.0
    np.testing.assert_allclose(back.data, r.data)


def test_load_synthetic_scene_pair(tmp_path, guard):
    bpath, apath, pair = write_synthetic_pair(tmp_path)
    prov = LocalSceneProvider(guard)
    b, a = prov.load_scene(bpath), prov.load_scene(apath)
    assert b.raster.band_names == ["VV", "VH"]
    assert b.ref.metadata.data_mode is DataMode.SYNTHETIC_TEST
    assert b.raster.same_grid(a.raster)
    assert b.warnings == []
    np.testing.assert_allclose(b.raster.data, pair.before.data)


def test_missing_sidecar_rejected(tmp_path, guard):
    r = GeoRaster(np.ones((2, 5, 5), np.float32), T, "EPSG:32645", band_names=["VV", "VH"])
    p = write_geotiff(r, tmp_path / "nosidecar.tif")
    with pytest.raises(InputDataError, match="sidecar"):
        LocalSceneProvider(guard).load_scene(p)


def test_band_count_mismatch_and_unit_warning(tmp_path, guard):
    bpath, _, _ = write_synthetic_pair(tmp_path)
    side = bpath.with_suffix(".meta.json")
    meta = json.loads(side.read_text())
    meta["bands"] = ["VV"]
    side.write_text(json.dumps(meta))
    with pytest.raises(InputDataError, match="bands"):
        LocalSceneProvider(guard).load_scene(bpath)
    # dB-looking data declared as linear -> warning, not silent
    meta["bands"] = ["VV", "VH"]
    side.write_text(json.dumps(meta))
    r = read_geotiff(bpath)
    write_geotiff(r.with_data(10 * np.log10(r.data)), bpath)
    loaded = LocalSceneProvider(guard).load_scene(bpath)
    assert any("looks like dB" in w for w in loaded.warnings)


def test_prohibited_scene_name_rejected(tmp_path, guard):
    bpath, _, _ = write_synthetic_pair(tmp_path / "EMSR927")
    with pytest.raises(DataRuleViolation):
        LocalSceneProvider(guard).load_scene(bpath)


def test_dem_loader_masks_invalid(tmp_path, guard):
    elev = np.full((1, 10, 10), 1500.0, np.float32)
    elev[0, 0, 0] = -32768.0   # typical DEM nodata
    p = write_geotiff(GeoRaster(elev, T, "EPSG:32645", nodata=-32768.0), tmp_path / "dem.tif")
    dem = LocalDEMProvider(guard).load_dem(p)
    assert dem.band_names == ["elevation"] and np.isnan(dem.data[0, 0, 0])
    assert np.nanmin(dem.data) == 1500.0
