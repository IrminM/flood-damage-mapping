import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

from dev_mocks.synthetic_scene import (LAND_VV, WATER_VV, make_synthetic_s1_pair,  # noqa: E402
                                       synthetic_aoi, synthetic_loaded_pair)
from floodmap.core.errors import InputDataError  # noqa: E402
from floodmap.satellite.grid import make_analysis_grid  # noqa: E402
from floodmap.satellite.preprocess import (db_to_linear, lee_filter, linear_to_db,  # noqa: E402
                                           preprocess_s1_pair)


def test_db_linear_roundtrip_and_floor():
    x = np.array([1.0, 0.1, 0.01, 0.0, np.nan], np.float32)
    db = linear_to_db(x, floor=1e-5)
    np.testing.assert_allclose(db[:3], [0, -10, -20], atol=1e-5)
    assert db[3] == pytest.approx(-50.0) and np.isnan(db[4])
    np.testing.assert_allclose(db_to_linear(db[:3]), x[:3], rtol=1e-5)


def test_lee_reduces_speckle_preserves_mean_and_edges():
    rng = np.random.default_rng(0)
    clean = np.full((100, 100), 0.08)
    clean[:, 50:] = 0.005                                  # sharp land/water edge
    noisy = (clean * rng.gamma(4.4, 1 / 4.4, clean.shape)).astype(np.float32)
    f = lee_filter(noisy, window=5, enl=4.4)
    land_n, land_f = noisy[:, 5:45], f[:, 5:45]
    assert land_f.std() < 0.6 * land_n.std()               # speckle reduced
    assert land_f.mean() == pytest.approx(land_n.mean(), rel=0.03)   # radiometry preserved
    # edge preserved: columns right next to the edge stay clearly separated
    assert np.median(f[:, 47]) > 4 * np.median(f[:, 52])


def test_lee_keeps_nodata_and_ignores_it():
    img = np.full((20, 20), 0.05, np.float32)
    img[:, :5] = np.nan
    f = lee_filter(img, 5, 4.4)
    assert np.isnan(f[:, :5]).all()
    np.testing.assert_allclose(f[:, 5:], 0.05, rtol=1e-5)   # NaN neighbours don't pull values


def test_lee_rejects_even_window():
    with pytest.raises(ValueError):
        lee_filter(np.ones((5, 5), np.float32), window=4)


@pytest.fixture
def pair_and_grid(settings):
    pair = make_synthetic_s1_pair(seed=3)
    aoi = synthetic_aoi(pair, inset_m=30)   # < 60 m nodata strip, so it stays inside the AOI
    grid = make_analysis_grid(aoi, settings.grid.resolution_m, margin_m=0)
    return pair, grid


def test_preprocess_pair_outputs(settings, pair_and_grid):
    pair, grid = pair_and_grid
    b, a = synthetic_loaded_pair(pair)
    pp = preprocess_s1_pair(b, a, grid, settings.preprocessing.s1)
    assert pp.bands == ["VV", "VH"]
    assert pp.before_db.same_grid(pp.after_db) and pp.before_db.shape == grid.spec.shape
    assert pp.after_db.metadata["units"] == "dB"
    # outside AOI and the after-image nodata strip are invalid in BOTH outputs
    assert not pp.valid[~grid.aoi_mask].any()
    assert np.isnan(pp.before_db.data[:, ~pp.valid]).all() and np.isnan(pp.after_db.data[:, ~pp.valid]).all()
    assert pp.nodata_in_aoi.any() and 0.8 < pp.stats["valid_fraction_both"] < 1.0
    # radiometry: land ≈ -11 dB, water ≈ -23 dB after filtering
    assert pp.stats["median_db_before"]["VV"] == pytest.approx(10 * np.log10(LAND_VV), abs=1.0)
    river_db = np.nanmedian(pp.before_db.band("VV")[:, grid.spec.width // 2 - 1: grid.spec.width // 2 + 1])
    assert river_db == pytest.approx(10 * np.log10(WATER_VV), abs=2.5)


def test_db_input_gives_same_result_as_linear(settings, pair_and_grid):
    pair, grid = pair_and_grid
    b, a = synthetic_loaded_pair(pair)
    ref = preprocess_s1_pair(b, a, grid, settings.preprocessing.s1)
    for scene in (b, a):
        scene.raster = scene.raster.with_data(linear_to_db(scene.raster.data, 1e-12))
        scene.ref.metadata.units = "dB"
    got = preprocess_s1_pair(b, a, grid, settings.preprocessing.s1)
    np.testing.assert_allclose(got.after_db.data, ref.after_db.data, atol=1e-3, equal_nan=True)


def test_no_overlap_raises(settings, pair_and_grid):
    pair, grid = pair_and_grid
    pair.after.data[:] = np.nan
    b, a = synthetic_loaded_pair(pair)
    with pytest.raises(InputDataError):
        preprocess_s1_pair(b, a, grid, settings.preprocessing.s1)


def test_missing_vv_raises(settings, pair_and_grid):
    pair, grid = pair_and_grid
    b, a = synthetic_loaded_pair(pair)
    a.raster = a.raster.select_bands(["VH"])
    with pytest.raises(InputDataError, match="VV"):
        preprocess_s1_pair(b, a, grid, settings.preprocessing.s1)


def test_filter_can_be_disabled(settings, pair_and_grid):
    pair, grid = pair_and_grid
    b, a = synthetic_loaded_pair(pair)
    cfg = settings.preprocessing.s1.model_copy(deep=True)
    cfg.speckle_filter.enabled = False
    raw = preprocess_s1_pair(b, a, grid, cfg)
    filt = preprocess_s1_pair(b, a, grid, settings.preprocessing.s1)
    assert np.nanstd(raw.before_db.band("VV")) > np.nanstd(filt.before_db.band("VV"))
