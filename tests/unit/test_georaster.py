import numpy as np
import pytest

from floodmap.core.errors import GridMismatchError
from floodmap.core.georaster import GeoRaster

T = (10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)  # 10 m pixels, top-left (1000, 2000)


def raster(rows=4, cols=5, bands=2, **kw):
    data = np.arange(bands * rows * cols, dtype=np.float32).reshape(bands, rows, cols)
    return GeoRaster(data, T, "epsg:32645", band_names=[f"b{i}" for i in range(bands)], **kw)


def test_basic_geometry():
    r = raster()
    assert r.crs == "EPSG:32645"
    assert r.shape == (4, 5) and r.count == 2
    assert r.res == (10.0, 10.0) and r.pixel_area == 100.0
    assert r.bounds == (1000.0, 1960.0, 1050.0, 2000.0)
    assert r.xy(0, 0) == (1005.0, 1995.0)
    assert r.rowcol(1005.0, 1995.0) == (0, 0)
    assert r.rowcol(1049.9, 1960.1) == (3, 4)


def test_2d_input_promoted():
    r = GeoRaster(np.zeros((3, 3)), T, "EPSG:4326")
    assert r.data.shape == (1, 3, 3) and r.band_names == ["band_1"]


def test_rejects_rotation_and_bad_names():
    with pytest.raises(ValueError):
        GeoRaster(np.zeros((3, 3)), (10, 1, 0, 0, -10, 0), "EPSG:4326")
    with pytest.raises(ValueError):
        GeoRaster(np.zeros((2, 3, 3)), T, "EPSG:4326", band_names=["only_one"])


def test_valid_mask_nodata_and_nan():
    r = raster(nodata=0.0)          # first pixel of band 0 is 0 -> nodata
    r.data[1, 2, 2] = np.nan
    m = r.valid_mask()
    assert not m[0, 0] and not m[2, 2] and m.sum() == 4 * 5 - 2


def test_band_access_and_select():
    r = raster()
    assert np.array_equal(r.band("b1"), r.data[1])
    with pytest.raises(KeyError):
        r.band("nope")
    s = r.select_bands(["b1"])
    assert s.count == 1 and s.same_grid(r)


def test_same_grid_and_mismatch():
    a, b = raster(), raster()
    assert a.same_grid(b)
    shifted = GeoRaster(a.data, (10.0, 0, 1010.0, 0, -10.0, 2000.0), a.crs, band_names=a.band_names)
    assert not a.same_grid(shifted)
    with pytest.raises(GridMismatchError):
        a.require_same_grid(shifted)
    other_crs = GeoRaster(a.data, T, "EPSG:32644", band_names=a.band_names)
    assert not a.same_grid(other_crs)


def test_with_data_checks_shape():
    r = raster()
    d = r.with_data(np.ones(r.shape, np.uint8), band_names=["mask"], nodata=None)
    assert d.same_grid(r) and d.band_names == ["mask"] and d.nodata is None
    with pytest.raises(GridMismatchError):
        r.with_data(np.ones((2, 2)))
    # same band count and no names given -> names kept; different count -> generic names
    assert r.with_data(r.data * 2).band_names == ["b0", "b1"]
    assert r.with_data(r.data[:1]).band_names == ["band_1"]


def test_crop_to_bounds_snaps_and_georeferences():
    r = raster(rows=10, cols=10)
    c = r.crop_to_bounds((1015.0, 1925.0, 1042.0, 1985.0))
    # cols 1..4 (1010-1050), rows 1..7 (1990-1920)
    assert c.shape == (7, 4)
    assert c.transform[2] == 1010.0 and c.transform[5] == 1990.0
    # same pixel values at the same location
    assert c.data[0, 0, 0] == r.data[0, 1, 1]
    x, y = c.xy(0, 0)
    assert r.rowcol(x, y) == (1, 1)
    with pytest.raises(ValueError):
        r.crop_to_bounds((0, 0, 10, 10))
