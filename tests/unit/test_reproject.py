import numpy as np
import pytest

pytest.importorskip("rasterio")
pyproj = pytest.importorskip("pyproj")

from floodmap.core.georaster import GeoRaster  # noqa: E402
from floodmap.satellite.grid import GridSpec  # noqa: E402
from floodmap.satellite.reproject import reproject_to_grid  # noqa: E402


def test_identity_reprojection_preserves_values():
    data = np.random.default_rng(0).random((2, 30, 40)).astype(np.float32)
    r = GeoRaster(data, (10, 0, 330000, 0, -10, 3100000), "EPSG:32645", band_names=["VV", "VH"])
    out = reproject_to_grid(r, GridSpec.from_raster(r), "bilinear")
    assert out.same_grid(r) and out.band_names == ["VV", "VH"]
    np.testing.assert_allclose(out.data, data, rtol=1e-5)


def test_upsampling_constant_field_and_coverage():
    # 20 m source covering only the western half of a 10 m grid
    src = GeoRaster(np.full((1, 10, 5), 0.05, np.float32), (20, 0, 330000, 0, -20, 3100000), "EPSG:32645")
    grid = GridSpec("EPSG:32645", (10, 0, 330000, 0, -10, 3100000), 20, 20)
    out = reproject_to_grid(src, grid, "bilinear")
    west, east = out.data[0, :, :8], out.data[0, :, 12:]
    np.testing.assert_allclose(west, 0.05, rtol=1e-6)
    assert np.isnan(east).all(), "pixels without source coverage must be NaN"


def test_source_nodata_becomes_nan_and_is_not_interpolated():
    d = np.full((1, 10, 10), 0.1, np.float32)
    d[0, :, 5:] = -9999.0
    src = GeoRaster(d, (10, 0, 0, 0, -10, 100), "EPSG:32645", nodata=-9999.0)
    out = reproject_to_grid(src, GridSpec.from_raster(src), "bilinear")
    assert np.isnan(out.data[0, :, 6:]).all()
    assert np.nanmax(out.data) == pytest.approx(0.1)   # -9999 never leaks into valid pixels


def test_lonlat_to_utm_puts_values_in_the_right_place():
    # source in EPSG:4326 whose value encodes longitude*1000
    lon0, lat0, step = 85.20, 28.05, 0.0005
    lons = lon0 + step * (np.arange(200) + 0.5)
    src = GeoRaster((np.tile(lons, (200, 1)) * 1000)[None].astype(np.float64),
                    (step, 0, lon0, 0, -step, lat0), "EPSG:4326")
    tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    x, y = tr.transform(85.25, 28.0)
    grid = GridSpec("EPSG:32645", (10, 0, round(x) - 500, 0, -10, round(y) + 500), 100, 100)
    out = reproject_to_grid(src, grid, "bilinear")
    lon_c, _ = tr.transform(*grid.template().xy(50, 50), direction="INVERSE")
    assert out.data[0, 50, 50] / 1000 == pytest.approx(lon_c, abs=step)


def test_categorical_nearest_keeps_codes():
    m = np.zeros((1, 10, 10), np.uint8)
    m[0, 2:5, 2:5] = 3
    src = GeoRaster(m, (30, 0, 0, 0, -30, 300), "EPSG:32645", nodata=255)
    grid = GridSpec("EPSG:32645", (10, 0, 0, 0, -10, 300), 40, 40)
    out = reproject_to_grid(src, grid, "nearest")
    assert out.data.dtype == np.uint8
    assert set(np.unique(out.data)) == {0, 3, 255}     # 255 only where the source does not reach
    assert out.data[0, 10, 10] == 3


def test_mask_with_dst_nodata_equal_to_a_valid_value_is_not_erased():
    # regression: GDAL used dst_nodata=0 as a source mask and erased every valid pixel
    m = np.zeros((1, 20, 20), np.uint8)
    m[0, 5:10, 5:10] = 1
    src = GeoRaster(m, (10, 0, 0, 0, -10, 200), "EPSG:32645", nodata=None)
    out = reproject_to_grid(src, GridSpec("EPSG:32645", (10, 0, -50, 0, -10, 200), 30, 20), "nearest", dst_nodata=0)
    assert out.data.sum() == 25 and out.nodata == 0
    assert (out.data[0, :, :5] == 0).all()        # no source coverage -> requested nodata (0)
