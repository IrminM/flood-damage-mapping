import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")
shapely_geometry = pytest.importorskip("shapely.geometry")

from floodmap.core.geo_utils import aoi_from_bounds  # noqa: E402
from floodmap.satellite.grid import GridSpec, make_analysis_grid, project_aoi  # noqa: E402

AOI = aoi_from_bounds((85.20, 28.00, 85.25, 28.04))  # ~5 x 4.4 km near the Trishuli valley


def test_grid_crs_resolution_and_snapping():
    g = make_analysis_grid(AOI, resolution_m=10, margin_m=200)
    s = g.spec
    assert s.crs == "EPSG:32645"
    assert s.transform[0] == 10 and s.transform[4] == -10
    assert s.transform[2] % 10 == 0 and s.transform[5] % 10 == 0   # origin snapped to the pixel size
    # AOI (~4.9 km wide) + 2 x 200 m margin -> roughly 530 px wide
    assert 500 < s.width < 560 and 450 < s.height < 520


def test_aoi_mask_area_matches_polygon_area():
    g = make_analysis_grid(AOI, resolution_m=10, margin_m=200)
    poly = shapely_geometry.shape(project_aoi(AOI, g.spec.crs))
    assert g.aoi_area_m2 == pytest.approx(poly.area, rel=0.01)
    # margin pixels are outside the AOI
    assert not g.aoi_mask[0, :].any() and not g.aoi_mask[:, 0].any()


def test_same_aoi_same_grid():
    a = make_analysis_grid(AOI, 10, 200).spec
    b = make_analysis_grid(AOI, 10, 200).spec
    assert a == b


def test_scaled_grid_covers_same_extent():
    s = make_analysis_grid(AOI, 10, 200).spec
    s30 = s.scaled(30)
    assert s30.transform[2] == s.transform[2] and s30.transform[5] == s.transform[5]
    assert s30.width * 30 >= s.width * 10 and s30.width * 30 - s.width * 10 < 30


def test_template():
    s = GridSpec("EPSG:32645", (10, 0, 0, 0, -10, 100), 4, 3)
    t = s.template(bands=2, band_names=["a", "b"])
    assert t.data.shape == (2, 3, 4) and np.isnan(t.data).all()
