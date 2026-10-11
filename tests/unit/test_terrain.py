import math

import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

from dev_mocks.synthetic_scene import make_synthetic_s1_pair, synthetic_aoi  # noqa: E402
from floodmap.satellite.grid import make_analysis_grid  # noqa: E402
from floodmap.satellite.terrain import (compute_terrain, flow_accumulation,  # noqa: E402
                                        height_above_nearest_drainage, look_geometry,
                                        priority_flood_routing, radar_shadow_layover, range_slope_deg,
                                        slope_aspect, terrain_flag_raster)


def plane(rows, cols, res, slope_deg, rises_towards):
    """Plane rising towards 'east' or 'north' with the given slope."""
    rr, cc = np.mgrid[0:rows, 0:cols].astype(float)
    t = math.tan(math.radians(slope_deg))
    if rises_towards == "east":
        return cc * res * t
    return (rows - rr) * res * t          # rows go south, so elevation grows as row decreases


@pytest.mark.parametrize("towards,downslope", [("east", 270.0), ("north", 180.0)])
def test_slope_and_aspect_on_planes(towards, downslope):
    s, a = slope_aspect(plane(20, 20, 30, 20, towards), 30, 30)
    np.testing.assert_allclose(s[2:-2, 2:-2], 20.0, atol=1e-3)
    np.testing.assert_allclose(a[2:-2, 2:-2], downslope, atol=1e-3)


def test_range_slope_sign_convention():
    look = 78.0   # ascending S1 looks roughly east
    assert range_slope_deg(np.array([30.0]), np.array([look + 180]), look)[0] == pytest.approx(30.0)  # faces radar
    assert range_slope_deg(np.array([30.0]), np.array([look]), look)[0] == pytest.approx(-30.0)       # faces away
    assert abs(range_slope_deg(np.array([30.0]), np.array([look + 90]), look)[0]) < 1e-4             # side-on


@pytest.mark.parametrize("slope,aspect_rel,shadow,layover", [
    (45.0, 180.0, False, True),    # faces radar, steeper than incidence 39° -> layover
    (30.0, 180.0, False, False),   # faces radar, gentler -> fine
    (55.0, 0.0, True, False),      # faces away, steeper than 90-39=51° -> shadow
    (45.0, 0.0, False, False),
    (60.0, 90.0, False, False),    # steep but side-on to the radar
])
def test_shadow_layover(slope, aspect_rel, shadow, layover):
    look = 78.0
    sh, lo = radar_shadow_layover(np.array([slope]), np.array([look + aspect_rel]), look, 39.0)
    assert bool(sh[0]) is shadow and bool(lo[0]) is layover


def test_look_geometry(settings):
    g = look_geometry("ascending", None, settings.terrain)
    assert g.look_azimuth_deg == pytest.approx(78.0) and g.incidence_deg == 39.0 and g.source == "nominal"
    assert look_geometry("descending", 41.5, settings.terrain).look_azimuth_deg == pytest.approx(282.0)
    assert look_geometry(None, None, settings.terrain).look_azimuth_deg is None


def _follow_to_outlet(parent, start, limit):
    c, steps = start, 0
    while parent[c] >= 0:
        c = parent[c]
        steps += 1
        assert steps < limit, "cycle in flow routing"
    return c


def test_priority_flood_fills_pits_and_routes_everything():
    dem = plane(15, 15, 30, 5, "east") + 100
    dem[7, 7] = 50.0                      # deep pit
    dem[0, 3] = np.nan                    # a hole on the border
    order, parent, filled = priority_flood_routing(dem, eps=1e-3)
    valid = np.isfinite(dem).ravel()
    assert len(order) == valid.sum() and len(set(order.tolist())) == len(order)
    assert np.all(filled[np.isfinite(dem)] >= dem[np.isfinite(dem)] - 1e-9)
    assert filled[7, 7] > 100             # pit filled up to its spill level
    pos = {c: i for i, c in enumerate(order.tolist())}
    for c in order.tolist():
        if parent[c] >= 0:
            assert pos[parent[c]] < pos[c], "receiver must be processed before donor"
            assert filled.ravel()[parent[c]] < filled.ravel()[c] + 1e-12
        _follow_to_outlet(parent, c, dem.size)


def test_flow_accumulation_conserves_area():
    dem = plane(10, 12, 30, 3, "east")    # drains to the west edge
    order, parent, _ = priority_flood_routing(dem)
    acc = flow_accumulation(order, parent, dem.size, 900.0)
    outlets = np.flatnonzero(parent < 0)
    assert acc[outlets].sum() == pytest.approx(dem.size * 900.0)


def test_hand_on_synthetic_valley():
    pair = make_synthetic_s1_pair()
    z = pair.dem.band("elevation").astype(float)
    order, parent, _ = priority_flood_routing(z)
    acc_km2 = flow_accumulation(order, parent, z.size, 900.0).reshape(z.shape) / 1e6
    hand = height_above_nearest_drainage(z, order, parent, acc_km2 >= 0.05)
    axis = z.shape[1] // 2
    xs = (np.arange(z.shape[1]) + 0.5) * 30 - z.shape[1] * 30 / 2
    lower = slice(z.shape[0] // 2, z.shape[0])   # downstream half: drainage network well developed
    assert np.nanmedian(hand[lower, axis - 1:axis + 1]) < 2.0
    floodplain = np.abs(xs) < 300
    assert np.nanmedian(hand[lower][:, floodplain]) < 8.0
    side = np.abs(xs) > 650
    assert np.nanmedian(hand[lower][:, side]) > 100.0


def test_compute_terrain_on_analysis_grid(settings):
    pair = make_synthetic_s1_pair()
    grid = make_analysis_grid(synthetic_aoi(pair, inset_m=30), 10, 0)
    cfg = settings.terrain.model_copy(update={"hand_stream_threshold_km2": 0.05, "margin_m": 0})
    tp = compute_terrain(pair.dem, grid, cfg, look_geometry("ascending", None, cfg))
    s, h = tp.slope_deg.data[0], tp.hand_m.data[0]
    assert tp.slope_deg.shape == grid.spec.shape
    cols = (np.arange(grid.spec.width) + 0.5) * 10
    x_axis = (pair.before.bounds[0] + pair.before.bounds[2]) / 2 - grid.spec.transform[2]
    d = np.abs(cols - x_axis)
    assert np.nanmedian(s[:, d < 300]) < 3.0
    assert np.nanmedian(s[:, d > 600]) == pytest.approx(30.0, abs=2.0)
    # 30° slopes are neither shadow (needs ≥51° facing away) nor layover (needs >39° facing radar)
    assert not tp.shadow.any() and not tp.layover.any()
    flags = terrain_flag_raster(tp, 15, 25, 35)
    assert flags.dtype == np.uint8
    assert (flags[:, d > 600] & 4).all()          # STEEP_FOR_WATER on valley sides
    assert not (flags[:, d < 200] & 4).any()
    # DEM covers the analysis extent but margin_m=0 was requested -> no coverage warning
    assert tp.stats["dem_coverage_fraction"] > 0.95


def test_compute_terrain_without_orbit_direction_warns(settings):
    pair = make_synthetic_s1_pair()
    grid = make_analysis_grid(synthetic_aoi(pair, inset_m=30), 10, 0)
    cfg = settings.terrain.model_copy(update={"margin_m": 0})
    tp = compute_terrain(pair.dem, grid, cfg, look_geometry(None, None, cfg))
    assert any("NOT masked" in w for w in tp.warnings)


def test_dem_smaller_than_margin_warns(settings):
    pair = make_synthetic_s1_pair()
    grid = make_analysis_grid(synthetic_aoi(pair, inset_m=30), 10, 0)
    tp = compute_terrain(pair.dem, grid, settings.terrain, look_geometry("ascending", None, settings.terrain))
    assert any("DEM covers only" in w for w in tp.warnings)
