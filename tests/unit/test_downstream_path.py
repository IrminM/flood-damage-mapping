"""Unit tests for Stage 6 DEM downstream flood path tracing and settlement detection."""
import numpy as np
import pytest
from shapely.geometry import Point

from floodmap.core.config import DownstreamConfig
from floodmap.core.georaster import GeoRaster
from floodmap.dem.settlements import trace_nearby_settlements
from floodmap.dem.tracer import parse_upstream_point, trace_downstream_path


def make_synthetic_ramp_dem(rows=50, cols=50, res_deg=0.001):
    """Synthetic DEM sloping downwards from North-West (1000m) to South-East (100m)."""
    r_idx, c_idx = np.mgrid[0:rows, 0:cols]
    elev = 1000.0 - (r_idx * 10.0 + c_idx * 5.0)
    # Bounds: left=85.0, top=28.0
    transform = (res_deg, 0.0, 85.0, 0.0, -res_deg, 28.0)
    data = elev.astype(np.float32)[np.newaxis, ...]
    return GeoRaster(data=data, transform=transform, crs="EPSG:4326", nodata=-9999.0)


def test_parse_upstream_point():
    assert parse_upstream_point((85.1, 27.9)) == (85.1, 27.9)
    assert parse_upstream_point([85.1, 27.9]) == (85.1, 27.9)
    assert parse_upstream_point(Point(85.1, 27.9)) == (85.1, 27.9)
    assert parse_upstream_point({"coordinates": [85.1, 27.9]}) == (85.1, 27.9)
    assert parse_upstream_point("85.1, 27.9") == (85.1, 27.9)

    with pytest.raises(ValueError):
        parse_upstream_point("invalid_point")


def test_upstream_point_out_of_bounds():
    dem = make_synthetic_ramp_dem()
    cfg = DownstreamConfig()
    # Point outside [85.0, 27.95] to [85.05, 28.0]
    with pytest.raises(ValueError, match="outside DEM bounds"):
        trace_downstream_path(dem, (80.0, 20.0), cfg)


def test_snapping_to_nearest_valid_cell():
    dem = make_synthetic_ramp_dem(rows=20, cols=20)
    # Set top-left corner cells to NaN
    dem.data[0, 0, 0] = np.nan
    dem.data[0, 0, 1] = np.nan

    cfg = DownstreamConfig()
    # Query the NaN pixel center (c=0, r=0)
    lon0 = 85.0 + 0.0005
    lat0 = 28.0 - 0.0005
    res = trace_downstream_path(dem, (lon0, lat0), cfg)

    assert res.snapping_distance_m > 0.0
    assert np.isfinite(res.start_elevation_m)
    assert res.start_elevation_m > 0.0


def test_downhill_movement_and_elevation_drop():
    dem = make_synthetic_ramp_dem(rows=30, cols=30)
    cfg = DownstreamConfig(max_steps=50, max_length_m=10000.0)

    # Start near top: c=5, r=5
    lon = 85.0 + 5 * 0.001 + 0.0005
    lat = 28.0 - 5 * 0.001 - 0.0005
    res = trace_downstream_path(dem, (lon, lat), cfg)

    assert len(res.coords_lonlat) > 5
    assert res.total_length_m > 0.0
    assert res.total_drop_m > 0.0
    assert res.start_elevation_m > res.end_elevation_m

    # Verify downhill trajectory
    for i in range(len(res.elevations_m) - 1):
        assert res.elevations_m[i] >= res.elevations_m[i + 1]


def test_flat_and_sink_handling():
    dem = make_synthetic_ramp_dem(rows=20, cols=20)
    # Create an artificial pit at r=5, c=5: surround with higher cells
    dem.data[0, 5, 5] = 200.0  # very low
    dem.data[0, 6, 5] = 800.0  # walls
    dem.data[0, 5, 6] = 800.0
    dem.data[0, 4, 5] = 800.0
    dem.data[0, 5, 4] = 800.0

    # Start at pit (5, 5) with sink_handling="stop"
    cfg_stop = DownstreamConfig(sink_handling="stop")
    lon = 85.0 + 5 * 0.001 + 0.0005
    lat = 28.0 - 5 * 0.001 - 0.0005
    res_stop = trace_downstream_path(dem, (lon, lat), cfg_stop)
    assert "depression" in res_stop.termination_reason.lower() or "sink" in res_stop.termination_reason.lower()

    # With sink_handling="fill", depression filling resolves the overflow
    cfg_fill = DownstreamConfig(sink_handling="fill")
    res_fill = trace_downstream_path(dem, (lon, lat), cfg_fill)
    assert len(res_fill.coords_lonlat) >= 1


def test_path_termination_limits():
    dem = make_synthetic_ramp_dem(rows=40, cols=40)

    # 1. Terminate on max_steps
    cfg_steps = DownstreamConfig(max_steps=5)
    lon = 85.0 + 2 * 0.001
    lat = 28.0 - 2 * 0.001
    res_steps = trace_downstream_path(dem, (lon, lat), cfg_steps)
    assert len(res_steps.coords_lonlat) <= 6

    # 2. Terminate on max_length_m
    cfg_len = DownstreamConfig(max_length_m=300.0)
    res_len = trace_downstream_path(dem, (lon, lat), cfg_len)
    assert res_len.total_length_m <= 600.0  # Stops on step exceeding limit


def test_settlement_proximity_and_ordering():
    dem = make_synthetic_ramp_dem(rows=30, cols=30)
    cfg = DownstreamConfig(settlement_proximity_radius_m=500.0, max_steps=50)

    lon_start = 85.0 + 2 * 0.001
    lat_start = 28.0 - 2 * 0.001
    path_res = trace_downstream_path(dem, (lon_start, lat_start), cfg)

    # Place 3 synthetic settlements along the path
    # Settlement 1: near step 3
    lon1, lat1 = path_res.coords_lonlat[3]
    # Settlement 2: near step 8
    lon2, lat2 = path_res.coords_lonlat[8]
    # Settlement 3: far off to the side (2000m away, should be excluded)
    lon3 = lon1 + 0.03
    lat3 = lat1 + 0.03

    places_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "place/2",
                "properties": {"osm_id": "place/2", "name": "Second Village", "place": "village"},
                "geometry": {"type": "Point", "coordinates": [lon2, lat2]},
            },
            {
                "type": "Feature",
                "id": "place/1",
                "properties": {"osm_id": "place/1", "name": "First Hamlet", "place": "hamlet"},
                "geometry": {"type": "Point", "coordinates": [lon1, lat1]},
            },
            {
                "type": "Feature",
                "id": "place/far",
                "properties": {"osm_id": "place/far", "name": "Distant Town", "place": "town"},
                "geometry": {"type": "Point", "coordinates": [lon3, lat3]},
            },
        ],
    }

    settlements_gdf, records = trace_nearby_settlements(path_res, places_fc, cfg)

    # Distant town must be excluded
    assert len(records) == 2
    assert len(settlements_gdf) == 2

    # Check chronological downstream sequence: First Hamlet must be order 1, Second Village order 2
    assert records[0].order == 1
    assert records[0].settlement_id == "place/1"
    assert records[1].order == 2
    assert records[1].settlement_id == "place/2"
    assert records[0].distance_along_path_m < records[1].distance_along_path_m
