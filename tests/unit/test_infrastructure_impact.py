"""Unit tests for infrastructure impact spatial assessment (floodmap.gis.impact)."""
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from floodmap.core.config import ImpactConfig
from floodmap.gis.impact import (
    assess_bridges,
    assess_buildings,
    assess_roads,
    assess_waterway_context,
)


@pytest.fixture
def metric_evidence_gdf():
    # Flood polygon covering [100, 100] to [200, 200]
    flood_poly = Polygon([(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)])
    # Debris polygon covering [300, 300] to [350, 350]
    debris_poly = Polygon([(300, 300), (350, 300), (350, 350), (300, 350), (300, 300)])

    gdf = gpd.GeoDataFrame(
        {
            "class_code": [1, 3],
            "class_name": ["flood_water", "debris_or_change"],
            "evidence_source": [3, 1],
            "evidence_name": ["confirmed_by_both", "change_detection_only"],
        },
        geometry=[flood_poly, debris_poly],
        crs="EPSG:32645",
    )
    return gdf


@pytest.fixture
def impact_config():
    return ImpactConfig(
        building_buffer_m=0.0,
        min_affected_building_area_m2=1.0,
        road_buffer_m=5.0,
        min_affected_road_length_m=5.0,
        bridge_buffer_m=15.0,
        waterway_buffer_m=50.0,
    )


def test_assess_buildings(metric_evidence_gdf, impact_config):
    # Create GeoJSON features in EPSG:4326 (convert from known coordinates)
    # We test with UTM coordinates directly reprojected or mocked:
    # Building 1: inside flood [120, 120] to [140, 140] (area = 400 m²)
    # Building 2: partially intersecting flood [180, 180] to [220, 220] (overlap [180, 180] to [200, 200], area = 400 m² overlap)
    # Building 3: far away [500, 500] to [520, 520] (outside)
    # Self-intersecting invalid building: bowtie polygon
    target_crs = "EPSG:32645"

    b1 = Polygon([(120, 120), (140, 120), (140, 140), (120, 140), (120, 120)])
    b2 = Polygon([(180, 180), (220, 180), (220, 220), (180, 220), (180, 180)])
    b3 = Polygon([(500, 500), (520, 500), (520, 520), (500, 520), (500, 520)])
    # Bowtie: (0,0) -> (10,10) -> (10,0) -> (0,10) -> (0,0)
    b_invalid = Polygon([(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)])

    # Construct GeoDataFrame in metric and convert to EPSG:4326 for the input fc
    b_metric = gpd.GeoDataFrame(
        {"id": ["way/1", "way/2", "way/3", "way/4"], "building": ["yes", "residential", "school", "yes"]},
        geometry=[b1, b2, b3, b_invalid],
        crs=target_crs,
    )
    b_4326 = b_metric.to_crs("EPSG:4326")
    fc = {"type": "FeatureCollection", "features": b_4326.__geo_interface__["features"]}

    res_gdf, stats = assess_buildings(fc, metric_evidence_gdf, target_crs, impact_config)

    assert stats.total_count == 4
    assert stats.potentially_affected_count == 2
    assert stats.affected_pct == 50.0

    b1_row = res_gdf[res_gdf["id"] == "way/1"].iloc[0]
    assert bool(b1_row["is_affected"]) is True
    assert b1_row["affected_area_m2"] == pytest.approx(400.0, abs=1.0)
    assert b1_row["affected_fraction"] == pytest.approx(1.0, abs=0.01)
    assert b1_row["impact_class"] == "flood_water"
    assert b1_row["evidence_source"] == "confirmed_by_both"

    b2_row = res_gdf[res_gdf["id"] == "way/2"].iloc[0]
    assert bool(b2_row["is_affected"]) is True
    assert b2_row["affected_area_m2"] == pytest.approx(400.0, abs=1.0)
    assert b2_row["affected_fraction"] == pytest.approx(0.25, abs=0.01)

    b3_row = res_gdf[res_gdf["id"] == "way/3"].iloc[0]
    assert bool(b3_row["is_affected"]) is False
    assert b3_row["affected_area_m2"] == 0.0


def test_assess_roads(metric_evidence_gdf, impact_config):
    target_crs = "EPSG:32645"
    # Road 1: crosses flood polygon from (50, 150) to (250, 150). Total length 200m.
    # Inside flood [100, 150] to [200, 150] = 100m.
    # With 5m buffer on flood polygon: inside dilated flood [95, 150] to [205, 150] = 110m.
    r1 = LineString([(50, 150), (250, 150)])
    # Road 2: entirely outside from (500, 500) to (500, 600). Length 100m.
    r2 = LineString([(500, 500), (500, 600)])

    r_metric = gpd.GeoDataFrame(
        {"id": ["way/101", "way/102"], "highway": ["primary", "track"]},
        geometry=[r1, r2],
        crs=target_crs,
    )
    r_4326 = r_metric.to_crs("EPSG:4326")
    fc = {"type": "FeatureCollection", "features": r_4326.__geo_interface__["features"]}

    res_gdf, stats = assess_roads(fc, metric_evidence_gdf, target_crs, impact_config)

    assert stats.total_segments == 2
    assert stats.potentially_affected_segments == 1
    assert stats.total_length_km == pytest.approx(0.30, abs=0.01)
    assert stats.potentially_affected_length_km > 0.10

    r1_row = res_gdf[res_gdf["id"] == "way/101"].iloc[0]
    assert bool(r1_row["is_affected"]) is True
    assert r1_row["affected_length_m"] >= 100.0
    assert r1_row["impact_class"] == "flood_water"
    assert r1_row["highway"] == "primary"

    r2_row = res_gdf[res_gdf["id"] == "way/102"].iloc[0]
    assert bool(r2_row["is_affected"]) is False
    assert r2_row["affected_length_m"] == 0.0


def test_assess_bridges(metric_evidence_gdf, impact_config):
    target_crs = "EPSG:32645"
    # Bridge 1: right near flood edge at (205, 150) to (215, 150) (within 15m bridge buffer)
    br1 = LineString([(205, 150), (215, 150)])
    # Bridge 2: far away at (600, 600) to (610, 600)
    br2 = LineString([(600, 600), (610, 600)])

    br_metric = gpd.GeoDataFrame(
        {"id": ["way/201", "way/202"], "man_made": ["bridge", "bridge"]},
        geometry=[br1, br2],
        crs=target_crs,
    )
    br_4326 = br_metric.to_crs("EPSG:4326")
    fc = {"type": "FeatureCollection", "features": br_4326.__geo_interface__["features"]}

    res_gdf, stats = assess_bridges(fc, metric_evidence_gdf, target_crs, impact_config)

    assert stats.total_count == 2
    assert stats.potentially_affected_count == 1
    assert stats.affected_pct == 50.0

    br1_row = res_gdf[res_gdf["id"] == "way/201"].iloc[0]
    assert bool(br1_row["is_affected"]) is True
    assert br1_row["impact_class"] == "flood_water"

    br2_row = res_gdf[res_gdf["id"] == "way/202"].iloc[0]
    assert bool(br2_row["is_affected"]) is False


def test_assess_waterway_context(metric_evidence_gdf, impact_config):
    target_crs = "EPSG:32645"
    # River line passing right through debris polygon at (320, 200) to (320, 400)
    river = LineString([(320, 200), (320, 400)])
    w_metric = gpd.GeoDataFrame({"id": ["way/301"], "waterway": ["river"]}, geometry=[river], crs=target_crs)
    fc = {"type": "FeatureCollection", "features": w_metric.to_crs("EPSG:4326").__geo_interface__["features"]}

    w_stats = assess_waterway_context(fc, metric_evidence_gdf, target_crs, impact_config)
    assert w_stats is not None
    assert w_stats.total_waterways_km == pytest.approx(0.20, abs=0.01)
    # The debris polygon at [300, 300] to [350, 350] is within 50m of x=320
    assert w_stats.class3_near_waterway_km2 > 0.0
