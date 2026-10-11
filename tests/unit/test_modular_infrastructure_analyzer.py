"""Unit tests for the modular Stage-4 Infrastructure Impact Analyzer.

Tests all components of floodmap.gis.infrastructure_analyzer:
- OSMLoader (date verification, missing meta, layer loading)
- InfrastructureNormalizer (geometry fixing, ID assignment, CRS projection)
- EvidenceSpatialIndex (category filtering, R-tree indexing)
- BuildingImpactAnalyzer (footprint intersection, affected fraction, tier hierarchy)
- RoadImpactAnalyzer (centerline buffering, length calculation, highway breakdown)
- BridgeImpactAnalyzer (proximity buffer, flood approach classification)
- InfrastructureImpactResult (export of 6 required deliverables)
- Full coordinate workflow via analyze_infrastructure_impact
"""
import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from floodmap.core.config import ImpactConfig
from floodmap.core.contracts import DataMode
from floodmap.core.errors import InputDataError
from floodmap.gis.infrastructure_analyzer import (
    BridgeImpactAnalyzer,
    BuildingImpactAnalyzer,
    EvidenceSpatialIndex,
    InfrastructureImpactResult,
    InfrastructureNormalizer,
    OSMLoader,
    RoadImpactAnalyzer,
    analyze_infrastructure_impact,
)


@pytest.fixture
def sample_event_date():
    return date(2026, 8, 26)


@pytest.fixture
def sample_evidence_gdf():
    """Evidence polygons in EPSG:32645."""
    # Poly 1: Dual evidence (high confidence flood) [100, 100] to [200, 200]
    p1 = Polygon([(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)])
    # Poly 2: AI only candidate [300, 300] to [400, 400]
    p2 = Polygon([(300, 300), (400, 300), (400, 400), (300, 400), (300, 300)])
    # Poly 3: SAR hillside change (unvalidated) [500, 500] to [600, 600]
    p3 = Polygon([(500, 500), (600, 500), (600, 600), (500, 600), (500, 500)])
    # Poly 4: Terrain rejected (should be filtered out by EvidenceSpatialIndex)
    p4 = Polygon([(700, 700), (800, 700), (800, 800), (700, 800), (700, 700)])

    return gpd.GeoDataFrame(
        {
            "category_name": [
                "ai_water_and_sar_valley_change",
                "ai_only_water",
                "sar_only_hillside_change",
                "terrain_rejected",
            ],
            "strength_name": [
                "high_confidence",
                "ai_only",
                "unvalidated",
                "rejected",
            ],
        },
        geometry=[p1, p2, p3, p4],
        crs="EPSG:32645",
    )


# ==============================================================================
# OSMLoader Tests
# ==============================================================================

def test_osm_loader_valid(tmp_path: Path, sample_event_date: date):
    meta = {
        "snapshot_date": "2026-07-27",
        "source": "ohsome API test",
        "bbox": [85.0, 27.8, 85.3, 28.0],
    }
    (tmp_path / "snapshot_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "buildings.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )

    loader = OSMLoader()
    data = loader.load_from_dir(tmp_path, sample_event_date)
    assert data.meta.snapshot_date == date(2026, 7, 27)
    assert data.meta.source == "ohsome API test"
    assert "buildings" in data.layers


def test_osm_loader_missing_meta(tmp_path: Path, sample_event_date: date):
    loader = OSMLoader()
    with pytest.raises(InputDataError, match="snapshot_meta.json missing"):
        loader.load_from_dir(tmp_path, sample_event_date)


def test_osm_loader_post_event_prohibited(tmp_path: Path, sample_event_date: date):
    meta = {"snapshot_date": "2026-08-30", "source": "post_event"}
    (tmp_path / "snapshot_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    loader = OSMLoader()
    with pytest.raises(InputDataError, match="Post-event OSM data is prohibited"):
        loader.load_from_dir(tmp_path, sample_event_date)


# ==============================================================================
# InfrastructureNormalizer Tests
# ==============================================================================

def test_normalizer_reprojection_and_cleaning():
    normalizer = InfrastructureNormalizer(target_crs="EPSG:32645")
    # Coordinates roughly in Trishuli (EPSG:4326)
    # Longitude ~85.16, Latitude ~27.97
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "way/101",
                "properties": {"building": "yes"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [85.160, 27.970],
                            [85.161, 27.970],
                            [85.161, 27.971],
                            [85.160, 27.971],
                            [85.160, 27.970],
                        ]
                    ],
                },
            },
            # Feature with no ID (should receive fallback)
            {
                "type": "Feature",
                "properties": {"highway": "residential"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[85.160, 27.970], [85.165, 27.975]],
                },
            },
        ],
    }

    b_gdf = normalizer.normalize(fc, "buildings", src_crs="EPSG:4326")
    assert len(b_gdf) == 2
    assert b_gdf.crs == "EPSG:32645"
    assert b_gdf.loc[0, "id"] == "way/101"
    assert b_gdf.loc[1, "id"] == "buildings/1"


def test_normalizer_empty_and_invalid():
    normalizer = InfrastructureNormalizer(target_crs="EPSG:32645")
    empty_fc = {"type": "FeatureCollection", "features": []}
    res = normalizer.normalize(empty_fc, "test")
    assert res.empty
    assert res.crs == "EPSG:32645"

    # Bowtie invalid polygon
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)])
    gdf_invalid = gpd.GeoDataFrame({"id": ["b1"]}, geometry=[bowtie], crs="EPSG:32645")
    normalized = normalizer.normalize(gdf_invalid, "test")
    assert len(normalized) == 1
    assert normalized.geometry.iloc[0].is_valid


# ==============================================================================
# EvidenceSpatialIndex Tests
# ==============================================================================

def test_evidence_spatial_index(sample_evidence_gdf: gpd.GeoDataFrame):
    idx = EvidenceSpatialIndex(sample_evidence_gdf, target_crs="EPSG:32645")
    assert not idx.is_empty
    # terrain_rejected should be excluded
    cats = set(idx.filtered_gdf["category_name"])
    assert "terrain_rejected" not in cats
    assert "ai_water_and_sar_valley_change" in cats
    assert len(idx.filtered_gdf) == 3


# ==============================================================================
# BuildingImpactAnalyzer Tests
# ==============================================================================

def test_building_impact_analyzer(sample_evidence_gdf: gpd.GeoDataFrame):
    idx = EvidenceSpatialIndex(sample_evidence_gdf, target_crs="EPSG:32645")
    cfg = ImpactConfig(building_buffer_m=0.0, min_affected_building_area_m2=1.0)
    analyzer = BuildingImpactAnalyzer(config=cfg)

    # Building 1: inside dual evidence [120, 120] to [150, 150] (area = 900 m²)
    b1 = Polygon([(120, 120), (150, 120), (150, 150), (120, 150), (120, 120)])
    # Building 2: inside AI-only [310, 310] to [330, 330] (area = 400 m²)
    b2 = Polygon([(310, 310), (330, 310), (330, 330), (310, 330), (310, 330)])
    # Building 3: far outside [900, 900] to [920, 920] (area = 400 m²)
    b3 = Polygon([(900, 900), (920, 900), (920, 920), (900, 920), (900, 920)])

    gdf = gpd.GeoDataFrame(
        {"id": ["b1", "b2", "b3"]},
        geometry=[b1, b2, b3],
        crs="EPSG:32645",
    )

    analyzed, stats = analyzer.analyze(gdf, idx)
    assert stats.total_count == 3
    assert stats.potentially_affected_count == 2
    assert stats.affected_pct == pytest.approx(66.67, 0.01)

    # Check building 1 (dual high confidence)
    row1 = analyzed.loc[analyzed["id"] == "b1"].iloc[0]
    assert bool(row1["is_affected"]) is True
    assert row1["evidence_tier"] == "confirmed_dual_evidence"
    assert row1["evidence_source"] == "high_confidence"
    assert row1["affected_area_m2"] == pytest.approx(900.0, 1.0)
    assert row1["affected_fraction"] == 1.0
    assert bool(row1["structural_damage_claimed"]) is False

    # Check building 2 (AI only candidate)
    row2 = analyzed.loc[analyzed["id"] == "b2"].iloc[0]
    assert bool(row2["is_affected"]) is True
    assert row2["evidence_tier"] == "ai_only_candidate"
    assert bool(row2["structural_damage_claimed"]) is False

    # Check building 3 (clear)
    row3 = analyzed.loc[analyzed["id"] == "b3"].iloc[0]
    assert bool(row3["is_affected"]) is False
    assert row3["evidence_tier"] == "none"


# ==============================================================================
# RoadImpactAnalyzer Tests
# ==============================================================================

def test_road_impact_analyzer(sample_evidence_gdf: gpd.GeoDataFrame):
    idx = EvidenceSpatialIndex(sample_evidence_gdf, target_crs="EPSG:32645")
    cfg = ImpactConfig(road_buffer_m=5.0, min_affected_road_length_m=5.0)
    analyzer = RoadImpactAnalyzer(config=cfg)

    # Road 1: crosses dual evidence polygon [100, 100] to [200, 200]
    # Line from (50, 150) to (250, 150) -> length = 200 m
    # With 5m buffer on polygon, intersection length along the line: [95, 205] = 110 m
    r1 = LineString([(50, 150), (250, 150)])
    # Road 2: clear road [900, 900] to [950, 950]
    r2 = LineString([(900, 900), (950, 950)])

    gdf = gpd.GeoDataFrame(
        {"id": ["r1", "r2"], "highway": ["primary", "residential"]},
        geometry=[r1, r2],
        crs="EPSG:32645",
    )

    analyzed, stats = analyzer.analyze(gdf, idx)
    assert stats.total_segments == 2
    assert stats.potentially_affected_segments == 1

    row1 = analyzed.loc[analyzed["id"] == "r1"].iloc[0]
    assert bool(row1["is_affected"]) is True
    assert row1["evidence_tier"] == "confirmed_dual_evidence"
    assert row1["affected_length_m"] > 90.0
    assert bool(row1["structural_damage_claimed"]) is False

    # Verify highway breakdown
    assert "primary" in stats.by_highway_type_km
    assert stats.by_highway_type_km["primary"]["affected_km"] > 0.09
    assert stats.by_highway_type_km["residential"]["affected_km"] == 0.0


# ==============================================================================
# BridgeImpactAnalyzer Tests
# ==============================================================================

def test_bridge_impact_analyzer(sample_evidence_gdf: gpd.GeoDataFrame):
    idx = EvidenceSpatialIndex(sample_evidence_gdf, target_crs="EPSG:32645")
    cfg = ImpactConfig(bridge_buffer_m=15.0)
    analyzer = BridgeImpactAnalyzer(config=cfg)

    # Bridge 1: Point at (205, 150) -> 5 m from dual evidence boundary (within 15m buffer)
    br1 = Point(205, 150)
    # Bridge 2: Point at (800, 800) -> far away
    br2 = Point(800, 800)

    gdf = gpd.GeoDataFrame(
        {"id": ["br1", "br2"], "bridge": ["yes", "yes"]},
        geometry=[br1, br2],
        crs="EPSG:32645",
    )

    analyzed, stats = analyzer.analyze(gdf, idx)
    assert stats.total_count == 2
    assert stats.potentially_affected_count == 1
    assert stats.affected_pct == 50.0

    row1 = analyzed.loc[analyzed["id"] == "br1"].iloc[0]
    assert bool(row1["is_affected"]) is True
    assert row1["evidence_tier"] == "confirmed_dual_evidence"
    assert bool(row1["structural_damage_claimed"]) is False
    assert "water passing beneath" in row1["impact_description"].lower()


# ==============================================================================
# Save Outputs & Coordination Tests
# ==============================================================================

def test_analyze_and_save_outputs(tmp_path: Path, sample_event_date: date):
    # Setup mock pre-event OSM directory
    osm_dir = tmp_path / "osm"
    osm_dir.mkdir()
    meta = {
        "snapshot_date": "2026-07-27",
        "source": "ohsome API test",
        "bbox": [85.15, 27.95, 85.20, 28.00],
    }
    (osm_dir / "snapshot_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # Buildings GeoJSON in realistic EPSG:4326 coords
    b_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "b1",
                "properties": {"building": "yes"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [85.160, 27.970],
                            [85.161, 27.970],
                            [85.161, 27.971],
                            [85.160, 27.971],
                            [85.160, 27.970],
                        ]
                    ],
                },
            }
        ],
    }
    (osm_dir / "buildings.geojson").write_text(json.dumps(b_fc), encoding="utf-8")
    (osm_dir / "roads.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    (osm_dir / "bridges.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")

    # Target output directory
    out_dir = tmp_path / "stage4_outputs"

    # Create evidence polygon that overlaps the building in EPSG:32645
    # First project building to get its bounds in EPSG:32645
    normalizer = InfrastructureNormalizer(target_crs="EPSG:32645")
    b_proj = normalizer.normalize(b_fc, "buildings")
    ev_geom = b_proj.geometry.iloc[0].buffer(10.0)

    evidence_gdf = gpd.GeoDataFrame(
        {
            "category_name": ["ai_water_and_sar_valley_change"],
            "strength_name": ["high_confidence"],
        },
        geometry=[ev_geom],
        crs="EPSG:32645",
    )

    result = analyze_infrastructure_impact(
        osm_source=osm_dir,
        evidence_source=evidence_gdf,
        event_date=sample_event_date,
        target_crs="EPSG:32645",
        data_mode=DataMode.SYNTHETIC_TEST,
        aoi_name="test_aoi",
    )

    outputs = result.save_outputs(out_dir)

    # Verify all 6 files are saved
    assert (out_dir / "infrastructure_impact.json").exists()
    assert (out_dir / "affected_buildings.geojson").exists()
    assert (out_dir / "affected_roads.geojson").exists()
    assert (out_dir / "affected_bridges.geojson").exists()
    assert (out_dir / "infrastructure_summary.json").exists()
    assert (out_dir / "provenance.json").exists()

    # Verify non-structural damage disclaimer in JSON
    impact_json = json.loads((out_dir / "infrastructure_impact.json").read_text(encoding="utf-8"))
    assert "scientific_disclaimer" in impact_json
    assert "structural building destruction" in impact_json["scientific_disclaimer"]

    # Verify provenance JSON
    prov_json = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    assert prov_json["scientific_principles"]["structural_damage_claimed"] is False
    assert prov_json["inputs"]["osm_pre_event"]["snapshot_date"] == "2026-07-27"
