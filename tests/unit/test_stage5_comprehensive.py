"""Comprehensive Unit Tests for Stage 5 Road Network Connectivity Analysis.

Tests all required validation areas:
1. Graph construction (nodes, edges, attributes, geometry preservation)
2. Road segment length (metric calculation and aggregation)
3. Node snapping (within tolerance vs exceeding max_snapping_distance)
4. Baseline route existence (path, distance, travel time)
5. Route removal after affected edge (hazard-aware disconnection)
6. Alternate route detection (detour bypass identified)
7. Disconnected component detection (component partitioning, isolated components)
8. No-baseline-route handling (pre-event isolated islands not labeled cut off)
9. Evidence provenance propagation (evidence tiers mapped to disrupting roads)
10. Deterministic results (repeatability verification)
"""
import math
from datetime import date
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Point

from floodmap.core.config import ConnectivityConfig
from floodmap.core.contracts import SettlementStatus
from floodmap.network.connectivity import (
    analyze_disconnected_components,
    evaluate_connectivity,
    extract_destinations,
    extract_settlements,
)
from floodmap.network.graph import RoadGraph, build_road_graph


@pytest.fixture
def test_network():
    """Build a multi-branch test road network in EPSG:32645:
    
    Nodes:
      A (1000, 1000) - Village Alpha
      B (2000, 1000) - Main Junction
      C (3000, 1000) - District Town (Destination 1)
      D (2000, 2000) - Northern bypass detour junction
      E (1000, 3000) - Hamlet Beta (Cul-de-sac)
      F (2000, 3000) - Cul-de-sac junction
      H (3000, 2000) - Regional Hospital (Destination 2)
      I (8000, 8000) - Island Road West
      J (8500, 8000) - Island Road East / Island Settlement
      Z (9999, 9999) - Remote Settlement (far outside network)
    """
    roads = [
        {"osm_id": "road_A_B", "highway": "primary", "geometry": LineString([(1000, 1000), (2000, 1000)])},
        {"osm_id": "road_B_C", "highway": "primary", "geometry": LineString([(2000, 1000), (3000, 1000)])},
        {"osm_id": "road_B_D", "highway": "secondary", "geometry": LineString([(2000, 1000), (2000, 2000)])},
        {"osm_id": "road_D_C", "highway": "secondary", "geometry": LineString([(2000, 2000), (3000, 1000)])},
        {"osm_id": "road_D_H", "highway": "tertiary", "geometry": LineString([(2000, 2000), (3000, 2000)])},
        {"osm_id": "road_E_F", "highway": "residential", "geometry": LineString([(1000, 3000), (2000, 3000)])},
        {"osm_id": "road_F_D", "highway": "residential", "geometry": LineString([(2000, 3000), (2000, 2000)])},
        {"osm_id": "road_island", "highway": "track", "geometry": LineString([(8000, 8000), (8500, 8000)])},
    ]
    roads_gdf = gpd.GeoDataFrame(roads, crs="EPSG:32645")

    places = [
        {"osm_id": "dest_town", "name": "Bazaar Town", "place": "town", "geometry": Point(3000, 1000)},
        {"osm_id": "settle_alpha", "name": "Village Alpha", "place": "village", "population": 500, "geometry": Point(1000, 1000)},
        {"osm_id": "settle_beta", "name": "Hamlet Beta", "place": "hamlet", "population": 120, "geometry": Point(1000, 3000)},
        {"osm_id": "settle_island", "name": "Island Hamlet", "place": "hamlet", "geometry": Point(8500, 8000)},
        {"osm_id": "settle_remote", "name": "Remote Peak", "place": "isolated_dwelling", "geometry": Point(9999, 9999)},
    ]
    places_gdf = gpd.GeoDataFrame(places, crs="EPSG:32645")

    health = [
        {"osm_id": "dest_hospital", "name": "Valley Hospital", "amenity": "hospital", "geometry": Point(3000, 2000)}
    ]
    health_gdf = gpd.GeoDataFrame(health, crs="EPSG:32645")

    return roads_gdf, places_gdf, health_gdf


# ==============================================================================
# 1. Graph Construction
# ==============================================================================

def test_graph_construction(test_network):
    roads_gdf, _, _ = test_network
    affected_meta = {
        "road_B_C": {
            "is_affected": True,
            "evidence_tier": "confirmed_dual_evidence",
            "impact_class": "ai_water_and_sar_valley_change",
            "affected_length_m": 1000.0,
        }
    }

    rg = build_road_graph(
        roads_gdf=roads_gdf,
        affected_road_ids={"road_B_C"},
        target_metric_crs="EPSG:32645",
        affected_roads_meta=affected_meta,
    )

    assert rg.graph.number_of_nodes() > 0
    assert rg.total_edges_count == len(roads_gdf)
    assert rg.affected_edges_count == 1
    assert rg.metric_crs == "EPSG:32645"

    # Verify edge attributes preserved
    u, v = (2000.0, 1000.0), (3000.0, 1000.0)
    assert rg.graph.has_edge(u, v)
    edge_d = rg.graph[u][v]
    assert edge_d["osm_id"] == "road_B_C"
    assert edge_d["highway"] == "primary"
    assert edge_d["affected"] is True
    assert edge_d["edge_state"] == "POTENTIALLY_AFFECTED"
    assert edge_d["evidence_tier"] == "confirmed_dual_evidence"
    assert edge_d["length_m"] == pytest.approx(1000.0, 0.1)

    # Convert to GeoDataFrame export
    gdf = rg.to_geodataframe("EPSG:4326")
    assert len(gdf) == len(roads_gdf)
    assert "edge_state" in gdf.columns
    assert "evidence_tier" in gdf.columns


# ==============================================================================
# 2. Road Segment Length
# ==============================================================================

def test_road_segment_length(test_network):
    roads_gdf, _, _ = test_network
    rg = build_road_graph(roads_gdf, target_metric_crs="EPSG:32645")

    # road_D_C runs from (2000, 2000) to (3000, 1000)
    # Euclidean length: hypot(1000, 1000) = 1414.21356...
    u, v = (2000.0, 2000.0), (3000.0, 1000.0)
    assert rg.graph.has_edge(u, v)
    edge_d = rg.graph[u][v]
    expected = math.hypot(1000.0, 1000.0)
    assert edge_d["length_m"] == pytest.approx(expected, 0.01)


# ==============================================================================
# 3. Node Snapping
# ==============================================================================

def test_node_snapping(test_network):
    roads_gdf, _, _ = test_network
    rg = build_road_graph(roads_gdf, target_metric_crs="EPSG:32645")

    # Point close to (1000, 1000) -> distance ~ 50m
    pt_near = Point(1030.0, 1040.0)
    node, dist = rg.snap_point(pt_near, src_crs="EPSG:32645", max_distance_m=200.0)
    assert node == (1000.0, 1000.0)
    assert dist == pytest.approx(50.0, 0.1)

    # Point far away -> exceeds tolerance
    pt_far = Point(5000.0, 5000.0)
    node_none, dist_large = rg.snap_point(pt_far, src_crs="EPSG:32645", max_distance_m=200.0)
    assert node_none is None
    assert dist_large > 200.0


# ==============================================================================
# 4. Baseline Route Existence
# ==============================================================================

def test_baseline_route_existence(test_network):
    roads_gdf, places_gdf, health_gdf = test_network
    rg = build_road_graph(roads_gdf, target_metric_crs="EPSG:32645")
    cfg = ConnectivityConfig(disruption_mode="remove", max_snapping_distance_m=500.0)

    all_gdf, cut_off, records, _ = evaluate_connectivity(rg, places_gdf, health_gdf, cfg)

    # Village Alpha to Bazaar Town: A -> B -> C = 2000m
    alpha = next(r for r in records if r.settlement_id == "settle_alpha")
    assert alpha.status == SettlementStatus.CONNECTED
    assert alpha.classification == "CONNECTED"
    assert alpha.nearest_destination_id == "dest_town"
    assert alpha.baseline_distance_m == pytest.approx(2000.0, 1.0)
    assert alpha.detour_distance_m == pytest.approx(0.0, 0.1)
    assert alpha.population == 500


# ==============================================================================
# 5. Route Removal After Affected Edge
# ==============================================================================

def test_route_removal_after_affected_edge(test_network):
    roads_gdf, places_gdf, health_gdf = test_network
    # Affect sole cul-de-sac road E->F
    affected_meta = {
        "road_E_F": {
            "is_affected": True,
            "evidence_tier": "confirmed_dual_evidence",
            "impact_class": "flood_water",
        }
    }
    rg = build_road_graph(
        roads_gdf,
        affected_road_ids={"road_E_F"},
        target_metric_crs="EPSG:32645",
        affected_roads_meta=affected_meta,
    )
    cfg = ConnectivityConfig(disruption_mode="remove", max_snapping_distance_m=500.0)

    all_gdf, cut_off, records, _ = evaluate_connectivity(rg, places_gdf, health_gdf, cfg)

    beta = next(r for r in records if r.settlement_id == "settle_beta")
    assert beta.status == SettlementStatus.DISCONNECTED
    assert beta.classification == "DISCONNECTED"
    assert beta.connectivity_lost is True
    assert "road_E_F" in beta.disrupting_road_ids
    assert "confirmed_dual_evidence" in beta.disrupting_evidence_tiers
    assert len(cut_off) == 1


# ==============================================================================
# 6. Alternate Route Detection
# ==============================================================================

def test_alternate_route_detection(test_network):
    roads_gdf, places_gdf, health_gdf = test_network
    # Disrupt direct road B->C, but detour B->D->C exists
    rg = build_road_graph(roads_gdf, affected_road_ids={"road_B_C"}, target_metric_crs="EPSG:32645")
    cfg = ConnectivityConfig(disruption_mode="remove", max_snapping_distance_m=500.0)

    all_gdf, cut_off, records, _ = evaluate_connectivity(rg, places_gdf, health_gdf, cfg)

    alpha = next(r for r in records if r.settlement_id == "settle_alpha")
    # Alpha still CONNECTED via detour A->B->D->C
    assert alpha.status == SettlementStatus.CONNECTED
    assert alpha.classification == "CONNECTED"
    assert alpha.baseline_distance_m == pytest.approx(2000.0, 1.0)
    # Detour: A->B: 1000m + B->D: 1000m + D->C: 1414.2m = 3414.2m
    assert alpha.post_disruption_distance_m == pytest.approx(3414.2, 1.0)
    assert alpha.detour_distance_m == pytest.approx(1414.2, 1.0)
    assert alpha.alternate_route_distance_m == pytest.approx(3414.2, 1.0)
    assert "road_B_C" in alpha.disrupting_road_ids


# ==============================================================================
# 7. Disconnected Component Detection
# ==============================================================================

def test_disconnected_component_detection(test_network):
    roads_gdf, places_gdf, health_gdf = test_network
    # Disrupt road_B_C and road_B_D: splits network into two components
    rg = build_road_graph(roads_gdf, affected_road_ids={"road_B_C", "road_B_D"}, target_metric_crs="EPSG:32645")
    cfg = ConnectivityConfig(disruption_mode="remove", max_snapping_distance_m=500.0)

    all_gdf, cut_off, records, _ = evaluate_connectivity(rg, places_gdf, health_gdf, cfg)

    destinations = extract_destinations(places_gdf, health_gdf, rg, cfg)
    snapped = {
        "settle_alpha": (1000.0, 1000.0),
        "settle_beta": (1000.0, 3000.0),
        "settle_island": (8500.0, 8000.0),
    }

    comp_summary = analyze_disconnected_components(rg, destinations, records, snapped)
    assert comp_summary.total_components > comp_summary.baseline_components
    assert comp_summary.isolated_components_count >= 1


# ==============================================================================
# 8. No-Baseline-Route Handling
# ==============================================================================

def test_no_baseline_route_handling(test_network):
    roads_gdf, places_gdf, health_gdf = test_network
    rg = build_road_graph(roads_gdf, target_metric_crs="EPSG:32645")
    cfg = ConnectivityConfig(disruption_mode="remove", max_snapping_distance_m=500.0)

    all_gdf, cut_off, records, _ = evaluate_connectivity(rg, places_gdf, health_gdf, cfg)

    # Island Hamlet snaps to road_island, but road_island never connects to any town/hospital
    island = next(r for r in records if r.settlement_id == "settle_island")
    assert island.classification == "NO_BASELINE_ROUTE"
    assert "isolated road component" in island.isolation_reason

    # Remote Peak is far from any mapped road (>500m)
    remote = next(r for r in records if r.settlement_id == "settle_remote")
    assert remote.classification == "OUTSIDE_NETWORK"
    assert "exceeds maximum snapping tolerance" in remote.isolation_reason


# ==============================================================================
# 9. Evidence Provenance Propagation
# ==============================================================================

def test_evidence_provenance_propagation(test_network):
    roads_gdf, places_gdf, health_gdf = test_network
    affected_meta = {
        "road_A_B": {
            "is_affected": True,
            "evidence_tier": "sar_only_evidence",
            "impact_class": "sar_only_valley_change",
        }
    }
    rg = build_road_graph(
        roads_gdf,
        affected_road_ids={"road_A_B"},
        target_metric_crs="EPSG:32645",
        affected_roads_meta=affected_meta,
    )
    cfg = ConnectivityConfig(disruption_mode="remove", max_snapping_distance_m=500.0)

    all_gdf, cut_off, records, _ = evaluate_connectivity(rg, places_gdf, health_gdf, cfg)

    alpha = next(r for r in records if r.settlement_id == "settle_alpha")
    assert alpha.status == SettlementStatus.DISCONNECTED
    assert "road_A_B" in alpha.disrupting_road_ids
    assert "sar_only_evidence" in alpha.disrupting_evidence_tiers


# ==============================================================================
# 10. Deterministic Results
# ==============================================================================

def test_deterministic_results(test_network):
    roads_gdf, places_gdf, health_gdf = test_network
    cfg = ConnectivityConfig(disruption_mode="remove", max_snapping_distance_m=500.0)

    rg1 = build_road_graph(roads_gdf, affected_road_ids={"road_B_C"}, target_metric_crs="EPSG:32645")
    _, _, records1, _ = evaluate_connectivity(rg1, places_gdf, health_gdf, cfg)

    rg2 = build_road_graph(roads_gdf, affected_road_ids={"road_B_C"}, target_metric_crs="EPSG:32645")
    _, _, records2, _ = evaluate_connectivity(rg2, places_gdf, health_gdf, cfg)

    assert len(records1) == len(records2)
    for r1, r2 in zip(records1, records2):
        assert r1.settlement_id == r2.settlement_id
        assert r1.status == r2.status
        assert r1.classification == r2.classification
        assert r1.baseline_distance_m == r2.baseline_distance_m
        assert r1.post_disruption_distance_m == r2.post_disruption_distance_m
