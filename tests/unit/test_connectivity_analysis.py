"""Unit tests for Stage 5 settlement connectivity and cut-off analysis."""
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from floodmap.core.config import ConnectivityConfig
from floodmap.core.contracts import SettlementStatus
from floodmap.network.connectivity import evaluate_connectivity
from floodmap.network.graph import build_road_graph


def create_mock_network_and_places(affected_road_ids=None):
    """Create a controlled synthetic network:

    Nodes:
      A: (1000, 1000) - Settlement 1 (village)
      B: (2000, 1000) - Junction
      C: (3000, 1000) - Main Destination (Town)
      D: (2000, 2000) - Detour route vertex
      E: (1000, 3000) - Settlement 2 (hamlet on cul-de-sac)
      F: (2000, 3000) - Cul-de-sac junction
      Z: (9000, 9000) - Remote Settlement 3 (far from road)

    Roads:
      road_A_B: A -> B
      road_B_C: B -> C (direct to town)
      road_B_D: B -> D (detour part 1)
      road_D_C: D -> C (detour part 2)
      road_E_F: E -> F (cul-de-sac to junction)
      road_F_B: F -> B (junction to network)
    """
    roads = [
        {"osm_id": "road_A_B", "highway": "primary", "geometry": LineString([(1000, 1000), (2000, 1000)])},
        {"osm_id": "road_B_C", "highway": "primary", "geometry": LineString([(2000, 1000), (3000, 1000)])},
        {"osm_id": "road_B_D", "highway": "secondary", "geometry": LineString([(2000, 1000), (2000, 2000)])},
        {"osm_id": "road_D_C", "highway": "secondary", "geometry": LineString([(2000, 2000), (3000, 1000)])},
        {"osm_id": "road_E_F", "highway": "residential", "geometry": LineString([(1000, 3000), (2000, 3000)])},
        {"osm_id": "road_F_B", "highway": "residential", "geometry": LineString([(2000, 3000), (2000, 1000)])},
    ]
    roads_gdf = gpd.GeoDataFrame(roads, crs="EPSG:32645")

    places = [
        {"osm_id": "place_town", "name": "Bazaar Town", "place": "town", "geometry": Point(3000, 1000)},
        {"osm_id": "place_v1", "name": "Village Alpha", "place": "village", "geometry": Point(1000, 1000)},
        {"osm_id": "place_v2", "name": "Hamlet Beta", "place": "hamlet", "geometry": Point(1000, 3000)},
        {"osm_id": "place_remote", "name": "Isolated Peak", "place": "isolated_dwelling", "geometry": Point(9000, 9000)},
    ]
    places_gdf = gpd.GeoDataFrame(places, crs="EPSG:32645")

    rg = build_road_graph(roads_gdf, affected_road_ids=affected_road_ids, target_metric_crs="EPSG:32645")
    return rg, places_gdf


def test_unaffected_network():
    rg, places_gdf = create_mock_network_and_places(affected_road_ids=set())
    cfg = ConnectivityConfig(disruption_mode="remove", max_snapping_distance_m=500.0)

    all_gdf, cut_off_gdf, records, warnings = evaluate_connectivity(
        road_graph=rg,
        places_fc=places_gdf,
        health_fc=None,
        cfg=cfg,
    )

    # Village Alpha and Hamlet Beta should be CONNECTED
    v1 = next(r for r in records if r.settlement_id == "place_v1")
    assert v1.status == SettlementStatus.CONNECTED
    assert v1.nearest_destination_id == "place_town"
    assert pytest.approx(v1.baseline_distance_m, abs=1.0) == 2000.0
    assert pytest.approx(v1.post_disruption_distance_m, abs=1.0) == 2000.0
    assert pytest.approx(v1.detour_distance_m, abs=1.0) == 0.0

    # Remote should be INSUFFICIENT_EVIDENCE (far from network)
    remote = next(r for r in records if r.settlement_id == "place_remote")
    assert remote.status == SettlementStatus.INSUFFICIENT_EVIDENCE
    assert "exceeds maximum snapping tolerance" in remote.isolation_reason

    # Cut off count should be 0
    assert len(cut_off_gdf) == 0


def test_alternative_route_detection():
    # Disrupt direct road B->C, but detour B->D->C exists
    rg, places_gdf = create_mock_network_and_places(affected_road_ids={"road_B_C"})
    cfg = ConnectivityConfig(disruption_mode="remove", max_snapping_distance_m=500.0)

    all_gdf, cut_off_gdf, records, warnings = evaluate_connectivity(
        road_graph=rg,
        places_fc=places_gdf,
        health_fc=None,
        cfg=cfg,
    )

    v1 = next(r for r in records if r.settlement_id == "place_v1")
    # Village Alpha is still CONNECTED via detour B->D->C
    assert v1.status == SettlementStatus.CONNECTED
    # Baseline was 2000m (A->B: 1000m + B->C: 1000m)
    assert pytest.approx(v1.baseline_distance_m, abs=1.0) == 2000.0
    # Detour is A->B: 1000m + B->D: 1000m + D->C: hypot(1000, 1000) ~ 1414.2m => ~ 3414m
    assert v1.post_disruption_distance_m > v1.baseline_distance_m
    assert v1.detour_distance_m > 0.0
    # Disrupting roads on primary route was recorded
    assert "road_B_C" in v1.disrupting_road_ids


def test_true_disconnection():
    # Disrupt road_E_F (sole access road for Hamlet Beta)
    rg, places_gdf = create_mock_network_and_places(affected_road_ids={"road_E_F"})
    cfg = ConnectivityConfig(disruption_mode="remove", max_snapping_distance_m=500.0)

    all_gdf, cut_off_gdf, records, warnings = evaluate_connectivity(
        road_graph=rg,
        places_fc=places_gdf,
        health_fc=None,
        cfg=cfg,
    )

    v2 = next(r for r in records if r.settlement_id == "place_v2")
    assert v2.status == SettlementStatus.DISCONNECTED
    assert "road_E_F" in v2.disrupting_road_ids
    assert len(cut_off_gdf) == 1
    assert cut_off_gdf.iloc[0]["settlement_id"] == "place_v2"


def test_penalty_disruption_mode():
    # In penalty mode, test that alternative route is preferred over penalized edge
    rg, places_gdf = create_mock_network_and_places(affected_road_ids={"road_B_C"})
    cfg = ConnectivityConfig(disruption_mode="penalty", penalty_multiplier=1000.0, max_snapping_distance_m=500.0)

    all_gdf, cut_off_gdf, records, warnings = evaluate_connectivity(
        road_graph=rg,
        places_fc=places_gdf,
        health_fc=None,
        cfg=cfg,
    )

    v1 = next(r for r in records if r.settlement_id == "place_v1")
    # Finds detour B->D->C without traversing affected road_B_C
    assert v1.status == SettlementStatus.CONNECTED
    assert v1.post_disruption_distance_m > v1.baseline_distance_m

    # Now disrupt cul-de-sac road_E_F in penalty mode
    rg2, places_gdf2 = create_mock_network_and_places(affected_road_ids={"road_E_F"})
    all_gdf2, cut_off_gdf2, records2, warnings2 = evaluate_connectivity(
        road_graph=rg2,
        places_fc=places_gdf2,
        health_fc=None,
        cfg=cfg,
    )

    v2 = next(r for r in records2 if r.settlement_id == "place_v2")
    # Hamlet Beta must traverse road_E_F, so classified as DISCONNECTED
    assert v2.status == SettlementStatus.DISCONNECTED
    assert "road_E_F" in v2.disrupting_road_ids


def test_island_graph_insufficient_evidence():
    # Road network has an island component with no destination
    roads = [
        {"osm_id": "island_road", "highway": "track", "geometry": LineString([(5000, 5000), (5100, 5000)])},
        {"osm_id": "main_road", "highway": "primary", "geometry": LineString([(1000, 1000), (2000, 1000)])},
    ]
    roads_gdf = gpd.GeoDataFrame(roads, crs="EPSG:32645")
    places = [
        {"osm_id": "island_settlement", "place": "hamlet", "geometry": Point(5000, 5000)},
        {"osm_id": "town", "place": "town", "geometry": Point(2000, 1000)},
    ]
    places_gdf = gpd.GeoDataFrame(places, crs="EPSG:32645")

    rg = build_road_graph(roads_gdf, target_metric_crs="EPSG:32645")
    cfg = ConnectivityConfig(disruption_mode="remove", max_snapping_distance_m=500.0)

    all_gdf, cut_off_gdf, records, warnings = evaluate_connectivity(
        road_graph=rg,
        places_fc=places_gdf,
        health_fc=None,
        cfg=cfg,
    )

    island_rec = next(r for r in records if r.settlement_id == "island_settlement")
    assert island_rec.status == SettlementStatus.INSUFFICIENT_EVIDENCE
    assert "isolated road component" in island_rec.isolation_reason
