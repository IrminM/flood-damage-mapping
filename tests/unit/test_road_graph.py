"""Unit tests for Stage 5 road graph construction and node snapping."""
import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Point

from floodmap.network.graph import RoadGraph, _densify_coords, build_road_graph


def test_densify_coords():
    coords = [(0.0, 0.0), (300.0, 0.0)]
    densified = _densify_coords(coords, max_step_m=100.0)
    assert len(densified) == 4
    assert densified[0] == (0.0, 0.0)
    assert densified[-1] == (300.0, 0.0)
    assert densified[1] == (100.0, 0.0)
    assert densified[2] == (200.0, 0.0)


def test_build_road_graph_basic():
    # Build a simple T-intersection in projected metric coordinates (EPSG:32645)
    r1 = LineString([(1000.0, 1000.0), (1000.0, 1500.0)])
    r2 = LineString([(1000.0, 1500.0), (1000.0, 2000.0)])
    r3 = LineString([(1000.0, 1500.0), (1500.0, 1500.0)])

    gdf = gpd.GeoDataFrame(
        [
            {"osm_id": "way/1", "highway": "primary", "geometry": r1},
            {"osm_id": "way/2", "highway": "secondary", "geometry": r2},
            {"osm_id": "way/3", "highway": "residential", "geometry": r3},
        ],
        crs="EPSG:32645",
    )

    rg = build_road_graph(gdf, affected_road_ids={"way/2"})
    G = rg.graph

    assert isinstance(G, nx.Graph)
    assert rg.metric_crs == "EPSG:32645"
    assert G.number_of_nodes() >= 4
    assert rg.affected_edges_count >= 1
    assert rg.total_edges_count >= 3

    # Check intersection node exists
    intersection = (1000.0, 1500.0)
    assert intersection in G.nodes

    # Check edge attributes
    u = (1000.0, 1000.0)
    v = (1000.0, 1500.0)
    assert G.has_edge(u, v)
    assert G[u][v]["osm_id"] == "way/1"
    assert G[u][v]["affected"] is False
    assert pytest.approx(G[u][v]["length_m"], abs=1.0) == 500.0

    # Check affected edge
    v2 = (1000.0, 2000.0)
    assert G.has_edge(v, v2)
    assert G[v][v2]["osm_id"] == "way/2"
    assert G[v][v2]["affected"] is True


def test_build_road_graph_missing_and_invalid_osm_ids():
    line = LineString([(0.0, 0.0), (100.0, 0.0)])
    # Row without osm_id, @osmId, or id
    gdf = gpd.GeoDataFrame(
        [
            {"highway": "residential", "geometry": line},
            {"osm_id": "", "highway": "service", "geometry": LineString([(100.0, 0.0), (200.0, 0.0)])},
        ],
        crs="EPSG:32645",
    )

    rg = build_road_graph(gdf)
    assert rg.total_edges_count >= 2
    for u, v, d in rg.graph.edges(data=True):
        assert d["osm_id"] != ""
        assert d["osm_id"] is not None


def test_point_snapping_tolerance():
    r1 = LineString([(100.0, 100.0), (500.0, 100.0)])
    gdf = gpd.GeoDataFrame([{"osm_id": "road/1", "geometry": r1}], crs="EPSG:32645")
    rg = build_road_graph(gdf)

    # Point close to road node (100, 100)
    p_near = Point(110.0, 110.0)
    node, dist = rg.snap_point(p_near, src_crs="EPSG:32645", max_distance_m=100.0)
    assert node is not None
    assert dist < 100.0

    # Point far from road (dist ~ 500m, exceeds max_distance_m=200m)
    p_far = Point(200.0, 600.0)
    node_far, dist_far = rg.snap_point(p_far, src_crs="EPSG:32645", max_distance_m=200.0)
    assert node_far is None
    assert dist_far > 200.0


def test_empty_roads_gdf():
    empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    rg = build_road_graph(empty_gdf)
    assert rg.total_edges_count == 0
    assert rg.affected_edges_count == 0
    node, dist = rg.snap_point(Point(85.0, 27.0))
    assert node is None
    assert dist == float("inf")
