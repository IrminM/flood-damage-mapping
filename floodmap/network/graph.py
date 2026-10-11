"""Road network graph construction and spatial snapping for Stage 5 connectivity analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pyproj
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, base


def _resolve_metric_crs(gdf: gpd.GeoDataFrame, target_crs: Optional[str] = None) -> str:
    """Ensure a projected metric CRS is used for metric graph weights (meters)."""
    if target_crs:
        try:
            crs_obj = pyproj.CRS(target_crs)
            if not crs_obj.is_geographic:
                return target_crs
        except Exception:
            pass

    if gdf.crs:
        crs_obj = pyproj.CRS(gdf.crs)
        if not crs_obj.is_geographic:
            return str(gdf.crs)
        try:
            estimated = gdf.estimate_utm_crs()
            if estimated:
                return str(estimated)
        except Exception:
            pass

    return "EPSG:3857"


def _densify_coords(coords: list[tuple[float, float]], max_step_m: float = 200.0) -> list[tuple[float, float]]:
    """Subdivide long segments so that nearest-node queries can accurately reach edges."""
    if len(coords) < 2:
        return coords
    densified = [coords[0]]
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist > max_step_m:
            steps = int(math.ceil(dist / max_step_m))
            for s in range(1, steps):
                frac = s / steps
                densified.append((x1 + frac * (x2 - x1), y1 + frac * (y2 - y1)))
        densified.append((x2, y2))
    return densified


@dataclass
class RoadGraph:
    """Container for the in-memory road network graph and spatial index."""
    graph: nx.Graph
    metric_crs: str
    node_tree: Optional[cKDTree]
    node_coords: list[tuple[float, float]]
    affected_edges_count: int
    total_edges_count: int

    def snap_point(
        self,
        geom: base.BaseGeometry,
        src_crs: str = "EPSG:4326",
        max_distance_m: float = 1000.0,
    ) -> tuple[Optional[tuple[float, float]], float]:
        """Snap a geometry (Point or Polygon centroid) to the nearest graph node.

        Returns (node_key, distance_in_meters). If distance exceeds max_distance_m,
        returns (None, distance_in_meters).
        """
        if self.node_tree is None or not self.node_coords or geom is None or geom.is_empty:
            return None, float("inf")

        pt = geom.centroid if geom.geom_type != "Point" else geom

        # Reproject to graph metric CRS if necessary
        try:
            if src_crs != self.metric_crs:
                transformer = pyproj.Transformer.from_crs(src_crs, self.metric_crs, always_xy=True)
                px, py = transformer.transform(pt.x, pt.y)
            else:
                px, py = pt.x, pt.y
        except Exception:
            return None, float("inf")

        if not (math.isfinite(px) and math.isfinite(py)):
            return None, float("inf")

        dist, idx = self.node_tree.query([px, py])
        dist = float(dist)
        if dist > max_distance_m:
            return None, dist
        return self.node_coords[idx], dist

    def to_geodataframe(self, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """Convert graph edges into a GeoDataFrame with edge states and attributes."""
        from shapely.geometry import LineString

        rows = []
        for u, v, d in self.graph.edges(data=True):
            geom = d.get("geometry")
            if geom is None:
                geom = LineString([u, v])
            rows.append({
                "osm_id": d.get("osm_id", ""),
                "highway": d.get("highway", "road"),
                "length_m": round(float(d.get("length_m", 0.0)), 2),
                "speed_kmh": round(float(d.get("speed_kmh", 25.0)), 1),
                "edge_state": d.get("edge_state", "POTENTIALLY_AFFECTED" if d.get("affected") else "NORMAL"),
                "is_affected": bool(d.get("affected", False)),
                "evidence_tier": d.get("evidence_tier", "none"),
                "impact_class": d.get("impact_class", "none"),
                "structural_damage_claimed": False,
                "routing_scenario": "hazard-aware connectivity",
                "geometry": geom,
            })

        if not rows:
            return gpd.GeoDataFrame(columns=["osm_id", "highway", "length_m", "edge_state", "geometry"], crs=target_crs)

        gdf = gpd.GeoDataFrame(rows, crs=self.metric_crs)
        if target_crs != self.metric_crs:
            gdf = gdf.to_crs(target_crs)
        return gdf


def build_road_graph(
    roads_gdf: gpd.GeoDataFrame,
    affected_road_ids: Optional[set[str]] = None,
    speeds_kmh: Optional[dict[str, float]] = None,
    target_metric_crs: Optional[str] = None,
    max_segment_length_m: Optional[float] = None,
    affected_roads_meta: Optional[dict[str, dict[str, Any]]] = None,
) -> RoadGraph:
    """Build an undirected NetworkX graph from a GeoDataFrame of roads.

    - Preserves stable OSM identifiers on graph edges.
    - Accurately marks affected edges using Stage 4 results or affected_road_ids.
    - Projects to metric CRS so weights are in meters.
    """
    if roads_gdf.empty:
        return RoadGraph(
            graph=nx.Graph(),
            metric_crs=target_metric_crs or "EPSG:3857",
            node_tree=None,
            node_coords=[],
            affected_edges_count=0,
            total_edges_count=0,
        )

    metric_crs = _resolve_metric_crs(roads_gdf, target_metric_crs)
    roads_metric = roads_gdf.to_crs(metric_crs)

    default_speeds = {
        "motorway": 80.0,
        "trunk": 60.0,
        "primary": 50.0,
        "secondary": 40.0,
        "tertiary": 30.0,
        "unclassified": 25.0,
        "residential": 20.0,
        "track": 10.0,
        "default": 25.0,
    }
    speed_map = {**default_speeds, **(speeds_kmh or {})}

    affected_set = set(affected_road_ids or [])
    G = nx.Graph()
    G.graph["crs"] = metric_crs

    affected_edges = 0
    total_edges = 0

    for idx, row in roads_metric.iterrows():
        osm_id = str(row.get("osm_id") or row.get("@osmId") or row.get("id") or f"road/{idx}")
        highway = str(row.get("highway") or "road")
        
        # Check if flagged as affected directly, via meta, or in affected_set
        meta = (affected_roads_meta or {}).get(osm_id, {})
        row_affected = bool(row.get("potentially_affected", False) or row.get("is_affected", False))
        is_affected = row_affected or (osm_id in affected_set) or bool(meta.get("is_affected", False))

        if is_affected:
            edge_state = "POTENTIALLY_AFFECTED"
            evidence_tier = str(meta.get("evidence_tier") or row.get("evidence_tier") or "candidate_evidence")
            impact_class = str(meta.get("impact_class") or row.get("impact_class") or "flood_or_change")
        else:
            edge_state = "NORMAL"
            evidence_tier = "none"
            impact_class = "none"

        speed = speed_map.get(highway, speed_map["default"])
        speed_mps = speed * 1000.0 / 3600.0

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            raw_coords = list(line.coords)
            if len(raw_coords) < 2:
                continue

            if max_segment_length_m is not None and max_segment_length_m > 0:
                coords = _densify_coords(raw_coords, max_step_m=max_segment_length_m)
            else:
                coords = raw_coords

            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                u = (round(x1, 2), round(y1, 2))
                v = (round(x2, 2), round(y2, 2))
                if u == v:
                    continue

                length_m = math.hypot(x2 - x1, y2 - y1)
                if length_m < 1e-4:
                    continue

                time_s = length_m / speed_mps

                edge_attrs = {
                    "osm_id": osm_id,
                    "highway": highway,
                    "length_m": float(length_m),
                    "time_s": float(time_s),
                    "speed_kmh": float(speed),
                    "affected": is_affected,
                    "edge_state": edge_state,
                    "evidence_tier": evidence_tier,
                    "impact_class": impact_class,
                    "weight": float(length_m),
                    "geometry": LineString([(x1, y1), (x2, y2)]),
                }

                if G.has_edge(u, v):
                    ex = G[u][v]
                    # If existing edge is affected and new edge is unaffected, overwrite
                    if ex["affected"] and not is_affected:
                        G.add_edge(u, v, **edge_attrs)
                    elif not ex["affected"] and is_affected:
                        pass  # Keep unaffected edge
                    else:
                        if length_m < ex["length_m"]:
                            G.add_edge(u, v, **edge_attrs)
                else:
                    G.add_edge(u, v, **edge_attrs)
                    G.nodes[u]["pos"] = u
                    G.nodes[v]["pos"] = v
                    total_edges += 1
                    if is_affected:
                        affected_edges += 1

    # Recalculate edge counts accurately after deduplication
    total_edges = G.number_of_edges()
    affected_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("affected", False))

    node_coords = list(G.nodes())
    node_tree = cKDTree(np.array(node_coords)) if node_coords else None

    return RoadGraph(
        graph=G,
        metric_crs=metric_crs,
        node_tree=node_tree,
        node_coords=node_coords,
        affected_edges_count=affected_edges,
        total_edges_count=total_edges,
    )
