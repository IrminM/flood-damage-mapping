"""Settlement and service destination network connectivity analysis for Stage 5.

Implements hazard-aware road network connectivity, baseline vs post-disruption routing,
4-tier settlement classification, disconnected components analysis, and provenance tracking.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point

from ..core.config import ConnectivityConfig
from ..core.contracts import (
    ConnectivitySummary,
    DataMode,
    SettlementConnectivityRecord,
    SettlementStatus,
)
from .graph import RoadGraph

DISCLAIMER_TEXT = (
    "Stage-5 road network connectivity represents graph-theoretic topological reachability "
    "under a hazard-disrupted scenario. Road segments intersecting detected hazard areas are modeled "
    "as potentially impassable; satellite imagery does NOT prove structural road destruction or "
    "bridge washout without on-ground inspection. Features are classified with explicit evidence provenance."
)


@dataclass
class CandidateDestination:
    """Service location on the road network (town or healthcare facility)."""
    id: str
    name: str
    dest_type: str
    node: tuple[float, float]
    snap_distance_m: float
    geom: Any
    category: str = "town"  # "town" | "health"
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class DisconnectedComponentSummary:
    """Summary of connected components in the hazard-disrupted graph."""
    total_components: int
    baseline_components: int
    new_components_count: int
    component_sizes: list[int]
    isolated_components_count: int
    components: list[dict[str, Any]]


def _to_gdf(fc_or_gdf: Any, src_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Normalize GeoJSON FeatureCollection or GeoDataFrame to GeoDataFrame."""
    if isinstance(fc_or_gdf, gpd.GeoDataFrame):
        return fc_or_gdf.copy()
    if not isinstance(fc_or_gdf, dict) or "features" not in fc_or_gdf:
        return gpd.GeoDataFrame(geometry=[], crs=src_crs)
    features = fc_or_gdf.get("features", [])
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=src_crs)
    gdf = gpd.GeoDataFrame.from_features(features, crs=src_crs)
    if "id" not in gdf.columns:
        gdf["id"] = [f.get("id", f"feature_{i}") for i, f in enumerate(features)]
    return gdf


def extract_destinations(
    places_gdf: gpd.GeoDataFrame,
    health_gdf: Optional[gpd.GeoDataFrame],
    road_graph: RoadGraph,
    cfg: ConnectivityConfig,
) -> list[CandidateDestination]:
    """Identify and snap relevant service destinations (towns, cities, hospitals)."""
    destinations: list[CandidateDestination] = []
    seen_ids: set[str] = set()
    places_src_crs = str(places_gdf.crs) if places_gdf.crs else "EPSG:4326"
    health_src_crs = str(health_gdf.crs) if health_gdf is not None and health_gdf.crs else "EPSG:4326"

    # 1. Urban destinations from places
    if not places_gdf.empty:
        for idx, row in places_gdf.iterrows():
            place_type = str(row.get("place", "")).lower()
            if place_type in cfg.destination_place_types:
                dest_id = str(row.get("osm_id") or row.get("@osmId") or row.get("id") or f"place/{idx}")
                if dest_id in seen_ids:
                    continue
                seen_ids.add(dest_id)
                name = str(row.get("name") or f"{place_type}_{dest_id}")
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                node, dist = road_graph.snap_point(geom, src_crs=places_src_crs, max_distance_m=cfg.max_snapping_distance_m)
                if node is not None:
                    tags = {k: v for k, v in row.items() if k != "geometry" and v is not None}
                    destinations.append(
                        CandidateDestination(
                            id=dest_id,
                            name=name,
                            dest_type=place_type,
                            node=node,
                            snap_distance_m=dist,
                            geom=geom,
                            category="town",
                            tags=tags,
                        )
                    )

    # 2. Medical destinations from health layer
    if cfg.include_health_destinations and health_gdf is not None and not health_gdf.empty:
        for idx, row in health_gdf.iterrows():
            dest_id = str(row.get("osm_id") or row.get("@osmId") or row.get("id") or f"health/{idx}")
            if dest_id in seen_ids:
                continue
            seen_ids.add(dest_id)
            amenity = str(row.get("amenity") or row.get("healthcare") or "hospital").lower()
            name = str(row.get("name") or f"{amenity}_{dest_id}")
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            node, dist = road_graph.snap_point(geom, src_crs=health_src_crs, max_distance_m=cfg.max_snapping_distance_m)
            if node is not None:
                tags = {k: v for k, v in row.items() if k != "geometry" and v is not None}
                destinations.append(
                    CandidateDestination(
                        id=dest_id,
                        name=name,
                        dest_type=amenity,
                        node=node,
                        snap_distance_m=dist,
                        geom=geom,
                        category="health",
                        tags=tags,
                    )
                )

    return destinations


def extract_settlements(
    places_gdf: gpd.GeoDataFrame,
    cfg: ConnectivityConfig,
) -> gpd.GeoDataFrame:
    """Identify settlement features to be evaluated for network connectivity."""
    if places_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    rows = []
    dest_types = set(cfg.destination_place_types)
    settlement_types = set(cfg.settlement_place_types)

    for idx, row in places_gdf.iterrows():
        place_type = str(row.get("place", "")).lower()
        # If explicitly a destination town/city, do not evaluate as an origin settlement
        if place_type in dest_types:
            continue
        # If place type matches configured settlement types, or if no explicit place type given
        if not place_type or place_type in settlement_types or place_type not in dest_types:
            rows.append(row)

    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs=places_gdf.crs)


def analyze_disconnected_components(
    road_graph: RoadGraph,
    destinations: list[CandidateDestination],
    records: list[SettlementConnectivityRecord],
    snapped_nodes: dict[str, tuple[float, float]],
) -> DisconnectedComponentSummary:
    """Analyze connected components in the hazard-disrupted graph."""
    G = road_graph.graph
    G_disrupted = nx.Graph()
    G_disrupted.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if not d.get("affected", False):
            G_disrupted.add_edge(u, v, **d)

    base_comps = nx.number_connected_components(G)
    dis_comps = sorted(list(nx.connected_components(G_disrupted)), key=len, reverse=True)
    total_dis = len(dis_comps)
    comp_sizes = [len(c) for c in dis_comps]

    node_to_comp: dict[tuple[float, float], int] = {}
    for i, c in enumerate(dis_comps):
        for n in c:
            node_to_comp[n] = i

    dest_by_comp: dict[int, list[CandidateDestination]] = {}
    for d in destinations:
        c_idx = node_to_comp.get(d.node)
        if c_idx is not None:
            dest_by_comp.setdefault(c_idx, []).append(d)

    rec_by_comp: dict[int, list[SettlementConnectivityRecord]] = {}
    for r in records:
        s_node = snapped_nodes.get(r.settlement_id)
        if s_node is not None:
            c_idx = node_to_comp.get(s_node)
            if c_idx is not None:
                r.component_id = c_idx
                rec_by_comp.setdefault(c_idx, []).append(r)

    component_details: list[dict[str, Any]] = []
    isolated_count = 0

    for i, c in enumerate(dis_comps):
        settlements_in_c = rec_by_comp.get(i, [])
        dests_in_c = dest_by_comp.get(i, [])

        towns_in_c = [d for d in dests_in_c if d.category == "town"]
        hospitals_in_c = [d for d in dests_in_c if d.category == "health"]

        is_isolated = len(settlements_in_c) > 0 and len(towns_in_c) == 0 and len(hospitals_in_c) == 0
        if is_isolated:
            isolated_count += 1

        nearest_service = None
        if towns_in_c:
            nearest_service = towns_in_c[0].name
        elif hospitals_in_c:
            nearest_service = hospitals_in_c[0].name

        component_details.append({
            "component_id": i,
            "node_count": len(c),
            "settlements_count": len(settlements_in_c),
            "settlement_ids": [r.settlement_id for r in settlements_in_c],
            "settlement_names": [r.settlement_name for r in settlements_in_c if r.settlement_name],
            "towns_count": len(towns_in_c),
            "town_names": [d.name for d in towns_in_c],
            "hospitals_count": len(hospitals_in_c),
            "hospital_names": [d.name for d in hospitals_in_c],
            "is_isolated": is_isolated,
            "nearest_reachable_service": nearest_service,
        })

    return DisconnectedComponentSummary(
        total_components=total_dis,
        baseline_components=base_comps,
        new_components_count=max(0, total_dis - base_comps),
        component_sizes=comp_sizes,
        isolated_components_count=isolated_count,
        components=component_details,
    )


def evaluate_connectivity(
    road_graph: RoadGraph,
    places_fc: Any,
    health_fc: Optional[Any],
    cfg: ConnectivityConfig,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[SettlementConnectivityRecord], list[str]]:
    """Evaluate road network connectivity and cut-off status for all settlements.

    Returns:
      (all_settlements_gdf, cut_off_settlements_gdf, records, warnings)
    """
    warnings: list[str] = []
    places_gdf = _to_gdf(places_fc, src_crs="EPSG:4326")
    health_gdf = _to_gdf(health_fc, src_crs="EPSG:4326") if health_fc is not None else None

    destinations = extract_destinations(places_gdf, health_gdf, road_graph, cfg)
    settlements_gdf = extract_settlements(places_gdf, cfg)

    if not destinations:
        warnings.append(
            "No candidate destination (town/hospital) features could be identified or snapped to the road network."
        )

    G = road_graph.graph

    # Pre-build disrupted graph representations according to disruption_mode
    if cfg.disruption_mode == "remove":
        G_disrupted = nx.Graph()
        G_disrupted.graph.update(G.graph)
        G_disrupted.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            if not d.get("affected", False):
                G_disrupted.add_edge(u, v, **d)
    elif cfg.disruption_mode == "penalty":
        G_penalty = nx.Graph()
        G_penalty.graph.update(G.graph)
        G_penalty.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            d_copy = dict(d)
            if d_copy.get("affected", False):
                d_copy["weight"] = d_copy["length_m"] * cfg.penalty_multiplier
            else:
                d_copy["weight"] = d_copy["length_m"]
            G_penalty.add_edge(u, v, **d_copy)
    else:
        raise ValueError(f"Unknown disruption_mode: {cfg.disruption_mode!r}. Expected 'remove' or 'penalty'.")

    # Precompute shortest path lengths from each destination in baseline and disrupted graphs
    base_dest_lengths: dict[tuple[float, float], dict[tuple[float, float], float]] = {}
    for d in destinations:
        if d.node in G:
            try:
                base_dest_lengths[d.node] = nx.single_source_dijkstra_path_length(G, d.node, weight="length_m")
            except Exception:
                pass

    disrupted_dest_lengths: dict[tuple[float, float], dict[tuple[float, float], float]] = {}
    if cfg.disruption_mode == "remove":
        target_G = G_disrupted
        target_weight = "length_m"
    else:
        target_G = G_penalty
        target_weight = "weight"

    for d in destinations:
        if d.node in target_G:
            try:
                disrupted_dest_lengths[d.node] = nx.single_source_dijkstra_path_length(target_G, d.node, weight=target_weight)
            except Exception:
                pass

    records: list[SettlementConnectivityRecord] = []
    gdf_rows: list[dict[str, Any]] = []
    settlements_src_crs = str(settlements_gdf.crs) if settlements_gdf.crs else "EPSG:4326"
    snapped_nodes: dict[str, tuple[float, float]] = {}

    # Separate towns and health destinations
    towns = [d for d in destinations if d.category == "town"]
    hospitals = [d for d in destinations if d.category == "health"]

    for idx, row in settlements_gdf.iterrows():
        settlement_id = str(row.get("osm_id") or row.get("@osmId") or row.get("id") or f"settlement/{idx}")
        name = row.get("name")
        place_type = str(row.get("place", "settlement"))
        raw_pop = row.get("population")
        population: Optional[int] = None
        if raw_pop not in (None, ""):
            try:
                pop_f = float(str(raw_pop).replace(",", "").strip())
                if math.isfinite(pop_f):
                    population = int(pop_f)
            except Exception:
                pass

        geom = row.geometry

        rec_base = {
            "settlement_id": settlement_id,
            "settlement_name": str(name) if name is not None else None,
            "place_type": place_type,
            "population": population,
        }

        if geom is None or geom.is_empty:
            records.append(
                SettlementConnectivityRecord(
                    **rec_base,
                    status=SettlementStatus.INSUFFICIENT_EVIDENCE,
                    classification="OUTSIDE_NETWORK",
                    isolation_reason="Missing or empty settlement geometry",
                )
            )
            continue

        pt_geom = geom if geom.geom_type == "Point" else geom.centroid

        snapped_node, snap_dist = road_graph.snap_point(
            geom, src_crs=settlements_src_crs, max_distance_m=cfg.max_snapping_distance_m
        )

        if snapped_node is None:
            # Settlement exceeds maximum snapping distance
            rec = SettlementConnectivityRecord(
                **rec_base,
                status=SettlementStatus.INSUFFICIENT_EVIDENCE,
                classification="OUTSIDE_NETWORK",
                isolation_reason=(
                    f"Settlement is {snap_dist:.1f} m from road network "
                    f"(exceeds maximum snapping tolerance {cfg.max_snapping_distance_m:.1f} m)"
                ),
                snapped_distance_to_road_m=float(snap_dist),
            )
            records.append(rec)
            gdf_rows.append(_make_gdf_row(rec, pt_geom))
            continue

        snapped_nodes[settlement_id] = snapped_node

        if not destinations:
            rec = SettlementConnectivityRecord(
                **rec_base,
                status=SettlementStatus.INSUFFICIENT_EVIDENCE,
                classification="NO_BASELINE_ROUTE",
                isolation_reason="No candidate service destinations (towns/hospitals) available in network",
                snapped_distance_to_road_m=float(snap_dist),
            )
            records.append(rec)
            gdf_rows.append(_make_gdf_row(rec, pt_geom))
            continue

        # 1. Baseline routing to nearest general service destination
        reachable_dests = [
            d for d in destinations
            if d.node in base_dest_lengths and snapped_node in base_dest_lengths[d.node]
        ]

        if not reachable_dests:
            # Settlement snaps to road, but its road component has no path to any service destination
            rec = SettlementConnectivityRecord(
                **rec_base,
                status=SettlementStatus.INSUFFICIENT_EVIDENCE,
                classification="NO_BASELINE_ROUTE",
                isolation_reason="No path to any service destination in baseline pre-event road network (isolated road component)",
                snapped_distance_to_road_m=float(snap_dist),
            )
            records.append(rec)
            gdf_rows.append(_make_gdf_row(rec, pt_geom))
            continue

        nearest_dest = min(reachable_dests, key=lambda d: base_dest_lengths[d.node][snapped_node])
        base_dist = float(base_dest_lengths[nearest_dest.node][snapped_node])
        base_path = nx.shortest_path(G, snapped_node, nearest_dest.node, weight="length_m")

        base_time_s = 0.0
        for u, v in zip(base_path[:-1], base_path[1:]):
            base_time_s += G[u][v].get("time_s", 0.0)

        # Baseline hospital routing if available
        reachable_hospitals = [
            h for h in hospitals
            if h.node in base_dest_lengths and snapped_node in base_dest_lengths[h.node]
        ]
        nearest_hospital = min(reachable_hospitals, key=lambda h: base_dest_lengths[h.node][snapped_node]) if reachable_hospitals else None
        hosp_base_dist = float(base_dest_lengths[nearest_hospital.node][snapped_node]) if nearest_hospital else None

        # Identify any affected road segments along the baseline route
        disrupting_roads: list[str] = []
        disrupting_tiers: list[str] = []
        for u, v in zip(base_path[:-1], base_path[1:]):
            if G[u][v].get("affected", False):
                r_id = G[u][v].get("osm_id")
                r_tier = G[u][v].get("evidence_tier", "candidate_evidence")
                if r_id and r_id not in disrupting_roads:
                    disrupting_roads.append(r_id)
                if r_tier and r_tier not in disrupting_tiers:
                    disrupting_tiers.append(r_tier)

        # 2. Hazard-aware routing evaluation
        status: SettlementStatus
        classification: str
        post_dist: Optional[float] = None
        post_time_s: Optional[float] = None
        alt_route_dist: Optional[float] = None
        detour_m: Optional[float] = None
        connectivity_lost: bool = False
        isolation_reason: Optional[str] = None

        if cfg.disruption_mode == "remove":
            has_valid_path = (
                nearest_dest.node in disrupted_dest_lengths
                and snapped_node in disrupted_dest_lengths[nearest_dest.node]
            )
            if has_valid_path:
                post_dist = float(disrupted_dest_lengths[nearest_dest.node][snapped_node])
                post_path = nx.shortest_path(G_disrupted, snapped_node, nearest_dest.node, weight="length_m")
                post_time_s = sum(G_disrupted[u][v].get("time_s", 0.0) for u, v in zip(post_path[:-1], post_path[1:]))
                detour_m = float(post_dist - base_dist)
                status = SettlementStatus.CONNECTED
                classification = "CONNECTED"
                if detour_m > 1.0:
                    alt_route_dist = post_dist
                    isolation_reason = f"Baseline route disrupted; alternative detour intact (+{detour_m:.1f} m)"
            else:
                # Primary route to nearest destination severed
                reachable_in_disrupted = [
                    d for d in destinations
                    if d.node in disrupted_dest_lengths and snapped_node in disrupted_dest_lengths[d.node]
                ]
                if reachable_in_disrupted:
                    alt_dest = min(reachable_in_disrupted, key=lambda d: disrupted_dest_lengths[d.node][snapped_node])
                    alt_dist = float(disrupted_dest_lengths[alt_dest.node][snapped_node])
                    post_path = nx.shortest_path(G_disrupted, snapped_node, alt_dest.node, weight="length_m")
                    post_time_s = sum(G_disrupted[u][v].get("time_s", 0.0) for u, v in zip(post_path[:-1], post_path[1:]))
                    alt_route_dist = alt_dist
                    detour_m = float(alt_dist - base_dist)
                    status = SettlementStatus.DISCONNECTED
                    classification = "DISCONNECTED"
                    connectivity_lost = False
                    isolation_reason = (
                        f"Road access to nearest destination '{nearest_dest.name}' severed; "
                        f"alternative route exists only to secondary destination '{alt_dest.name}'"
                    )
                else:
                    status = SettlementStatus.DISCONNECTED
                    classification = "DISCONNECTED"
                    connectivity_lost = True
                    isolation_reason = (
                        f"All road access to service locations severed by affected road segments "
                        f"({len(disrupting_roads)} affected roads on primary route)"
                    )

        elif cfg.disruption_mode == "penalty":
            has_valid_path = (
                nearest_dest.node in disrupted_dest_lengths
                and snapped_node in disrupted_dest_lengths[nearest_dest.node]
            )
            if not has_valid_path:
                status = SettlementStatus.DISCONNECTED
                classification = "DISCONNECTED"
                connectivity_lost = True
                isolation_reason = "No path exists in network"
            else:
                pen_path = nx.shortest_path(G_penalty, snapped_node, nearest_dest.node, weight="weight")
                affected_in_path = [
                    G[u][v].get("osm_id")
                    for u, v in zip(pen_path[:-1], pen_path[1:])
                    if G[u][v].get("affected", False)
                ]
                affected_in_path = [r for r in affected_in_path if r]

                if not affected_in_path:
                    post_dist = sum(G[u][v].get("length_m", 0.0) for u, v in zip(pen_path[:-1], pen_path[1:]))
                    post_time_s = sum(G[u][v].get("time_s", 0.0) for u, v in zip(pen_path[:-1], pen_path[1:]))
                    detour_m = float(post_dist - base_dist)
                    status = SettlementStatus.CONNECTED
                    classification = "CONNECTED"
                    if detour_m > 1.0:
                        alt_route_dist = post_dist
                else:
                    status = SettlementStatus.DISCONNECTED
                    classification = "DISCONNECTED"
                    connectivity_lost = True
                    isolation_reason = (
                        f"No unaffected route exists; path traverses affected road segments "
                        f"({', '.join(set(affected_in_path))})"
                    )
                    for r in affected_in_path:
                        if r not in disrupting_roads:
                            disrupting_roads.append(r)

        # Hospital post-disruption assessment
        hosp_post_dist: Optional[float] = None
        hosp_status: Optional[str] = None
        if nearest_hospital:
            if nearest_hospital.node in disrupted_dest_lengths and snapped_node in disrupted_dest_lengths[nearest_hospital.node]:
                hosp_post_dist = float(disrupted_dest_lengths[nearest_hospital.node][snapped_node])
                hosp_status = "connected"
            else:
                hosp_status = "disconnected"

        rec = SettlementConnectivityRecord(
            **rec_base,
            status=status,
            classification=classification,
            nearest_destination_id=nearest_dest.id,
            nearest_destination_name=nearest_dest.name,
            destination_type=nearest_dest.dest_type,
            baseline_distance_m=base_dist,
            baseline_time_s=base_time_s,
            post_disruption_distance_m=post_dist,
            post_disruption_time_s=post_time_s,
            alternate_route_distance_m=alt_route_dist,
            detour_distance_m=detour_m,
            connectivity_lost=connectivity_lost,
            disrupting_road_ids=disrupting_roads,
            disrupting_evidence_tiers=disrupting_tiers,
            isolation_reason=isolation_reason,
            snapped_distance_to_road_m=float(snap_dist),
            nearest_hospital_id=nearest_hospital.id if nearest_hospital else None,
            nearest_hospital_name=nearest_hospital.name if nearest_hospital else None,
            hospital_baseline_distance_m=hosp_base_dist,
            hospital_post_disruption_distance_m=hosp_post_dist,
            hospital_status=hosp_status,
        )
        records.append(rec)
        gdf_rows.append(_make_gdf_row(rec, pt_geom))

    # Assemble GeoDataFrames
    if gdf_rows:
        all_gdf = gpd.GeoDataFrame(gdf_rows, crs="EPSG:4326")
    else:
        all_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    cut_off_gdf = all_gdf[all_gdf["status"] == SettlementStatus.DISCONNECTED.value].copy() if not all_gdf.empty else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Perform disconnected components analysis and attach component IDs
    comp_summary = analyze_disconnected_components(road_graph, destinations, records, snapped_nodes)

    # Propagate component_id to GeoDataFrame rows
    comp_map = {r.settlement_id: r.component_id for r in records}
    if not all_gdf.empty and "settlement_id" in all_gdf.columns:
        all_gdf["component_id"] = all_gdf["settlement_id"].map(comp_map)
    if not cut_off_gdf.empty and "settlement_id" in cut_off_gdf.columns:
        cut_off_gdf["component_id"] = cut_off_gdf["settlement_id"].map(comp_map)

    return all_gdf, cut_off_gdf, records, warnings


def _make_gdf_row(rec: SettlementConnectivityRecord, geom: Point) -> dict[str, Any]:
    """Format record as a flat dictionary suitable for GeoJSON serialization."""
    base_dist_km = round(rec.baseline_distance_m / 1000.0, 3) if rec.baseline_distance_m is not None else None
    base_time_min = round(rec.baseline_time_s / 60.0, 1) if rec.baseline_time_s is not None else None
    post_dist_km = round(rec.post_disruption_distance_m / 1000.0, 3) if rec.post_disruption_distance_m is not None else None
    post_time_min = round(rec.post_disruption_time_s / 60.0, 1) if rec.post_disruption_time_s is not None else None
    alt_dist_km = round(rec.alternate_route_distance_m / 1000.0, 3) if rec.alternate_route_distance_m is not None else None
    detour_km = round(rec.detour_distance_m / 1000.0, 3) if rec.detour_distance_m is not None else None
    detour_ratio = None
    if rec.post_disruption_distance_m is not None and rec.baseline_distance_m and rec.baseline_distance_m > 0:
        detour_ratio = round(rec.post_disruption_distance_m / rec.baseline_distance_m, 2)

    hosp_base_km = round(rec.hospital_baseline_distance_m / 1000.0, 3) if rec.hospital_baseline_distance_m is not None else None
    hosp_post_km = round(rec.hospital_post_disruption_distance_m / 1000.0, 3) if rec.hospital_post_disruption_distance_m is not None else None

    return {
        "geometry": geom,
        "settlement_id": rec.settlement_id,
        "settlement_name": rec.settlement_name or "",
        "place_type": rec.place_type,
        "population": rec.population,
        "classification": rec.classification,
        "status": rec.status.value,
        "nearest_destination_id": rec.nearest_destination_id or "",
        "nearest_destination_name": rec.nearest_destination_name or "",
        "destination_type": rec.destination_type or "",
        "baseline_distance_km": base_dist_km,
        "baseline_time_min": base_time_min,
        "post_disruption_distance_km": post_dist_km,
        "post_disruption_time_min": post_time_min,
        "alternate_route_distance_km": alt_dist_km,
        "detour_distance_km": detour_km,
        "detour_ratio": detour_ratio,
        "connectivity_lost": rec.connectivity_lost,
        "disrupting_road_ids": json.dumps(rec.disrupting_road_ids),
        "disrupting_evidence_tiers": json.dumps(rec.disrupting_evidence_tiers),
        "isolation_reason": rec.isolation_reason or "",
        "snapped_distance_to_road_m": rec.snapped_distance_to_road_m,
        "nearest_hospital_id": rec.nearest_hospital_id or "",
        "nearest_hospital_name": rec.nearest_hospital_name or "",
        "hospital_baseline_distance_km": hosp_base_km,
        "hospital_post_disruption_distance_km": hosp_post_km,
        "hospital_status": rec.hospital_status or "",
        "component_id": rec.component_id,
        "structural_damage_claimed": False,
        "scientific_disclaimer": DISCLAIMER_TEXT,
    }
