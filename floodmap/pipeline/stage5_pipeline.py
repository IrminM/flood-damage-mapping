"""Stage-5 pipeline: Cut-Off Settlement / Road Network Connectivity Analysis.

Inputs:
  1. Stage 4 affected_roads.geojson (or GeoDataFrame of affected roads)
  2. Pre-event OSM Snapshot (LocalOSMProvider with snapshot_date < event_date guard)

Outputs:
  1. cut_off_settlements.geojson (settlements that lose road connectivity to town/hospital)
  2. settlement_connectivity.geojson (all evaluated settlements with route metrics & status)
  3. connectivity_summary.json (Pydantic ConnectivitySummary contract)
  4. provenance.json (full lineage, input hashes, parameters)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd

from ..core.config import Settings
from ..core.contracts import (
    ConnectivitySummary,
    DataMode,
    SettlementStatus,
    SourceType,
)
from ..core.data_rules import DataRulesGuard
from ..core.errors import InputDataError
from ..core.provenance import ProvenanceLog
from ..data_sources.base import OSMData
from ..data_sources.osm_local import LocalOSMProvider
from ..network.connectivity import _to_gdf, evaluate_connectivity
from ..network.graph import RoadGraph, build_road_graph

STAGE5_INTERPRETATION_NOTES = [
    "Stage-5 road network connectivity: Identifies settlements that lose road access to their nearest service town/hospital due to detected flood/change disruptions.",
    "Network analysis is built strictly upon pre-event OSM roads; no post-event OSM edits or external damage maps were used.",
    "Road segments are marked disrupted according to Stage 4 infrastructure impact evidence.",
    "Disconnections reflect graph topological reachability over the mapped road network; unmapped footpaths, informal trails, or waterways are not represented.",
    "Settlements situated beyond the configurable snapping distance or in disconnected baseline components are classified as 'insufficient_network_evidence' rather than cut off.",
]


@dataclass
class Stage5PipelineResult:
    run_dir: Path
    summary: ConnectivitySummary
    outputs: dict[str, Path]
    provenance_path: Path
    cut_off_gdf: gpd.GeoDataFrame
    settlements_gdf: gpd.GeoDataFrame
    road_graph: RoadGraph
    warnings: list[str] = field(default_factory=list)


def run_stage5_pipeline(
    affected_roads: gpd.GeoDataFrame | Path | str,
    osm_source: OSMData | Path | str,
    settings: Settings,
    run_dir: Path | str,
    event_date: date,
    analysis_id: Optional[str] = None,
    aoi_name: Optional[str] = None,
    data_mode: DataMode = DataMode.REAL,
    disruption_mode: Optional[str] = None,
) -> Stage5PipelineResult:
    """Execute Stage 5 cut-off settlement and road network connectivity analysis."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_id = analysis_id or uuid.uuid4().hex[:12]
    guard = DataRulesGuard(settings.data_rules)
    prov = ProvenanceLog(analysis_id)
    prov.note("Stage-5 cut-off settlement and road network connectivity analysis (pre-event OSM)")
    prov.data_mode = data_mode
    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)
        prov.warn(msg)

    # 1. Load / validate pre-event OSM data
    if isinstance(osm_source, (str, Path)):
        osm_path = Path(osm_source)
        loader = LocalOSMProvider(guard, settings.osm)
        osm_data = loader.load(osm_path, event_date)
        prov.add_input(
            path=osm_path,
            source_type=SourceType.OSM_PRE_EVENT.value,
            role="osm_pre_event",
            metadata={
                "snapshot_date": osm_data.meta.snapshot_date.isoformat(),
                "source": osm_data.meta.source,
                "counts": {layer: osm_data.count(layer) for layer in osm_data.layers},
            },
        )
    elif isinstance(osm_source, OSMData):
        osm_data = osm_source
        guard.check_osm_snapshot(osm_data.meta.snapshot_date, event_date)
        prov.add_input(
            path=osm_data.source_dir or "in_memory",
            source_type=SourceType.OSM_PRE_EVENT.value,
            role="osm_pre_event",
            metadata={
                "snapshot_date": osm_data.meta.snapshot_date.isoformat(),
                "source": osm_data.meta.source,
                "counts": {layer: osm_data.count(layer) for layer in osm_data.layers},
            },
        )
    else:
        raise InputDataError(f"Invalid osm_source: {type(osm_source)}")

    for w in osm_data.warnings:
        warn(f"osm: {w}")

    # 2. Load / inspect affected roads from Stage 4
    if isinstance(affected_roads, (str, Path)):
        aff_path = Path(affected_roads)
        guard.check_input(aff_path, declared_source=SourceType.OSM_PRE_EVENT)
        prov.add_input(
            path=aff_path,
            source_type=SourceType.OSM_PRE_EVENT.value,
            role="affected_roads",
            metadata={"filename": aff_path.name},
        )
        affected_roads_gdf = gpd.read_file(aff_path)
    elif isinstance(affected_roads, gpd.GeoDataFrame):
        affected_roads_gdf = affected_roads
        prov.add_input(
            path="in_memory",
            source_type=SourceType.OSM_PRE_EVENT.value,
            role="affected_roads",
            metadata={"count": len(affected_roads_gdf)},
        )
    else:
        raise InputDataError(f"Invalid affected_roads: {type(affected_roads)}")

    # Extract set of affected road OSM IDs and metadata
    affected_road_ids: set[str] = set()
    affected_roads_meta: dict[str, dict[str, Any]] = {}
    if not affected_roads_gdf.empty:
        if "is_affected" in affected_roads_gdf.columns:
            aff_subset = affected_roads_gdf[affected_roads_gdf["is_affected"] == True]
        elif "potentially_affected" in affected_roads_gdf.columns:
            aff_subset = affected_roads_gdf[affected_roads_gdf["potentially_affected"] == True]
        else:
            aff_subset = affected_roads_gdf

        for idx, row in aff_subset.iterrows():
            r_id = str(row.get("osm_id") or row.get("@osmId") or row.get("id") or f"road/{idx}")
            affected_road_ids.add(r_id)
            affected_roads_meta[r_id] = {
                "is_affected": True,
                "evidence_tier": str(row.get("evidence_tier") or "confirmed_dual_evidence"),
                "impact_class": str(row.get("impact_class") or "flood_or_change"),
                "affected_length_m": float(row.get("affected_length_m") or 0.0),
            }

    # 3. Determine base road network
    roads_layer = osm_data.layers.get("roads", {})
    roads_gdf = _to_gdf(roads_layer, src_crs="EPSG:4326")
    if roads_gdf.empty and not affected_roads_gdf.empty:
        roads_gdf = affected_roads_gdf

    # Configure connectivity options
    cfg = settings.connectivity.model_copy()
    if disruption_mode:
        cfg.disruption_mode = disruption_mode

    # 4. Build Road Graph
    with prov.step("build_road_graph", total_roads=len(roads_gdf), affected_roads_count=len(affected_road_ids)) as st:
        road_graph = build_road_graph(
            roads_gdf=roads_gdf,
            affected_road_ids=affected_road_ids,
            speeds_kmh=cfg.speeds_kmh,
            affected_roads_meta=affected_roads_meta,
        )
        st["outputs"].append({
            "total_nodes": road_graph.graph.number_of_nodes(),
            "total_edges": road_graph.total_edges_count,
            "affected_edges": road_graph.affected_edges_count,
            "metric_crs": road_graph.metric_crs,
        })

    # 5. Evaluate Connectivity
    places_layer = osm_data.layers.get("places", {})
    health_layer = osm_data.layers.get("health", {})

    with prov.step("evaluate_connectivity", disruption_mode=cfg.disruption_mode) as st:
        settlements_gdf, cut_off_gdf, records, eval_warnings = evaluate_connectivity(
            road_graph=road_graph,
            places_fc=places_layer,
            health_fc=health_layer,
            cfg=cfg,
        )
        for w in eval_warnings:
            warn(w)

        from ..network.connectivity import analyze_disconnected_components, extract_destinations
        places_gdf_norm = _to_gdf(places_layer, src_crs="EPSG:4326")
        health_gdf_norm = _to_gdf(health_layer, src_crs="EPSG:4326") if health_layer else None
        destinations = extract_destinations(places_gdf_norm, health_gdf_norm, road_graph, cfg)

        snapped_nodes = {}
        for r in records:
            if not settlements_gdf.empty and (settlements_gdf["settlement_id"] == r.settlement_id).any():
                g = settlements_gdf.loc[settlements_gdf["settlement_id"] == r.settlement_id, "geometry"].iloc[0]
                s_node, _ = road_graph.snap_point(g, src_crs="EPSG:4326", max_distance_m=cfg.max_snapping_distance_m)
                if s_node:
                    snapped_nodes[r.settlement_id] = s_node

        comp_summary = analyze_disconnected_components(road_graph, destinations, records, snapped_nodes)

        connected_count = sum(1 for r in records if r.classification == "CONNECTED" or r.status == SettlementStatus.CONNECTED)
        disconnected_count = sum(1 for r in records if r.classification == "DISCONNECTED" or r.status == SettlementStatus.DISCONNECTED)
        no_baseline_count = sum(1 for r in records if r.classification == "NO_BASELINE_ROUTE")
        outside_network_count = sum(1 for r in records if r.classification == "OUTSIDE_NETWORK")
        insufficient_count = no_baseline_count + outside_network_count
        if insufficient_count == 0:
            insufficient_count = sum(1 for r in records if r.status == SettlementStatus.INSUFFICIENT_EVIDENCE)

        alt_route_count = sum(1 for r in records if r.classification == "CONNECTED" and r.alternate_route_distance_m is not None)

        st["outputs"].append({
            "total_settlements": len(records),
            "connected": connected_count,
            "disconnected": disconnected_count,
            "no_baseline_route": no_baseline_count,
            "outside_network": outside_network_count,
            "alternate_routes": alt_route_count,
            "disconnected_components": comp_summary.total_components,
            "isolated_components": comp_summary.isolated_components_count,
        })

    # 6. Save GeoJSON and Summary outputs
    outputs: dict[str, Path] = {}
    with prov.step("write_connectivity_outputs") as st:
        # 1. connectivity_results.json
        results_data = {
            "analysis_id": analysis_id,
            "data_mode": data_mode.value,
            "event_date": event_date.isoformat(),
            "osm_snapshot_date": osm_data.meta.snapshot_date.isoformat(),
            "disruption_mode": cfg.disruption_mode,
            "total_settlements": len(records),
            "metrics_summary": {
                "connected": connected_count,
                "disconnected": disconnected_count,
                "no_baseline_route": no_baseline_count,
                "outside_network": outside_network_count,
                "alternate_route_count": alt_route_count,
            },
            "disconnected_components_summary": {
                "total_components": comp_summary.total_components,
                "baseline_components": comp_summary.baseline_components,
                "new_components_formed": comp_summary.new_components_count,
                "isolated_components_count": comp_summary.isolated_components_count,
            },
            "settlements": [r.model_dump() for r in records],
            "disconnected_components": comp_summary.components,
            "scientific_disclaimer": STAGE5_INTERPRETATION_NOTES[0],
        }
        conn_res_path = run_dir / "connectivity_results.json"
        conn_res_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
        outputs["connectivity_results"] = conn_res_path

        # 2. disconnected_settlements.geojson
        disc_path = run_dir / "disconnected_settlements.geojson"
        cut_off_gdf.to_crs("EPSG:4326").to_file(disc_path, driver="GeoJSON")
        outputs["disconnected_settlements"] = disc_path
        outputs["cut_off_settlements"] = disc_path  # backward compatibility alias

        # 3. road_network.geojson
        road_net_path = run_dir / "road_network.geojson"
        road_net_gdf = road_graph.to_geodataframe(target_crs="EPSG:4326")
        road_net_gdf.to_file(road_net_path, driver="GeoJSON")
        outputs["road_network"] = road_net_path

        # Also save legacy settlement_connectivity.geojson
        conn_path = run_dir / "settlement_connectivity.geojson"
        settlements_gdf.to_crs("EPSG:4326").to_file(conn_path, driver="GeoJSON")
        outputs["settlement_connectivity"] = conn_path

        dest_types = list(cfg.destination_place_types)
        if cfg.include_health_destinations:
            dest_types.append("health")

        # 4. connectivity_summary.json
        summary = ConnectivitySummary(
            analysis_id=analysis_id,
            data_mode=data_mode,
            event_date=event_date,
            osm_snapshot_date=osm_data.meta.snapshot_date,
            aoi_name=aoi_name,
            disruption_mode=cfg.disruption_mode,
            total_settlements=len(records),
            connected_count=connected_count,
            disconnected_count=disconnected_count,
            no_baseline_route_count=no_baseline_count,
            outside_network_count=outside_network_count,
            insufficient_evidence_count=insufficient_count,
            alternate_route_count=alt_route_count,
            disconnected_components_count=comp_summary.total_components,
            isolated_components_count=comp_summary.isolated_components_count,
            affected_road_segments_count=len(affected_road_ids),
            destination_types=dest_types,
            parameters=cfg.model_dump(),
            outputs={k: str(v.name) for k, v in outputs.items()},
            warnings=warnings,
            interpretation_notes=STAGE5_INTERPRETATION_NOTES,
        )

        sum_path = run_dir / "connectivity_summary.json"
        sum_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        outputs["connectivity_summary"] = sum_path
        outputs["summary"] = sum_path  # backward compatibility alias
        st["outputs"].extend(str(p) for p in outputs.values())

    # 5. connectivity_provenance.json
    prov_path = prov.write_json(run_dir / "connectivity_provenance.json")
    outputs["connectivity_provenance"] = prov_path
    # Also save standard provenance.json for pipeline consistency
    legacy_prov_path = prov.write_json(run_dir / "provenance.json")
    outputs["provenance"] = legacy_prov_path

    return Stage5PipelineResult(
        run_dir=run_dir,
        summary=summary,
        outputs=outputs,
        provenance_path=prov_path,
        cut_off_gdf=cut_off_gdf,
        settlements_gdf=settlements_gdf,
        road_graph=road_graph,
        warnings=warnings,
    )
