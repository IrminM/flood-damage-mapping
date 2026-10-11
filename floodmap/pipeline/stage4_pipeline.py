"""Stage-4 pipeline: Pre-event OSM Data + Raster Flood/Change Evidence -> Infrastructure Impact Assessment.

    Inputs:
      1. Raster Flood/Change Evidence (from Stage 2 ChangeResult or Stage 3 FusedDetectionResult)
      2. Pre-event OSM Snapshot (LocalOSMProvider with snapshot_date < event_date guard)

    Outputs:
      1. infrastructure_summary.json (Pydantic InfrastructureSummary contract)
      2. affected_buildings.geojson (buildings with footprint affected metrics)
      3. affected_roads.geojson (individual road segments with affected length & flags)
      4. affected_bridges.geojson (bridge points/lines/polygons with impact status)
      5. flood_evidence_vector.geojson (vectorized flood & change polygons)
      6. provenance.json (full lineage, input hashes, parameters)
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
    DataMode,
    InfrastructureSummary,
    SourceType,
)
from ..core.data_rules import DataRulesGuard
from ..core.errors import InputDataError
from ..core.georaster import GeoRaster
from ..core.provenance import ProvenanceLog
from ..data_sources.base import OSMData
from ..data_sources.osm_local import LocalOSMProvider
from ..gis.impact import (
    assess_bridges,
    assess_buildings,
    assess_roads,
    assess_waterway_context,
)
from ..gis.vectorize import vectorize_evidence

STAGE4_INTERPRETATION_NOTES = [
    "Stage-4 infrastructure impact: Identifies pre-event OSM features that spatially intersect detected flood/change evidence.",
    "Features are reported as 'potentially affected' / 'intersecting detected evidence'; physical structural destruction cannot be claimed without ground-truth structural inspection.",
    "Buildings are assessed by footprint polygon intersection; roads are assessed by centerline intersection with configurable half-width buffer.",
    "Bridges are reported as 'potentially affected / approach flooded'; water passing beneath an intact bridge deck cannot be distinguished from structural failure using SAR backscatter alone.",
    "Road segments retain individual identifiers and affected length/fraction for network disruption analysis in Stage 5.",
    "OSM data is strictly pre-event; no post-event OSM edits or published disaster maps were used.",
]


@dataclass
class Stage4PipelineResult:
    run_dir: Path
    summary: InfrastructureSummary
    outputs: dict[str, Path]
    provenance_path: Path
    buildings_gdf: gpd.GeoDataFrame
    roads_gdf: gpd.GeoDataFrame
    bridges_gdf: gpd.GeoDataFrame
    evidence_gdf: gpd.GeoDataFrame
    warnings: list[str] = field(default_factory=list)


def run_stage4_pipeline(
    classes_raster: GeoRaster,
    osm_source: OSMData | Path | str,
    settings: Settings,
    run_dir: Path | str,
    event_date: date,
    evidence_raster: Optional[GeoRaster] = None,
    analysis_id: Optional[str] = None,
    aoi_name: Optional[str] = None,
    data_mode: DataMode = DataMode.REAL,
) -> Stage4PipelineResult:
    """Execute Stage 4 infrastructure impact analysis."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_id = analysis_id or uuid.uuid4().hex[:12]
    guard = DataRulesGuard(settings.data_rules)
    prov = ProvenanceLog(analysis_id)
    prov.note("Stage-4 OSM infrastructure impact analysis (pre-event OSM)")
    prov.data_mode = data_mode
    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)
        prov.warn(msg)

    # 1. Load / validate OSM data
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

    # 2. Vectorize flood & change evidence
    with prov.step("vectorize_evidence", target_classes=settings.impact.include_flood_classes) as st:
        evidence_gdf = vectorize_evidence(
            classes_raster=classes_raster,
            evidence_raster=evidence_raster,
            target_classes=settings.impact.include_flood_classes,
        )
        st["outputs"].append({
            "polygon_count": len(evidence_gdf),
            "crs": classes_raster.crs,
            "classes": evidence_gdf["class_name"].unique().tolist() if not evidence_gdf.empty else [],
        })

    # 3. Assess Buildings
    with prov.step("assess_buildings", buffer_m=settings.impact.building_buffer_m) as st:
        buildings_gdf, b_stats = assess_buildings(
            buildings_fc=osm_data.layers.get("buildings", {}),
            evidence_gdf=evidence_gdf,
            target_crs=classes_raster.crs,
            cfg=settings.impact,
        )
        st["outputs"].append(b_stats.model_dump())

    # 4. Assess Roads
    with prov.step("assess_roads", buffer_m=settings.impact.road_buffer_m) as st:
        roads_gdf, r_stats = assess_roads(
            roads_fc=osm_data.layers.get("roads", {}),
            evidence_gdf=evidence_gdf,
            target_crs=classes_raster.crs,
            cfg=settings.impact,
        )
        st["outputs"].append(r_stats.model_dump())

    # 5. Assess Bridges
    with prov.step("assess_bridges", buffer_m=settings.impact.bridge_buffer_m) as st:
        bridges_gdf, br_stats = assess_bridges(
            bridges_fc=osm_data.layers.get("bridges", {}),
            evidence_gdf=evidence_gdf,
            target_crs=classes_raster.crs,
            cfg=settings.impact,
        )
        st["outputs"].append(br_stats.model_dump())

    # 6. Contextual Waterways
    with prov.step("assess_waterway_context", buffer_m=settings.impact.waterway_buffer_m) as st:
        w_stats = assess_waterway_context(
            waterways_fc=osm_data.layers.get("waterways"),
            evidence_gdf=evidence_gdf,
            target_crs=classes_raster.crs,
            cfg=settings.impact,
        )
        if w_stats:
            st["outputs"].append(w_stats.model_dump())

    # 7. Write outputs
    outputs: dict[str, Path] = {}
    with prov.step("write_infrastructure_outputs") as st:
        # Write GeoJSON files in EPSG:4326 for web / Leaflet interoperability
        if not evidence_gdf.empty:
            ev_path = run_dir / "flood_evidence_vector.geojson"
            evidence_gdf.to_crs("EPSG:4326").to_file(ev_path, driver="GeoJSON")
            outputs["flood_evidence_vector"] = ev_path

        b_path = run_dir / "affected_buildings.geojson"
        buildings_gdf.to_crs("EPSG:4326").to_file(b_path, driver="GeoJSON")
        outputs["affected_buildings"] = b_path

        r_path = run_dir / "affected_roads.geojson"
        roads_gdf.to_crs("EPSG:4326").to_file(r_path, driver="GeoJSON")
        outputs["affected_roads"] = r_path

        br_path = run_dir / "affected_bridges.geojson"
        bridges_gdf.to_crs("EPSG:4326").to_file(br_path, driver="GeoJSON")
        outputs["affected_bridges"] = br_path

        summary = InfrastructureSummary(
            analysis_id=analysis_id,
            data_mode=data_mode,
            event_date=event_date,
            aoi_name=aoi_name,
            osm_snapshot_date=osm_data.meta.snapshot_date,
            crs=classes_raster.crs,
            buildings=b_stats,
            roads=r_stats,
            bridges=br_stats,
            waterways_context=w_stats,
            parameters=settings.impact.model_dump(),
            outputs={k: str(v.name) for k, v in outputs.items()},
            warnings=warnings,
            interpretation_notes=STAGE4_INTERPRETATION_NOTES,
        )

        sum_path = run_dir / "infrastructure_summary.json"
        sum_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        outputs["summary"] = sum_path
        st["outputs"].extend(str(p) for p in outputs.values())

    prov_path = prov.write_json(run_dir / "provenance.json")
    outputs["provenance"] = prov_path

    return Stage4PipelineResult(
        run_dir=run_dir,
        summary=summary,
        outputs=outputs,
        provenance_path=prov_path,
        buildings_gdf=buildings_gdf,
        roads_gdf=roads_gdf,
        bridges_gdf=bridges_gdf,
        evidence_gdf=evidence_gdf,
        warnings=warnings,
    )
