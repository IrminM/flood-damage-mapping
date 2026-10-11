"""Stage-6 pipeline: DEM-Based Downstream Flood Path & Settlement Tracing.

Inputs:
  1. Copernicus DEM / GLO-30 (LocalDEMProvider with guard)
  2. Upstream point (lat/lon, GeoJSON Point, or (lon, lat) tuple)
  3. Pre-event OSM Snapshot (LocalOSMProvider with snapshot_date < event_date guard)

Outputs:
  1. downstream_path.geojson (derived terrain flow corridor LineString)
  2. downstream_path_settlements.geojson (settlements encountered in order along path)
  3. downstream_path_summary.json (Pydantic DownstreamPathSummary contract)
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
    DataMode,
    DownstreamPathSummary,
    DownstreamSettlementRecord,
    SourceType,
)
from ..core.data_rules import DataRulesGuard
from ..core.errors import InputDataError
from ..core.georaster import GeoRaster
from ..core.provenance import ProvenanceLog
from ..data_sources.base import OSMData
from ..data_sources.dem_local import LocalDEMProvider
from ..data_sources.osm_local import LocalOSMProvider
from ..dem.settlements import trace_nearby_settlements
from ..dem.tracer import DownstreamPathResult, parse_upstream_point, trace_downstream_path

STAGE6_METHODOLOGY_AND_LIMITATIONS = [
    "Methodology: Deterministic D8 terrain-following path derived from Copernicus DEM (GLO-30) elevation and drainage routing.",
    "Candidate path: Represents a likely topographical downstream flow corridor based strictly on terrain slope, not an observed inundation polygon.",
    "Hydraulic disclaimer: This is NOT a hydraulic or hydrodynamic simulation. Flow velocity, discharge volume, flood depth, backwater effects, and flood wave attenuation are NOT modeled.",
    "Sediment & debris disclaimer: Landslide damming, avulsions, debris clogging, culvert blockages, and infrastructure barriers are not accounted for.",
    "Settlement proximity: Settlements reported are pre-event OSM features situated within the configurable proximity corridor of the centerline path.",
    "Data integrity: Analysis utilizes exclusively pre-event OSM data and Copernicus DEM; no post-event OSM edits or published damage assessments were used.",
]


@dataclass
class Stage6PipelineResult:
    run_dir: Path
    summary: DownstreamPathSummary
    outputs: dict[str, Path]
    provenance_path: Path
    path_result: DownstreamPathResult
    path_gdf: gpd.GeoDataFrame
    settlements_gdf: gpd.GeoDataFrame
    warnings: list[str] = field(default_factory=list)


def run_stage6_pipeline(
    dem_source: GeoRaster | Path | str,
    upstream_point: Any,
    osm_source: Optional[OSMData | Path | str] = None,
    settings: Optional[Settings] = None,
    run_dir: Path | str = ".",
    event_date: Optional[date] = None,
    analysis_id: Optional[str] = None,
    aoi_name: Optional[str] = None,
    data_mode: DataMode = DataMode.REAL,
) -> Stage6PipelineResult:
    """Execute Stage 6 DEM downstream flood path and settlement tracing."""
    settings = settings or Settings()
    event_date = event_date or date.today()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_id = analysis_id or uuid.uuid4().hex[:12]
    guard = DataRulesGuard(settings.data_rules)
    prov = ProvenanceLog(analysis_id)
    prov.note("Stage-6 DEM-based downstream flood path and settlement tracing")
    prov.data_mode = data_mode
    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)
        prov.warn(msg)

    # 1. Load / validate DEM
    if isinstance(dem_source, (str, Path)):
        dem_path = Path(dem_source)
        loader = LocalDEMProvider(guard)
        dem = loader.load_dem(dem_path)
        prov.add_input(
            path=dem_path,
            source_type=dem.metadata.get("source_type", SourceType.COPERNICUS_DEM.value),
            role="dem",
            metadata={"bounds": list(dem.bounds), "crs": dem.crs, "shape": list(dem.shape)},
        )
    elif isinstance(dem_source, GeoRaster):
        dem = dem_source
        src_type = dem.metadata.get("source_type", SourceType.COPERNICUS_DEM.value)
        guard.check_input("in_memory_dem", declared_source=src_type)
        prov.add_input(
            path="in_memory",
            source_type=src_type,
            role="dem",
            metadata={"bounds": list(dem.bounds), "crs": dem.crs, "shape": list(dem.shape)},
        )
    else:
        raise InputDataError(f"Invalid dem_source: {type(dem_source)}")

    # 2. Load / validate pre-event OSM (optional)
    osm_data: Optional[OSMData] = None
    if osm_source is not None:
        if isinstance(osm_source, (str, Path)):
            osm_path = Path(osm_source)
            osm_loader = LocalOSMProvider(guard, settings.osm)
            osm_data = osm_loader.load(osm_path, event_date)
            prov.add_input(
                path=osm_path,
                source_type=SourceType.OSM_PRE_EVENT.value,
                role="osm_pre_event",
                metadata={
                    "snapshot_date": osm_data.meta.snapshot_date.isoformat(),
                    "counts": {l: osm_data.count(l) for l in osm_data.layers},
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
                    "counts": {l: osm_data.count(l) for l in osm_data.layers},
                },
            )
        else:
            raise InputDataError(f"Invalid osm_source: {type(osm_source)}")

        for w in osm_data.warnings:
            warn(f"osm: {w}")
    else:
        warn("No pre-event OSM source provided; downstream path traced without settlement proximity detection.")

    # 3. Parse upstream point
    orig_lon, orig_lat = parse_upstream_point(upstream_point)
    cfg = settings.downstream

    # 4. Trace Downstream Path
    with prov.step("trace_downstream_path", upstream=(orig_lon, orig_lat)) as st:
        path_result = trace_downstream_path(
            dem=dem,
            upstream_point=(orig_lon, orig_lat),
            cfg=cfg,
        )
        st["outputs"].append({
            "path_length_km": round(path_result.total_length_m / 1000.0, 2),
            "vertices": len(path_result.coords_lonlat),
            "start_elev_m": round(path_result.start_elevation_m, 1),
            "end_elev_m": round(path_result.end_elevation_m, 1),
            "termination_reason": path_result.termination_reason,
        })

    # Assemble GeoDataFrame for downstream path
    path_props = {
        "geometry": path_result.line_geom_lonlat,
        "path_length_km": round(path_result.total_length_m / 1000.0, 2),
        "start_elevation_m": round(path_result.start_elevation_m, 1),
        "end_elevation_m": round(path_result.end_elevation_m, 1),
        "total_drop_m": round(path_result.total_drop_m, 1),
        "vertices_count": len(path_result.coords_lonlat),
        "termination_reason": path_result.termination_reason,
        "snapping_distance_m": round(path_result.snapping_distance_m, 1),
        "upstream_lon": round(path_result.original_coords_lonlat[0], 5),
        "upstream_lat": round(path_result.original_coords_lonlat[1], 5),
        "snapped_lon": round(path_result.snapped_coords_lonlat[0], 5),
        "snapped_lat": round(path_result.snapped_coords_lonlat[1], 5),
    }
    path_gdf = gpd.GeoDataFrame([path_props], crs="EPSG:4326")

    # 5. Trace Nearby Settlements along Path
    places_layer = osm_data.layers.get("places", {}) if osm_data is not None else {"type": "FeatureCollection", "features": []}
    with prov.step("trace_nearby_settlements", radius_m=cfg.settlement_proximity_radius_m) as st:
        settlements_gdf, settlement_records = trace_nearby_settlements(
            path_result=path_result,
            places_fc=places_layer,
            cfg=cfg,
        )
        st["outputs"].append({
            "settlements_found": len(settlement_records),
            "proximity_radius_m": cfg.settlement_proximity_radius_m,
        })

    # 6. Save GeoJSON and Summary outputs
    outputs: dict[str, Path] = {}
    with prov.step("write_downstream_outputs") as st:
        path_file = run_dir / "downstream_path.geojson"
        path_gdf.to_file(path_file, driver="GeoJSON")
        outputs["downstream_path"] = path_file

        settlements_file = run_dir / "downstream_path_settlements.geojson"
        settlements_gdf.to_file(settlements_file, driver="GeoJSON")
        outputs["downstream_path_settlements"] = settlements_file

        summary = DownstreamPathSummary(
            analysis_id=analysis_id,
            data_mode=data_mode,
            event_date=event_date,
            aoi_name=aoi_name,
            upstream_input_coords=(round(orig_lon, 5), round(orig_lat, 5)),
            upstream_snapped_coords=(
                round(path_result.snapped_coords_lonlat[0], 5),
                round(path_result.snapped_coords_lonlat[1], 5),
            ),
            snapping_distance_m=round(path_result.snapping_distance_m, 1),
            start_elevation_m=round(path_result.start_elevation_m, 1),
            end_elevation_m=round(path_result.end_elevation_m, 1),
            total_elevation_drop_m=round(path_result.total_drop_m, 1),
            path_length_km=round(path_result.total_length_m / 1000.0, 2),
            path_vertices_count=len(path_result.coords_lonlat),
            termination_reason=path_result.termination_reason,
            settlements_count=len(settlement_records),
            settlements=settlement_records,
            parameters=cfg.model_dump(),
            outputs={k: str(v.name) for k, v in outputs.items()},
            warnings=warnings,
            methodology_and_limitations=(
                STAGE6_METHODOLOGY_AND_LIMITATIONS
                if osm_data is not None
                else STAGE6_METHODOLOGY_AND_LIMITATIONS
                + ["Data limitation: Pre-event OSM places layer was unavailable; no settlement proximity detection was performed along the path."]
            ),
        )

        sum_path = run_dir / "downstream_path_summary.json"
        sum_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        outputs["summary"] = sum_path
        st["outputs"].extend(str(p) for p in outputs.values())

    prov_path = prov.write_json(run_dir / "provenance.json")
    outputs["provenance"] = prov_path

    return Stage6PipelineResult(
        run_dir=run_dir,
        summary=summary,
        outputs=outputs,
        provenance_path=prov_path,
        path_result=path_result,
        path_gdf=path_gdf,
        settlements_gdf=settlements_gdf,
        warnings=warnings,
    )
