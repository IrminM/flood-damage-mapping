"""End-to-end pipeline orchestrator chaining Stages 1 through 6.

Orchestration Flow:
  Stage 1 (SAR Ingestion & Grid) ->
  Stage 2 (Change Detection) ->
  Stage 3 (AI Flood Segmentation & Fusion) ->
  Stage 4 (OSM Infrastructure Impact Assessment - if pre-event OSM available) ->
  Stage 5 (Road Network Connectivity & Cut-Off Settlements - if OSM available) ->
  Stage 6 (DEM Downstream Flow Path & Settlement Tracing - if upstream point provided)

Data Integrity Guarantees:
  - Unavailable OSM data is explicitly reported as UNAVAILABLE; zero counts are NEVER fabricated.
  - Baseline AI model (s1_mini_unet_v1) is explicitly recorded as an uncalibrated baseline development model.
  - Candidate Model A (best_flood_unet_best.pth) is preserved as a future plug-in and blocked from unverified inference.
  - Complete multi-source provenance and limitations are preserved for every stage.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..core.config import Settings, load_settings
from ..core.contracts import (
    AOI,
    AnalysisRequest,
    AnalysisRunConfig,
    AnalysisStatus,
    AnalysisSubmitRequest,
    ConnectivitySummary,
    DataAvailabilityStatus,
    DataAvailabilitySummary,
    DataMode,
    DownstreamPathSummary,
    FullAnalysisResult,
    InfrastructureSummary,
    LayerMetadata,
    SituationReport,
    Stage3Summary,
)
from ..core.errors import FloodmapError, InputDataError
from .stage3_pipeline import Stage3PipelineResult, run_stage3_pipeline
from .stage4_pipeline import Stage4PipelineResult, run_stage4_pipeline
from .stage5_pipeline import Stage5PipelineResult, run_stage5_pipeline
from .stage6_pipeline import Stage6PipelineResult, run_stage6_pipeline

logger = logging.getLogger(__name__)


def _resolve_aoi(aoi_input: Any, name: Optional[str] = None) -> AOI:
    """Normalize flexible AOI inputs (AOI, GeoJSON dict, Feature, bbox list, or file path) into AOI."""
    if isinstance(aoi_input, AOI):
        return aoi_input
    if isinstance(aoi_input, dict):
        if "geometry" in aoi_input and isinstance(aoi_input["geometry"], dict):
            return AOI(geometry=aoi_input["geometry"], name=aoi_input.get("name") or name)
        from ..core.geo_utils import aoi_from_geojson
        return aoi_from_geojson(aoi_input, name=name)
    if isinstance(aoi_input, (list, tuple)) and len(aoi_input) == 4:
        from ..core.geo_utils import aoi_from_bounds
        return aoi_from_bounds(tuple(aoi_input), name=name)
    if isinstance(aoi_input, (str, Path)):
        from ..core.geo_utils import load_aoi
        return load_aoi(str(aoi_input))
    raise ValueError(f"Cannot resolve AOI from input of type {type(aoi_input)}")



def _check_osm_availability(osm_path: Optional[Path]) -> tuple[bool, Optional[str]]:
    """Check if a directory contains genuine pre-event OSM data files."""
    if osm_path is None or not osm_path.exists():
        return False, "OSM directory not provided or does not exist on disk."

    meta_file = osm_path / "snapshot_meta.json"
    if not meta_file.exists() or meta_file.stat().st_size == 0:
        return False, f"OSM directory {osm_path} is missing snapshot_meta.json metadata file."

    # Check for presence of required GeoJSON layers
    required = ["roads.geojson", "buildings.geojson"]
    found = [f for f in required if (osm_path / f).is_file() and (osm_path / f).stat().st_size > 0]
    if len(found) < len(required):
        return False, (
            f"OSM directory {osm_path} is missing pre-event layers (found: {found}, expected: {required}). "
            "Historical OSM data must be downloaded prior to running infrastructure analysis."
        )
    return True, None


def run_full_pipeline(
    submit_req: AnalysisSubmitRequest,
    settings: Optional[Settings] = None,
    base_runs_dir: Path | str = "data/runs",
    analysis_id: Optional[str] = None,
) -> FullAnalysisResult:
    """Execute end-to-end disaster analysis across Stages 1–6."""
    start_time = time.time()
    created_at = datetime.now(timezone.utc)
    settings = settings or load_settings()
    analysis_id = analysis_id or uuid.uuid4().hex[:12]
    run_dir = Path(base_runs_dir) / analysis_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Apply optional runtime configuration overrides
    if submit_req.config:
        cfg = submit_req.config
        if cfg.ai_model:
            settings.ai.model = cfg.ai_model
        if cfg.ai_threshold is not None:
            settings.ai.s1_mini_unet_v1.threshold = cfg.ai_threshold
        if cfg.change_threshold_db is not None:
            settings.change_detection.drop_threshold_db = cfg.change_threshold_db
        if cfg.sink_handling:
            settings.downstream.sink_handling = cfg.sink_handling
        if cfg.downstream_max_length_km:
            settings.downstream.max_length_m = cfg.downstream_max_length_km * 1000.0
        if cfg.downstream_proximity_radius_m:
            settings.downstream.settlement_proximity_radius_m = cfg.downstream_proximity_radius_m

    # 2. Resolve input data locations
    before_scene = Path(submit_req.before_scene) if submit_req.before_scene else None
    after_scene = Path(submit_req.after_scene) if submit_req.after_scene else None
    dem_path = Path(submit_req.dem) if submit_req.dem else None
    osm_dir = Path(submit_req.osm_dir) if submit_req.osm_dir else None

    # Fallback to dev defaults if unprovided
    if before_scene is None and (Path("data/raw/dev/s1_before_20260816.tif")).exists():
        before_scene = Path("data/raw/dev/s1_before_20260816.tif")
    if after_scene is None and (Path("data/raw/dev/s1_after_20260828.tif")).exists():
        after_scene = Path("data/raw/dev/s1_after_20260828.tif")
    if dem_path is None and (Path("data/dem/dev/dem_glo30.tif")).exists():
        dem_path = Path("data/dem/dev/dem_glo30.tif")
    if osm_dir is None and (Path("data/osm/dev")).exists():
        osm_dir = Path("data/osm/dev")

    # 3. Assess Data Availability
    osm_available, osm_reason = _check_osm_availability(osm_dir)
    data_avail = DataAvailabilitySummary(
        sentinel1=DataAvailabilityStatus.AVAILABLE if (before_scene and after_scene) else DataAvailabilityStatus.UNAVAILABLE,
        sentinel2=DataAvailabilityStatus.NOT_PROVIDED,
        copernicus_dem=DataAvailabilityStatus.AVAILABLE if (dem_path and dem_path.exists()) else DataAvailabilityStatus.UNAVAILABLE,
        osm_preevent=DataAvailabilityStatus.AVAILABLE if osm_available else DataAvailabilityStatus.UNAVAILABLE,
        ai_segmentation=DataAvailabilityStatus.AVAILABLE,
        ai_model_name=settings.ai.model,
        ai_model_role=(
            "Baseline development model (uncalibrated, not teammate's Model A)"
            if settings.ai.model == "s1_mini_unet_v1"
            else "Plug-in model candidate"
        ),
        osm_unavailable_reason=osm_reason,
    )

    stages_executed: dict[str, str] = {}
    warnings: list[str] = []
    limitations: list[str] = []
    layers: dict[str, LayerMetadata] = {}

    stage3_res: Optional[Stage3PipelineResult] = None
    stage4_res: Optional[Stage4PipelineResult] = None
    stage5_res: Optional[Stage5PipelineResult] = None
    stage6_res: Optional[Stage6PipelineResult] = None

    # 4. Execute Stages 1 -> 2 -> 3 (SAR, Preprocessing, Terrain, Change Detection, AI Segmentation, Fusion)
    try:
        analysis_req = AnalysisRequest(
            aoi=_resolve_aoi(submit_req.aoi, submit_req.name),
            event_date=submit_req.event_date,
            before_scene=before_scene,
            after_scene=after_scene,
            dem=dem_path,
            osm_dir=osm_dir,
            name=submit_req.name,
        )
        stage3_res = run_stage3_pipeline(
            request=analysis_req,
            settings=settings,
            run_dir=run_dir,
            analysis_id=analysis_id,
            write_quicklooks=True,
        )
        stages_executed["stage1_sar_ingestion"] = "COMPLETED"
        stages_executed["stage2_change_detection"] = "COMPLETED"
        stages_executed["stage3_ai_fusion"] = "COMPLETED"
        warnings.extend(stage3_res.warnings)
        limitations.extend(stage3_res.summary.interpretation_notes)
    except Exception as e:
        logger.error(f"Stage 1-3 execution failed: {e}", exc_info=True)
        return FullAnalysisResult(
            analysis_id=analysis_id,
            status=AnalysisStatus.FAILED,
            data_mode=DataMode.REAL,
            event_date=submit_req.event_date,
            aoi_name=submit_req.name,
            created_at=created_at,
            completed_at=datetime.now(timezone.utc),
            execution_time_seconds=round(time.time() - start_time, 2),
            data_availability=data_avail,
            stages_executed=stages_executed,
            warnings=[f"Pipeline error: {str(e)}"],
            uncertainty_and_limitations=["Analysis aborted due to input or processing failure."],
        )

    # Register Stage 3 Layers
    for key, path in stage3_res.outputs.items():
        ext = path.suffix.lower()
        l_type = "raster_geotiff" if ext in (".tif", ".tiff") else "quicklook_png" if ext == ".png" else "data_json"
        layers[key] = LayerMetadata(
            layer_id=key,
            name=key.replace("_", " ").title(),
            layer_type=l_type,
            file_path=str(path),
            url=f"/api/v1/analyses/{analysis_id}/layers/{key}",
            description=f"Stage 3 output: {key}",
            data_available=True,
        )

    # 5. Execute Stage 4: OSM Infrastructure Impact (if pre-event OSM available)
    if osm_available and osm_dir is not None:
        try:
            stage4_dir = run_dir / "infrastructure"
            stage4_res = run_stage4_pipeline(
                classes_raster=stage3_res.fused.classes,
                evidence_raster=stage3_res.fused.evidence_source,
                osm_source=osm_dir,
                settings=settings,
                run_dir=stage4_dir,
                event_date=submit_req.event_date,
                analysis_id=f"{analysis_id}_s4",
                aoi_name=submit_req.name,
                data_mode=stage3_res.summary.data_mode,
            )
            stages_executed["stage4_infrastructure"] = "COMPLETED"
            warnings.extend(stage4_res.warnings)
            limitations.extend(stage4_res.summary.interpretation_notes)

            for key, path in stage4_res.outputs.items():
                l_type = "vector_geojson" if path.suffix.lower() == ".geojson" else "data_json"
                layers[f"infra_{key}"] = LayerMetadata(
                    layer_id=f"infra_{key}",
                    name=f"Infrastructure: {key.replace('_', ' ').title()}",
                    layer_type=l_type,
                    file_path=str(path),
                    url=f"/api/v1/analyses/{analysis_id}/layers/infra_{key}",
                    description=f"Stage 4 infrastructure layer: {key}",
                    data_available=True,
                )
        except Exception as e:
            logger.warning(f"Stage 4 failed: {e}")
            stages_executed["stage4_infrastructure"] = f"FAILED: {str(e)}"
            warnings.append(f"Stage 4 infrastructure impact failed: {str(e)}")
    else:
        stages_executed["stage4_infrastructure"] = "UNAVAILABLE_NO_PREEVENT_OSM"
        limitations.append(
            "STAGE 4 NOTE: Pre-event OpenStreetMap infrastructure data was unavailable. "
            "Impact statistics on buildings, roads, and bridges were NOT computed to avoid fabricating false zero-impact figures."
        )

    # 6. Execute Stage 5: Road Network Connectivity & Cut-Off Settlements
    if stage4_res is not None and "affected_roads" in stage4_res.outputs and osm_available and osm_dir is not None:
        try:
            stage5_dir = run_dir / "connectivity"
            stage5_res = run_stage5_pipeline(
                affected_roads=stage4_res.outputs["affected_roads"],
                osm_source=osm_dir,
                settings=settings,
                run_dir=stage5_dir,
                event_date=submit_req.event_date,
                analysis_id=f"{analysis_id}_s5",
                aoi_name=submit_req.name,
                data_mode=stage3_res.summary.data_mode,
                disruption_mode=submit_req.config.disruption_mode if submit_req.config else "remove",
            )
            stages_executed["stage5_connectivity"] = "COMPLETED"
            warnings.extend(stage5_res.warnings)
            limitations.extend(stage5_res.summary.interpretation_notes)

            for key, path in stage5_res.outputs.items():
                l_type = "vector_geojson" if path.suffix.lower() == ".geojson" else "data_json"
                layers[f"conn_{key}"] = LayerMetadata(
                    layer_id=f"conn_{key}",
                    name=f"Connectivity: {key.replace('_', ' ').title()}",
                    layer_type=l_type,
                    file_path=str(path),
                    url=f"/api/v1/analyses/{analysis_id}/layers/conn_{key}",
                    description=f"Stage 5 connectivity layer: {key}",
                    data_available=True,
                )
        except Exception as e:
            logger.warning(f"Stage 5 failed: {e}")
            stages_executed["stage5_connectivity"] = f"FAILED: {str(e)}"
            warnings.append(f"Stage 5 connectivity analysis failed: {str(e)}")
    else:
        stages_executed["stage5_connectivity"] = "UNAVAILABLE_NO_ROAD_NETWORK"
        limitations.append(
            "STAGE 5 NOTE: Road network connectivity analysis was omitted because pre-event road network / Stage 4 outputs were unavailable."
        )

    # 7. Execute Stage 6: Downstream Flow Path & Settlement Tracing (if upstream point provided)
    if submit_req.upstream_point is not None:
        if dem_path is not None and dem_path.exists():
            try:
                stage6_dir = run_dir / "downstream"
                stage6_res = run_stage6_pipeline(
                    dem_source=dem_path,
                    upstream_point=submit_req.upstream_point,
                    osm_source=osm_dir if osm_available else None,
                    settings=settings,
                    run_dir=stage6_dir,
                    event_date=submit_req.event_date,
                    analysis_id=f"{analysis_id}_s6",
                    aoi_name=submit_req.name,
                    data_mode=stage3_res.summary.data_mode,
                )
                stages_executed["stage6_downstream"] = "COMPLETED"
                warnings.extend(stage6_res.warnings)
                limitations.extend(stage6_res.summary.methodology_and_limitations)

                for key, path in stage6_res.outputs.items():
                    l_type = "vector_geojson" if path.suffix.lower() == ".geojson" else "data_json"
                    layers[f"downstream_{key}"] = LayerMetadata(
                        layer_id=f"downstream_{key}",
                        name=f"Downstream: {key.replace('_', ' ').title()}",
                        layer_type=l_type,
                        file_path=str(path),
                        url=f"/api/v1/analyses/{analysis_id}/layers/downstream_{key}",
                        description=f"Stage 6 downstream layer: {key}",
                        data_available=True,
                    )
            except Exception as e:
                logger.warning(f"Stage 6 failed: {e}")
                stages_executed["stage6_downstream"] = f"FAILED: {str(e)}"
                warnings.append(f"Stage 6 downstream path tracing failed: {str(e)}")
        else:
            stages_executed["stage6_downstream"] = "UNAVAILABLE_NO_DEM"
            limitations.append("STAGE 6 NOTE: Downstream path tracing omitted because DEM raster is unavailable.")
    else:
        stages_executed["stage6_downstream"] = "SKIPPED_NO_UPSTREAM_POINT"

    # 8. Load provenance if written
    prov_data = None
    prov_file = run_dir / "provenance.json"
    if prov_file.exists():
        try:
            prov_data = json.loads(prov_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 9. Assemble Full Result
    full_result = FullAnalysisResult(
        analysis_id=analysis_id,
        status=AnalysisStatus.COMPLETED,
        data_mode=stage3_res.summary.data_mode,
        event_date=submit_req.event_date,
        aoi_name=submit_req.name,
        created_at=created_at,
        completed_at=datetime.now(timezone.utc),
        execution_time_seconds=round(time.time() - start_time, 2),
        data_availability=data_avail,
        stages_executed=stages_executed,
        stage3_summary=stage3_res.summary if stage3_res else None,
        stage4_summary=stage4_res.summary if stage4_res else None,
        stage5_summary=stage5_res.summary if stage5_res else None,
        stage6_summary=stage6_res.summary if stage6_res else None,
        available_layers=layers,
        provenance=prov_data,
        uncertainty_and_limitations=limitations,
        warnings=warnings,
    )

    # Persist machine-readable result
    res_path = run_dir / "full_analysis_result.json"
    res_path.write_text(full_result.model_dump_json(indent=2), encoding="utf-8")

    # 10. Assemble Situation Report
    sitrep = generate_situation_report(full_result)
    sitrep_path = run_dir / "situation_report.json"
    sitrep_path.write_text(sitrep.model_dump_json(indent=2), encoding="utf-8")

    return full_result


def generate_situation_report(result: FullAnalysisResult) -> SituationReport:
    """Synthesize operational situation report from complete analysis results."""
    flood_findings: dict[str, Any] = {}
    if result.stage3_summary:
        s3 = result.stage3_summary
        flood_findings = {
            "total_flood_water_km2": s3.class_areas_km2.get("new_open_water", 0.0),
            "permanent_water_km2": s3.class_areas_km2.get("water_both_dates", 0.0),
            "surface_change_debris_km2": s3.class_areas_km2.get("non_water_backscatter_change", 0.0),
            "evidence_breakdown_km2": s3.evidence_areas_km2,
            "ai_confirmed_flood_km2": s3.evidence_areas_km2.get("confirmed_by_both", 0.0),
            "confidence_ratio_both": (
                round(s3.evidence_areas_km2.get("confirmed_by_both", 0.0) / s3.class_areas_km2.get("new_open_water", 1.0), 3)
                if s3.class_areas_km2.get("new_open_water", 0.0) > 0 else 0.0
            ),
        }

    infra_findings: dict[str, Any] = {}
    if result.stage4_summary:
        s4 = result.stage4_summary
        infra_findings = {
            "status": "COMPLETED",
            "buildings_potentially_affected": s4.buildings.potentially_affected_count,
            "buildings_total_evaluated": s4.buildings.total_count,
            "roads_affected_km": s4.roads.potentially_affected_length_km,
            "roads_total_km": s4.roads.total_length_km,
            "bridges_potentially_affected": s4.bridges.potentially_affected_count,
            "bridges_total_evaluated": s4.bridges.total_count,
        }
    else:
        infra_findings = {
            "status": "UNAVAILABLE",
            "message": "Pre-event OpenStreetMap data was not available; impact figures not evaluated.",
        }

    conn_findings: dict[str, Any] = {}
    if result.stage5_summary:
        s5 = result.stage5_summary
        conn_findings = {
            "status": "COMPLETED",
            "settlements_cut_off": s5.disconnected_count,
            "settlements_connected": s5.connected_count,
            "settlements_insufficient_network_evidence": s5.insufficient_evidence_count,
            "affected_road_segments": s5.affected_road_segments_count,
        }
    else:
        conn_findings = {
            "status": "UNAVAILABLE",
            "message": "Road network connectivity analysis was omitted because pre-event road network was unavailable.",
        }

    downstream_findings: Optional[dict[str, Any]] = None
    if result.stage6_summary:
        s6 = result.stage6_summary
        downstream_findings = {
            "status": "COMPLETED",
            "path_length_km": s6.path_length_km,
            "total_elevation_drop_m": s6.total_elevation_drop_m,
            "start_elevation_m": s6.start_elevation_m,
            "end_elevation_m": s6.end_elevation_m,
            "termination_reason": s6.termination_reason,
            "settlements_along_path_count": s6.settlements_count,
            "settlements_detail": [s.model_dump() for s in s6.settlements],
        }

    exec_summary_parts = [
        f"Rapid Disaster Assessment for event on {result.event_date.isoformat()} (ID: {result.analysis_id}).",
    ]
    if flood_findings:
        exec_summary_parts.append(
            f"Detected {flood_findings.get('total_flood_water_km2', 0.0):.2f} km² of new flood water, "
            f"including {flood_findings.get('ai_confirmed_flood_km2', 0.0):.2f} km² jointly confirmed by both "
            "temporal SAR change detection and AI spatial segmentation."
        )
    if infra_findings.get("status") == "COMPLETED":
        exec_summary_parts.append(
            f"Infrastructure impact: {infra_findings.get('buildings_potentially_affected', 0)} buildings, "
            f"{infra_findings.get('roads_affected_km', 0.0):.2f} km of roads, and "
            f"{infra_findings.get('bridges_potentially_affected', 0)} bridges intersect detected flood/change evidence."
        )
    else:
        exec_summary_parts.append(
            "Infrastructure impact assessment was omitted due to unavailability of verified pre-event OSM data."
        )
    if downstream_findings and downstream_findings.get("status") == "COMPLETED":
        exec_summary_parts.append(
            f"Downstream terrain flow path traced {downstream_findings.get('path_length_km', 0.0):.1f} km "
            f"with a {downstream_findings.get('total_elevation_drop_m', 0.0):.0f} m vertical descent to the catchment terminus."
        )

    key_findings = []
    if flood_findings:
        key_findings.append(
            f"Flood Inundation: {flood_findings.get('total_flood_water_km2', 0.0):.2f} km² newly inundated; "
            f"{flood_findings.get('permanent_water_km2', 0.0):.2f} km² standing water."
        )
        if flood_findings.get("surface_change_debris_km2", 0.0) > 0:
            key_findings.append(
                f"Non-water Surface Change: {flood_findings.get('surface_change_debris_km2', 0.0):.2f} km² "
                "candidate debris, landslide, or vegetation disturbance."
            )
    if conn_findings.get("status") == "COMPLETED":
        key_findings.append(f"Connectivity Disruption: {conn_findings.get('settlements_cut_off', 0)} settlements cut off from services.")
    if downstream_findings and downstream_findings.get("status") == "COMPLETED":
        key_findings.append(
            f"Downstream Corridor: {downstream_findings.get('path_length_km', 0.0):.1f} km flow path identified from upstream point."
        )

    prov_highlights = {
        "analysis_id": result.analysis_id,
        "data_mode": result.data_mode.value,
        "satellite_platform": "Sentinel-1 IW GRD",
        "dem_source": "Copernicus DEM GLO-30 (30m)",
        "ai_segmenter": result.data_availability.ai_model_name,
        "ai_role": result.data_availability.ai_model_role,
        "osm_status": result.data_availability.osm_preevent.value,
        "data_rules_guard": "Enforced (prohibited sources and post-event edits blocked)",
    }

    return SituationReport(
        analysis_id=result.analysis_id,
        event_date=result.event_date,
        data_mode=result.data_mode,
        generated_at=datetime.now(timezone.utc),
        executive_summary=" ".join(exec_summary_parts),
        key_findings=key_findings,
        flood_findings=flood_findings,
        infrastructure_findings=infra_findings,
        connectivity_findings=conn_findings,
        downstream_findings=downstream_findings,
        data_provenance_highlights=prov_highlights,
        uncertainty_and_limitations=result.uncertainty_and_limitations,
    )
