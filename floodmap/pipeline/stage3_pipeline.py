"""Stage-3 pipeline: Sentinel-1 -> Preprocessing -> Terrain -> (Change Detection + AI Segmentation) -> Fusion.

    AnalysisRequest (AOI, event date, before/after scene paths, optional DEM)
      1. load_inputs        scenes + sidecars, DEM; data-rules guard on every input
      2. pair_check         same relative orbit / direction, dates around the event
      3. analysis_grid      AOI -> UTM grid + AOI mask
      4. preprocess_s1      align, speckle filter, dB, shared valid mask
      5. terrain            slope, shadow/layover, HAND (skipped with a warning if no DEM)
      6. change_detection   Path A: selected ChangeDetector (config change_detection.method)
      7. ai_segmentation    Path B: selected AISegmentor (config ai.model, e.g. s1_mini_unet_v1)
      8. fusion             fused classes + per-pixel evidence source flags
      9. write_outputs      GeoTIFFs, summary.json, quicklooks, provenance.json
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..ai.interface import AISegmentationResult, get_ai_segmentor
from ..core.config import Settings
from ..core.contracts import (
    AnalysisRequest,
    ChangeContext,
    DataMode,
    EvidenceSource,
    ExclusionFlag,
    OutputClass,
    SourceType,
    Stage3Summary,
)
from ..core.data_rules import DataRulesGuard
from ..core.errors import InputDataError, ScenePairError
from ..core.georaster import GeoRaster
from ..core.provenance import ProvenanceLog
from ..data_sources.base import LoadedScene
from ..data_sources.dem_local import LocalDEMProvider
from ..data_sources.satellite_local import LocalSceneProvider
from ..detection.fusion import FusedDetectionResult, fuse_change_and_ai
from ..detection.interface import CHANGE_CLASS_LABELS, ChangeResult, get_change_detector
from ..io.geotiff import write_geotiff
from ..satellite.grid import AnalysisGrid, make_analysis_grid
from ..satellite.pairing import check_pair
from ..satellite.preprocess import PreprocessedPair, preprocess_s1_pair
from ..satellite.terrain import TerrainProducts, compute_terrain, look_geometry, terrain_flag_raster

CLASS_KEYS = {
    int(OutputClass.FLOOD_WATER): "new_open_water",
    int(OutputClass.PERMANENT_WATER): "water_both_dates",
    int(OutputClass.DEBRIS_OR_CHANGE): "non_water_backscatter_change",
    int(OutputClass.BACKGROUND): "no_detected_change",
}

EVIDENCE_LABELS = {
    "no_flood_detected": "No flood detected",
    "change_detection_only": "Change detection only (temporal backscatter drop)",
    "ai_segmentation_only": "AI segmentation only (spatial morphology candidate)",
    "confirmed_by_both": "Confirmed by both (highest confidence)",
}

STAGE3_INTERPRETATION_NOTES = [
    "Stage-3 multi-evidence pipeline: Traditional Change Detection (Path A) fused with AI Flood Segmentation (Path B).",
    "'confirmed_by_both' (EvidenceSource.BOTH = 3) indicates flood water independently flagged by both SAR backscatter drop and AI spatial segmentation, representing maximum confidence.",
    "'change_detection_only' (EvidenceSource.CHANGE_DETECTION = 1) detects flood water with backscatter drop below threshold, but AI model probability was below decision threshold.",
    "'ai_segmentation_only' (EvidenceSource.AI_SEGMENTATION = 2) detects flood water classified by AI spatial morphology, but temporal backscatter drop did not cross change-detection threshold.",
    "'water_both_dates' (OutputClass.PERMANENT_WATER = 2) represents multi-temporal standing water and is preserved from change detection.",
    "'non_water_backscatter_change' (OutputClass.DEBRIS_OR_CHANGE = 3) is candidate surface damage/debris from change detection; the AI model does not emit debris classes.",
    "Thresholds and AI models use explicit parameters from config/default.yaml.",
]


@dataclass
class Stage3PipelineResult:
    run_dir: Path
    summary: Stage3Summary
    outputs: dict[str, Path]
    provenance_path: Path
    fused: FusedDetectionResult
    change: ChangeResult
    ai: AISegmentationResult
    pair: PreprocessedPair
    terrain: Optional[TerrainProducts]
    grid: AnalysisGrid
    warnings: list[str] = field(default_factory=list)


def _scene_info(s: LoadedScene) -> dict[str, Any]:
    m = s.ref.metadata
    return {
        "path": str(s.ref.path),
        "platform": m.platform,
        "acquisition_time": m.acquisition_time.isoformat(),
        "relative_orbit": m.relative_orbit,
        "orbit_direction": m.orbit_direction,
        "units": m.units,
        "processing": m.processing,
        "product_ids": m.product_ids,
        "data_mode": m.data_mode.value,
    }


def _km2(mask: np.ndarray, px_area: float) -> float:
    return round(float(mask.sum()) * px_area / 1e6, 6)


def run_stage3_pipeline(
    request: AnalysisRequest,
    settings: Settings,
    run_dir: Path | str,
    analysis_id: Optional[str] = None,
    write_quicklooks: bool = True,
) -> Stage3PipelineResult:
    run_dir = Path(run_dir)
    analysis_id = analysis_id or uuid.uuid4().hex[:12]
    guard = DataRulesGuard(settings.data_rules)
    prov = ProvenanceLog(analysis_id)
    prov.note("Stage-3 AI flood segmentation + evidence fusion pipeline")
    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)
        prov.warn(msg)

    if request.before_scene is None or request.after_scene is None:
        raise InputDataError("Stage-3 needs explicit before_scene and after_scene files")

    try:
        # 1. inputs
        with prov.step("load_inputs") as st:
            scenes = LocalSceneProvider(guard)
            before = scenes.load_scene(request.before_scene)
            after = scenes.load_scene(request.after_scene)
            for role, sc in (("before", before), ("after", after)):
                prov.add_input(
                    sc.ref.path,
                    sc.ref.metadata.source_type.value,
                    role,
                    json.loads(sc.ref.metadata.model_dump_json()),
                )
                for w in sc.warnings:
                    warn(f"{role} scene: {w}")
            dem = None
            if request.dem is not None and settings.terrain.enabled:
                dem = LocalDEMProvider(guard).load_dem(request.dem)
                prov.add_input(request.dem, dem.metadata["source_type"], "dem", dem.metadata.get("sidecar", {}))
            elif request.dem is None:
                warn("No DEM supplied: terrain masking (shadow/layover/slope/HAND) will NOT be applied")

            s2_scene = None
            if request.s2_after_scene is not None:
                try:
                    s2_scene = scenes.load_scene(request.s2_after_scene)
                    prov.add_input(
                        s2_scene.ref.path,
                        s2_scene.ref.metadata.source_type.value,
                        "s2_after",
                        json.loads(s2_scene.ref.metadata.model_dump_json()),
                    )
                    for w in s2_scene.warnings:
                        warn(f"s2_after scene: {w}")
                except Exception as e:
                    warn(f"s2_after scene load failed: {e}")

            st["outputs"].append({
                "before": str(before.ref.path),
                "after": str(after.ref.path),
                "dem": str(request.dem) if dem is not None else None,
                "s2_after": str(request.s2_after_scene) if s2_scene is not None else None,
            })

        synthetic = (
            before.ref.metadata.data_mode is DataMode.SYNTHETIC_TEST
            or after.ref.metadata.data_mode is DataMode.SYNTHETIC_TEST
            or (dem is not None and dem.metadata["source_type"] == SourceType.SYNTHETIC_TEST.value)
        )
        data_mode = DataMode.SYNTHETIC_TEST if synthetic else DataMode.REAL
        prov.data_mode = data_mode

        # 2. pair check
        with prov.step("pair_check", **settings.satellite.s1.model_dump()) as st:
            pc = check_pair(before.ref.metadata, after.ref.metadata, request.event_date, settings.satellite.s1)
            st["outputs"].append(pc.model_dump())
            for w in pc.warnings:
                warn(f"pair: {w}")
            if not pc.valid:
                raise ScenePairError("Invalid before/after pair: " + "; ".join(pc.errors))

        # 3. grid
        with prov.step(
            "analysis_grid",
            resolution_m=settings.grid.resolution_m,
            margin_m=settings.grid.margin_m,
            crs_rule=settings.crs.analysis,
        ) as st:
            grid = make_analysis_grid(
                request.aoi,
                settings.grid.resolution_m,
                settings.grid.margin_m,
                settings.crs.analysis,
            )
            st["outputs"].append({
                "crs": grid.spec.crs,
                "shape": list(grid.spec.shape),
                "transform": list(grid.spec.transform),
                "aoi_area_km2": grid.aoi_area_m2 / 1e6,
            })

        # 4. preprocess
        with prov.step("preprocess_s1", **settings.preprocessing.s1.model_dump()) as st:
            pair = preprocess_s1_pair(before, after, grid, settings.preprocessing.s1)
            st["outputs"].append(pair.stats)
            for w in pair.warnings:
                warn(f"preprocess: {w}")

        # 5. terrain
        terrain = None
        if dem is not None:
            m = after.ref.metadata
            geom = look_geometry(m.orbit_direction, m.extra.get("incidence_angle_deg"), settings.terrain)
            with prov.step("terrain", **settings.terrain.model_dump()) as st:
                terrain = compute_terrain(dem, grid, settings.terrain, geom)
                st["outputs"].append(terrain.stats)
                for w in terrain.warnings:
                    warn(f"terrain: {w}")

        # 6. Path A: change detection
        detector = get_change_detector(settings)
        with prov.step(
            "change_detection",
            method=detector.name,
            **getattr(settings.change_detection, detector.name).model_dump(),
        ) as st:
            change = detector.detect(pair, terrain)
            st["outputs"].append({"thresholds": change.thresholds, "stats": change.stats})
            for w in change.warnings:
                warn(f"change detection: {w}")

        # 7. Path B: AI flood segmentation
        precomputed_ai = request.options.get("ai_result") or request.options.get("ai_segmentation_result")
        if precomputed_ai is not None and isinstance(precomputed_ai, AISegmentationResult):
            ai = precomputed_ai
            with prov.step("ai_segmentation", model=ai.model_name, precomputed=True) as st:
                st["outputs"].append({
                    "model_name": ai.model_name,
                    "threshold": ai.threshold,
                    "metadata": ai.metadata,
                    "stats": ai.stats,
                })
        else:
            segmentor = get_ai_segmentor(settings)
            ai_cfg = getattr(settings.ai, segmentor.name) if hasattr(settings.ai, segmentor.name) else {}
            ai_meta = ai_cfg.model_dump() if hasattr(ai_cfg, "model_dump") else {}
            with prov.step("ai_segmentation", model=segmentor.name, **ai_meta) as st:
                s2_data = s2_scene.raster.data if s2_scene is not None else None
                ai = segmentor.segment(pair, terrain, s2_data=s2_data)
                st["outputs"].append({
                    "model_name": ai.model_name,
                    "threshold": ai.threshold,
                    "metadata": ai.metadata,
                    "stats": ai.stats,
                })

        # 8. Fusion
        with prov.step(
            "evidence_fusion",
            mode=settings.ai.fusion.mode,
            preserve_permanent_water=settings.ai.fusion.preserve_permanent_water,
            preserve_debris=settings.ai.fusion.preserve_debris,
        ) as st:
            fused = fuse_change_and_ai(
                change=change,
                ai=ai,
                preserve_permanent_water=settings.ai.fusion.preserve_permanent_water,
                preserve_debris=settings.ai.fusion.preserve_debris,
            )
            st["outputs"].append(fused.stats)
            for w in fused.warnings:
                warn(f"fusion: {w}")

        # 9. Outputs
        with prov.step("write_outputs") as st:
            outputs = _write_outputs(
                run_dir=run_dir,
                request=request,
                settings=settings,
                data_mode=data_mode,
                pair=pair,
                change=change,
                fused=fused,
                terrain=terrain,
                ai=ai,
                write_quicklooks=write_quicklooks,
            )
            summary = _make_stage3_summary(
                analysis_id=analysis_id,
                request=request,
                settings=settings,
                data_mode=data_mode,
                before=before,
                after=after,
                pc=pc,
                grid=grid,
                pair=pair,
                terrain=terrain,
                change=change,
                ai=ai,
                fused=fused,
                outputs=outputs,
                run_dir=run_dir,
                warnings=warnings,
            )
            spath = run_dir / "change" / "summary.json"
            spath.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
            outputs["summary"] = spath
            st["outputs"].extend(str(p) for p in outputs.values())

    finally:
        prov_path = prov.write_json(run_dir / "provenance.json")

    return Stage3PipelineResult(
        run_dir=run_dir,
        summary=summary,
        outputs=outputs,
        provenance_path=prov_path,
        fused=fused,
        change=change,
        ai=ai,
        pair=pair,
        terrain=terrain,
        grid=grid,
        warnings=warnings,
    )


def _write_outputs(
    run_dir: Path,
    request: AnalysisRequest,
    settings: Settings,
    data_mode: DataMode,
    pair: PreprocessedPair,
    change: ChangeResult,
    fused: FusedDetectionResult,
    terrain: Optional[TerrainProducts],
    ai: AISegmentationResult,
    write_quicklooks: bool,
) -> dict[str, Path]:
    tag = {"data_mode": data_mode.value}
    out: dict[str, Path] = {}

    def w(r: GeoRaster, rel: str, key: str) -> None:
        out[key] = write_geotiff(r, run_dir / rel)

    w(pair.before_db, "preprocessed/s1_before_db.tif", "s1_before_db")
    w(pair.after_db, "preprocessed/s1_after_db.tif", "s1_after_db")
    w(change.log_ratio_db, "change/log_ratio_db.tif", "log_ratio_db")
    w(fused.classes, "change/change_classes.tif", "change_classes")
    w(fused.evidence_source, "change/evidence_source.tif", "evidence_source")
    w(fused.ai_probability, "change/ai_probability.tif", "ai_probability")
    w(
        change.classes.with_data(fused.exclusion, band_names=["exclusion_flags"], nodata=None),
        "change/exclusion_flags.tif",
        "exclusion_flags",
    )

    if change.context is not None:
        w(
            change.classes.with_data(change.context, band_names=["change_context"], nodata=255),
            "change/change_context.tif",
            "change_context",
        )

    if terrain is not None:
        cd = getattr(settings.change_detection, settings.change_detection.method)
        w(terrain.slope_deg, "terrain/slope_deg.tif", "slope_deg")
        w(terrain.hand_m, "terrain/hand_m.tif", "hand_m")
        flags = terrain_flag_raster(terrain, cd.water_max_slope_deg, cd.water_max_hand_m, cd.change_max_slope_deg)
        w(
            terrain.slope_deg.with_data(flags, band_names=["terrain_flags"], nodata=None),
            "terrain/terrain_flags.tif",
            "terrain_flags",
        )

    aoi_path = run_dir / "aoi.geojson"
    aoi_path.write_text(
        json.dumps({
            "type": "Feature",
            "properties": {"name": request.aoi.name, **tag},
            "geometry": request.aoi.geometry,
        }),
        encoding="utf-8",
    )
    out["aoi"] = aoi_path

    if write_quicklooks:
        from ..io import quicklook as ql

        wm = "SYNTHETIC_TEST" if data_mode is DataMode.SYNTHETIC_TEST else None
        q = run_dir / "quicklooks"
        out["ql_before_vv"] = ql.save_db_image(
            pair.before_db.band("VV"), q / "before_vv_db.png", "Before VV (dB)", watermark=wm
        )
        out["ql_after_vv"] = ql.save_db_image(
            pair.after_db.band("VV"), q / "after_vv_db.png", "After VV (dB)", watermark=wm
        )
        out["ql_log_ratio"] = ql.save_log_ratio(
            change.log_ratio_db.band("VV_log_ratio"),
            q / "log_ratio_vv.png",
            "VV log-ratio (after − before)",
            watermark=wm,
        )
        out["ql_classes"] = ql.save_classes(
            fused.classes.data[0], q / "change_classes.png", "Fused Detection (Change + AI)", watermark=wm
        )
        out["ql_evidence"] = ql.save_evidence_source(
            fused.evidence_source.data[0], q / "evidence_source.png", "Evidence Sources", watermark=wm
        )
        out["ql_probability"] = ql.save_probability(
            fused.ai_probability.data[0],
            q / "ai_probability.png",
            f"AI Flood Probability ({ai.model_name})",
            watermark=wm,
        )
        if change.context is not None:
            out["ql_context"] = ql.save_change_context(
                change.context, q / "change_context.png", "Non-water change by terrain context", watermark=wm
            )
    return out


def _make_stage3_summary(
    analysis_id: str,
    request: AnalysisRequest,
    settings: Settings,
    data_mode: DataMode,
    before: LoadedScene,
    after: LoadedScene,
    pc: Any,
    grid: AnalysisGrid,
    pair: PreprocessedPair,
    terrain: Optional[TerrainProducts],
    change: ChangeResult,
    ai: AISegmentationResult,
    fused: FusedDetectionResult,
    outputs: dict[str, Path],
    run_dir: Path,
    warnings: list[str],
) -> Stage3Summary:
    px = pair.before_db.pixel_area
    aoi = grid.aoi_mask
    c = fused.classes.data[0]
    ex = fused.exclusion
    ev = fused.evidence_source.data[0]

    class_areas = {name: _km2((c == code) & aoi, px) for code, name in CLASS_KEYS.items()}
    evidence_areas = {
        "confirmed_by_both": _km2((ev == int(EvidenceSource.BOTH)) & aoi, px),
        "change_detection_only": _km2((ev == int(EvidenceSource.CHANGE_DETECTION)) & aoi, px),
        "ai_segmentation_only": _km2((ev == int(EvidenceSource.AI_SEGMENTATION)) & aoi, px),
        "total_flood": _km2((c == int(OutputClass.FLOOD_WATER)) & aoi, px),
    }

    excl = {
        "no_data": _km2(((ex & ExclusionFlag.NO_DATA) > 0) & aoi, px),
        "radar_shadow": _km2(((ex & ExclusionFlag.RADAR_SHADOW) > 0) & aoi, px),
        "radar_layover": _km2(((ex & ExclusionFlag.RADAR_LAYOVER) > 0) & aoi, px),
        "rejected_water_by_terrain_gate": _km2(((ex & ExclusionFlag.TERRAIN_WATER_GATE) > 0) & aoi, px),
        "rejected_change_by_terrain_gate": _km2(((ex & ExclusionFlag.TERRAIN_CHANGE_GATE) > 0) & aoi, px),
        "removed_as_small_blobs": _km2(((ex & ExclusionFlag.SMALL_BLOB) > 0) & aoi, px),
    }
    ctx = {}
    if change.context is not None:
        ctx = {
            "valley_bottom": _km2((change.context == ChangeContext.VALLEY_BOTTOM) & aoi, px),
            "hillside": _km2((change.context == ChangeContext.HILLSIDE) & aoi, px),
            "unknown": _km2((change.context == ChangeContext.UNKNOWN) & aoi, px),
        }

    return Stage3Summary(
        analysis_id=analysis_id,
        data_mode=data_mode,
        method=f"fused_{change.method}_{ai.model_name}",
        event_date=request.event_date,
        aoi_name=request.aoi.name,
        crs=grid.spec.crs,
        resolution_m=grid.spec.resolution,
        scenes={"before": _scene_info(before), "after": _scene_info(after)},
        pair_check=pc,
        aoi_area_km2=round(grid.aoi_area_m2 / 1e6, 6),
        observable_area_km2=_km2((c != OutputClass.NODATA) & aoi, px),
        class_areas_km2=class_areas,
        class_labels={CLASS_KEYS[k]: v for k, v in CHANGE_CLASS_LABELS.items() if k in CLASS_KEYS},
        class3_context_km2=ctx,
        excluded_area_km2=excl,
        thresholds=change.thresholds,
        terrain_applied=terrain is not None,
        terrain=terrain.stats if terrain is not None else {},
        preprocessing=pair.stats,
        detection_stats=change.stats,
        outputs={k: str(Path(v).relative_to(run_dir)) for k, v in outputs.items()},
        warnings=warnings,
        interpretation_notes=STAGE3_INTERPRETATION_NOTES,
        ai_model=ai.model_name,
        ai_threshold=ai.threshold,
        ai_stats=ai.stats,
        ai_metadata=ai.metadata,
        fusion_mode=settings.ai.fusion.mode,
        evidence_areas_km2=evidence_areas,
        evidence_labels=EVIDENCE_LABELS,
    )
