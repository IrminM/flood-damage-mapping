"""Stage-2 pipeline: Sentinel-1 before/after -> preprocessing -> terrain -> change raster.

    AnalysisRequest (AOI, event date, before/after scene paths, optional DEM)
      1. load_inputs      scenes + sidecars, DEM; data-rules guard on every input
      2. pair_check       same relative orbit / direction, dates around the event
      3. analysis_grid    AOI -> UTM grid + AOI mask
      4. preprocess_s1    align, speckle filter, dB, shared valid mask
      5. terrain          slope, shadow/layover, HAND   (skipped with a warning if no DEM)
      6. change_detection selected ChangeDetector (config change_detection.method)
      7. write_outputs    GeoTIFFs, summary.json, quicklooks, provenance.json

Every step is recorded in provenance with its parameters. This module is reused by
the full orchestrator later (stage 5); it contains no AI or GIS logic.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core.config import Settings
from ..core.contracts import (AnalysisRequest, ChangeContext, ChangeSummary, DataMode, ExclusionFlag,
                              OutputClass, SourceType)
from ..core.data_rules import DataRulesGuard
from ..core.errors import InputDataError, ScenePairError
from ..core.georaster import GeoRaster
from ..core.provenance import ProvenanceLog
from ..data_sources.base import LoadedScene
from ..data_sources.dem_local import LocalDEMProvider
from ..data_sources.satellite_local import LocalSceneProvider
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

INTERPRETATION_NOTES = [
    "Baseline traditional change detection (no AI). AI segmentation is a separate path (stage 3).",
    "'new_open_water' detects open, calm water only; flooded vegetation and flooded built-up areas "
    "(which brighten in SAR) are not captured.",
    "'non_water_backscatter_change' is NOT a debris map: it includes debris/mud, damage and landslides, "
    "but also vegetation/moisture change and noise. It is evidence for later analysis, not a verdict.",
    "Radar shadow/layover are local approximations from the DEM with nominal geometry (no ray tracing).",
    "class3_context_km2 splits the non-water change into valley-bottom (low HAND, gentle slope: where floods "
    "and debris flows deposit) and hillside. Valley-bottom change is candidate event-related change; it "
    "remains unvalidated until checked against a pre-event null pair or independent reference.",
    "Thresholds are untuned starting values from config/default.yaml.",
]


@dataclass
class ChangePipelineResult:
    run_dir: Path
    summary: ChangeSummary
    outputs: dict[str, Path]
    provenance_path: Path
    change: ChangeResult
    pair: PreprocessedPair
    terrain: Optional[TerrainProducts]
    grid: AnalysisGrid
    warnings: list[str] = field(default_factory=list)


def _scene_info(s: LoadedScene) -> dict[str, Any]:
    m = s.ref.metadata
    return {"path": str(s.ref.path), "platform": m.platform, "acquisition_time": m.acquisition_time.isoformat(),
            "relative_orbit": m.relative_orbit, "orbit_direction": m.orbit_direction, "units": m.units,
            "processing": m.processing, "product_ids": m.product_ids, "data_mode": m.data_mode.value}


def _km2(mask: np.ndarray, px_area: float) -> float:
    return round(float(mask.sum()) * px_area / 1e6, 6)


def run_change_pipeline(
    request: AnalysisRequest,
    settings: Settings,
    run_dir: Path | str,
    analysis_id: Optional[str] = None,
    write_quicklooks: bool = True,
) -> ChangePipelineResult:
    run_dir = Path(run_dir)
    analysis_id = analysis_id or uuid.uuid4().hex[:12]
    guard = DataRulesGuard(settings.data_rules)
    prov = ProvenanceLog(analysis_id)
    prov.note("Stage-2 change-detection pipeline (traditional remote sensing, no AI)")
    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)
        prov.warn(msg)

    if request.before_scene is None or request.after_scene is None:
        raise InputDataError("Stage 2 needs explicit before_scene and after_scene files "
                             "(automatic scene search/download is a later stage)")

    try:
        # 1. inputs ---------------------------------------------------------
        with prov.step("load_inputs") as st:
            scenes = LocalSceneProvider(guard)
            before = scenes.load_scene(request.before_scene)
            after = scenes.load_scene(request.after_scene)
            for role, sc in (("before", before), ("after", after)):
                prov.add_input(sc.ref.path, sc.ref.metadata.source_type.value, role,
                               json.loads(sc.ref.metadata.model_dump_json()))
                for w in sc.warnings:
                    warn(f"{role} scene: {w}")
            dem = None
            if request.dem is not None and settings.terrain.enabled:
                dem = LocalDEMProvider(guard).load_dem(request.dem)
                prov.add_input(request.dem, dem.metadata["source_type"], "dem", dem.metadata.get("sidecar", {}))
            elif request.dem is None:
                warn("No DEM supplied: terrain masking (shadow/layover/slope/HAND) will NOT be applied")
            st["outputs"].append({"before": str(before.ref.path), "after": str(after.ref.path),
                                  "dem": str(request.dem) if dem is not None else None})
        synthetic = (before.ref.metadata.data_mode is DataMode.SYNTHETIC_TEST
                     or after.ref.metadata.data_mode is DataMode.SYNTHETIC_TEST
                     or (dem is not None and dem.metadata["source_type"] == SourceType.SYNTHETIC_TEST.value))
        data_mode = DataMode.SYNTHETIC_TEST if synthetic else DataMode.REAL
        prov.data_mode = data_mode

        # 2. pair -----------------------------------------------------------
        with prov.step("pair_check", **settings.satellite.s1.model_dump()) as st:
            pc = check_pair(before.ref.metadata, after.ref.metadata, request.event_date, settings.satellite.s1)
            st["outputs"].append(pc.model_dump())
            for w in pc.warnings:
                warn(f"pair: {w}")
            if not pc.valid:
                raise ScenePairError("Invalid before/after pair: " + "; ".join(pc.errors))

        # 3. grid -----------------------------------------------------------
        with prov.step("analysis_grid", resolution_m=settings.grid.resolution_m,
                       margin_m=settings.grid.margin_m, crs_rule=settings.crs.analysis) as st:
            grid = make_analysis_grid(request.aoi, settings.grid.resolution_m, settings.grid.margin_m,
                                      settings.crs.analysis)
            st["outputs"].append({"crs": grid.spec.crs, "shape": list(grid.spec.shape),
                                  "transform": list(grid.spec.transform), "aoi_area_km2": grid.aoi_area_m2 / 1e6})

        # 4. preprocess -----------------------------------------------------
        with prov.step("preprocess_s1", **settings.preprocessing.s1.model_dump()) as st:
            pair = preprocess_s1_pair(before, after, grid, settings.preprocessing.s1)
            st["outputs"].append(pair.stats)
            for w in pair.warnings:
                warn(f"preprocess: {w}")

        # 5. terrain --------------------------------------------------------
        terrain = None
        if dem is not None:
            m = after.ref.metadata
            geom = look_geometry(m.orbit_direction, m.extra.get("incidence_angle_deg"), settings.terrain)
            with prov.step("terrain", **settings.terrain.model_dump()) as st:
                terrain = compute_terrain(dem, grid, settings.terrain, geom)
                st["outputs"].append(terrain.stats)
                for w in terrain.warnings:
                    warn(f"terrain: {w}")

        # 6. detect ---------------------------------------------------------
        detector = get_change_detector(settings)
        with prov.step("change_detection", method=detector.name,
                       **getattr(settings.change_detection, detector.name).model_dump()) as st:
            change = detector.detect(pair, terrain)
            st["outputs"].append({"thresholds": change.thresholds, "stats": change.stats})
            for w in change.warnings:
                warn(f"change detection: {w}")

        # 7. outputs --------------------------------------------------------
        with prov.step("write_outputs") as st:
            outputs = _write_outputs(run_dir, request, pair, terrain, change, settings, data_mode, write_quicklooks)
            st["outputs"].extend(str(p) for p in outputs.values())
            summary = _summary(analysis_id, request, settings, data_mode, before, after, pc, grid, pair,
                               terrain, change, outputs, run_dir, warnings)
            spath = run_dir / "change" / "summary.json"
            spath.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
            outputs["summary"] = spath
    finally:
        prov_path = prov.write_json(run_dir / "provenance.json")

    return ChangePipelineResult(run_dir=run_dir, summary=summary, outputs=outputs, provenance_path=prov_path,
                                change=change, pair=pair, terrain=terrain, grid=grid, warnings=warnings)


def _write_outputs(run_dir: Path, request: AnalysisRequest, pair: PreprocessedPair,
                   terrain: Optional[TerrainProducts], change: ChangeResult, settings: Settings,
                   data_mode: DataMode, write_quicklooks: bool) -> dict[str, Path]:
    tag = {"data_mode": data_mode.value}
    out: dict[str, Path] = {}

    def w(r: GeoRaster, rel: str, key: str) -> None:
        out[key] = write_geotiff(r, run_dir / rel)

    w(pair.before_db, "preprocessed/s1_before_db.tif", "s1_before_db")
    w(pair.after_db, "preprocessed/s1_after_db.tif", "s1_after_db")
    w(change.log_ratio_db, "change/log_ratio_db.tif", "log_ratio_db")
    w(change.classes, "change/change_classes.tif", "change_classes")
    w(change.classes.with_data(change.exclusion, band_names=["exclusion_flags"], nodata=None),
      "change/exclusion_flags.tif", "exclusion_flags")
    if change.context is not None:
        w(change.classes.with_data(change.context, band_names=["change_context"], nodata=255),
          "change/change_context.tif", "change_context")
    if terrain is not None:
        cd = getattr(settings.change_detection, settings.change_detection.method)
        w(terrain.slope_deg, "terrain/slope_deg.tif", "slope_deg")
        w(terrain.hand_m, "terrain/hand_m.tif", "hand_m")
        flags = terrain_flag_raster(terrain, cd.water_max_slope_deg, cd.water_max_hand_m, cd.change_max_slope_deg)
        w(terrain.slope_deg.with_data(flags, band_names=["terrain_flags"], nodata=None),
          "terrain/terrain_flags.tif", "terrain_flags")
    aoi_path = run_dir / "aoi.geojson"
    aoi_path.write_text(json.dumps({"type": "Feature", "properties": {"name": request.aoi.name, **tag},
                                    "geometry": request.aoi.geometry}), encoding="utf-8")
    out["aoi"] = aoi_path

    if write_quicklooks:
        from ..io import quicklook as ql
        wm = "SYNTHETIC_TEST" if data_mode is DataMode.SYNTHETIC_TEST else None
        q = run_dir / "quicklooks"
        out["ql_before_vv"] = ql.save_db_image(pair.before_db.band("VV"), q / "before_vv_db.png", "Before VV (dB)", watermark=wm)
        out["ql_after_vv"] = ql.save_db_image(pair.after_db.band("VV"), q / "after_vv_db.png", "After VV (dB)", watermark=wm)
        out["ql_log_ratio"] = ql.save_log_ratio(change.log_ratio_db.band("VV_log_ratio"), q / "log_ratio_vv.png",
                                                "VV log-ratio (after − before)", watermark=wm)
        out["ql_classes"] = ql.save_classes(change.classes.data[0], q / "change_classes.png",
                                            f"Change detection ({change.method})", watermark=wm)
        if change.context is not None:
            out["ql_context"] = ql.save_change_context(change.context, q / "change_context.png",
                                                       "Non-water change by terrain context", watermark=wm)
    return out


def _summary(analysis_id, request, settings, data_mode, before, after, pc, grid, pair, terrain, change,
             outputs, run_dir, warnings) -> ChangeSummary:
    px = pair.before_db.pixel_area
    aoi = grid.aoi_mask
    c = change.classes.data[0]
    ex = change.exclusion
    class_areas = {name: _km2((c == code) & aoi, px) for code, name in CLASS_KEYS.items()}
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
        ctx = {"valley_bottom": _km2((change.context == ChangeContext.VALLEY_BOTTOM) & aoi, px),
               "hillside": _km2((change.context == ChangeContext.HILLSIDE) & aoi, px),
               "unknown": _km2((change.context == ChangeContext.UNKNOWN) & aoi, px)}
    return ChangeSummary(
        analysis_id=analysis_id, data_mode=data_mode, method=change.method, event_date=request.event_date,
        aoi_name=request.aoi.name, crs=grid.spec.crs, resolution_m=grid.spec.resolution,
        scenes={"before": _scene_info(before), "after": _scene_info(after)}, pair_check=pc,
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
        interpretation_notes=INTERPRETATION_NOTES,
    )
