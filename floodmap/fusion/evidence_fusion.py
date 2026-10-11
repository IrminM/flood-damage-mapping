"""Multimodal Evidence Fusion Layer.

Integrates:
  1. Multimodal AI Flood Probability and Binary Mask (Attention_UNet_S1_S2)
  2. Baseline Sentinel-1 Change Detection (s1_logratio_v1)
  3. Physical Terrain Constraints (Copernicus DEM GLO-30 slope, HAND, radar shadow/layover)

Produces a rigorous pixel-level evidence and provenance classification distinguishing:
  - Confirmed flood / open water
  - Possible event-related surface change (valley bottom)
  - Unvalidated change (hillside)
  - Terrain-rejected candidates
  - Observable background / No data
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional, Sequence

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import shape
from shapely.validation import make_valid

from ..core.contracts import (
    ChangeContext,
    EvidenceSource,
    ExclusionFlag,
    OutputClass,
    TerrainFlag,
)
from ..core.georaster import GeoRaster
from ..io.geotiff import write_geotiff
from ..satellite.grid import GridSpec
from ..satellite.reproject import reproject_to_grid


class EvidenceCategory(IntEnum):
    """Canonical per-pixel multimodal evidence categories."""
    BACKGROUND = 0
    AI_AND_SAR_FLOOD = 1
    AI_WATER_AND_SAR_VALLEY_CHANGE = 2
    AI_ONLY_WATER = 3
    SAR_ONLY_VALLEY_CHANGE = 4
    SAR_ONLY_HILLSIDE_CHANGE = 5
    PERMANENT_WATER = 6
    TERRAIN_REJECTED = 7
    INVALID_OR_NODATA = 255


CATEGORY_DESCRIPTIONS: dict[int, str] = {
    int(EvidenceCategory.BACKGROUND): "Observable area; no detected flood water or surface change",
    int(EvidenceCategory.AI_AND_SAR_FLOOD): "Multi-sensor confirmed open water (AI water candidate + SAR new open water, terrain-validated)",
    int(EvidenceCategory.AI_WATER_AND_SAR_VALLEY_CHANGE): "Dual-evidence event change (AI water candidate + SAR valley-bottom temporal change, terrain-validated)",
    int(EvidenceCategory.AI_ONLY_WATER): "AI water candidate only (terrain-plausible, unconfirmed by temporal SAR change)",
    int(EvidenceCategory.SAR_ONLY_VALLEY_CHANGE): "SAR valley-bottom surface change only (not classified as open water by AI)",
    int(EvidenceCategory.SAR_ONLY_HILLSIDE_CHANGE): "SAR hillside surface change (unvalidated hillside change; vegetation/moisture or noise)",
    int(EvidenceCategory.PERMANENT_WATER): "Permanent standing water (river/lake present in both SAR dates)",
    int(EvidenceCategory.TERRAIN_REJECTED): "Candidate rejected by physical terrain constraints (steep slope, high HAND, or radar shadow/layover)",
    int(EvidenceCategory.INVALID_OR_NODATA): "Outside AOI or invalid/missing sensor data",
}

CATEGORY_NAMES: dict[int, str] = {
    int(EvidenceCategory.BACKGROUND): "background",
    int(EvidenceCategory.AI_AND_SAR_FLOOD): "ai_and_sar_flood",
    int(EvidenceCategory.AI_WATER_AND_SAR_VALLEY_CHANGE): "ai_water_and_sar_valley_change",
    int(EvidenceCategory.AI_ONLY_WATER): "ai_only_water",
    int(EvidenceCategory.SAR_ONLY_VALLEY_CHANGE): "sar_only_valley_change",
    int(EvidenceCategory.SAR_ONLY_HILLSIDE_CHANGE): "sar_only_hillside_change",
    int(EvidenceCategory.PERMANENT_WATER): "permanent_water",
    int(EvidenceCategory.TERRAIN_REJECTED): "terrain_rejected",
    int(EvidenceCategory.INVALID_OR_NODATA): "invalid_or_nodata",
}


class EvidenceStrength(IntEnum):
    """Categorical confidence level of detected flood or surface change."""
    NONE = 0
    UNVALIDATED = 1
    AI_ONLY = 2
    SAR_ONLY = 3
    HIGH_CONFIDENCE = 4
    BASELINE_WATER = 5
    NODATA = 255


STRENGTH_DESCRIPTIONS: dict[int, str] = {
    int(EvidenceStrength.NONE): "No detected flood or surface change",
    int(EvidenceStrength.UNVALIDATED): "Unvalidated change (hillside change or terrain-rejected candidate)",
    int(EvidenceStrength.AI_ONLY): "AI spatial water candidate only (unconfirmed by SAR temporal change)",
    int(EvidenceStrength.SAR_ONLY): "Temporal SAR surface change only (unconfirmed as water by AI)",
    int(EvidenceStrength.HIGH_CONFIDENCE): "High confidence: multi-sensor corroborated event change/flood, terrain-validated",
    int(EvidenceStrength.BASELINE_WATER): "Baseline permanent water bodies (not new flood hazard)",
    int(EvidenceStrength.NODATA): "Outside AOI or invalid/missing data",
}

STRENGTH_NAMES: dict[int, str] = {
    int(EvidenceStrength.NONE): "none",
    int(EvidenceStrength.UNVALIDATED): "unvalidated",
    int(EvidenceStrength.AI_ONLY): "ai_only",
    int(EvidenceStrength.SAR_ONLY): "sar_only",
    int(EvidenceStrength.HIGH_CONFIDENCE): "high_confidence",
    int(EvidenceStrength.BASELINE_WATER): "baseline_water",
    int(EvidenceStrength.NODATA): "nodata",
}


@dataclass
class MultimodalFusionResult:
    """Outcome of fusing AI flood segmentation, SAR change detection, and terrain gates."""
    category_raster: GeoRaster
    strength_raster: GeoRaster
    ai_probability: GeoRaster
    summary: dict[str, Any]
    provenance: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _ensure_grid_aligned(
    raster: GeoRaster,
    target_spec: GridSpec,
    resampling: str = "nearest",
    dst_nodata: Optional[float] = None,
) -> GeoRaster:
    """Ensure a GeoRaster is strictly aligned to the target grid."""
    if (
        raster.shape == target_spec.shape
        and raster.crs == target_spec.crs
        and np.allclose(raster.transform, target_spec.transform, atol=1e-5)
    ):
        return raster
    return reproject_to_grid(raster, target_spec, resampling=resampling, dst_nodata=dst_nodata)


def fuse_multimodal_evidence(
    ai_probability: GeoRaster,
    stage2_classes: GeoRaster,
    ai_mask: Optional[GeoRaster | np.ndarray] = None,
    stage2_exclusion: Optional[GeoRaster | np.ndarray] = None,
    stage2_context: Optional[GeoRaster | np.ndarray] = None,
    terrain_slope: Optional[GeoRaster | np.ndarray] = None,
    terrain_hand: Optional[GeoRaster | np.ndarray] = None,
    terrain_flags: Optional[GeoRaster | np.ndarray] = None,
    ai_threshold: float = 0.40,
    water_max_slope_deg: float = 10.0,
    water_max_hand_m: float = 30.0,
    change_max_slope_deg: float = 25.0,
    source_metadata: Optional[dict[str, Any]] = None,
) -> MultimodalFusionResult:
    """Fuse AI flood segmentation, SAR change detection, and terrain gates.

    Parameters:
        ai_probability: GeoRaster (1, H, W) float32 probabilities in [0.0, 1.0].
        stage2_classes: GeoRaster (1, H, W) uint8 canonical OutputClass values.
        ai_mask: Optional binary GeoRaster or bool ndarray (H, W).
        stage2_exclusion: Optional uint8 ExclusionFlag bitmask.
        stage2_context: Optional uint8 ChangeContext raster.
        terrain_slope: Optional float32 slope raster in degrees.
        terrain_hand: Optional float32 HAND raster in meters.
        terrain_flags: Optional uint8 TerrainFlag bitmask raster.
        ai_threshold: Decision threshold for AI water candidates (default 0.40).
        water_max_slope_deg: Maximum slope for standing water (default 10.0°).
        water_max_hand_m: Maximum HAND for flood water (default 30.0 m).
        change_max_slope_deg: Maximum slope for surface change (default 25.0°).
        source_metadata: Optional dict documenting source files and lineages.

    Returns:
        MultimodalFusionResult containing classified rasters, summary, and provenance.
    """
    warnings: list[str] = []
    target_spec = GridSpec(
        crs=ai_probability.crs,
        transform=ai_probability.transform,
        width=ai_probability.width,
        height=ai_probability.height,
    )

    H, W = target_spec.shape
    px_res = target_spec.resolution
    px_area_m2 = px_res * px_res
    total_pixels = H * W
    total_area_km2 = round(total_pixels * px_area_m2 / 1e6, 4)

    # 1. Align Stage 2 classes to reference grid
    s2_cls_aligned = _ensure_grid_aligned(stage2_classes, target_spec, resampling="nearest", dst_nodata=255)
    c_data = s2_cls_aligned.data[0]

    # 2. Extract / align AI probability and mask
    prob_data = ai_probability.data[0]
    if ai_mask is None:
        ai_water_bool = (prob_data >= ai_threshold) & np.isfinite(prob_data)
    elif isinstance(ai_mask, GeoRaster):
        mask_aligned = _ensure_grid_aligned(ai_mask, target_spec, resampling="nearest", dst_nodata=0)
        ai_water_bool = (mask_aligned.data[0] > 0) & np.isfinite(prob_data)
    else:
        ai_water_bool = (np.asarray(ai_mask) > 0) & np.isfinite(prob_data)

    # 3. Extract / align Stage 2 exclusion flags
    if stage2_exclusion is not None:
        if isinstance(stage2_exclusion, GeoRaster):
            excl_aligned = _ensure_grid_aligned(stage2_exclusion, target_spec, resampling="nearest", dst_nodata=int(ExclusionFlag.OUTSIDE_AOI))
            excl_data = excl_aligned.data[0]
        else:
            excl_data = np.asarray(stage2_exclusion, dtype=np.uint8)
    else:
        excl_data = np.zeros((H, W), dtype=np.uint8)
        excl_data[c_data == int(OutputClass.NODATA)] = int(ExclusionFlag.OUTSIDE_AOI)

    # 4. Extract / align Stage 2 context
    if stage2_context is not None:
        if isinstance(stage2_context, GeoRaster):
            ctx_aligned = _ensure_grid_aligned(stage2_context, target_spec, resampling="nearest", dst_nodata=int(ChangeContext.NODATA))
            ctx_data = ctx_aligned.data[0]
        else:
            ctx_data = np.asarray(stage2_context, dtype=np.uint8)
    else:
        ctx_data = np.full((H, W), int(ChangeContext.NOT_CHANGE), dtype=np.uint8)
        ctx_data[c_data == int(OutputClass.NODATA)] = int(ChangeContext.NODATA)

    # 5. Extract / align Terrain products
    if terrain_slope is not None and isinstance(terrain_slope, GeoRaster):
        slope_aligned = _ensure_grid_aligned(terrain_slope, target_spec, resampling="nearest", dst_nodata=np.nan)
        slope_data = slope_aligned.data[0]
    elif terrain_slope is not None:
        slope_data = np.asarray(terrain_slope, dtype=np.float32)
    else:
        slope_data = None

    if terrain_hand is not None and isinstance(terrain_hand, GeoRaster):
        hand_aligned = _ensure_grid_aligned(terrain_hand, target_spec, resampling="nearest", dst_nodata=np.nan)
        hand_data = hand_aligned.data[0]
    elif terrain_hand is not None:
        hand_data = np.asarray(terrain_hand, dtype=np.float32)
    else:
        hand_data = None

    if terrain_flags is not None and isinstance(terrain_flags, GeoRaster):
        tflags_aligned = _ensure_grid_aligned(terrain_flags, target_spec, resampling="nearest", dst_nodata=0)
        tflags_data = tflags_aligned.data[0]
    elif terrain_flags is not None:
        tflags_data = np.asarray(terrain_flags, dtype=np.uint8)
    else:
        tflags_data = np.zeros((H, W), dtype=np.uint8)

    # 6. Observability and NoData mask
    nodata_mask = (
        (c_data == int(OutputClass.NODATA))
        | ~np.isfinite(prob_data)
        | ((excl_data & int(ExclusionFlag.OUTSIDE_AOI)) > 0)
        | ((excl_data & int(ExclusionFlag.NO_DATA)) > 0)
    )
    observable = ~nodata_mask

    # 7. Physical terrain gate masks
    terrain_water_fail = np.zeros((H, W), dtype=bool)
    terrain_change_fail = np.zeros((H, W), dtype=bool)

    if slope_data is not None:
        with np.errstate(invalid="ignore"):
            terrain_water_fail |= np.isfinite(slope_data) & (slope_data > water_max_slope_deg)
            terrain_change_fail |= np.isfinite(slope_data) & (slope_data > change_max_slope_deg)
    if hand_data is not None:
        with np.errstate(invalid="ignore"):
            terrain_water_fail |= np.isfinite(hand_data) & (hand_data > water_max_hand_m)

    # Incorporate radar layover/shadow and exclusion bitmask
    radar_distortion = ((tflags_data & int(TerrainFlag.RADAR_SHADOW)) > 0) | ((tflags_data & int(TerrainFlag.RADAR_LAYOVER)) > 0)
    radar_distortion |= ((excl_data & int(ExclusionFlag.RADAR_SHADOW)) > 0) | ((excl_data & int(ExclusionFlag.RADAR_LAYOVER)) > 0)
    terrain_water_fail |= radar_distortion
    terrain_change_fail |= radar_distortion

    terrain_water_fail |= ((excl_data & int(ExclusionFlag.TERRAIN_WATER_GATE)) > 0)
    terrain_change_fail |= ((excl_data & int(ExclusionFlag.TERRAIN_CHANGE_GATE)) > 0)

    # 8. Classification canvas
    cat_arr = np.zeros((H, W), dtype=np.uint8)
    strength_arr = np.zeros((H, W), dtype=np.uint8)

    # Default observable background
    cat_arr[observable] = int(EvidenceCategory.BACKGROUND)
    strength_arr[observable] = int(EvidenceStrength.NONE)

    # Permanent water (pre-existing water bodies from SAR)
    perm_mask = observable & (c_data == int(OutputClass.PERMANENT_WATER))
    cat_arr[perm_mask] = int(EvidenceCategory.PERMANENT_WATER)
    strength_arr[perm_mask] = int(EvidenceStrength.BASELINE_WATER)

    # SAR changes outside permanent water
    sar_cand = observable & ~perm_mask
    sar_new_water = sar_cand & (c_data == int(OutputClass.FLOOD_WATER))
    sar_change = sar_cand & (c_data == int(OutputClass.DEBRIS_OR_CHANGE))
    sar_valley_change = sar_change & (ctx_data == int(ChangeContext.VALLEY_BOTTOM))
    sar_hillside_change = sar_change & (ctx_data == int(ChangeContext.HILLSIDE))

    # AI water candidates outside permanent water
    ai_cand = observable & ~perm_mask & ai_water_bool
    ai_terrain_fail = ai_cand & terrain_water_fail
    ai_terrain_valid = ai_cand & ~terrain_water_fail

    # Terrain-rejected candidates:
    # 1. AI water candidates failing water terrain gate (slope > 10°, HAND > 30m, radar distortion)
    # 2. SAR change candidates failing change terrain gate (slope > 25° or flagged in exclusion)
    # 3. SAR water candidates failing water terrain gate (flagged in exclusion)
    sar_change_cand_fail = (sar_change & terrain_change_fail) | ((excl_data & int(ExclusionFlag.TERRAIN_CHANGE_GATE)) > 0)
    sar_water_cand_fail = ((excl_data & int(ExclusionFlag.TERRAIN_WATER_GATE)) > 0)
    terrain_rejected_mask = (ai_terrain_fail & ~sar_change) | sar_change_cand_fail | sar_water_cand_fail
    cat_arr[terrain_rejected_mask] = int(EvidenceCategory.TERRAIN_REJECTED)
    strength_arr[terrain_rejected_mask] = int(EvidenceStrength.NONE)

    # Case 1: AI_AND_SAR_FLOOD (AI water + SAR new water, terrain-valid)
    both_water = ai_terrain_valid & sar_new_water
    cat_arr[both_water] = int(EvidenceCategory.AI_AND_SAR_FLOOD)
    strength_arr[both_water] = int(EvidenceStrength.HIGH_CONFIDENCE)

    # Case 2: AI_WATER_AND_SAR_VALLEY_CHANGE (AI water + SAR valley change, terrain-valid)
    ai_sar_valley = ai_terrain_valid & sar_valley_change
    cat_arr[ai_sar_valley] = int(EvidenceCategory.AI_WATER_AND_SAR_VALLEY_CHANGE)
    strength_arr[ai_sar_valley] = int(EvidenceStrength.HIGH_CONFIDENCE)

    # Case 3: AI_ONLY_WATER (AI water, terrain-valid, no SAR change or water)
    ai_only = ai_terrain_valid & ~sar_new_water & ~sar_change
    cat_arr[ai_only] = int(EvidenceCategory.AI_ONLY_WATER)
    strength_arr[ai_only] = int(EvidenceStrength.AI_ONLY)

    # Case 4: SAR_ONLY_VALLEY_CHANGE (SAR valley change, no AI water or AI rejected by terrain)
    sar_only_valley = sar_valley_change & ~terrain_change_fail & (~ai_water_bool | ai_terrain_fail)
    cat_arr[sar_only_valley] = int(EvidenceCategory.SAR_ONLY_VALLEY_CHANGE)
    strength_arr[sar_only_valley] = int(EvidenceStrength.SAR_ONLY)

    # Case 5: SAR_ONLY_HILLSIDE_CHANGE (SAR hillside change, terrain-valid, unvalidated change)
    sar_only_hill = sar_hillside_change & ~terrain_change_fail
    cat_arr[sar_only_hill] = int(EvidenceCategory.SAR_ONLY_HILLSIDE_CHANGE)
    strength_arr[sar_only_hill] = int(EvidenceStrength.UNVALIDATED)

    # Set NoData pixels
    cat_arr[nodata_mask] = int(EvidenceCategory.INVALID_OR_NODATA)
    strength_arr[nodata_mask] = int(EvidenceStrength.NODATA)

    # 9. Compute detailed statistics
    def _km2(mask: np.ndarray) -> float:
        return round(float(mask.sum()) * px_area_m2 / 1e6, 4)

    def _count(mask: np.ndarray) -> int:
        return int(mask.sum())

    def _pct(mask: np.ndarray, base: int) -> float:
        return round(100.0 * float(mask.sum()) / max(base, 1), 4)

    obs_count = int(observable.sum())

    categories_stats = {}
    for code, name in CATEGORY_NAMES.items():
        m = (cat_arr == code)
        categories_stats[name] = {
            "code": code,
            "description": CATEGORY_DESCRIPTIONS[code],
            "pixels": _count(m),
            "area_km2": _km2(m),
            "pct_observable": _pct(m, obs_count) if code != int(EvidenceCategory.INVALID_OR_NODATA) else 0.0,
            "pct_total_scene": _pct(m, total_pixels),
        }

    strength_stats = {}
    for code, name in STRENGTH_NAMES.items():
        m = (strength_arr == code)
        strength_stats[name] = {
            "code": code,
            "description": STRENGTH_DESCRIPTIONS[code],
            "pixels": _count(m),
            "area_km2": _km2(m),
            "pct_observable": _pct(m, obs_count) if code != int(EvidenceStrength.NODATA) else 0.0,
            "pct_total_scene": _pct(m, total_pixels),
        }

    # High-level scientific aggregations
    scientific_groupings = {
        "flood_open_water_km2": {
            "dual_confirmed_flood": categories_stats["ai_and_sar_flood"]["area_km2"],
            "ai_only_water_candidate": categories_stats["ai_only_water"]["area_km2"],
            "permanent_standing_water": categories_stats["permanent_water"]["area_km2"],
            "total_open_water_candidates": round(
                categories_stats["ai_and_sar_flood"]["area_km2"]
                + categories_stats["ai_only_water"]["area_km2"]
                + categories_stats["permanent_water"]["area_km2"],
                4,
            ),
        },
        "possible_event_surface_change_km2": {
            "dual_evidence_valley_change": categories_stats["ai_water_and_sar_valley_change"]["area_km2"],
            "sar_only_valley_change": categories_stats["sar_only_valley_change"]["area_km2"],
            "total_valley_corridor_change": round(
                categories_stats["ai_water_and_sar_valley_change"]["area_km2"]
                + categories_stats["sar_only_valley_change"]["area_km2"],
                4,
            ),
        },
        "unvalidated_change_km2": {
            "sar_hillside_change": categories_stats["sar_only_hillside_change"]["area_km2"],
            "terrain_rejected_candidates": categories_stats["terrain_rejected"]["area_km2"],
        },
        "observable_background_km2": categories_stats["background"]["area_km2"],
        "invalid_or_nodata_km2": categories_stats["invalid_or_nodata"]["area_km2"],
    }

    summary = {
        "task": "Multimodal Evidence Fusion",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spatial_reference": {
            "crs": target_spec.crs,
            "resolution_m": px_res,
            "shape": list(target_spec.shape),
            "pixel_area_m2": px_area_m2,
            "total_pixels": total_pixels,
            "total_scene_area_km2": total_area_km2,
            "observable_pixels": obs_count,
            "observable_area_km2": round(obs_count * px_area_m2 / 1e6, 4),
        },
        "thresholds_and_parameters": {
            "ai_probability_threshold": ai_threshold,
            "water_max_slope_deg": water_max_slope_deg,
            "water_max_hand_m": water_max_hand_m,
            "change_max_slope_deg": change_max_slope_deg,
        },
        "evidence_categories": categories_stats,
        "evidence_strengths": strength_stats,
        "scientific_groupings": scientific_groupings,
        "scientific_disclaimer": (
            "This fused layer is an evidence and provenance product combining independent optical/SAR AI predictions, "
            "SAR temporal change detection, and physical DEM terrain gates. It is NOT ground-truth flood extent. "
            "High-confidence areas represent multi-sensor corroborated inundation/change in terrain-plausible valley corridors. "
            "AI_ONLY and SAR_ONLY categories represent unconfirmed candidates. "
            "This product does NOT classify structural building damage, mud deposition, or landslide debris."
        ),
    }

    provenance = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "fusion_engine": "floodmap.fusion.evidence_fusion",
        "inputs": source_metadata or {},
        "grid": {
            "crs": target_spec.crs,
            "shape": list(target_spec.shape),
            "transform": list(target_spec.transform),
            "resolution_m": px_res,
        },
        "parameters": {
            "ai_threshold": ai_threshold,
            "water_max_slope_deg": water_max_slope_deg,
            "water_max_hand_m": water_max_hand_m,
            "change_max_slope_deg": change_max_slope_deg,
        },
    }

    cat_raster = ai_probability.with_data(
        cat_arr[np.newaxis, :, :],
        band_names=["fused_evidence_category"],
        nodata=int(EvidenceCategory.INVALID_OR_NODATA),
        metadata={"units": "evidence_category", "description": "Multimodal evidence category"},
    )

    strength_raster = ai_probability.with_data(
        strength_arr[np.newaxis, :, :],
        band_names=["evidence_strength"],
        nodata=int(EvidenceStrength.NODATA),
        metadata={"units": "evidence_strength", "description": "Evidence strength/confidence level"},
    )

    return MultimodalFusionResult(
        category_raster=cat_raster,
        strength_raster=strength_raster,
        ai_probability=ai_probability,
        summary=summary,
        provenance=provenance,
        warnings=warnings,
    )


def vectorize_fused_evidence(
    category_raster: GeoRaster,
    strength_raster: Optional[GeoRaster] = None,
    target_categories: Sequence[int] = (
        int(EvidenceCategory.AI_AND_SAR_FLOOD),
        int(EvidenceCategory.AI_WATER_AND_SAR_VALLEY_CHANGE),
        int(EvidenceCategory.AI_ONLY_WATER),
        int(EvidenceCategory.SAR_ONLY_VALLEY_CHANGE),
        int(EvidenceCategory.SAR_ONLY_HILLSIDE_CHANGE),
        int(EvidenceCategory.PERMANENT_WATER),
    ),
) -> gpd.GeoDataFrame:
    """Vectorize target evidence classes into GeoPandas GeoDataFrame.

    Parameters:
        category_raster: GeoRaster containing EvidenceCategory integer codes.
        strength_raster: Optional GeoRaster containing EvidenceStrength integer codes.
        target_categories: Categories to vectorize into polygon features.

    Returns:
        GeoDataFrame with columns: geometry, category_code, category_name, strength_code, strength_name, area_m2.
    """
    crs = category_raster.crs
    transform = category_raster.transform
    cat_data = category_raster.data[0]
    str_data = strength_raster.data[0] if strength_raster is not None else None

    target_set = set(target_categories)
    mask = np.isin(cat_data, list(target_set)) & (cat_data != int(EvidenceCategory.INVALID_OR_NODATA))

    columns = [
        "geometry",
        "category_code",
        "category_name",
        "strength_code",
        "strength_name",
        "area_m2",
    ]
    if not mask.any():
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)

    if str_data is not None:
        coded = (cat_data.astype(np.int32) * 1000 + str_data.astype(np.int32)).astype(np.int32)
    else:
        coded = cat_data.astype(np.int32)

    geoms = []
    c_codes = []
    c_names = []
    s_codes = []
    s_names = []
    areas = []

    for geom_dict, val in rasterio.features.shapes(coded, mask=mask, transform=transform):
        val_int = int(val)
        if str_data is not None:
            c_code = val_int // 1000
            s_code = val_int % 1000
        else:
            c_code = val_int
            s_code = int(EvidenceStrength.NONE)

        g = shape(geom_dict)
        if not g.is_valid:
            g = make_valid(g)
        if g.is_empty:
            continue

        geoms.append(g)
        c_codes.append(c_code)
        c_names.append(CATEGORY_NAMES.get(c_code, f"category_{c_code}"))
        s_codes.append(s_code)
        s_names.append(STRENGTH_NAMES.get(s_code, f"strength_{s_code}"))
        areas.append(round(float(g.area), 2))

    if not geoms:
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)

    return gpd.GeoDataFrame(
        {
            "category_code": c_codes,
            "category_name": c_names,
            "strength_code": s_codes,
            "strength_name": s_names,
            "area_m2": areas,
        },
        geometry=geoms,
        crs=crs,
    )


def write_fusion_outputs(
    result: MultimodalFusionResult,
    out_dir: Path | str,
    write_vector: bool = True,
) -> dict[str, Path]:
    """Persist all fusion outputs (GeoTIFFs, GeoJSON, summary, provenance) to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # 1. Category GeoTIFF
    cat_tif = out_dir / "fused_evidence.tif"
    write_geotiff(result.category_raster, cat_tif)
    with rasterio.open(cat_tif, "r+") as dst:
        dst.set_band_description(1, "fused_evidence_category")
        dst.update_tags(
            PRODUCT="Multimodal Evidence Fusion",
            CATEGORIES=json.dumps(CATEGORY_NAMES),
            THRESHOLD="0.40",
            AREA_OR_POINT="Area",
        )
    outputs["fused_evidence"] = cat_tif

    # 2. Strength GeoTIFF
    str_tif = out_dir / "evidence_strength.tif"
    write_geotiff(result.strength_raster, str_tif)
    with rasterio.open(str_tif, "r+") as dst:
        dst.set_band_description(1, "evidence_strength")
        dst.update_tags(
            PRODUCT="Multimodal Evidence Strength",
            STRENGTHS=json.dumps(STRENGTH_NAMES),
            AREA_OR_POINT="Area",
        )
    outputs["evidence_strength"] = str_tif

    # 3. Vector GeoJSON
    if write_vector:
        vec_gdf = vectorize_fused_evidence(result.category_raster, result.strength_raster)
        vec_path = out_dir / "fused_evidence_vector.geojson"
        vec_gdf.to_file(vec_path, driver="GeoJSON")
        outputs["fused_evidence_vector"] = vec_path

    # 4. Summary JSON
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    outputs["summary"] = summary_path

    # 5. Provenance JSON
    prov_path = out_dir / "provenance.json"
    prov_path.write_text(json.dumps(result.provenance, indent=2), encoding="utf-8")
    outputs["provenance"] = prov_path

    return outputs
