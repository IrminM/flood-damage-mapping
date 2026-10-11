"""Fusion of Path A (Change Detection) and Path B (AI Flood Segmentation).

Combines independent physical evidence sources with per-pixel provenance tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ai.interface import AISegmentationResult
from ..core.contracts import EvidenceSource, OutputClass
from ..core.georaster import GeoRaster
from .interface import ChangeResult


@dataclass
class FusedDetectionResult:
    """Consolidated outcome of fusing Change Detection and AI Segmentation."""
    classes: GeoRaster              # uint8 canonical OutputClass codes (0..3, 255)
    evidence_source: GeoRaster      # uint8 canonical EvidenceSource flags (0..3)
    ai_probability: GeoRaster       # float32 continuous probability map
    exclusion: np.ndarray           # uint8 ExclusionFlag bitmask
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def fuse_change_and_ai(
    change: ChangeResult,
    ai: AISegmentationResult,
    preserve_permanent_water: bool = True,
    preserve_debris: bool = True,
) -> FusedDetectionResult:
    """Merge ChangeResult and AISegmentationResult into a unified raster with per-pixel evidence.

    Fusion Rules:
    1. Permanent water (Class 2 from change detection) remains Class 2 (evidence: CHANGE_DETECTION).
    2. Debris / Surface change (Class 3 from change detection) remains Class 3 (evidence: CHANGE_DETECTION).
       AI segmentation does NOT emit Class 3.
    3. Flood water (Class 1):
       - If Change=1 and AI=1: Class 1, evidence = BOTH (3)  [Highest confidence].
       - If Change=1 and AI=0: Class 1, evidence = CHANGE_DETECTION (1).
       - If Change=0 and AI=1: Class 1, evidence = AI_SEGMENTATION (2).
    4. Observable background (Class 0): Neither detected water or change; evidence = NONE (0).
    5. Nodata (Class 255): outside AOI or radar shadow/layover.
    """
    c_data = change.classes.data[0]   # uint8 (H, W)
    H, W = c_data.shape
    prob_raster = ai.probability

    if ai.water_mask.shape != (H, W):
        from ..satellite.grid import GridSpec
        from ..satellite.reproject import reproject_to_grid
        target_spec = GridSpec(
            crs=change.classes.crs,
            transform=change.classes.transform,
            width=change.classes.width,
            height=change.classes.height,
        )
        prob_aligned = reproject_to_grid(prob_raster, target_spec, resampling="bilinear")
        prob_raster = prob_aligned
        ai_water = (prob_raster.data[0] >= ai.threshold) & np.isfinite(prob_raster.data[0])
    else:
        ai_water = ai.water_mask


    fused_classes = np.zeros((H, W), dtype=np.uint8)
    evidence = np.zeros((H, W), dtype=np.uint8)

    # 1. Start with non-observable / nodata mask
    nodata_mask = (c_data == int(OutputClass.NODATA))
    fused_classes[nodata_mask] = int(OutputClass.NODATA)
    evidence[nodata_mask] = int(EvidenceSource.NONE)

    observable = ~nodata_mask

    # 2. Permanent water (water in both dates)
    perm_mask = observable & (c_data == int(OutputClass.PERMANENT_WATER))
    if preserve_permanent_water:
        fused_classes[perm_mask] = int(OutputClass.PERMANENT_WATER)
        evidence[perm_mask] = int(EvidenceSource.CHANGE_DETECTION)

    # 3. Non-water surface change (Class 3)
    # The AI does NOT emit debris. Preserve Class 3 from change detection.
    change3_mask = observable & ~perm_mask & (c_data == int(OutputClass.DEBRIS_OR_CHANGE))
    if preserve_debris:
        fused_classes[change3_mask] = int(OutputClass.DEBRIS_OR_CHANGE)
        evidence[change3_mask] = int(EvidenceSource.CHANGE_DETECTION)

    # 4. Flood water detection (candidates outside permanent water and debris)
    flood_candidates = observable & ~perm_mask & (~change3_mask if preserve_debris else np.ones_like(observable))
    change_flood = flood_candidates & (c_data == int(OutputClass.FLOOD_WATER))
    ai_flood = flood_candidates & ai_water

    both_flood = change_flood & ai_flood
    change_only_flood = change_flood & ~ai_flood
    ai_only_flood = ~change_flood & ai_flood

    any_flood = change_flood | ai_flood
    fused_classes[any_flood] = int(OutputClass.FLOOD_WATER)

    evidence[both_flood] = int(EvidenceSource.BOTH)
    evidence[change_only_flood] = int(EvidenceSource.CHANGE_DETECTION)
    evidence[ai_only_flood] = int(EvidenceSource.AI_SEGMENTATION)

    # Remaining observable pixels are background (0)
    # fused_classes and evidence remain 0

    # Calculate statistics
    px_res = abs(change.classes.transform[0])
    px_area = px_res * px_res

    def _km2(mask: np.ndarray) -> float:
        return round(float(mask.sum()) * px_area / 1e6, 4)

    stats = {
        "fusion_mode": "union_with_provenance",
        "flood_water_km2": {
            "total": _km2(any_flood),
            "confirmed_by_both": _km2(both_flood),
            "change_detection_only": _km2(change_only_flood),
            "ai_segmentation_only": _km2(ai_only_flood),
        },
        "pixels": {
            "confirmed_by_both": int(both_flood.sum()),
            "change_detection_only": int(change_only_flood.sum()),
            "ai_segmentation_only": int(ai_only_flood.sum()),
            "total_flood": int(any_flood.sum()),
        },
        "class_areas_km2": {
            "new_open_water": _km2(any_flood),
            "water_both_dates": _km2(perm_mask),
            "non_water_backscatter_change": _km2(change3_mask),
            "no_detected_change": _km2(observable & ~any_flood & ~perm_mask & ~change3_mask),
        },
        "ai_metadata": ai.metadata,
        "ai_stats": ai.stats,
    }

    fused_geo = change.classes.with_data(
        fused_classes[np.newaxis, :, :],
        band_names=["classes"],
        nodata=int(OutputClass.NODATA),
    )
    evidence_geo = change.classes.with_data(
        evidence[np.newaxis, :, :],
        band_names=["evidence_source"],
        nodata=int(EvidenceSource.NONE),
    )

    return FusedDetectionResult(
        classes=fused_geo,
        evidence_source=evidence_geo,
        ai_probability=prob_raster,
        exclusion=change.exclusion,
        stats=stats,
        warnings=list(change.warnings),
    )
