"""Unit tests for Evidence Fusion (Stage 3 Path A + Path B fusion)."""
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio.transform

from floodmap.ai.interface import AISegmentationResult
from floodmap.core.contracts import EvidenceSource, ExclusionFlag, OutputClass
from floodmap.core.georaster import GeoRaster
from floodmap.detection.fusion import fuse_change_and_ai
from floodmap.detection.interface import ChangeResult


@pytest.fixture
def fusion_test_rasters():
    # Construct a synthetic 3x3 grid testing every permutation:
    # Row 0: [Change 0, AI 0], [Change 1, AI 0], [Change 0, AI 1]
    # Row 1: [Change 1, AI 1], [Change 2, AI 1], [Change 2, AI 0]
    # Row 2: [Change 3, AI 1], [Change 3, AI 0], [Change 255, AI 1]
    change_classes = np.array([
        [0, 1, 0],
        [1, 2, 2],
        [3, 3, 255]
    ], dtype=np.uint8)

    ai_water = np.array([
        [False, False, True],
        [True,  True,  False],
        [True,  False, True]
    ], dtype=bool)

    prob = np.array([
        [0.1, 0.2, 0.8],
        [0.9, 0.7, 0.3],
        [0.85, 0.4, 0.95]
    ], dtype=np.float32)

    transform = (10.0, 0.0, 0.0, 0.0, -10.0, 0.0)  # 10m res, 100 m² per pixel
    crs = "EPSG:32645"

    c_geo = GeoRaster(
        data=change_classes[np.newaxis, :, :],
        transform=transform,
        crs=crs,
        band_names=["classes"],
        nodata=255,
    )
    prob_geo = GeoRaster(
        data=prob[np.newaxis, :, :],
        transform=transform,
        crs=crs,
        band_names=["water_prob"],
        nodata=np.nan,
    )

    change_res = MagicMock(spec=ChangeResult)
    change_res.classes = c_geo
    change_res.exclusion = np.zeros((3, 3), dtype=np.uint8)
    change_res.warnings = ["Change warning 1"]
    change_res.method = "rca"

    ai_res = AISegmentationResult(
        probability=prob_geo,
        water_mask=ai_water,
        model_name="s1_mini_unet_v1",
        threshold=0.50,
        metadata={"test": True},
        stats={"test_stat": 123},
    )

    return change_res, ai_res


def test_fusion_truth_table(fusion_test_rasters):
    change_res, ai_res = fusion_test_rasters
    fused = fuse_change_and_ai(change_res, ai_res)

    f_classes = fused.classes.data[0]
    evidence = fused.evidence_source.data[0]

    # Row 0:
    # (0, 0): Change=0, AI=0 -> class=0, evidence=NONE (0)
    assert f_classes[0, 0] == int(OutputClass.BACKGROUND)
    assert evidence[0, 0] == int(EvidenceSource.NONE)

    # (0, 1): Change=1, AI=0 -> class=1, evidence=CHANGE_DETECTION (1)
    assert f_classes[0, 1] == int(OutputClass.FLOOD_WATER)
    assert evidence[0, 1] == int(EvidenceSource.CHANGE_DETECTION)

    # (0, 2): Change=0, AI=1 -> class=1, evidence=AI_SEGMENTATION (2)
    assert f_classes[0, 2] == int(OutputClass.FLOOD_WATER)
    assert evidence[0, 2] == int(EvidenceSource.AI_SEGMENTATION)

    # Row 1:
    # (1, 0): Change=1, AI=1 -> class=1, evidence=BOTH (3)
    assert f_classes[1, 0] == int(OutputClass.FLOOD_WATER)
    assert evidence[1, 0] == int(EvidenceSource.BOTH)

    # (1, 1): Change=2 (Permanent water), AI=1 -> class=2, evidence=CHANGE_DETECTION (1)
    assert f_classes[1, 1] == int(OutputClass.PERMANENT_WATER)
    assert evidence[1, 1] == int(EvidenceSource.CHANGE_DETECTION)

    # (1, 2): Change=2 (Permanent water), AI=0 -> class=2, evidence=CHANGE_DETECTION (1)
    assert f_classes[1, 2] == int(OutputClass.PERMANENT_WATER)
    assert evidence[1, 2] == int(EvidenceSource.CHANGE_DETECTION)

    # Row 2:
    # (2, 0): Change=3 (Debris), AI=1 -> class=3, evidence=CHANGE_DETECTION (1) (AI does NOT override debris)
    assert f_classes[2, 0] == int(OutputClass.DEBRIS_OR_CHANGE)
    assert evidence[2, 0] == int(EvidenceSource.CHANGE_DETECTION)

    # (2, 1): Change=3 (Debris), AI=0 -> class=3, evidence=CHANGE_DETECTION (1)
    assert f_classes[2, 1] == int(OutputClass.DEBRIS_OR_CHANGE)
    assert evidence[2, 1] == int(EvidenceSource.CHANGE_DETECTION)

    # (2, 2): Change=255 (Nodata), AI=1 -> class=255, evidence=NONE (0)
    assert f_classes[2, 2] == int(OutputClass.NODATA)
    assert evidence[2, 2] == int(EvidenceSource.NONE)


def test_fusion_pixel_counts_and_stats(fusion_test_rasters):
    change_res, ai_res = fusion_test_rasters
    fused = fuse_change_and_ai(change_res, ai_res)

    stats = fused.stats
    assert stats["pixels"]["confirmed_by_both"] == 1
    assert stats["pixels"]["change_detection_only"] == 1
    assert stats["pixels"]["ai_segmentation_only"] == 1
    assert stats["pixels"]["total_flood"] == 3

    # Resolution 10m x 10m = 100 m² = 0.0001 km² per pixel
    assert stats["flood_water_km2"]["confirmed_by_both"] == pytest.approx(0.0001)
    assert stats["flood_water_km2"]["total"] == pytest.approx(0.0003)

    assert "Change warning 1" in fused.warnings
