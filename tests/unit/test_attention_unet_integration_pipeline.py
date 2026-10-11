"""Unit tests verifying Attention_UNet_S1_S2 pipeline integration interfaces & contracts.

Guarantees:
  1. Sentinel-2 preparation adapter strictly refuses unresolved provenance.
  2. Band omission, fabrication, or zero-padding is strictly rejected.
  3. AI segmentation output carries explicit required provenance:
       source = attention_unet
       output_type = flood_water_probability / flood_water_mask
       threshold = 0.40
       model_id = Attention_UNet_S1_S2
       validation_status = not_validated_on_Trishuli
  4. Downstream evidence fusion distinguishes:
       - AI flood evidence
       - SAR change evidence
       - confirmed dual evidence
       - unvalidated hillside SAR change
       - other/no evidence
  5. AI flood mask is NEVER marked as confirmed structural damage / debris.
  6. Stage 3 pipeline safely refuses execution when S2 provenance is unresolved.
  7. Precomputed AI flood mask connects cleanly into Stage 3 pipeline & fusion.
"""
from datetime import date
from pathlib import Path
import numpy as np
import pytest

from dev_mocks.synthetic_scene import synthetic_aoi, write_synthetic_pair
from floodmap.ai.adapters.attention_unet_adapter import (
    DEFAULT_INFERENCE_THRESHOLD,
    EXPECTED_15_CHANNELS,
    SENTINEL_1_CHANNELS,
    SENTINEL_2_CHANNELS,
    AttentionUNetS1S2Segmentor,
    ChannelContractError,
    Sentinel2PreparationMetadata,
    Sentinel2PreparationUnresolvedError,
    create_attention_unet_result,
)
from floodmap.ai.adapters.s2_preparation_adapter import Sentinel2PreparationAdapter
from floodmap.ai.interface import AISegmentationResult
from floodmap.core.config import Settings, load_settings
from floodmap.core.contracts import (
    AnalysisRequest,
    ChangeContext,
    DataMode,
    EvidenceSource,
    OutputClass,
)
from floodmap.core.georaster import GeoRaster
from floodmap.fusion.evidence_fusion import (
    EvidenceCategory,
    EvidenceStrength,
    fuse_multimodal_evidence,
)
from floodmap.pipeline.stage3_pipeline import run_stage3_pipeline
from floodmap.satellite.preprocess import PreprocessedPair

EVENT_DATE = date(2026, 8, 26)


# ==============================================================================
# 1. Sentinel2PreparationAdapter Unresolved Refusal
# ==============================================================================

def test_s2_prep_adapter_refuses_unresolved():
    """Verify Sentinel2PreparationAdapter refuses execution when any field is unresolved."""
    adapter = Sentinel2PreparationAdapter()
    assert not adapter.is_resolved

    unresolved = adapter.get_unresolved_fields()
    assert "product_level" in unresolved
    assert "b10_source" in unresolved
    assert "target_resolution_m" in unresolved
    assert "resampling_method" in unresolved
    assert "co_registration_method" in unresolved
    assert "temporal_pairing_rule" in unresolved

    # Calling assert_ready raises error
    with pytest.raises(Sentinel2PreparationUnresolvedError, match="preparation provenance is unresolved"):
        adapter.assert_ready()

    # Calling prepare_s2_array raises error
    fake_s2 = np.ones((13, 256, 256), dtype=np.float32)
    with pytest.raises(Sentinel2PreparationUnresolvedError, match="preparation provenance is unresolved"):
        adapter.prepare_s2_array(fake_s2)

    # Calling assemble_multimodal_scene_tensor raises error
    fake_s1 = np.ones((2, 256, 256), dtype=np.float32)
    with pytest.raises(Sentinel2PreparationUnresolvedError, match="preparation provenance is unresolved"):
        adapter.assemble_multimodal_scene_tensor(fake_s1, fake_s2)


# ==============================================================================
# 2. Sentinel2PreparationAdapter Band Contract & Preprocessing
# ==============================================================================

def test_s2_prep_adapter_channel_validation():
    """Verify adapter enforces the exact 13 Sentinel-2 bands and rejects missing or reordered channels."""
    meta = Sentinel2PreparationMetadata(
        product_level="L1C",
        b10_source="L1C_native_resampled",
        target_resolution_m=10.0,
        resampling_method="nearest",
        co_registration_method="s1_grid_aligned",
        temporal_pairing_rule="same_day_closest_cloud_free",
    )
    assert meta.is_resolved
    adapter = Sentinel2PreparationAdapter(metadata=meta)

    # Missing band B12
    with pytest.raises(ChannelContractError, match="missing required bands"):
        adapter.validate_s2_bands(SENTINEL_2_CHANNELS[:12])

    # Reordered bands
    wrong_order = list(SENTINEL_2_CHANNELS)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    with pytest.raises(ChannelContractError, match="sequence mismatch"):
        adapter.validate_s2_bands(wrong_order)

    # Valid execution
    valid_raw_s2 = np.ones((13, 100, 100), dtype=np.float32) * 2500.0
    valid_raw_s2[0, 0, 0] = 12000.0  # clipped to 1.0
    valid_raw_s2[1, 0, 0] = -100.0   # clipped to 0.0

    prepared = adapter.prepare_s2_array(valid_raw_s2)
    assert prepared.shape == (13, 100, 100)
    assert prepared.dtype == np.float32
    assert prepared[0, 0, 0] == 1.0
    assert prepared[1, 0, 0] == 0.0
    assert prepared[2, 0, 0] == 0.25

    # Assemble multimodal tensor
    valid_raw_s1 = np.ones((2, 100, 100), dtype=np.float32) * 0.05
    tensor_hwc = adapter.assemble_multimodal_scene_tensor(valid_raw_s1, prepared)
    assert tensor_hwc.shape == (100, 100, 15)
    assert tensor_hwc.dtype == np.float32
    assert np.allclose(tensor_hwc[:, :, 0], 0.05)
    assert np.allclose(tensor_hwc[:, :, 1], 0.05)
    assert tensor_hwc[0, 0, 2] == 1.0  # B1 at (0, 0)
    assert np.allclose(tensor_hwc[1:, 1:, 2], 0.25)



# ==============================================================================
# 3. Explicit Provenance Contract on AI Output
# ==============================================================================

def test_ai_output_explicit_provenance_contract():
    """Verify that Attention U-Net outputs carry the exact required provenance fields."""
    prob_arr = np.array([
        [0.10, 0.45],
        [0.55, 0.05],
    ], dtype=np.float32)[np.newaxis, ...]
    raster = GeoRaster(
        data=prob_arr,
        transform=(10.0, 0, 0, 0, -10.0, 0),
        crs="EPSG:32645",
        band_names=["water_prob"],
    )

    ai_res = create_attention_unet_result(
        probability_raster=raster,
        threshold=0.40,
    )

    assert isinstance(ai_res, AISegmentationResult)
    assert ai_res.model_name == "attention_unet_s1_s2"
    assert ai_res.threshold == 0.40
    assert ai_res.water_mask.shape == (2, 2)
    # Threshold 0.40 -> pixels >= 0.40: (0, 1) and (1, 0)
    assert not ai_res.water_mask[0, 0]
    assert ai_res.water_mask[0, 1]
    assert ai_res.water_mask[1, 0]
    assert not ai_res.water_mask[1, 1]

    # Required provenance keys
    meta = ai_res.metadata
    assert meta["source"] == "attention_unet"
    assert meta["output_type"] == "flood_water_probability / flood_water_mask"
    assert meta["threshold"] == 0.40
    assert meta["model_id"] == "Attention_UNet_S1_S2"
    assert meta["validation_status"] == "not_validated_on_Trishuli"
    assert meta["channels"] == list(EXPECTED_15_CHANNELS)


# ==============================================================================
# 4. Downstream Evidence Fusion Distinguishes All 5 Categories
# ==============================================================================

def test_downstream_evidence_fusion_distinguishes_categories():
    """Verify evidence fusion distinguishes:

    1. AI flood evidence (ai_only_water)
    2. SAR change evidence (sar_only_valley_change)
    3. confirmed dual evidence (ai_water_and_sar_valley_change / ai_and_sar_flood)
    4. unvalidated hillside SAR change (sar_only_hillside_change)
    5. other/no evidence (background / terrain_rejected)
    """
    # Grid 1x5: each pixel tests one condition
    # Pixel 0: Dual confirmed (AI water + SAR valley change)
    # Pixel 1: AI only water (AI water, no SAR change, terrain passed)
    # Pixel 2: SAR valley change only (no AI water)
    # Pixel 3: SAR hillside change (unvalidated)
    # Pixel 4: Background (no AI, no SAR)

    H, W = 1, 5
    prob_data = np.array([[[0.80, 0.70, 0.10, 0.10, 0.05]]], dtype=np.float32)
    cls_data = np.array([[[
        int(OutputClass.DEBRIS_OR_CHANGE),  # px 0: SAR change
        int(OutputClass.BACKGROUND),        # px 1: no SAR change
        int(OutputClass.DEBRIS_OR_CHANGE),  # px 2: SAR change
        int(OutputClass.DEBRIS_OR_CHANGE),  # px 3: SAR change
        int(OutputClass.BACKGROUND),        # px 4: background
    ]]], dtype=np.uint8)

    ctx_data = np.array([[[
        int(ChangeContext.VALLEY_BOTTOM),   # px 0: valley bottom
        int(ChangeContext.NOT_CHANGE),      # px 1: not change
        int(ChangeContext.VALLEY_BOTTOM),   # px 2: valley bottom
        int(ChangeContext.HILLSIDE),        # px 3: hillside
        int(ChangeContext.NOT_CHANGE),      # px 4: not change
    ]]], dtype=np.uint8)

    slope_data = np.zeros((1, H, W), dtype=np.float32)  # flat terrain (< 10°)
    hand_data = np.zeros((1, H, W), dtype=np.float32)   # 0m HAND (< 30m)

    transform = (10.0, 0, 0, 0, -10.0, 0)
    crs = "EPSG:32645"

    ai_prob_r = GeoRaster(data=prob_data, transform=transform, crs=crs, band_names=["water_prob"])
    s2_cls_r = GeoRaster(data=cls_data, transform=transform, crs=crs, band_names=["classes"])
    ctx_r = GeoRaster(data=ctx_data, transform=transform, crs=crs, band_names=["context"])
    slope_r = GeoRaster(data=slope_data, transform=transform, crs=crs, band_names=["slope"])
    hand_r = GeoRaster(data=hand_data, transform=transform, crs=crs, band_names=["hand"])

    res = fuse_multimodal_evidence(
        ai_probability=ai_prob_r,
        stage2_classes=s2_cls_r,
        stage2_context=ctx_r,
        terrain_slope=slope_r,
        terrain_hand=hand_r,
        ai_threshold=0.40,
    )

    cat = res.category_raster.data[0, 0]
    strn = res.strength_raster.data[0, 0]

    # Pixel 0: Confirmed dual evidence
    assert cat[0] == int(EvidenceCategory.AI_WATER_AND_SAR_VALLEY_CHANGE)
    assert strn[0] == int(EvidenceStrength.HIGH_CONFIDENCE)

    # Pixel 1: AI flood evidence only (candidate water, not confirmed damage)
    assert cat[1] == int(EvidenceCategory.AI_ONLY_WATER)
    assert strn[1] == int(EvidenceStrength.AI_ONLY)

    # Pixel 2: SAR change evidence only
    assert cat[2] == int(EvidenceCategory.SAR_ONLY_VALLEY_CHANGE)
    assert strn[2] == int(EvidenceStrength.SAR_ONLY)

    # Pixel 3: Unvalidated hillside SAR change
    assert cat[3] == int(EvidenceCategory.SAR_ONLY_HILLSIDE_CHANGE)
    assert strn[3] == int(EvidenceStrength.UNVALIDATED)

    # Pixel 4: Background (other/no evidence)
    assert cat[4] == int(EvidenceCategory.BACKGROUND)
    assert strn[4] == int(EvidenceStrength.NONE)


# ==============================================================================
# 5. AI Flood Mask NEVER Becomes "Confirmed Damage"
# ==============================================================================

def test_ai_flood_mask_never_becomes_confirmed_damage():
    """Verify that an AI water candidate is NEVER classified as structural damage or debris."""
    prob_data = np.array([[[0.95]]], dtype=np.float32)
    cls_data = np.array([[[int(OutputClass.BACKGROUND)]]], dtype=np.uint8)

    transform = (10.0, 0, 0, 0, -10.0, 0)
    crs = "EPSG:32645"

    ai_prob_r = GeoRaster(data=prob_data, transform=transform, crs=crs, band_names=["water_prob"])
    s2_cls_r = GeoRaster(data=cls_data, transform=transform, crs=crs, band_names=["classes"])

    res = fuse_multimodal_evidence(
        ai_probability=ai_prob_r,
        stage2_classes=s2_cls_r,
        ai_threshold=0.40,
    )

    # Must be categorized as AI_ONLY_WATER, NOT high confidence and NOT debris/damage
    cat = res.category_raster.data[0, 0, 0]
    strn = res.strength_raster.data[0, 0, 0]

    assert cat == int(EvidenceCategory.AI_ONLY_WATER)
    assert strn == int(EvidenceStrength.AI_ONLY)
    assert strn != int(EvidenceStrength.HIGH_CONFIDENCE)
    assert cat != int(EvidenceCategory.AI_WATER_AND_SAR_VALLEY_CHANGE)


# ==============================================================================
# 6. Stage 3 Pipeline Refusal When S2 Provenance is Unresolved
# ==============================================================================

def test_stage3_pipeline_attention_unet_unresolved_refusal(tmp_path):
    """Verify Stage 3 pipeline strictly refuses execution if Attention_UNet_S1_S2 is selected with unresolved S2."""
    bpath, apath, pair = write_synthetic_pair(tmp_path / "in")
    aoi = synthetic_aoi(pair, inset_m=30)
    dpath = tmp_path / "in" / "SYNTHETIC_dem.tif"

    req = AnalysisRequest(
        aoi=aoi,
        event_date=EVENT_DATE,
        before_scene=bpath,
        after_scene=apath,
        dem=dpath,
    )

    settings = load_settings()
    settings.ai.model = "attention_unet_s1_s2"

    run_dir = tmp_path / "s3_refusal_run"

    # Must raise Sentinel2PreparationUnresolvedError
    with pytest.raises(Sentinel2PreparationUnresolvedError, match="Sentinel-2 preparation provenance is unresolved"):
        run_stage3_pipeline(req, settings, run_dir, analysis_id="s3_refusal")


# ==============================================================================
# 7. Stage 3 Pipeline Ingestion of Precomputed AI Result
# ==============================================================================

def test_stage3_pipeline_ingests_precomputed_ai_result(tmp_path):
    """Verify Stage 3 pipeline ingests a verified precomputed AI result into the full pipeline flow."""
    bpath, apath, pair = write_synthetic_pair(tmp_path / "in")
    aoi = synthetic_aoi(pair, inset_m=30)
    dpath = tmp_path / "in" / "SYNTHETIC_dem.tif"

    # Pre-build an AI result with verified contract provenance
    # Shape matching the synthetic analysis grid (approx 57x57)
    from floodmap.satellite.grid import make_analysis_grid
    grid = make_analysis_grid(aoi, 10.0, 0, "EPSG:32645")
    H, W = grid.spec.shape

    prob_arr = np.zeros((1, H, W), dtype=np.float32)
    prob_arr[0, 10:20, 10:20] = 0.85  # flood candidate patch

    ai_raster = GeoRaster(
        data=prob_arr,
        transform=grid.spec.transform,
        crs=grid.spec.crs,
        band_names=["water_prob"],
    )
    ai_result = create_attention_unet_result(
        probability_raster=ai_raster,
        threshold=0.40,
    )

    req = AnalysisRequest(
        aoi=aoi,
        event_date=EVENT_DATE,
        before_scene=bpath,
        after_scene=apath,
        dem=dpath,
        options={"ai_result": ai_result},
    )

    settings = load_settings()
    run_dir = tmp_path / "s3_precomputed_run"

    res = run_stage3_pipeline(req, settings, run_dir, analysis_id="s3_precomputed")
    assert res.ai.model_name == "attention_unet_s1_s2"
    assert res.ai.metadata["source"] == "attention_unet"
    assert res.ai.metadata["validation_status"] == "not_validated_on_Trishuli"

    # Verify output rasters written
    assert (res.run_dir / "change" / "change_classes.tif").is_file()
    assert (res.run_dir / "change" / "evidence_source.tif").is_file()
    assert (res.run_dir / "change" / "ai_probability.tif").is_file()
