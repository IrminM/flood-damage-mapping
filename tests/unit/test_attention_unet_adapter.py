"""Unit tests for multimodal Attention U-Net (Attention_UNet_S1_S2) adapter and contracts."""
from pathlib import Path
import numpy as np
import pytest

from floodmap.ai.adapters.attention_unet_adapter import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_INFERENCE_THRESHOLD,
    EXPECTED_15_CHANNELS,
    EXPECTED_TENSOR_SHAPE,
    SENTINEL_1_CHANNELS,
    SENTINEL_2_CHANNELS,
    AttentionUNetS1S2Segmentor,
    ChannelContractError,
    Sentinel2PreparationMetadata,
    Sentinel2PreparationUnresolvedError,
    assemble_full_scene_tensor,
    assemble_multimodal_tensor,
    get_attention_unet_metadata,
    load_attention_unet_model,
    preprocess_s1,
    preprocess_s2,
    verify_checkpoint_execution,
)
from floodmap.ai.inference import MultimodalTiledInferenceRunner
from floodmap.ai.interface import AISegmentationResult, get_ai_segmentor
from floodmap.core.config import Settings
from floodmap.core.georaster import GeoRaster
from floodmap.satellite.preprocess import PreprocessedPair


def test_channel_constants():
    """Verify channel constants match the proven 15-channel contract."""
    assert len(SENTINEL_1_CHANNELS) == 2
    assert SENTINEL_1_CHANNELS == ("VV", "VH")

    assert len(SENTINEL_2_CHANNELS) == 13
    assert SENTINEL_2_CHANNELS == (
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
    )

    assert len(EXPECTED_15_CHANNELS) == 15
    assert EXPECTED_15_CHANNELS[:2] == ("VV", "VH")
    assert EXPECTED_15_CHANNELS[2:] == SENTINEL_2_CHANNELS
    assert EXPECTED_TENSOR_SHAPE == (256, 256, 15)
    assert DEFAULT_INFERENCE_THRESHOLD == 0.40


def test_s1_preprocessing_nan_inf_handling():
    """Verify S1 preprocessing performs per-channel median replacement and 0.0 fallback."""
    # Channel 0: valid finite values with median = 20.0, plus NaN, +Inf, -Inf
    ch0 = np.array([
        [10.0, 20.0, np.nan],
        [30.0, np.inf, -np.inf],
    ], dtype=np.float32)

    # Channel 1: all non-finite (NaN/Inf) -> fallback to 0.0
    ch1 = np.array([
        [np.nan, np.inf, -np.inf],
        [np.nan, np.nan, np.inf],
    ], dtype=np.float32)

    raw_s1 = np.stack([ch0, ch1], axis=0)
    proc_s1 = preprocess_s1(raw_s1)

    assert proc_s1.dtype == np.float32
    assert np.isfinite(proc_s1).all()

    # In ch0, finite median of [10.0, 20.0, 30.0] is 20.0
    assert proc_s1[0, 0, 0] == 10.0
    assert proc_s1[0, 0, 1] == 20.0
    assert proc_s1[0, 0, 2] == 20.0  # was NaN
    assert proc_s1[0, 1, 0] == 30.0
    assert proc_s1[0, 1, 1] == 20.0  # was +Inf
    assert proc_s1[0, 1, 2] == 20.0  # was -Inf

    # In ch1, all non-finite -> all 0.0
    assert np.all(proc_s1[1] == 0.0)


def test_s2_preprocessing_scaling_and_clipping():
    """Verify S2 values are divided by 10000.0 and clipped to [0.0, 1.0]."""
    # 13 channels of shape (2, 2)
    raw_s2 = np.ones((13, 2, 2), dtype=np.float32) * 5000.0
    raw_s2[0, 0, 0] = 15000.0   # exceeds 10000 -> clipped to 1.0
    raw_s2[1, 0, 0] = -500.0    # negative -> clipped to 0.0
    raw_s2[2, 0, 0] = 2500.0    # 2500 / 10000 -> 0.25

    proc_s2 = preprocess_s2(raw_s2)

    assert proc_s2.dtype == np.float32
    assert proc_s2.shape == (13, 2, 2)
    assert proc_s2[0, 0, 0] == 1.0
    assert proc_s2[1, 0, 0] == 0.0
    assert proc_s2[2, 0, 0] == 0.25
    assert proc_s2[3, 0, 0] == 0.50
    assert np.all(proc_s2 >= 0.0) and np.all(proc_s2 <= 1.0)


def test_assemble_multimodal_tensor_shape_and_cropping():
    """Verify tensor assembly, channels-last transpose, and top-left 256x256 cropping."""
    # Create 512x512 inputs
    s1_data = np.ones((2, 512, 512), dtype=np.float32) * -12.0
    s2_data = np.ones((13, 512, 512), dtype=np.float32) * 4000.0

    tensor = assemble_multimodal_tensor(s1_data, s2_data)

    assert tensor.shape == (256, 256, 15)
    assert tensor.dtype == np.float32

    # Channel 0 & 1 are S1 (-12.0)
    assert np.all(tensor[:, :, 0] == -12.0)
    assert np.all(tensor[:, :, 1] == -12.0)

    # Channels 2..14 are S2 (4000 / 10000 = 0.40)
    for c in range(2, 15):
        assert np.allclose(tensor[:, :, c], 0.40)


def test_channel_contract_violations():
    """Verify strict refusal when channel counts or orderings are violated."""
    s1_ok = np.ones((2, 256, 256), dtype=np.float32)
    s2_ok = np.ones((13, 256, 256), dtype=np.float32)

    # Wrong S1 channel ordering (e.g. reversed)
    with pytest.raises(ChannelContractError, match="Invalid S1 channel ordering"):
        assemble_multimodal_tensor(s1_ok, s2_ok, s1_channels=("VH", "VV"))

    # Wrong S2 channel ordering (e.g. missing bands in channel sequence)
    with pytest.raises(ChannelContractError, match="Invalid S2 channel ordering"):
        assemble_multimodal_tensor(s1_ok, s2_ok, s2_channels=SENTINEL_2_CHANNELS[:12])

    # Wrong S2 channel count in array (e.g. 12 channels instead of 13)
    s2_wrong = np.ones((12, 256, 256), dtype=np.float32)
    with pytest.raises(ChannelContractError, match="S2 channel count mismatch"):
        assemble_multimodal_tensor(s1_ok, s2_wrong, s2_channels=SENTINEL_2_CHANNELS)

    # Wrong S1 channel count in array (e.g. 1 channel instead of 2)
    s1_wrong = np.ones((1, 256, 256), dtype=np.float32)
    with pytest.raises(ChannelContractError, match="S1 channel count mismatch"):
        assemble_multimodal_tensor(s1_wrong, s2_ok, s1_channels=SENTINEL_1_CHANNELS)

    # Spatial dimension mismatch between S1 and S2
    s1_mismatch = np.ones((2, 256, 256), dtype=np.float32)
    s2_mismatch = np.ones((13, 128, 128), dtype=np.float32)
    with pytest.raises(ChannelContractError, match="Spatial dimension mismatch"):
        assemble_multimodal_tensor(s1_mismatch, s2_mismatch)

    # Input smaller than required 256x256
    s1_small = np.ones((2, 128, 128), dtype=np.float32)
    s2_small = np.ones((13, 128, 128), dtype=np.float32)
    with pytest.raises(ChannelContractError, match="smaller than required"):
        assemble_multimodal_tensor(s1_small, s2_small)


def test_s2_metadata_unresolved_refusal():
    """Verify adapter strictly refuses execution when Sentinel-2 preparation is unresolved."""
    # Default metadata has unresolved fields
    meta = Sentinel2PreparationMetadata()
    assert not meta.is_resolved
    unresolved = meta.get_unresolved_fields()
    assert "product_level" in unresolved
    assert "b10_source" in unresolved
    assert "target_resolution_m" in unresolved
    assert "resampling_method" in unresolved
    assert "co_registration_method" in unresolved
    assert "temporal_pairing_rule" in unresolved

    adapter = AttentionUNetS1S2Segmentor(s2_metadata=meta)
    assert not adapter.is_operational()

    # Calling assert_ready_to_run raises Sentinel2PreparationUnresolvedError
    with pytest.raises(Sentinel2PreparationUnresolvedError, match="Sentinel-2 preparation provenance is unresolved"):
        adapter.assert_ready_to_run()

    # Calling segment with dummy pair also refuses execution
    data = np.ones((2, 256, 256), dtype=np.float32) * -15.0
    raster = GeoRaster(data=data, transform=(1.0, 0, 0, 0, -1.0, 0), crs="EPSG:32645", band_names=["VV", "VH"])
    valid = np.ones((256, 256), dtype=bool)
    pair = PreprocessedPair(
        before_db=raster,
        after_db=raster,
        valid=valid,
        aoi_mask=valid,
        nodata_in_aoi=~valid,
        bands=["VV", "VH"],
    )

    with pytest.raises(Sentinel2PreparationUnresolvedError, match="Sentinel-2 preparation provenance is unresolved"):
        adapter.segment(pair, s2_data=np.ones((13, 256, 256), dtype=np.float32))


def test_decision_threshold_40():
    """Verify decision threshold is 0.40 and correctly binarizes probability maps."""
    adapter = AttentionUNetS1S2Segmentor(threshold=0.40)
    assert adapter.threshold == 0.40

    prob = np.array([
        [0.00, 0.39, 0.399],
        [0.40, 0.401, 0.85],
    ], dtype=np.float32)

    mask = adapter.apply_threshold(prob)
    expected = np.array([
        [False, False, False],
        [True, True, True],
    ], dtype=bool)

    assert np.array_equal(mask, expected)


def test_mock_model_execution_when_metadata_resolved():
    """Verify that once S2 preparation metadata is resolved, the adapter executes cleanly with a mock model."""
    resolved_meta = Sentinel2PreparationMetadata(
        product_level="L2A",
        b10_source="L1C_TOA",
        target_resolution_m=10.0,
        resampling_method="bilinear",
        co_registration_method="s1_affine_warp",
        temporal_pairing_rule="nearest_cloud_free_pre_event",
    )
    assert resolved_meta.is_resolved
    assert len(resolved_meta.get_unresolved_fields()) == 0

    # Mock model: returns constant flood probability of 0.65
    def mock_model(x):
        # x is (1, 256, 256, 15)
        return np.ones((x.shape[0], 256, 256, 1), dtype=np.float32) * 0.65

    adapter = AttentionUNetS1S2Segmentor(
        model=mock_model,
        s2_metadata=resolved_meta,
        threshold=0.40,
    )
    assert adapter.is_operational()

    data = np.ones((2, 256, 256), dtype=np.float32) * -15.0
    raster = GeoRaster(data=data, transform=(10.0, 0, 500000, 0, -10.0, 3100000), crs="EPSG:32645", band_names=["VV", "VH"])
    valid = np.ones((256, 256), dtype=bool)
    pair = PreprocessedPair(
        before_db=raster,
        after_db=raster,
        valid=valid,
        aoi_mask=valid,
        nodata_in_aoi=~valid,
        bands=["VV", "VH"],
    )
    s2_data = np.ones((13, 256, 256), dtype=np.float32) * 3000.0

    result = adapter.segment(pair, s2_data=s2_data)

    assert isinstance(result, AISegmentationResult)
    assert result.model_name == "attention_unet_s1_s2"
    assert result.threshold == 0.40
    assert result.probability.shape == (256, 256)
    assert np.allclose(result.probability.data[0], 0.65)
    # Threshold 0.40 -> 0.65 >= 0.40 -> True
    assert result.water_mask.shape == (256, 256)
    assert np.all(result.water_mask)
    assert result.stats["flood_pixel_count"] == 256 * 256


def test_load_attention_unet_model_missing_file():
    """Verify model loader raises FileNotFoundError if checkpoint does not exist."""
    fake_path = Path("nonexistent_best_flood_attention_unet.keras")
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_attention_unet_model(fake_path)


def test_ai_registry_resolution():
    """Verify attention_unet_s1_s2 is registered and resolves through Settings."""
    settings = Settings()
    settings.ai.model = "attention_unet_s1_s2"
    segmentor = get_ai_segmentor(settings)
    assert isinstance(segmentor, AttentionUNetS1S2Segmentor)
    assert segmentor.threshold == 0.40
    # Newly created from default settings is not operational because S2 is unresolved
    assert not segmentor.is_operational()


def test_get_attention_unet_metadata():
    """Verify metadata contract values for Attention U-Net."""
    meta = get_attention_unet_metadata()
    assert meta["model_name"] == "Attention_UNet_S1_S2"
    assert meta["decision_threshold"] == 0.40
    assert meta["input_shape"] == [None, 256, 256, 15]
    assert meta["output_shape"] == [None, 256, 256, 1]
    assert meta["s1_channels"] == ["VV", "VH"]
    assert len(meta["s2_channels"]) == 13
    assert meta["operational_readiness"] == "BLOCKED_PENDING_VALID_S2_INPUT"
    assert meta["s2_readiness"] == "BLOCKED_PENDING_VALID_S2_INPUT"
    assert meta["himalayan_validation"] is False
    assert meta["trishuli_validation_status"] == "UNVALIDATED / BASELINE"
    assert meta["predicted_classes"] == ["flood_water"]
    assert "debris" in meta["not_predicted_classes"]
    assert meta["checkpoint_exists"] is True


def test_multimodal_tiled_inference_runner_arbitrary_dimensions():
    """Verify sliding-window tiled inference operates on arbitrary non-256 scene shapes."""
    runner = MultimodalTiledInferenceRunner(tile_size=256, overlap=32)

    # 300 x 400 scene with 15 channels
    scene = np.ones((300, 400, 15), dtype=np.float32) * 0.5
    valid_mask = np.ones((300, 400), dtype=bool)
    valid_mask[:10, :10] = False  # masked edge

    ref_raster = GeoRaster(
        data=np.ones((1, 300, 400), dtype=np.float32),
        transform=(10.0, 0, 500000, 0, -10.0, 3100000),
        crs="EPSG:32645",
        band_names=["band1"],
    )

    # Predict function returns constant 0.70
    def mock_predict(batch):
        return np.ones((batch.shape[0], 256, 256, 1), dtype=np.float32) * 0.70

    out_raster = runner.predict(mock_predict, scene, ref_raster, valid_mask=valid_mask)

    assert out_raster.shape == (300, 400)
    assert out_raster.crs == "EPSG:32645"
    assert out_raster.transform == ref_raster.transform
    # Masked pixels must be NaN
    assert np.isnan(out_raster.data[0, 0, 0])
    # Unmasked pixels should be blended correctly (0.70)
    assert np.isclose(out_raster.data[0, 100, 100], 0.70)


def test_preserve_s1_mini_unet_fallback():
    """Verify that s1_mini_unet_v1 remains the intact default operational model."""
    settings = Settings()
    assert settings.ai.model == "s1_mini_unet_v1"
    segmentor = get_ai_segmentor(settings)
    assert segmentor.name == "s1_mini_unet_v1"


def test_real_checkpoint_smoke_verification():
    """Verify that the actual data/models/best_flood_attention_unet.keras loads and executes."""
    assert DEFAULT_CHECKPOINT_PATH.is_file(), f"Expected checkpoint file at {DEFAULT_CHECKPOINT_PATH}"
    report = verify_checkpoint_execution(DEFAULT_CHECKPOINT_PATH)
    assert report["status"] == "VERIFIED_LOADABLE"
    assert report["model_name"] == "Attention_UNet_S1_S2"
    assert report["input_shape"] == [None, 256, 256, 15]
    assert report["output_shape"] == [None, 256, 256, 1]
    assert report["synthetic_prediction_shape"] == [1, 256, 256, 1]
    assert report["is_sigmoid_probability"] is True


def test_attention_unet_smoke_test_run_artifacts():
    """Verify that the single real patch smoke test artifacts exist and satisfy contracts."""
    import json
    import rasterio

    run_dir = Path("data/runs/attention_unet_smoke_test")
    summary_path = run_dir / "summary.json"
    meta_path = run_dir / "input_metadata.json"
    prob_path = run_dir / "probability_patch.tif"
    mask_path = run_dir / "flood_mask_patch.tif"

    assert summary_path.is_file(), f"Missing {summary_path}"
    assert meta_path.is_file(), f"Missing {meta_path}"
    assert prob_path.is_file(), f"Missing {prob_path}"
    assert mask_path.is_file(), f"Missing {mask_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["input_shape"] == [256, 256, 15]
    assert summary["channel_order"] == list(EXPECTED_15_CHANNELS)
    assert 0.0 <= summary["probability_stats"]["min"] <= 1.0
    assert 0.0 <= summary["probability_stats"]["max"] <= 1.0
    assert summary["classification_results"]["total_pixels"] == 65536
    assert "disclaimer" in summary

    with rasterio.open(prob_path) as src:
        assert src.shape == (256, 256)
        assert src.crs == "EPSG:32645"
        assert src.dtypes[0] == "float32"

    with rasterio.open(mask_path) as src:
        assert src.shape == (256, 256)
        assert src.crs == "EPSG:32645"
        assert src.dtypes[0] == "uint8"
        data = src.read(1)
        unique_vals = set(np.unique(data))
        assert unique_vals.issubset({0, 1})


