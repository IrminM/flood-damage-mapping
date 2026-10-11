"""Unit tests for the production Sentinel-2 Preparation Adapter.

Validates all contracts established by the scientific provenance audit:
  1. All 13 Sentinel-2 bands must be present in canonical order.
  2. Band 10 (cirrus, native 60 m resampled to 10 m) cannot be omitted or synthesized.
  3. Product level must be Level-1C (Level-2A strictly rejected).
  4. Resampling method must strictly be nearest-neighbor ('nearest').
  5. Target grid resolution is 10 m.
  6. Scaling by 10,000 and clipping to [0.0, 1.0].
  7. Temporal separation check (Δt <= max_temporal_separation_days).
  8. Final multimodal shape (H, W, 15) with channel order [VV, VH, B1..B12].
  9. Spatial grid mismatch refusal (CRS, shape, or transform differences).
  10. Full diagnostic metadata generation.
"""
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np
import pytest

from floodmap.ai.adapters.attention_unet_adapter import (
    EXPECTED_15_CHANNELS,
    SENTINEL_1_CHANNELS,
    SENTINEL_2_CHANNELS,
    ChannelContractError,
    Sentinel2PreparationMetadata,
    Sentinel2PreparationUnresolvedError,
)
from floodmap.ai.adapters.s2_preparation_adapter import (
    ProductLevelError,
    ResamplingMethodError,
    Sentinel2AcquisitionConfig,
    Sentinel2PreparationAdapter,
    Sentinel2PreparationDiagnostics,
    TemporalPairingError,
    normalize_s2_band_name,
)
from floodmap.core.errors import GridMismatchError
from floodmap.core.georaster import GeoRaster
from floodmap.satellite.grid import GridSpec


# Helper to build mock GeoRasters
def make_mock_s2_raster(
    shape=(100, 100),
    bands=SENTINEL_2_CHANNELS,
    crs="EPSG:32645",
    transform=(10.0, 0, 318950.0, 0, -10.0, 3119510.0),
    fill_dn=3000.0,
) -> GeoRaster:
    nbands = len(bands)
    data = np.ones((nbands, shape[0], shape[1]), dtype=np.float32) * fill_dn
    return GeoRaster(
        data=data,
        transform=transform,
        crs=crs,
        band_names=list(bands),
    )


def make_mock_s1_raster(
    shape=(100, 100),
    bands=SENTINEL_1_CHANNELS,
    crs="EPSG:32645",
    transform=(10.0, 0, 318950.0, 0, -10.0, 3119510.0),
    fill_power=0.08,
) -> GeoRaster:
    nbands = len(bands)
    data = np.ones((nbands, shape[0], shape[1]), dtype=np.float32) * fill_power
    return GeoRaster(
        data=data,
        transform=transform,
        crs=crs,
        band_names=list(bands),
    )


# ==============================================================================
# 1. Band Name Normalization & All 13 Bands Present
# ==============================================================================

def test_band_name_normalization():
    """Verify band name standardizer correctly maps various S2 naming formats."""
    assert normalize_s2_band_name("B01") == "B1"
    assert normalize_s2_band_name("b01") == "B1"
    assert normalize_s2_band_name("B1") == "B1"
    assert normalize_s2_band_name("B08") == "B8"
    assert normalize_s2_band_name("B8A") == "B8A"
    assert normalize_s2_band_name("B08A") == "B8A"
    assert normalize_s2_band_name("b8a") == "B8A"
    assert normalize_s2_band_name("B09") == "B9"
    assert normalize_s2_band_name("B10") == "B10"
    assert normalize_s2_band_name("B11") == "B11"
    assert normalize_s2_band_name("B12") == "B12"


def test_all_13_bands_required():
    """Verify that omitting any of the 13 required bands raises ChannelContractError."""
    adapter = Sentinel2PreparationAdapter.create_production_adapter()

    # Missing B12
    with pytest.raises(ChannelContractError, match="missing required band"):
        incomplete_raster = make_mock_s2_raster(bands=SENTINEL_2_CHANNELS[:12])
        adapter.prepare_s2_raster(incomplete_raster)

    # Missing B1
    with pytest.raises(ChannelContractError, match="missing required band"):
        incomplete_raster = make_mock_s2_raster(bands=SENTINEL_2_CHANNELS[1:])
        adapter.prepare_s2_raster(incomplete_raster)


# ==============================================================================
# 2. B10 Cannot Be Silently Omitted or Fabricated
# ==============================================================================

def test_b10_cannot_be_silently_omitted():
    """Verify that omitting B10 raises ChannelContractError specifically mentioning B10."""
    adapter = Sentinel2PreparationAdapter.create_production_adapter()
    bands_without_b10 = [b for b in SENTINEL_2_CHANNELS if b != "B10"]
    raster_without_b10 = make_mock_s2_raster(bands=bands_without_b10)

    with pytest.raises(ChannelContractError, match="B10"):
        adapter.prepare_s2_raster(raster_without_b10)


def test_l2a_product_level_rejected():
    """Verify Level-2A products are strictly rejected because they lack B10 and alter TOA reflectance."""
    adapter = Sentinel2PreparationAdapter.create_production_adapter()
    s2_raster = make_mock_s2_raster()

    # Declared as Level-2A in metadata
    with pytest.raises(ProductLevelError, match="Level-2A"):
        adapter.prepare_s2_raster(s2_raster, s2_metadata={"product_type": "L2A"})

    with pytest.raises(ProductLevelError, match="Level-2A"):
        adapter.validate_product_level("MSIL2A")


# ==============================================================================
# 3. Nearest-Neighbor Resampling is Strictly Enforced
# ==============================================================================

def test_nearest_neighbor_enforced():
    """Verify that non-nearest resampling methods (bilinear, cubic, average) are strictly rejected."""
    # Prohibited bilinear
    cfg_bilinear = Sentinel2AcquisitionConfig(resampling_method="bilinear")
    adapter_bilinear = Sentinel2PreparationAdapter.create_production_adapter(config=cfg_bilinear)
    with pytest.raises(ResamplingMethodError, match="bilinear"):
        adapter_bilinear.validate_resampling_method("bilinear")

    # Prohibited cubic
    with pytest.raises(ResamplingMethodError, match="cubic"):
        adapter_bilinear.validate_resampling_method("cubic")

    # Allowed nearest
    adapter = Sentinel2PreparationAdapter.create_production_adapter()
    adapter.validate_resampling_method("nearest")
    adapter.validate_resampling_method("nearest-neighbour")


# ==============================================================================
# 4. Target Grid and 10 m Resolution Reprojection
# ==============================================================================

def test_10m_target_grid_reprojection():
    """Verify that bands at native coarse resolution (e.g. 20m, 60m) are resampled to 10 m target grid."""
    adapter = Sentinel2PreparationAdapter.create_production_adapter()

    # Create target grid at 10m (100 x 100 pixels)
    target_grid = GridSpec(
        crs="EPSG:32645",
        transform=(10.0, 0, 318950.0, 0, -10.0, 3119510.0),
        width=100,
        height=100,
    )

    # Input raster at 20m (50 x 50 pixels)
    coarse_raster = make_mock_s2_raster(
        shape=(50, 50),
        transform=(20.0, 0, 318950.0, 0, -20.0, 3119510.0),
        fill_dn=4000.0,
    )

    result = adapter.prepare_s2_raster(coarse_raster, target_grid=target_grid)
    prepared = result.s2_raster

    assert prepared.shape == (100, 100)
    assert np.allclose(prepared.res, (10.0, 10.0))
    assert np.allclose(prepared.data, 0.40)  # 4000 / 10000 = 0.40


# ==============================================================================
# 5. Scaling by 10,000 and Clipping to [0, 1]
# ==============================================================================

def test_scaling_and_clipping():
    """Verify raw integer DN values are scaled by 10000.0 and clipped to [0.0, 1.0]."""
    adapter = Sentinel2PreparationAdapter.create_production_adapter()

    raw_raster = make_mock_s2_raster(shape=(10, 10), fill_dn=2500.0)
    # Modify specific pixels in band 0 (B1)
    raw_raster.data[0, 0, 0] = 15000.0  # > 10000 -> clipped to 1.0
    raw_raster.data[0, 0, 1] = -500.0   # < 0 -> clipped to 0.0
    raw_raster.data[0, 0, 2] = 7500.0   # 7500 / 10000 -> 0.75

    result = adapter.prepare_s2_raster(raw_raster)
    data = result.s2_raster.data

    assert data.dtype == np.float32
    assert data[0, 0, 0] == 1.0
    assert data[0, 0, 1] == 0.0
    assert np.isclose(data[0, 0, 2], 0.75)
    assert np.isclose(data[1, 5, 5], 0.25)  # 2500 / 10000 = 0.25
    assert np.all(data >= 0.0) and np.all(data <= 1.0)


# ==============================================================================
# 6. Canonical Channel Order
# ==============================================================================

def test_canonical_channel_order_reordering():
    """Verify that scrambled band orders are restored to exact canonical order."""
    adapter = Sentinel2PreparationAdapter.create_production_adapter()

    # Scramble band order
    shuffled_bands = list(SENTINEL_2_CHANNELS)
    shuffled_bands[0], shuffled_bands[-1] = shuffled_bands[-1], shuffled_bands[0]  # swap B1 and B12

    scrambled_raster = make_mock_s2_raster(bands=shuffled_bands)
    result = adapter.prepare_s2_raster(scrambled_raster)

    assert result.s2_raster.band_names == list(SENTINEL_2_CHANNELS)
    assert result.diagnostics.bands == list(SENTINEL_2_CHANNELS)


# ==============================================================================
# 7. Temporal Separation Enforcement
# ==============================================================================

def test_temporal_separation_validation():
    """Verify temporal separation between S1 and S2 is validated and enforced."""
    cfg = Sentinel2AcquisitionConfig(max_temporal_separation_days=1.5, strict_temporal_check=True)
    adapter = Sentinel2PreparationAdapter.create_production_adapter(config=cfg)

    # Valid pair: S1 on 2026-08-28 12:00, S2 on 2026-08-27 05:00 (gap ~ 1.29 days <= 1.5)
    s1_time = "2026-08-28T12:00:00Z"
    s2_time = "2026-08-27T05:00:00Z"
    gap = adapter.validate_temporal_separation(s1_time, s2_time)
    assert np.isclose(gap, 1.291666, atol=0.01)

    # Invalid pair: S1 on 2026-08-28 12:00, S2 on 2026-08-20 05:00 (gap ~ 8.29 days > 1.5)
    s2_too_old = "2026-08-20T05:00:00Z"
    with pytest.raises(TemporalPairingError, match="Temporal separation"):
        adapter.validate_temporal_separation(s1_time, s2_too_old)


# ==============================================================================
# 8. Final Shape Contract & Combination with S1
# ==============================================================================

def test_combine_with_s1_and_final_shape():
    """Verify combining S1 and S2 produces (H, W, 15) tensor with exact channel layout."""
    adapter = Sentinel2PreparationAdapter.create_production_adapter()

    s1_raster = make_mock_s1_raster(shape=(50, 50), fill_power=0.05)
    s2_raster = make_mock_s2_raster(shape=(50, 50), fill_dn=3000.0)

    s1_meta = {"acquisition_time": "2026-08-28T12:00:00Z"}
    s2_meta = {"acquisition_time": "2026-08-27T05:00:00Z", "product_type": "L1C"}

    s2_res = adapter.prepare_s2_raster(s2_raster, s2_metadata=s2_meta)
    tensor_15ch, diag = adapter.combine_with_s1(
        s1_input=s1_raster,
        s2_input=s2_res,
        s1_metadata=s1_meta,
    )

    assert tensor_15ch.shape == (50, 50, 15)
    assert tensor_15ch.dtype == np.float32

    # Channel 0 (VV) & 1 (VH)
    assert np.allclose(tensor_15ch[:, :, 0], 0.05)
    assert np.allclose(tensor_15ch[:, :, 1], 0.05)

    # Channels 2..14 (B1..B12)
    for c in range(2, 15):
        assert np.allclose(tensor_15ch[:, :, c], 0.30)  # 3000 / 10000 = 0.30

    assert diag.s1_acquisition_date == "2026-08-28T12:00:00Z"
    assert diag.s2_acquisition_date == "2026-08-27T05:00:00Z"
    assert diag.delta_t_days is not None and np.isclose(diag.delta_t_days, 1.29, atol=0.02)
    assert diag.product_level == "Level-1C"
    assert diag.b10_provenance == "native_l1c_band10_60m_resampled_nearest_10m"


# ==============================================================================
# 9. Spatial Grid Mismatch Refusal
# ==============================================================================

def test_grid_mismatch_refusal():
    """Verify adapter strictly refuses combining S1 and S2 if CRS, shape, or transform differ."""
    adapter = Sentinel2PreparationAdapter.create_production_adapter()

    s1_raster = make_mock_s1_raster(shape=(50, 50), crs="EPSG:32645")

    # Mismatch 1: Different CRS
    s2_diff_crs = make_mock_s2_raster(shape=(50, 50), crs="EPSG:4326")
    with pytest.raises(GridMismatchError, match="not on the same grid"):
        adapter.combine_with_s1(s1_raster, s2_diff_crs)

    # Mismatch 2: Different Shape
    s2_diff_shape = make_mock_s2_raster(shape=(60, 60), crs="EPSG:32645")
    with pytest.raises(GridMismatchError, match="not on the same grid"):
        adapter.combine_with_s1(s1_raster, s2_diff_shape)

    # Mismatch 3: Different Transform
    s2_diff_transform = make_mock_s2_raster(
        shape=(50, 50),
        crs="EPSG:32645",
        transform=(10.0, 0, 999999.0, 0, -10.0, 888888.0),
    )
    with pytest.raises(GridMismatchError, match="not on the same grid"):
        adapter.combine_with_s1(s1_raster, s2_diff_transform)


# ==============================================================================
# 10. Real Local Trishuli Data Smoke Test (Without Model Inference)
# ==============================================================================

def test_real_local_trishuli_s2_preparation():
    """Verify adapter can ingest real local Trishuli S1 and S2 files without running inference."""
    s1_path = Path("data/raw/dev/s1_after_20260828.tif")
    s2_path = Path("data/raw/dev/s2_after_20260827_l1c.tif")

    if not s1_path.is_file() or not s2_path.is_file():
        pytest.skip("Local Trishuli dev files not present on disk.")

    cfg = Sentinel2AcquisitionConfig(max_temporal_separation_days=2.0)
    adapter = Sentinel2PreparationAdapter.create_production_adapter(config=cfg)

    # 1. Prepare S2 raster
    s2_res = adapter.prepare_s2_raster(s2_path)
    assert s2_res.s2_raster.shape == (2778, 2108)
    assert s2_res.s2_raster.count == 13
    assert s2_res.s2_raster.crs == "EPSG:32645"
    assert s2_res.s2_raster.band_names == list(SENTINEL_2_CHANNELS)
    assert "B10" in s2_res.s2_raster.band_names

    # Value range check
    assert s2_res.diagnostics.product_level == "Level-1C"
    assert "B10" in s2_res.diagnostics.value_ranges
    assert s2_res.diagnostics.nodata_statistics["finite_fraction"] > 0.99

    # 2. Combine with S1
    tensor_15ch, diag = adapter.combine_with_s1(s1_path, s2_res)
    assert tensor_15ch.shape == (2778, 2108, 15)
    assert tensor_15ch.dtype == np.float32
    assert diag.delta_t_days is not None and np.isclose(diag.delta_t_days, 1.31, atol=0.05)
    assert diag.s1_acquisition_date == "2026-08-28T12:21:41.762029Z"
    assert diag.s2_acquisition_date == "2026-08-27T04:56:59.024000Z"
