"""Unit tests for S1DualPolAdapter (Stage 3 Preprocessing & Tensor conversion)."""
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio.transform
import torch

from floodmap.ai.adapters.s1_adapter import S1DualPolAdapter
from floodmap.core.errors import InputDataError
from floodmap.core.georaster import GeoRaster
from floodmap.satellite.preprocess import PreprocessedPair


@pytest.fixture
def dummy_pair():
    # Shape: 2 bands (VV, VH), 10x10 pixels
    H, W = 10, 10
    vv_data = np.full((H, W), -15.0, dtype=np.float32)
    vh_data = np.full((H, W), -22.0, dtype=np.float32)

    # Some out-of-range values to test clamping
    vv_data[0, 0] = -50.0  # below -35.0
    vv_data[0, 1] = 5.0    # above 0.0

    # Some NaN values to test masking
    vv_data[1, 1] = np.nan
    vh_data[1, 1] = np.nan

    data = np.stack([vv_data, vh_data], axis=0)
    transform = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
    raster = GeoRaster(
        data=data,
        transform=transform,
        crs="EPSG:32645",
        band_names=["VV", "VH"],
        nodata=np.nan,
    )

    valid_mask = np.ones((H, W), dtype=bool)
    valid_mask[1, 1] = False
    valid_mask[2, 2] = False  # valid_mask False even if finite

    pair = MagicMock(spec=PreprocessedPair)
    pair.after_db = raster
    pair.valid = valid_mask
    return pair


def test_adapter_channel_order_and_shape(dummy_pair):
    adapter = S1DualPolAdapter(channels=["VV", "VH"], db_range=(-35.0, 0.0))
    tensor, valid = adapter.transform(dummy_pair)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 2, 10, 10)
    assert valid.shape == (10, 10)
    assert valid[1, 1] is np.False_ or not valid[1, 1]


def test_adapter_reversed_channels(dummy_pair):
    adapter_rev = S1DualPolAdapter(channels=["VH", "VV"], db_range=(-35.0, 0.0))
    tensor_rev, _ = adapter_rev.transform(dummy_pair)

    adapter_fwd = S1DualPolAdapter(channels=["VV", "VH"], db_range=(-35.0, 0.0))
    tensor_fwd, _ = adapter_fwd.transform(dummy_pair)

    # Channel 0 in rev should match Channel 1 in fwd
    torch.testing.assert_close(tensor_rev[:, 0], tensor_fwd[:, 1])
    torch.testing.assert_close(tensor_rev[:, 1], tensor_fwd[:, 0])


def test_adapter_clamping_and_normalization_math(dummy_pair):
    adapter = S1DualPolAdapter(channels=["VV", "VH"], db_range=(-35.0, 0.0), normalization="min_max")
    tensor, _ = adapter.transform(dummy_pair)

    arr = tensor.numpy()[0]
    # -50.0 should clamp to -35.0 -> normalized to (-35 - (-35)) / 35 = 0.0
    assert arr[0, 0, 0] == pytest.approx(0.0)

    # 5.0 should clamp to 0.0 -> normalized to (0 - (-35)) / 35 = 1.0
    assert arr[0, 0, 1] == pytest.approx(1.0)

    # -17.5 should normalize to (-17.5 - (-35)) / 35 = 0.5
    # For default -15.0: (-15 - (-35)) / 35 = 20/35 = 0.5714285
    assert arr[0, 5, 5] == pytest.approx(20.0 / 35.0, rel=1e-5)

    # Invalid pixels (e.g. at [2, 2]) should be filled with nan_fill (0.0)
    assert arr[0, 2, 2] == 0.0
    assert not np.isnan(arr).any()


def test_adapter_missing_channel(dummy_pair):
    adapter = S1DualPolAdapter(channels=["HH", "HV"])
    with pytest.raises(InputDataError, match="Required AI input band"):
        adapter.transform(dummy_pair)


def test_adapter_invalid_db_range():
    with pytest.raises(ValueError, match="db_range min must be less than max"):
        S1DualPolAdapter(db_range=(0.0, -35.0))
