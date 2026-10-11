"""Unit tests for TiledInferenceRunner (sliding-window tiling and blending)."""
import numpy as np
import pytest
import rasterio.transform
import torch
import torch.nn as nn

from floodmap.ai.inference import TiledInferenceRunner, _generate_weight_window
from floodmap.core.georaster import GeoRaster


class ConstantModel(nn.Module):
    """Toy model returning constant logits for testing."""
    def __init__(self, val: float = 0.0):
        super().__init__()
        self.val = val

    def forward(self, x):
        B, _, H, W = x.shape
        return torch.full((B, 1, H, W), self.val, dtype=torch.float32, device=x.device)


@pytest.fixture
def ref_raster():
    H, W = 150, 150
    transform = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
    data = np.zeros((1, H, W), dtype=np.float32)
    return GeoRaster(data=data, transform=transform, crs="EPSG:32645", band_names=["VV"])


def test_weight_window():
    w_none = _generate_weight_window(64, blend_mode="none")
    assert w_none.shape == (64, 64)
    assert np.all(w_none == 1.0)

    w_ramp = _generate_weight_window(64, blend_mode="ramp")
    assert w_ramp.shape == (64, 64)
    # Center should be higher than corners
    assert w_ramp[32, 32] > w_ramp[0, 0]
    assert w_ramp.min() >= 1e-4
    assert w_ramp.max() == pytest.approx(1.0)


def test_tiled_inference_smaller_than_tile(ref_raster):
    # Scene is 150x150, tile_size is 256
    runner = TiledInferenceRunner(tile_size=256, overlap=32, blend_mode="ramp", device="cpu")
    model = ConstantModel(val=0.0)  # sigmoid(0.0) = 0.5

    H, W = ref_raster.shape
    tensor = torch.zeros((1, 2, H, W), dtype=torch.float32)
    valid_mask = np.ones((H, W), dtype=bool)
    valid_mask[:10, :10] = False  # top-left corner invalid

    prob_raster = runner.predict(model, tensor, ref_raster, valid_mask)

    assert prob_raster.shape == (H, W)
    assert prob_raster.band_names == ["water_prob"]
    assert prob_raster.crs == ref_raster.crs

    probs = prob_raster.data[0]
    # Valid pixels should all be 0.5 (sigmoid of 0.0)
    np.testing.assert_allclose(probs[valid_mask], 0.5, atol=1e-5)
    # Invalid pixels should be NaN
    assert np.isnan(probs[~valid_mask]).all()


def test_tiled_inference_multiple_tiles():
    # Scene is 200x200, tile_size is 128, overlap 32
    H, W = 200, 200
    transform = (10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    data = np.zeros((1, H, W), dtype=np.float32)
    ref = GeoRaster(data=data, transform=transform, crs="EPSG:32645", band_names=["VV"])

    runner = TiledInferenceRunner(tile_size=128, overlap=32, blend_mode="ramp", device="cpu")
    model = ConstantModel(val=2.1972)  # sigmoid(2.1972) ≈ 0.90

    tensor = torch.randn((1, 2, H, W), dtype=torch.float32)
    valid_mask = np.ones((H, W), dtype=bool)

    prob_raster = runner.predict(model, tensor, ref, valid_mask)
    probs = prob_raster.data[0]

    assert probs.shape == (200, 200)
    # Blending of identical 0.9 outputs should yield uniform 0.9 everywhere
    np.testing.assert_allclose(probs, 0.90, atol=1e-3)


def test_tiled_inference_invalid_overlap():
    with pytest.raises(ValueError, match="must be strictly less than tile_size"):
        TiledInferenceRunner(tile_size=128, overlap=128)
