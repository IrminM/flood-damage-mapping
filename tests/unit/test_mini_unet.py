"""Tests for S1-MiniUNet-v1 architecture, parameter counts, and forward passes."""
import pytest
import torch

from floodmap.ai.models.mini_unet import S1MiniUNetV1


def test_mini_unet_architecture_and_param_count():
    model = S1MiniUNetV1(in_channels=2, out_channels=1, base_channels=16)
    num_params = sum(p.numel() for p in model.parameters())
    # Baseline mini unet with base=16 has ~1.93M parameters
    assert 1_500_000 < num_params < 2_500_000, f"Unexpected parameter count: {num_params}"


def test_mini_unet_forward_pass_shape():
    model = S1MiniUNetV1(in_channels=2, out_channels=1, base_channels=16)
    model.eval()

    # Standard tile size 256x256
    x = torch.randn(1, 2, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 256, 256)


def test_mini_unet_different_batch_and_spatial_sizes():
    model = S1MiniUNetV1(in_channels=2, out_channels=1, base_channels=16)
    model.eval()

    # Smaller batch/spatial dimensions
    x = torch.randn(2, 2, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 64, 64)

    # Sigmoid maps logits to [0, 1]
    probs = torch.sigmoid(out)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_mini_unet_configurable_channels():
    # 3 in channels, 2 out channels
    model = S1MiniUNetV1(in_channels=3, out_channels=2, base_channels=8)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2, 32, 32)
