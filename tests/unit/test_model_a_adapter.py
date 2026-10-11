"""Unit tests for candidate Model A plug-in adapter and safety guards."""
from datetime import date
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from floodmap.ai.adapters.model_a_adapter import (
    ModelAPreprocessingContract,
    ModelASegmentor,
    ModelAUnverifiedError,
    get_model_a_metadata,
)
from floodmap.ai.interface import AISegmentor, get_ai_segmentor
from floodmap.core.config import Settings
from floodmap.core.georaster import GeoRaster
from floodmap.satellite.preprocess import PreprocessedPair


def test_model_a_metadata_audit_preservation():
    """Verify Model A metadata reflects the forensic audit findings accurately."""
    meta = get_model_a_metadata()
    assert meta["model_name"] == "model_a_custom_unet"
    assert meta["checkpoint_candidate"] == "best_flood_unet_best.pth"
    assert meta["tensor_count"] == 118
    assert meta["architecture_status"] == "RECOVERED_VERIFIED"
    assert meta["channel_order_status"] == "UNVERIFIED"
    assert meta["preprocessing_status"] == "UNVERIFIED"
    assert meta["operational_readiness"] == "DISABLED_PENDING_PROVENANCE"
    assert meta["reported_metrics"]["calibrated_threshold"] == 0.4100


def test_model_a_safety_guard_blocks_unverified_inference():
    """Ensure unverified Model A cannot be executed for operational inference."""
    segmentor = ModelASegmentor()
    assert not segmentor.is_operational()

    # Create dummy pair
    data = np.ones((2, 32, 32), dtype=np.float32) * -15.0
    raster = GeoRaster(data=data, transform=(1.0, 0, 0, 0, -1.0, 0), crs="EPSG:32645", band_names=["VV", "VH"])
    valid = np.ones((data.shape[1], data.shape[2]), dtype=bool)
    pair = PreprocessedPair(
        before_db=raster,
        after_db=raster,
        valid=valid,
        aoi_mask=valid,
        nodata_in_aoi=~valid,
        bands=["VV", "VH"],
    )

    with pytest.raises(ModelAUnverifiedError, match="cannot be executed for operational inference"):
        segmentor.segment(pair)


def test_model_a_plug_in_with_verified_contract():
    """Verify that once a verified contract and model are provided, Model A executes cleanly."""
    class MockUNet(nn.Module):
        def forward(self, x):
            # x shape (B, 2, H, W) -> return logits (B, 1, H, W)
            return torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))

    contract = ModelAPreprocessingContract(
        channels=("VV", "VH"),
        db_range=(-35.0, 0.0),
        normalization="min_max",
        tile_size=64,
        overlap=8,
        threshold=0.41,
        is_verified=True,
    )

    segmentor = ModelASegmentor(contract=contract, model=MockUNet())
    assert segmentor.is_operational()

    data = np.ones((2, 64, 64), dtype=np.float32) * -15.0
    raster = GeoRaster(data=data, transform=(1.0, 0, 0, 0, -1.0, 0), crs="EPSG:32645", band_names=["VV", "VH"])
    valid = np.ones((64, 64), dtype=bool)
    pair = PreprocessedPair(
        before_db=raster,
        after_db=raster,
        valid=valid,
        aoi_mask=valid,
        nodata_in_aoi=~valid,
        bands=["VV", "VH"],
    )

    res = segmentor.segment(pair)
    assert res.model_name == "model_a_custom_unet"
    assert res.threshold == 0.41
    assert res.probability.shape == (64, 64)
    assert res.water_mask.shape == (64, 64)


def test_model_a_registry_resolution():
    """Verify Model A is registered in the AI registry."""
    settings = Settings()
    settings.ai.model = "model_a_custom_unet"
    seg = get_ai_segmentor(settings)
    assert isinstance(seg, ModelASegmentor)
    assert not seg.is_operational()
