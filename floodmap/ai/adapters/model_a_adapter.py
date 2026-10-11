"""Plug-in adapter interface for candidate Model A (best_flood_unet_best.pth).

Model A status per forensic audit:
  - Architecture: Recovered & structurally verified (custom 31M-param U-Net, 118 tensors)
  - Weights checkpoint: best_flood_unet_best.pth
  - Channel order: UNKNOWN (pending empirical verification)
  - Preprocessing / normalization: UNKNOWN (pending empirical verification)
  - Operational safety: BLOCKED / DISABLED for operational inference until contract is proven.

This adapter provides a strict, pluggable interface conforming to `AISegmentor` so that
once exact preprocessing and channel parameters are confirmed, the model can be enabled
without touching downstream fusion or pipeline code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from ...core.errors import FloodmapError
from ...satellite.preprocess import PreprocessedPair
from ...satellite.terrain import TerrainProducts
from ..interface import AISegmentationResult, AISegmentor, register_ai


class ModelAUnverifiedError(FloodmapError):
    """Raised when attempting to execute Model A without a proven preprocessing contract."""
    pass


@dataclass
class ModelAPreprocessingContract:
    """Explicit parameters required to safely run Model A."""
    channels: Sequence[str] = ("VV", "VH")
    db_range: Tuple[float, float] = (-35.0, 0.0)
    normalization: str = "min_max"
    tile_size: int = 256
    overlap: int = 32
    blend_mode: str = "ramp"
    threshold: float = 0.4100  # Reported evaluation threshold from forensic audit
    is_verified: bool = False
    verification_notes: str = (
        "Model A channel order and normalization equations remain unverified. "
        "Do not enable for operational production inference without empirical validation."
    )


def get_model_a_metadata() -> dict[str, Any]:
    """Return the current forensic and operational status of Model A."""
    return {
        "model_name": "model_a_custom_unet",
        "checkpoint_candidate": "best_flood_unet_best.pth",
        "parameter_count": "~31M",
        "tensor_count": 118,
        "architecture_status": "RECOVERED_VERIFIED",
        "channel_order_status": "UNVERIFIED",
        "preprocessing_status": "UNVERIFIED",
        "operational_readiness": "DISABLED_PENDING_PROVENANCE",
        "reported_metrics": {
            "test_dice_f1": 0.6807,
            "test_iou": 0.5160,
            "test_precision": 0.7558,
            "test_recall": 0.6192,
            "calibrated_threshold": 0.4100,
        },
    }


class ModelASegmentor(AISegmentor):
    """Plug-in adapter for Model A conforming to AISegmentor.

    Guards against unsafe unverified inference while providing the exact slot
    for plugging in `best_flood_unet_best.pth`.
    """

    name: str = "model_a_custom_unet"

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        contract: Optional[ModelAPreprocessingContract] = None,
        model: Optional[Any] = None,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.contract = contract or ModelAPreprocessingContract()
        self.model = model
        self.device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")

    def is_operational(self) -> bool:
        """Check if Model A has a proven contract and loaded weights."""
        return self.contract.is_verified and (self.model is not None or self.checkpoint_path is not None)

    def segment(
        self,
        pair: PreprocessedPair,
        terrain: Optional[TerrainProducts] = None,
        s2_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> AISegmentationResult:
        """Run Model A inference.

        Raises ModelAUnverifiedError if contract is not explicitly verified.
        """
        if not self.contract.is_verified:
            meta = get_model_a_metadata()
            raise ModelAUnverifiedError(
                f"Model A ({meta['model_name']}) cannot be executed for operational inference: "
                f"Architecture is {meta['architecture_status']}, but channel order and preprocessing "
                f"are {meta['channel_order_status']}. Operational status: {meta['operational_readiness']}. "
                "Use the baseline model (s1_mini_unet_v1) or provide an explicitly verified contract."
            )

        # When verified, execution delegates to S1DualPolAdapter and TiledInferenceRunner
        from .s1_adapter import S1DualPolAdapter
        from ..inference import TiledInferenceRunner

        adapter = S1DualPolAdapter(
            channels=self.contract.channels,
            db_range=self.contract.db_range,
            normalization=self.contract.normalization,
        )
        runner = TiledInferenceRunner(
            tile_size=self.contract.tile_size,
            overlap=self.contract.overlap,
            blend_mode=self.contract.blend_mode,
            device=self.device,
        )

        tensor, valid_mask = adapter.transform(pair)
        prob_raster = runner.predict(
            model=self.model,
            tensor=tensor,
            reference_raster=pair.after_db,
            valid_mask=valid_mask,
        )
        prob = prob_raster.data[0]
        water_mask = (prob >= self.contract.threshold) & valid_mask

        return AISegmentationResult(
            probability=prob_raster,
            water_mask=water_mask,
            model_name=self.name,
            threshold=self.contract.threshold,
            metadata={
                "contract": self.contract.__dict__,
                "model_status": get_model_a_metadata(),
            },
        )


@register_ai("model_a_custom_unet")
def make_model_a_segmentor(settings: Any) -> ModelASegmentor:
    """Factory creating ModelASegmentor from configuration."""
    return ModelASegmentor()
