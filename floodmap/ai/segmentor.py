"""S1-MiniUNet-v1 Segmentor implementation conforming to AISegmentor protocol."""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False
from scipy import ndimage

from ..core.contracts import OutputClass
from ..satellite.preprocess import PreprocessedPair
from ..satellite.terrain import TerrainProducts
from .adapters.s1_adapter import S1DualPolAdapter
from .inference import TiledInferenceRunner
from .interface import AISegmentationResult, AISegmentor, register_ai
from .models.mini_unet import S1MiniUNetV1


class S1MiniUNetSegmentor(AISegmentor):
    """Stage 3 S1-MiniUNet-v1 AI Flood Water Segmentor.

    Parameters:
        model: PyTorch model instance (defaults to freshly initialized S1MiniUNetV1).
        channels: Radar band input order, e.g. ["VV", "VH"].
        db_range: Clamping interval (min_db, max_db).
        tile_size: Patch size in pixels (default 256).
        overlap: Tile overlap in pixels (default 32).
        blend_mode: Blending method ('ramp' or 'none').
        threshold: Initial decision threshold (default 0.50; uncalibrated).
        water_max_slope_deg: Maximum slope angle for calm water (default 15.0 deg).
        water_max_hand_m: Maximum Height Above Nearest Drainage for water (default 25.0 m).
        min_area_m2: Minimum connected flood area to retain (default 2000.0 m²).
        device: 'cpu' or 'cuda'.
    """

    name: str = "s1_mini_unet_v1"

    def __init__(
        self,
        model: Optional[Any] = None,
        channels: Sequence[str] = ("VV", "VH"),
        db_range: Tuple[float, float] = (-35.0, 0.0),
        tile_size: int = 256,
        overlap: int = 32,
        blend_mode: str = "ramp",
        threshold: float = 0.50,
        water_max_slope_deg: float = 15.0,
        water_max_hand_m: float = 25.0,
        min_area_m2: float = 2000.0,
        device: Optional[str] = None,
    ):
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is not installed in this environment. "
                "S1MiniUNetSegmentor requires PyTorch. Please install requirements-ml.txt."
            )
        self.model = model or S1MiniUNetV1(in_channels=len(channels), out_channels=1)
        self.channels = list(channels)
        self.db_range = db_range
        self.threshold = float(threshold)
        self.water_max_slope_deg = float(water_max_slope_deg)
        self.water_max_hand_m = float(water_max_hand_m)
        self.min_area_m2 = float(min_area_m2)

        self.adapter = S1DualPolAdapter(
            channels=self.channels,
            db_range=self.db_range,
            normalization="min_max",
        )
        self.runner = TiledInferenceRunner(
            tile_size=tile_size,
            overlap=overlap,
            blend_mode=blend_mode,
            device=device,
        )

    def segment(
        self,
        pair: PreprocessedPair,
        terrain: Optional[TerrainProducts] = None,
        s2_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> AISegmentationResult:
        """Run inference on the preprocessed pair and return an AISegmentationResult."""
        tensor, valid_mask = self.adapter.transform(pair)

        # 1. Tiled forward pass -> continuous probability surface
        prob_raster = self.runner.predict(
            model=self.model,
            tensor=tensor,
            reference_raster=pair.after_db,
            valid_mask=valid_mask,
        )
        prob = prob_raster.data[0]  # shape (H, W)

        # 2. Probability thresholding (using initial/default threshold)
        water_cand = (prob >= self.threshold) & valid_mask

        # 3. Optional physical terrain gating
        if terrain is not None:
            slope = terrain.slope_deg.data[0]
            hand = terrain.hand_m.data[0]
            slope_ok = ~np.isfinite(slope) | (slope <= self.water_max_slope_deg)
            hand_ok = ~np.isfinite(hand) | (hand <= self.water_max_hand_m)
            water_cand = water_cand & slope_ok & hand_ok

        # 4. Blob filter
        px_res = abs(pair.after_db.transform[0])
        px_area = px_res * px_res
        min_pixels = int(np.ceil(self.min_area_m2 / max(px_area, 1.0)))

        if min_pixels > 1 and water_cand.any():
            labeled, n_features = ndimage.label(water_cand)
            if n_features > 0:
                counts = np.bincount(labeled.ravel())
                keep = counts >= min_pixels
                keep[0] = False
                water_mask = keep[labeled]
            else:
                water_mask = water_cand
        else:
            water_mask = water_cand

        water_px = int(water_mask.sum())
        water_km2 = round(float(water_px * px_area / 1e6), 4)
        mean_p = float(np.nanmean(prob[valid_mask])) if valid_mask.any() else 0.0

        metadata = {
            "model_architecture": "S1MiniUNetV1",
            "channels": self.channels,
            "db_range": list(self.db_range),
            "normalization": "min_max",
            "tile_size": self.runner.tile_size,
            "overlap": self.runner.overlap,
            "blend_mode": self.runner.blend_mode,
            "threshold_status": "initial_uncalibrated",
        }
        stats = {
            "threshold": self.threshold,
            "water_pixels": water_px,
            "water_km2": water_km2,
            "mean_probability": round(mean_p, 4),
        }

        return AISegmentationResult(
            probability=prob_raster,
            water_mask=water_mask,
            model_name=self.name,
            threshold=self.threshold,
            metadata=metadata,
            stats=stats,
        )


@register_ai("s1_mini_unet_v1")
def make_s1_mini_unet(settings: Any) -> S1MiniUNetSegmentor:
    """Instantiate S1MiniUNetSegmentor from validated Settings."""
    cfg = settings.ai.s1_mini_unet_v1

    model = S1MiniUNetV1(in_channels=len(cfg.channels), out_channels=1)
    if hasattr(settings.ai, "models") and isinstance(settings.ai.models, dict):
        baseline_info = settings.ai.models.get("baseline_unet")
        if baseline_info and isinstance(baseline_info, dict):
            w_path = baseline_info.get("weights")
            if w_path:
                resolved = settings.resolve_path(w_path)
                if resolved.exists():
                    sd = torch.load(resolved, map_location="cpu")
                    model.load_state_dict(sd)

    return S1MiniUNetSegmentor(
        model=model,
        channels=cfg.channels,
        db_range=tuple(cfg.db_range),
        tile_size=cfg.tile_size,
        overlap=cfg.overlap,
        blend_mode=cfg.blend_mode,
        threshold=cfg.threshold,
        water_max_slope_deg=cfg.water_max_slope_deg,
        water_max_hand_m=cfg.water_max_hand_m,
        min_area_m2=cfg.min_area_m2,
    )

