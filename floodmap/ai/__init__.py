"""AI flood segmentation module (Stage 3, Path B).

Provides the S1-MiniUNet-v1 baseline model, adapter contracts, tiled inference,
and registration for alternative model architectures.
"""
from .interface import AISegmentationResult, AISegmentor
from .models.mini_unet import S1MiniUNetV1
from .segmentor import S1MiniUNetSegmentor

__all__ = ["AISegmentationResult", "AISegmentor", "S1MiniUNetSegmentor", "S1MiniUNetV1"]
