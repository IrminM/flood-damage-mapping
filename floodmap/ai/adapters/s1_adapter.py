"""Adapter transforming Stage 2 PreprocessedPair into model-ready tensors.

All transformations (channel ordering, decibel clamping, normalization equations)
are strictly parameterized and explicit.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
try:
    import torch
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    torch = None  # type: ignore
    HAS_TORCH = False

from ...core.errors import InputDataError
from ...satellite.preprocess import PreprocessedPair


class S1DualPolAdapter:
    """Prepares dual-pol Sentinel-1 rasters for neural network inference.

    Parameters:
        channels: Expected channel order, e.g. ["VV", "VH"] or ["VH", "VV"].
        db_range: Clamping interval (min_db, max_db) in decibels.
        normalization: Normalization formula ('min_max' scales to [0, 1]).
        nan_fill: Replacement value for nodata / non-finite pixels.
    """

    def __init__(
        self,
        channels: Sequence[str] = ("VV", "VH"),
        db_range: Tuple[float, float] = (-35.0, 0.0),
        normalization: str = "min_max",
        nan_fill: float = 0.0,
    ):
        self.channels = list(channels)
        self.db_min, self.db_max = float(db_range[0]), float(db_range[1])
        if self.db_min >= self.db_max:
            raise ValueError(f"db_range min must be less than max, got {db_range}")
        self.normalization = normalization
        self.nan_fill = nan_fill

    def transform(self, pair: PreprocessedPair) -> Tuple[torch.Tensor, np.ndarray]:
        """Transform post-event S1 backscatter into an inference tensor.

        Returns:
            tensor: torch.Tensor of shape (1, C, H, W), float32.
            valid_mask: np.ndarray bool of shape (H, W).
        """
        raster = pair.after_db
        available = raster.band_names
        for ch in self.channels:
            if ch not in available:
                raise InputDataError(
                    f"Required AI input band {ch!r} not in available after-event bands {available}"
                )

        # Extract selected channels in explicit requested order
        indices = [available.index(ch) for ch in self.channels]
        data = raster.data[indices].astype(np.float32)  # shape (C, H, W)

        valid_mask = pair.valid.copy()

        if self.normalization == "min_max":
            # Normalization formula: clamp(dB, min, max), then scale linearly to [0, 1]
            clamped = np.clip(data, self.db_min, self.db_max)
            norm = (clamped - self.db_min) / (self.db_max - self.db_min)
        else:
            raise ValueError(f"Unsupported normalization formula: {self.normalization!r}")

        # Fill invalid or NaN pixels with deterministic constant
        norm[:, ~valid_mask] = self.nan_fill
        norm[~np.isfinite(norm)] = self.nan_fill

        # Add batch dimension: (1, C, H, W)
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is not installed in this environment. "
                "S1DualPolAdapter.transform requires PyTorch. Please install requirements-ml.txt."
            )
        tensor = torch.from_numpy(norm).unsqueeze(0)
        return tensor, valid_mask
