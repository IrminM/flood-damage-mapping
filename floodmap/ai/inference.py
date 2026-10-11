"""Tiled window inference runner for satellite flood segmentation.

Evaluates arbitrary scene dimensions via sliding-window overlapping patches
with smooth border blending to prevent tile seam discontinuities.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from ..core.georaster import GeoRaster


def _generate_weight_window(tile_size: int, blend_mode: str = "ramp") -> np.ndarray:
    """Generate 2D spatial blending weights for a tile."""
    if blend_mode == "none":
        return np.ones((tile_size, tile_size), dtype=np.float32)

    # 1D triangular / ramp window
    ramp = np.ones(tile_size, dtype=np.float32)
    margin = tile_size // 4
    for i in range(margin):
        w = float(i + 1) / float(margin + 1)
        ramp[i] = w
        ramp[tile_size - 1 - i] = w
    window = np.outer(ramp, ramp).astype(np.float32)
    return np.maximum(window, 1e-4)


class TiledInferenceRunner:
    """Runs sliding-window inference over full scenes using PyTorch models.

    Parameters:
        tile_size: Square tile dimension in pixels (default 256x256).
        overlap: Overlap in pixels between adjacent tiles (default 32 px).
        blend_mode: Blending method ('ramp' or 'none').
        device: 'cpu' or 'cuda' (defaults to automatic selection).
    """

    def __init__(
        self,
        tile_size: int = 256,
        overlap: int = 32,
        blend_mode: str = "ramp",
        device: Optional[str] = None,
    ):
        if overlap >= tile_size:
            raise ValueError(f"Overlap ({overlap}) must be strictly less than tile_size ({tile_size})")
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.stride = self.tile_size - self.overlap
        self.blend_mode = blend_mode
        self.device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
        self.weight_window = _generate_weight_window(self.tile_size, self.blend_mode)

    def predict(
        self,
        model: Any,
        tensor: Any,
        reference_raster: GeoRaster,
        valid_mask: np.ndarray,
    ) -> GeoRaster:
        """Run tiled inference and produce a continuous probability GeoRaster.

        Parameters:
            model: PyTorch segmentation model producing 1-channel logits.
            tensor: Input tensor of shape (1, C, H, W).
            reference_raster: Source GeoRaster for CRS and spatial transform.
            valid_mask: Boolean array (H, W) indicating observable AOI pixels.

        Returns:
            GeoRaster: (1, H, W) float32 probabilities in [0.0, 1.0], NaN for nodata.
        """
        model = model.to(self.device)
        model.eval()

        _, _, H, W = tensor.shape

        # Calculate padding needed to cover the scene with tile_size
        pad_h = max(0, self.tile_size - H) if H < self.tile_size else (self.stride - (H - self.tile_size) % self.stride) % self.stride
        pad_w = max(0, self.tile_size - W) if W < self.tile_size else (self.stride - (W - self.tile_size) % self.stride) % self.stride

        if pad_h > 0 or pad_w > 0:
            padded_tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect" if min(H, W) > 1 else "constant")
        else:
            padded_tensor = tensor

        _, _, p_H, p_W = padded_tensor.shape

        prob_accum = np.zeros((p_H, p_W), dtype=np.float32)
        weight_accum = np.zeros((p_H, p_W), dtype=np.float32)

        y_steps = range(0, p_H - self.tile_size + 1, self.stride)
        x_steps = range(0, p_W - self.tile_size + 1, self.stride)

        with torch.no_grad():
            for y in y_steps:
                for x in x_steps:
                    patch = padded_tensor[:, :, y : y + self.tile_size, x : x + self.tile_size].to(self.device)
                    logits = model(patch)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)

                    prob_accum[y : y + self.tile_size, x : x + self.tile_size] += probs * self.weight_window
                    weight_accum[y : y + self.tile_size, x : x + self.tile_size] += self.weight_window

        # Normalize overlapping tiles
        weight_accum = np.maximum(weight_accum, 1e-6)
        full_prob = prob_accum / weight_accum

        # Crop back to original dimensions
        full_prob = full_prob[:H, :W]

        # Mask invalid pixels to NaN
        full_prob[~valid_mask] = np.nan

        out_data = full_prob[np.newaxis, :, :].astype(np.float32)
        return reference_raster.with_data(
            out_data,
            band_names=["water_prob"],
            nodata=np.nan,
            metadata={**reference_raster.metadata, "units": "probability"},
        )


class MultimodalTiledInferenceRunner:
    """Sliding-window inference runner for multimodal 15-channel scenes.

    Supports arbitrary scene spatial dimensions via tiled inference with overlap
    and smooth 2D ramp blending. Edge tiles are padded where necessary and
    probability outputs are stitched back to original scene dimensions and georeferencing.

    NOTE: This sliding-window tiling and stitching logic is the project's operational
    integration runner, distinct from the teammate's single top-left crop training snippet.
    """

    def __init__(
        self,
        tile_size: int = 256,
        overlap: int = 32,
        blend_mode: str = "ramp",
    ):
        if overlap >= tile_size:
            raise ValueError(f"Overlap ({overlap}) must be strictly less than tile_size ({tile_size})")
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.stride = self.tile_size - self.overlap
        self.blend_mode = blend_mode
        self.weight_window = _generate_weight_window(self.tile_size, self.blend_mode)

    def predict(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        scene_hwc: np.ndarray,
        reference_raster: GeoRaster,
        valid_mask: Optional[np.ndarray] = None,
    ) -> GeoRaster:
        """Run tiled inference over a full scene tensor (H, W, 15) and produce a GeoRaster.

        Parameters:
            predict_fn: Callable taking batch array (B, 256, 256, 15) and returning probability (B, 256, 256) or (B, 256, 256, 1).
            scene_hwc: Full scene array of shape (H, W, 15) in channels-last order.
            reference_raster: Source GeoRaster for CRS and spatial transform.
            valid_mask: Optional boolean array (H, W) indicating observable pixels.

        Returns:
            GeoRaster: (1, H, W) float32 probabilities in [0.0, 1.0], NaN for nodata.
        """
        if scene_hwc.ndim != 3:
            raise ValueError(f"Expected 3D scene array (H, W, C), got shape {scene_hwc.shape}")

        H, W, C = scene_hwc.shape
        if C != 15:
            raise ValueError(f"Expected 15 channels, got {C}")

        pad_h = max(0, self.tile_size - H) if H < self.tile_size else (self.stride - (H - self.tile_size) % self.stride) % self.stride
        pad_w = max(0, self.tile_size - W) if W < self.tile_size else (self.stride - (W - self.tile_size) % self.stride) % self.stride

        if pad_h > 0 or pad_w > 0:
            padded_scene = np.pad(
                scene_hwc,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="edge" if min(H, W) > 1 else "constant",
            )
        else:
            padded_scene = scene_hwc

        p_H, p_W, _ = padded_scene.shape
        prob_accum = np.zeros((p_H, p_W), dtype=np.float32)
        weight_accum = np.zeros((p_H, p_W), dtype=np.float32)

        y_steps = list(range(0, p_H - self.tile_size + 1, self.stride))
        x_steps = list(range(0, p_W - self.tile_size + 1, self.stride))

        for y in y_steps:
            for x in x_steps:
                patch = padded_scene[y : y + self.tile_size, x : x + self.tile_size, :]
                batch = np.expand_dims(patch, axis=0)
                pred = predict_fn(batch)
                tile_prob = np.squeeze(pred).astype(np.float32)

                prob_accum[y : y + self.tile_size, x : x + self.tile_size] += tile_prob * self.weight_window
                weight_accum[y : y + self.tile_size, x : x + self.tile_size] += self.weight_window

        weight_accum = np.maximum(weight_accum, 1e-6)
        full_prob = prob_accum / weight_accum
        full_prob = full_prob[:H, :W]

        if valid_mask is not None:
            full_prob[~valid_mask] = np.nan

        out_data = full_prob[np.newaxis, :, :].astype(np.float32)
        return reference_raster.with_data(
            out_data,
            band_names=["water_prob"],
            nodata=np.nan,
            metadata={**reference_raster.metadata, "units": "probability"},
        )
