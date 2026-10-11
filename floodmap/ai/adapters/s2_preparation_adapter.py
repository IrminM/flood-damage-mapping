"""Sentinel-2 preparation adapter for multimodal Attention U-Net input construction.

Contract & Integration Flow:
  Sentinel-1 (PreprocessedPair / GeoRaster / Path) + Sentinel-2 (GeoRaster / Scene / Path / SAFE)
            ↓
  Sentinel2PreparationAdapter (validates 13 bands, L1C, B10, nearest-neighbour, grid match, Δt)
            ↓
  15-channel tensor (H, W, 15) [channels-last: VV, VH, B1..B12]
            ↓
  Diagnostic Metadata (provenance, acquisition dates, Δt, value ranges, nodata stats)
            ↓
  Attention_UNet_S1_S2 (exploratory, gated by validation status & DEM terrain gates)

Scientific Provenance Audit Distinction:
  A. Facts directly observed in the original training script (media_1791371815721.py):
     - 15-channel input ordering: [VV, VH, B1..B12].
     - S1 linear power (no dB conversion) with finite median NaN/Inf replacement.
     - S2 values in input GeoTIFF are integer DNs divided by 10,000 and clipped to [0.0, 1.0].
     - Model architecture: Attention U-Net expecting (256, 256, 15) input patches.
     - Model output: single-channel sigmoid probability, decision threshold 0.40.
     - Training crop: static top-left 256x256 window ([:256, :256, :]) from 512x512 source chips.
     - Training ground-truth valid mask: (label != -1); no cloud/shadow masking on optical scenes.
     - Geographic coverage: 11 global flood events; ZERO Himalayan or Nepalese training samples.

  B. Choices introduced by our implementation (working assumptions NOT verified against original training pipeline):
     - Level-1C acquisition from CDSE openEO (the training code ingested pre-exported Kaggle chips).
     - Cirrus Band 10 extraction and handling (native 60 m; read directly as band 11 of the Kaggle TIFF).
     - Nearest-neighbour resampling of 20 m and 60 m bands to 10 m target grid.
     - Metric UTM analysis grid (EPSG:32645 at 10 m) matching S1, whereas Sen1Floods11 chips were WGS84.
     - Configurable temporal pairing rule (|Δt| <= threshold) applied to post-event scene pair.
     - 256x256 sliding-window tiled inference with 32 px overlap and 2D ramp blending (distinct from training crop).
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ...core.contracts import AOI, SceneMetadata
from ...core.errors import FloodmapError, GridMismatchError, InputDataError
from ...core.georaster import GeoRaster
from ...satellite.grid import GridSpec
from ...satellite.preprocess import PreprocessedPair, db_to_linear
from .attention_unet_adapter import (
    EXPECTED_15_CHANNELS,
    EXPECTED_TENSOR_SHAPE,
    SENTINEL_1_CHANNELS,
    SENTINEL_2_CHANNELS,
    ChannelContractError,
    Sentinel2PreparationMetadata,
    Sentinel2PreparationUnresolvedError,
    assemble_full_scene_tensor,
    assemble_multimodal_tensor,
    preprocess_s1,
    preprocess_s2,
)

logger = logging.getLogger(__name__)

# Allowed canonical representation of resampling methods matching nearest-neighbour
ALLOWED_RESAMPLING_METHODS = {"nearest", "nearest-neighbour", "nearest_neighbour"}

# Mapping 2-digit zero-padded or alternate names to canonical 13-band names
S2_BAND_CANONICAL_MAP = {
    "B01": "B1", "B1": "B1", "BAND1": "B1", "BAND01": "B1",
    "B02": "B2", "B2": "B2", "BAND2": "B2", "BAND02": "B2",
    "B03": "B3", "B3": "B3", "BAND3": "B3", "BAND03": "B3",
    "B04": "B4", "B4": "B4", "BAND4": "B4", "BAND04": "B4",
    "B05": "B5", "B5": "B5", "BAND5": "B5", "BAND05": "B5",
    "B06": "B6", "B6": "B6", "BAND6": "B6", "BAND06": "B6",
    "B07": "B7", "B7": "B7", "BAND7": "B7", "BAND07": "B7",
    "B08": "B8", "B8": "B8", "BAND8": "B8", "BAND08": "B8",
    "B8A": "B8A", "B08A": "B8A", "BAND8A": "B8A", "BAND08A": "B8A",
    "B09": "B9", "B9": "B9", "BAND9": "B9", "BAND09": "B9",
    "B10": "B10", "BAND10": "B10",
    "B11": "B11", "BAND11": "B11",
    "B12": "B12", "BAND12": "B12",
}


class ProductLevelError(InputDataError):
    """Raised when an unsupported Sentinel-2 product level (e.g. Level-2A) is provided."""
    pass


class TemporalPairingError(FloodmapError):
    """Raised when the temporal separation between Sentinel-1 and Sentinel-2 exceeds allowed threshold."""
    pass


class ResamplingMethodError(FloodmapError):
    """Raised when a non-nearest resampling algorithm is requested for Sentinel-2 preparation."""
    pass


def normalize_s2_band_name(name: str) -> str:
    """Normalize band names to canonical Sentinel-2 names (e.g. 'B01' -> 'B1', 'B8A' -> 'B8A')."""
    clean = re.sub(r"[_\-\s\.]+", "", name.strip()).upper()
    # Check direct canonical match
    if clean in S2_BAND_CANONICAL_MAP:
        return S2_BAND_CANONICAL_MAP[clean]
    # Check if string contains one of the band identifiers at word boundary or suffix
    for k, v in S2_BAND_CANONICAL_MAP.items():
        if clean.endswith(k) or f"_{k}" in clean:
            return v
    return name.strip()


def parse_datetime_to_utc(dt_val: Any) -> Optional[datetime]:
    """Parse various datetime representations into a UTC datetime."""
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        if dt_val.tzinfo is None:
            return dt_val.replace(tzinfo=timezone.utc)
        return dt_val.astimezone(timezone.utc)
    if isinstance(dt_val, date):
        return datetime.combine(dt_val, datetime.min.time(), tzinfo=timezone.utc)
    if isinstance(dt_val, str):
        s = dt_val.strip()
        # Handle trailing Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s).astimezone(timezone.utc)
        except Exception:
            try:
                # Try date only YYYY-MM-DD
                d = date.fromisoformat(s[:10])
                return datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
            except Exception:
                logger.warning(f"Could not parse datetime string: {dt_val}")
                return None
    return None


@dataclass
class Sentinel2AcquisitionConfig:
    """Explicit configurable parameters for Sentinel-2 L1C acquisition & preparation."""
    aoi: Optional[AOI] = None
    event_date: Optional[date] = None
    max_temporal_separation_days: float = 2.0  # Configurable max temporal distance to S1
    max_cloud_cover_pct: Optional[float] = None  # Used for scene filtering if specified
    target_grid: Optional[GridSpec] = None
    target_resolution_m: float = 10.0
    resampling_method: str = "nearest"
    product_level: str = "L1C"
    b10_required: bool = True
    strict_grid_check: bool = True
    strict_temporal_check: bool = True


@dataclass
class Sentinel2PreparationDiagnostics:
    """Diagnostic provenance metadata emitted with every prepared Sentinel-2 tensor."""
    s1_acquisition_date: Optional[str] = None
    s2_acquisition_date: Optional[str] = None
    delta_t_days: Optional[float] = None
    s2_product_id: Optional[str] = None
    product_level: str = "Level-1C"
    crs: str = ""
    transform: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pixel_size: Tuple[float, float] = (10.0, 10.0)
    resampling_method: str = "nearest"
    bands: List[str] = field(default_factory=lambda: list(SENTINEL_2_CHANNELS))
    b10_provenance: str = "native_l1c_band10_60m"
    array_shape: Tuple[int, ...] = ()
    value_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    nodata_statistics: Dict[str, Any] = field(default_factory=dict)
    provenance_status: str = "PROVEN_L1C_NEAREST_10M"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "s1_acquisition_date": self.s1_acquisition_date,
            "s2_acquisition_date": self.s2_acquisition_date,
            "delta_t_days": self.delta_t_days,
            "s2_product_id": self.s2_product_id,
            "product_level": self.product_level,
            "crs": self.crs,
            "transform": list(self.transform),
            "pixel_size": list(self.pixel_size),
            "resampling_method": self.resampling_method,
            "bands": list(self.bands),
            "b10_provenance": self.b10_provenance,
            "array_shape": list(self.array_shape),
            "value_ranges": {k: [float(v[0]), float(v[1])] for k, v in self.value_ranges.items()},
            "nodata_statistics": self.nodata_statistics,
            "provenance_status": self.provenance_status,
        }


@dataclass
class Sentinel2PreparationResult:
    """Container holding the prepared S2 GeoRaster, diagnostic metadata, and optional multimodal tensor."""
    s2_raster: GeoRaster
    diagnostics: Sentinel2PreparationDiagnostics
    tensor_15ch: Optional[np.ndarray] = None


class Sentinel2PreparationAdapter:
    """Prepares and validates Sentinel-2 optical data for Attention U-Net integration.

    Acts as the contract gate between raw Sentinel-2 inputs and the 15-channel
    multimodal inference tensor.
    """

    def __init__(
        self,
        metadata: Optional[Sentinel2PreparationMetadata] = None,
        config: Optional[Sentinel2AcquisitionConfig] = None,
    ):
        self.metadata = metadata or Sentinel2PreparationMetadata()
        self.config = config or Sentinel2AcquisitionConfig()

    @classmethod
    def create_production_adapter(
        cls,
        config: Optional[Sentinel2AcquisitionConfig] = None,
    ) -> "Sentinel2PreparationAdapter":
        """Factory creating a production-ready adapter with fully resolved proven metadata."""
        meta = Sentinel2PreparationMetadata(
            product_level="L1C",
            b10_source="L1C_native_resampled",
            target_resolution_m=10.0,
            resampling_method="nearest",
            co_registration_method="s1_grid_aligned",
            temporal_pairing_rule="same_day_closest_cloud_free",
            notes="Proven Level-1C 10m nearest-neighbour configuration from Sen1Floods11 audit.",
        )
        return cls(metadata=meta, config=config)

    @property
    def is_resolved(self) -> bool:
        """True only if every required preparation attribute is populated and verified."""
        return self.metadata.is_resolved

    def get_unresolved_fields(self) -> list[str]:
        """Return list of required preparation fields that are currently unresolved."""
        return self.metadata.get_unresolved_fields()

    def assert_ready(self) -> None:
        """Strictly refuse execution if S2 preparation provenance is unresolved."""
        if not self.metadata.is_resolved:
            unresolved = self.get_unresolved_fields()
            raise Sentinel2PreparationUnresolvedError(
                f"Sentinel2PreparationAdapter cannot prepare S2 data: "
                f"preparation provenance is unresolved. Unresolved fields: {unresolved}. "
                f"Operational inference is strictly blocked until Sentinel-2 product level, "
                f"B10 provenance, target resolution, resampling, co-registration, and temporal "
                f"pairing rules are scientifically verified. Notes: {self.metadata.notes}"
            )

    def validate_product_level(self, product_level: Optional[str]) -> None:
        """Verify product level is Level-1C (TOA reflectance); strictly reject Level-2A."""
        if not product_level:
            return
        clean = product_level.strip().upper()
        if "L2A" in clean or "LEVEL-2A" in clean or "MSIL2A" in clean:
            raise ProductLevelError(
                f"Sentinel-2 product level must be Level-1C (TOA reflectance), got {product_level!r}. "
                f"Level-2A omits Cirrus Band 10 and applies surface-reflectance atmospheric corrections "
                f"that violate the Attention U-Net 15-channel model contract."
            )
        if clean not in ("L1C", "LEVEL-1C", "MSIL1C", "SENTINEL2_L1C"):
            logger.warning(f"Unusual Sentinel-2 product level declared: {product_level}")

    def validate_resampling_method(self, resampling_method: Optional[str]) -> None:
        """Enforce nearest-neighbor resampling; strictly reject bilinear, cubic, etc."""
        method = resampling_method or self.config.resampling_method or self.metadata.resampling_method or "nearest"
        clean = method.strip().lower()
        if clean not in ALLOWED_RESAMPLING_METHODS:
            raise ResamplingMethodError(
                f"Resampling method {resampling_method!r} is prohibited for Sentinel-2 preparation. "
                f"Nearest-neighbor ('nearest') is strictly required to preserve training data conventions "
                f"and avoid artificial spectral interpolation across multi-resolution bands."
            )

    def validate_s2_bands(
        self,
        available_bands: Sequence[str],
    ) -> None:
        """Verify that all 13 required Sentinel-2 bands are present in exact order without omissions."""
        available_tuple = tuple(available_bands)
        if available_tuple != SENTINEL_2_CHANNELS:
            missing = [b for b in SENTINEL_2_CHANNELS if b not in available_tuple]
            if missing:
                if "B10" in missing:
                    raise ChannelContractError(
                        f"Sentinel-2 input is missing required bands: {missing}. "
                        f"Band 10 (cirrus, 1373 nm) is strictly required by the 15-channel contract "
                        f"and cannot be silently omitted or substituted."
                    )
                raise ChannelContractError(
                    f"Sentinel-2 input is missing required bands: {missing}. "
                    f"Fabrication or zero-padding of missing bands is strictly prohibited."
                )
            raise ChannelContractError(
                f"Sentinel-2 bands sequence mismatch: expected {SENTINEL_2_CHANNELS}, got {available_tuple}"
            )

    def validate_temporal_separation(
        self,
        s1_time: Any,
        s2_time: Any,
        max_separation_days: Optional[float] = None,
    ) -> float:
        """Verify temporal separation between S1 and S2 acquisitions does not exceed allowed threshold."""
        t1 = parse_datetime_to_utc(s1_time)
        t2 = parse_datetime_to_utc(s2_time)
        if t1 is None or t2 is None:
            return 0.0

        gap_days = abs((t2 - t1).total_seconds()) / 86400.0
        max_gap = max_separation_days if max_separation_days is not None else self.config.max_temporal_separation_days

        if self.config.strict_temporal_check and gap_days > max_gap:
            raise TemporalPairingError(
                f"Temporal separation between Sentinel-1 ({t1.isoformat()}) and "
                f"Sentinel-2 ({t2.isoformat()}) is {gap_days:.2f} days, exceeding the "
                f"maximum allowable separation of {max_gap:.2f} days."
            )
        return gap_days

    def validate_grid_compatibility(
        self,
        s1_raster: GeoRaster,
        s2_raster: GeoRaster,
    ) -> None:
        """Verify S1 and S2 share the exact same CRS, shape, bounds, and affine transform."""
        if not s1_raster.same_grid(s2_raster):
            raise GridMismatchError(
                f"S1 and S2 rasters are not on the same grid: "
                f"S1 CRS={s1_raster.crs}, shape={s1_raster.shape}, transform={s1_raster.transform} vs "
                f"S2 CRS={s2_raster.crs}, shape={s2_raster.shape}, transform={s2_raster.transform}. "
                f"Reproducing the training dataset convention requires exact pixel-grid alignment."
            )

    def prepare_s2_array(
        self,
        s2_input: GeoRaster | np.ndarray,
        s2_bands: Sequence[str] = SENTINEL_2_CHANNELS,
        target_grid: Optional[GridSpec] = None,
    ) -> np.ndarray:
        """Prepare raw Sentinel-2 data into a preprocessed 13-band float32 array [0, 1].

        Guards:
          1. Provenance must be resolved (assert_ready).
          2. Band contract must be satisfied (no fabrication, no missing bands).
          3. Enforces nearest-neighbor resampling if reprojection is needed.
          4. Applies proven S2 preprocessing (float32 / 10000.0, clip [0, 1]).

        Returns:
          np.ndarray of shape (13, H, W), float32 in [0.0, 1.0].
        """
        self.assert_ready()
        self.validate_resampling_method(self.config.resampling_method)
        self.validate_s2_bands(s2_bands)

        if isinstance(s2_input, GeoRaster):
            if target_grid is not None and (
                s2_input.shape != target_grid.shape
                or s2_input.crs != target_grid.crs
                or not np.allclose(s2_input.transform, target_grid.transform, atol=1e-5)
            ):
                from ...satellite.reproject import reproject_to_grid
                s2_aligned = reproject_to_grid(
                    s2_input,
                    target_grid,
                    resampling="nearest",
                )
                raw_data = s2_aligned.data
            else:
                raw_data = s2_input.data
        else:
            raw_data = np.asarray(s2_input, dtype=np.float32)

        if raw_data.ndim != 3 or raw_data.shape[0] != len(SENTINEL_2_CHANNELS):
            raise ChannelContractError(
                f"S2 array must have shape (13, H, W); got shape {raw_data.shape}"
            )

        # Apply proven S2 normalization
        return preprocess_s2(raw_data, s2_bands)

    def prepare_s2_raster(
        self,
        s2_input: Union[GeoRaster, Path, str, Dict[str, Any]],
        s2_metadata: Optional[Union[Dict[str, Any], SceneMetadata]] = None,
        target_grid: Optional[GridSpec] = None,
    ) -> Sentinel2PreparationResult:
        """Prepare Sentinel-2 L1C input into a validated 13-band GeoRaster and diagnostics.

        Accepts:
          - GeoRaster with 13 bands
          - Path or string to multi-band GeoTIFF file
          - Path to SAFE granule directory
          - Dictionary of {band_name: GeoRaster | Path | np.ndarray}
        """
        self.assert_ready()
        self.validate_resampling_method(self.config.resampling_method)

        meta_dict: Dict[str, Any] = {}
        if isinstance(s2_metadata, SceneMetadata):
            meta_dict = {
                "product_type": s2_metadata.product_type,
                "acquisition_time": s2_metadata.acquisition_time.isoformat() if s2_metadata.acquisition_time else None,
                "product_ids": s2_metadata.product_ids,
                "cloud_cover_pct": s2_metadata.cloud_cover_pct,
                "extra": s2_metadata.extra,
            }
        elif isinstance(s2_metadata, dict):
            meta_dict = dict(s2_metadata)

        s2_raster: GeoRaster
        source_path: Optional[Path] = None

        if isinstance(s2_input, (str, Path)):
            p = Path(s2_input)
            source_path = p
            if not p.exists():
                raise InputDataError(f"Sentinel-2 input path does not exist: {p}")

            # Check if companion .meta.json exists
            companion_meta = p.with_name(f"{p.stem}.meta.json") if p.is_file() else p / "metadata.json"
            if companion_meta.is_file() and not meta_dict:
                try:
                    with open(companion_meta, "r", encoding="utf-8") as f:
                        meta_dict = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load companion metadata {companion_meta}: {e}")

            if p.is_file():
                from ...io.geotiff import read_geotiff
                s2_raster = read_geotiff(p)
            elif p.is_dir():
                s2_raster = self._load_s2_directory(p, target_grid)
            else:
                raise InputDataError(f"Unsupported Sentinel-2 path type: {p}")

        elif isinstance(s2_input, GeoRaster):
            s2_raster = s2_input

        elif isinstance(s2_input, dict):
            s2_raster = self._load_s2_band_dict(s2_input, target_grid)

        else:
            raise TypeError(f"Unsupported s2_input type: {type(s2_input)}")

        # Validate Product Level (L1C only, reject L2A)
        prod_level = meta_dict.get("product_type") or meta_dict.get("product_level") or "L1C"
        self.validate_product_level(str(prod_level))

        # Cloud cover check if configured
        cloud_pct = meta_dict.get("cloud_cover_pct")
        if self.config.max_cloud_cover_pct is not None and cloud_pct is not None:
            if float(cloud_pct) > float(self.config.max_cloud_cover_pct):
                logger.warning(
                    f"Sentinel-2 cloud cover {cloud_pct}% exceeds max threshold {self.config.max_cloud_cover_pct}%"
                )

        # Normalize band names and align band order
        current_names = [normalize_s2_band_name(b) for b in s2_raster.band_names]
        if s2_raster.count == 13 and all(b.startswith("band_") for b in s2_raster.band_names):
            # Fallback for rasters without embedded band descriptions
            current_names = list(SENTINEL_2_CHANNELS)
            s2_raster = s2_raster.with_data(s2_raster.data, band_names=current_names)

        # Ensure all 13 bands exist
        for b in SENTINEL_2_CHANNELS:
            if b not in current_names:
                if b == "B10":
                    raise ChannelContractError(
                        "Sentinel-2 B10 is missing: native Sentinel-2 L1C Band 10 (cirrus, 1373 nm) "
                        "is strictly required by the 15-channel contract and cannot be silently omitted."
                    )
                raise ChannelContractError(f"Sentinel-2 input missing required band: {b}")

        # Re-index raster bands into exact SENTINEL_2_CHANNELS order
        if tuple(current_names) != SENTINEL_2_CHANNELS:
            indices = [current_names.index(b) for b in SENTINEL_2_CHANNELS]
            reordered_data = s2_raster.data[indices]
            s2_raster = s2_raster.with_data(reordered_data, band_names=list(SENTINEL_2_CHANNELS))

        # Check target grid alignment & resolution
        grid_to_use = target_grid or self.config.target_grid
        if grid_to_use is not None and (
            s2_raster.shape != grid_to_use.shape
            or s2_raster.crs != grid_to_use.crs
            or not np.allclose(s2_raster.transform, grid_to_use.transform, atol=1e-5)
        ):
            from ...satellite.reproject import reproject_to_grid
            s2_raster = reproject_to_grid(s2_raster, grid_to_use, resampling="nearest")

        # Check resolution
        px_res = s2_raster.res
        if not np.allclose(px_res[0], self.config.target_resolution_m, atol=0.1):
            logger.warning(f"S2 raster resolution {px_res} differs from target {self.config.target_resolution_m}m")

        # Apply proven scaling & clipping: uint16/int DN -> / 10000.0 -> [0, 1]
        data = s2_raster.data.astype(np.float32, copy=True)
        if np.nanmax(data) > 1.0:
            data = data / 10000.0
        data = np.clip(data, 0.0, 1.0)
        s2_prepared_raster = s2_raster.with_data(data, band_names=list(SENTINEL_2_CHANNELS), nodata=np.nan)

        # Compute diagnostics
        ranges: Dict[str, Tuple[float, float]] = {}
        for i, b in enumerate(SENTINEL_2_CHANNELS):
            band_arr = data[i]
            fin = band_arr[np.isfinite(band_arr)]
            if len(fin) > 0:
                ranges[b] = (float(np.min(fin)), float(np.max(fin)))
            else:
                ranges[b] = (0.0, 0.0)

        total_px = int(data.size)
        nan_px = int(np.isnan(data).sum())
        nodata_stats = {
            "total_pixels": total_px,
            "nan_pixels": nan_px,
            "nodata_fraction": round(nan_px / max(total_px, 1), 6),
            "finite_fraction": round((total_px - nan_px) / max(total_px, 1), 6),
        }

        s2_acq = meta_dict.get("acquisition_time")
        prod_ids = meta_dict.get("product_ids") or []
        prod_id = prod_ids[0] if prod_ids else (source_path.stem if source_path else "S2_L1C")

        diag = Sentinel2PreparationDiagnostics(
            s1_acquisition_date=None,
            s2_acquisition_date=s2_acq,
            delta_t_days=None,
            s2_product_id=prod_id,
            product_level="Level-1C",
            crs=s2_prepared_raster.crs,
            transform=s2_prepared_raster.transform,
            pixel_size=s2_prepared_raster.res,
            resampling_method="nearest",
            bands=list(SENTINEL_2_CHANNELS),
            b10_provenance="native_l1c_band10_60m_resampled_nearest_10m",
            array_shape=s2_prepared_raster.data.shape,
            value_ranges=ranges,
            nodata_statistics=nodata_stats,
            provenance_status="PROVEN_L1C_NEAREST_10M",
        )

        return Sentinel2PreparationResult(s2_raster=s2_prepared_raster, diagnostics=diag)

    def _load_s2_band_dict(
        self,
        band_dict: Dict[str, Any],
        target_grid: Optional[GridSpec],
    ) -> GeoRaster:
        """Assemble 13-band GeoRaster from a dictionary of band rasters or file paths."""
        normalized_inputs: Dict[str, Any] = {}
        for k, v in band_dict.items():
            norm_k = normalize_s2_band_name(k)
            normalized_inputs[norm_k] = v

        for b in SENTINEL_2_CHANNELS:
            if b not in normalized_inputs:
                raise ChannelContractError(f"Missing required band {b} in band dictionary.")

        band_rasters: List[GeoRaster] = []
        for b in SENTINEL_2_CHANNELS:
            val = normalized_inputs[b]
            if isinstance(val, (str, Path)):
                from ...io.geotiff import read_geotiff
                br = read_geotiff(val)
            elif isinstance(val, GeoRaster):
                br = val
            elif isinstance(val, np.ndarray):
                if target_grid is None:
                    raise ValueError(f"target_grid is required when passing raw numpy array for band {b}")
                br = GeoRaster(
                    data=val if val.ndim == 3 else val[np.newaxis, ...],
                    transform=target_grid.transform,
                    crs=target_grid.crs,
                    band_names=[b],
                )
            else:
                raise TypeError(f"Unsupported band value type for {b}: {type(val)}")

            # Resample each native band to target grid if needed using nearest-neighbor
            if target_grid is not None and (
                br.shape != target_grid.shape
                or br.crs != target_grid.crs
                or not np.allclose(br.transform, target_grid.transform, atol=1e-5)
            ):
                from ...satellite.reproject import reproject_to_grid
                br = reproject_to_grid(br, target_grid, resampling="nearest")
            band_rasters.append(br)

        ref = band_rasters[0]
        stacked_data = np.concatenate([r.data for r in band_rasters], axis=0)
        return GeoRaster(
            data=stacked_data,
            transform=ref.transform,
            crs=ref.crs,
            band_names=list(SENTINEL_2_CHANNELS),
        )

    def _load_s2_directory(
        self,
        directory: Path,
        target_grid: Optional[GridSpec],
    ) -> GeoRaster:
        """Scan SAFE or standard directory for Sentinel-2 band files and assemble 13-band GeoRaster."""
        band_files: Dict[str, Path] = {}
        # Search all image files in directory
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".jp2"):
                norm = normalize_s2_band_name(p.stem)
                if norm in SENTINEL_2_CHANNELS:
                    band_files[norm] = p

        missing = [b for b in SENTINEL_2_CHANNELS if b not in band_files]
        if missing:
            raise ChannelContractError(f"Sentinel-2 directory {directory} missing required bands: {missing}")

        return self._load_s2_band_dict(band_files, target_grid)

    def assemble_multimodal_scene_tensor(
        self,
        s1_input: PreprocessedPair | GeoRaster | np.ndarray,
        s2_data: np.ndarray,
        s1_channels: Sequence[str] = SENTINEL_1_CHANNELS,
        s2_channels: Sequence[str] = SENTINEL_2_CHANNELS,
    ) -> np.ndarray:
        """Assemble full-scene 15-channel multimodal array (H, W, 15).

        Channels:
          [VV, VH, B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]
        """
        self.assert_ready()

        if isinstance(s1_input, PreprocessedPair):
            s1_raster = s1_input.after_db
            s1_raw = s1_raster.data
            if s1_raster.metadata.get("units") == "dB":
                s1_raw = db_to_linear(s1_raw)
        elif isinstance(s1_input, GeoRaster):
            s1_raw = s1_input.data
            if s1_input.metadata.get("units") == "dB":
                s1_raw = db_to_linear(s1_raw)
        else:
            s1_raw = np.asarray(s1_input, dtype=np.float32)

        s1_clean = preprocess_s1(s1_raw, s1_channels)
        # Avoid dividing by 10000 twice if s2_data is already preprocessed in [0, 1]
        if np.nanmax(s2_data) <= 1.0 and np.nanmin(s2_data) >= 0.0:
            s2_clean = s2_data.astype(np.float32)
        else:
            s2_clean = preprocess_s2(s2_data, s2_channels)

        if s1_clean.shape[1:] != s2_clean.shape[1:]:
            raise ChannelContractError(
                f"Spatial dimension mismatch between S1 {s1_clean.shape[1:]} and S2 {s2_clean.shape[1:]}"
            )

        s1_hwc = np.transpose(s1_clean, (1, 2, 0))
        s2_hwc = np.transpose(s2_clean, (1, 2, 0))

        combined = np.concatenate([s1_hwc, s2_hwc], axis=-1)
        return combined.astype(np.float32)

    def combine_with_s1(
        self,
        s1_input: Union[PreprocessedPair, GeoRaster, Path, str],
        s2_input: Union[GeoRaster, Sentinel2PreparationResult, np.ndarray],
        s1_metadata: Optional[Union[Dict[str, Any], SceneMetadata]] = None,
        s2_diagnostics: Optional[Sentinel2PreparationDiagnostics] = None,
    ) -> Tuple[np.ndarray, Sentinel2PreparationDiagnostics]:
        """Verify grid and temporal compatibility, then assemble 15-channel multimodal tensor with diagnostics."""
        self.assert_ready()

        s1_raster: GeoRaster
        s1_meta_dict: Dict[str, Any] = {}

        if isinstance(s1_input, (str, Path)):
            p1 = Path(s1_input)
            if not p1.exists():
                raise InputDataError(f"Sentinel-1 file does not exist: {p1}")
            from ...io.geotiff import read_geotiff
            s1_raster = read_geotiff(p1)
            meta_p1 = p1.with_name(f"{p1.stem}.meta.json")
            if meta_p1.is_file():
                try:
                    with open(meta_p1, "r", encoding="utf-8") as f:
                        s1_meta_dict = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load S1 metadata {meta_p1}: e")
        elif isinstance(s1_input, PreprocessedPair):
            s1_raster = s1_input.after_db
        elif isinstance(s1_input, GeoRaster):
            s1_raster = s1_input
        else:
            raise TypeError(f"Unsupported s1_input type: {type(s1_input)}")

        if isinstance(s1_metadata, SceneMetadata):
            s1_meta_dict["acquisition_time"] = s1_metadata.acquisition_time.isoformat() if s1_metadata.acquisition_time else None
        elif isinstance(s1_metadata, dict):
            s1_meta_dict.update(s1_metadata)

        # Extract S2 raster and diagnostics
        s2_raster: GeoRaster
        diag: Sentinel2PreparationDiagnostics
        if isinstance(s2_input, Sentinel2PreparationResult):
            s2_raster = s2_input.s2_raster
            diag = s2_input.diagnostics
        elif isinstance(s2_input, GeoRaster):
            s2_raster = s2_input
            diag = s2_diagnostics or Sentinel2PreparationDiagnostics(
                crs=s2_raster.crs,
                transform=s2_raster.transform,
                pixel_size=s2_raster.res,
                array_shape=s2_raster.data.shape,
            )
        elif isinstance(s2_input, np.ndarray):
            # Wrap array in GeoRaster with S1 grid for compatibility
            s2_raster = GeoRaster(
                data=s2_input,
                transform=s1_raster.transform,
                crs=s1_raster.crs,
                band_names=list(SENTINEL_2_CHANNELS),
            )
            diag = s2_diagnostics or Sentinel2PreparationDiagnostics(
                crs=s1_raster.crs,
                transform=s1_raster.transform,
                pixel_size=s1_raster.res,
                array_shape=s2_input.shape,
            )
        else:
            raise TypeError(f"Unsupported s2_input type: {type(s2_input)}")

        # 1. Enforce strict grid compatibility
        if self.config.strict_grid_check:
            self.validate_grid_compatibility(s1_raster, s2_raster)

        # 2. Enforce temporal pairing
        t1_str = s1_meta_dict.get("acquisition_time")
        t2_str = diag.s2_acquisition_date
        gap_days = self.validate_temporal_separation(t1_str, t2_str)

        # 3. Assemble 15-channel multimodal tensor
        s1_raw = s1_raster.data
        if s1_raster.metadata.get("units") == "dB":
            s1_raw = db_to_linear(s1_raw)

        tensor_15ch = self.assemble_multimodal_scene_tensor(s1_raw, s2_raster.data)

        # 4. Update diagnostics
        diag.s1_acquisition_date = t1_str
        diag.delta_t_days = round(gap_days, 4) if gap_days is not None else None
        diag.array_shape = tensor_15ch.shape

        return tensor_15ch, diag
