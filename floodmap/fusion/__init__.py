"""Multimodal flood and change evidence fusion package."""
from .evidence_fusion import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_NAMES,
    STRENGTH_DESCRIPTIONS,
    STRENGTH_NAMES,
    EvidenceCategory,
    EvidenceStrength,
    MultimodalFusionResult,
    fuse_multimodal_evidence,
    vectorize_fused_evidence,
    write_fusion_outputs,
)

__all__ = [
    "EvidenceCategory",
    "EvidenceStrength",
    "CATEGORY_DESCRIPTIONS",
    "CATEGORY_NAMES",
    "STRENGTH_DESCRIPTIONS",
    "STRENGTH_NAMES",
    "MultimodalFusionResult",
    "fuse_multimodal_evidence",
    "vectorize_fused_evidence",
    "write_fusion_outputs",
]
