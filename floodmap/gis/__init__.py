"""GIS and Infrastructure Impact Assessment module (Stage 4).

Vectorizes raster flood/change evidence and evaluates spatial intersection
with pre-event OpenStreetMap buildings, roads, bridges, and waterways.
"""
from .impact import (
    assess_bridges,
    assess_buildings,
    assess_roads,
    assess_waterway_context,
)
from .infrastructure_analyzer import (
    BridgeImpactAnalyzer,
    BuildingImpactAnalyzer,
    EvidenceSpatialIndex,
    InfrastructureImpactResult,
    InfrastructureNormalizer,
    OSMLoader,
    RoadImpactAnalyzer,
    analyze_infrastructure_impact,
)
from .vectorize import vectorize_evidence

__all__ = [
    "assess_bridges",
    "assess_buildings",
    "assess_roads",
    "assess_waterway_context",
    "vectorize_evidence",
    "OSMLoader",
    "InfrastructureNormalizer",
    "EvidenceSpatialIndex",
    "BuildingImpactAnalyzer",
    "RoadImpactAnalyzer",
    "BridgeImpactAnalyzer",
    "InfrastructureImpactResult",
    "analyze_infrastructure_impact",
]
