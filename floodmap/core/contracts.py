"""Data contracts shared between modules.

These are the *only* objects passed across module boundaries (satellite → detection →
GIS → network → API). Swapping an implementation (better preprocessing, the teammate's
model, a better connectivity algorithm) must not change these shapes.

Later stages add result contracts (infrastructure, cut-off, analysis result) here.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum, IntEnum, IntFlag
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ------------------------------------------------------------------ enums

class SourceType(str, Enum):
    """Declared origin of an input. Checked by floodmap.core.data_rules."""
    SENTINEL_1 = "sentinel-1"
    SENTINEL_2 = "sentinel-2"
    COPERNICUS_DEM = "copernicus-dem"
    OSM_PRE_EVENT = "osm-pre-event"
    TRAINING_KUROSIWO = "training-kurosiwo"
    TRAINING_SEN1FLOODS11 = "training-sen1floods11"
    SYNTHETIC_TEST = "synthetic-test"


class DataMode(str, Enum):
    """Whether results come from real data or from dev_mocks. Shown on every output."""
    REAL = "REAL"
    SYNTHETIC_TEST = "SYNTHETIC_TEST"


class OutputClass(IntEnum):
    """Canonical per-pixel classes used by everything downstream of detection.

    Any segmentation model (baseline or teammate) is mapped onto these by its adapter.
    DEBRIS_OR_CHANGE is produced by the change-detection path (surface change that is
    not open water). A water-segmentation model must NOT emit it unless it was trained
    and validated on debris labels.
    """
    BACKGROUND = 0
    FLOOD_WATER = 1
    PERMANENT_WATER = 2
    DEBRIS_OR_CHANGE = 3
    NODATA = 255


class EvidenceSource(IntFlag):
    """Per-pixel provenance of the final affected-area layer (bit flags)."""
    NONE = 0
    CHANGE_DETECTION = 1
    AI_SEGMENTATION = 2
    BOTH = 3


class TerrainFlag(IntFlag):
    """Per-pixel terrain conditions derived from the DEM (bit flags, uint8 raster)."""
    NONE = 0
    RADAR_SHADOW = 1      # approx.: slope faces away from the radar steeper than (90° - incidence)
    RADAR_LAYOVER = 2     # approx.: slope faces the radar steeper than the incidence angle
    STEEP_FOR_WATER = 4   # slope > water_max_slope_deg
    HIGH_HAND = 8         # height above nearest drainage > water_max_hand_m
    STEEP_FOR_CHANGE = 16 # slope > change_max_slope_deg
    NO_DEM = 32           # DEM missing/invalid at this pixel


class ExclusionFlag(IntFlag):
    """Why a pixel is not classified in the change-detection output (bit flags, uint8 raster)."""
    NONE = 0
    OUTSIDE_AOI = 1
    NO_DATA = 2           # invalid in before or after image
    RADAR_SHADOW = 4
    RADAR_LAYOVER = 8
    TERRAIN_WATER_GATE = 16   # water candidate rejected by slope/HAND gate
    TERRAIN_CHANGE_GATE = 32  # change candidate rejected by steep-slope gate
    SMALL_BLOB = 64           # removed by minimum-area / opening clean-up


class ChangeContext(IntEnum):
    """Terrain context of each non-water change (class 3) pixel. Labels only; classes are unchanged."""
    NOT_CHANGE = 0
    VALLEY_BOTTOM = 1     # HAND <= valley_max_hand_m and slope <= valley_max_slope_deg
    HILLSIDE = 2          # observable class-3 pixel outside the valley bottom
    UNKNOWN = 3           # class 3 but no DEM / unknown HAND or slope
    NODATA = 255


class AnalysisStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


# ------------------------------------------------------------------ AOI

class AOI(_Model):
    """Area of interest as a GeoJSON Polygon/MultiPolygon geometry in EPSG:4326 (lon, lat)."""
    geometry: dict[str, Any]
    name: Optional[str] = None

    @field_validator("geometry")
    @classmethod
    def _check_geometry(cls, g: dict[str, Any]) -> dict[str, Any]:
        gtype = g.get("type")
        if gtype not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"AOI geometry must be Polygon or MultiPolygon, got {gtype!r}")
        coords = g.get("coordinates")
        if not coords:
            raise ValueError("AOI geometry has no coordinates")
        polys = [coords] if gtype == "Polygon" else coords
        for poly in polys:
            for ring in poly:
                if len(ring) < 4:
                    raise ValueError("AOI ring needs at least 4 positions (closed)")
                for pt in ring:
                    lon, lat = float(pt[0]), float(pt[1])
                    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                        raise ValueError(f"AOI coordinate out of lon/lat range: {pt}")
        return g


# ------------------------------------------------------------------ scenes

class SceneMetadata(_Model):
    """Metadata of one satellite scene. Comes from a sidecar *.meta.json, never from filename guessing."""
    sensor: Literal["S1", "S2"]
    platform: str = Field(description="e.g. sentinel-1a, sentinel-1c, sentinel-2b")
    product_type: str = Field(description="e.g. GRD, L2A")
    acquisition_time: datetime
    bands: list[str]
    units: Literal["linear", "dB", "reflectance", "dimensionless"]
    processing: str = Field(description="Human-readable processing chain, e.g. 'sigma0-ellipsoid, terrain corrected, CDSE openEO'")
    source_type: SourceType
    relative_orbit: Optional[int] = None
    orbit_direction: Optional[Literal["ascending", "descending"]] = None
    product_ids: list[str] = Field(default_factory=list, description="Original product name(s) for traceability")
    cloud_cover_pct: Optional[float] = None
    data_mode: DataMode = DataMode.REAL
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bands")
    @classmethod
    def _bands_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("bands must not be empty")
        return v


class SceneRef(_Model):
    """Pointer to a scene file on disk plus its metadata."""
    path: Path
    metadata: SceneMetadata


class PairCheckResult(_Model):
    """Outcome of validating a before/after pair for change detection."""
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    gap_days: Optional[float] = None
    before_days_from_event: Optional[float] = None
    after_days_from_event: Optional[float] = None


# ------------------------------------------------------------------ change detection summary

class ChangeSummary(_Model):
    """Machine-readable summary of a change-detection run (written to change/summary.json).

    Every number is computed from the output rasters; nothing is entered by hand.
    """
    analysis_id: str
    data_mode: DataMode
    method: str
    event_date: date
    aoi_name: Optional[str] = None
    crs: str
    resolution_m: float
    scenes: dict[str, dict[str, Any]]
    pair_check: PairCheckResult
    aoi_area_km2: float
    observable_area_km2: float
    class_areas_km2: dict[str, float]
    class_labels: dict[str, str]
    class3_context_km2: dict[str, float] = Field(default_factory=dict)
    excluded_area_km2: dict[str, float]
    thresholds: dict[str, Any]
    terrain_applied: bool
    terrain: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    detection_stats: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    interpretation_notes: list[str] = Field(default_factory=list)


class Stage3Summary(ChangeSummary):
    """Machine-readable summary of a Stage 3 fused run (Change Detection + AI Flood Segmentation)."""
    ai_model: str = "s1_mini_unet_v1"
    ai_threshold: float = 0.50
    ai_stats: dict[str, Any] = Field(default_factory=dict)
    ai_metadata: dict[str, Any] = Field(default_factory=dict)
    fusion_mode: str = "union_with_provenance"
    evidence_areas_km2: dict[str, float] = Field(default_factory=dict)
    evidence_labels: dict[str, str] = Field(default_factory=dict)


# ------------------------------------------------------------------ stage 4 infrastructure summary

class BuildingImpactStats(_Model):
    total_count: int = 0
    potentially_affected_count: int = 0
    affected_pct: float = 0.0
    total_footprint_m2: float = 0.0
    affected_footprint_m2: float = 0.0
    by_class: dict[str, int] = Field(default_factory=dict)
    by_evidence: dict[str, int] = Field(default_factory=dict)


class RoadImpactStats(_Model):
    total_length_km: float = 0.0
    potentially_affected_length_km: float = 0.0
    affected_pct: float = 0.0
    total_segments: int = 0
    potentially_affected_segments: int = 0
    by_highway_type_km: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_class_km: dict[str, float] = Field(default_factory=dict)
    by_evidence_km: dict[str, float] = Field(default_factory=dict)


class BridgeImpactStats(_Model):
    total_count: int = 0
    potentially_affected_count: int = 0
    affected_pct: float = 0.0
    by_class: dict[str, int] = Field(default_factory=dict)
    by_evidence: dict[str, int] = Field(default_factory=dict)


class WaterwayContextStats(_Model):
    total_waterways_km: float = 0.0
    class3_near_waterway_km2: float = 0.0
    class3_away_from_waterway_km2: float = 0.0
    buffer_distance_m: float = 50.0


class InfrastructureSummary(_Model):
    """Machine-readable summary of Stage 4 infrastructure impact analysis."""
    analysis_id: str
    data_mode: DataMode
    event_date: date
    aoi_name: Optional[str] = None
    osm_snapshot_date: date
    crs: str
    buildings: BuildingImpactStats = Field(default_factory=BuildingImpactStats)
    roads: RoadImpactStats = Field(default_factory=RoadImpactStats)
    bridges: BridgeImpactStats = Field(default_factory=BridgeImpactStats)
    waterways_context: Optional[WaterwayContextStats] = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    interpretation_notes: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ stage 5 connectivity summary

class SettlementStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    NO_BASELINE_ROUTE = "no_baseline_route"
    OUTSIDE_NETWORK = "outside_network"
    INSUFFICIENT_EVIDENCE = "insufficient_network_evidence"


class SettlementConnectivityRecord(_Model):
    settlement_id: str
    settlement_name: Optional[str] = None
    place_type: str = "settlement"
    population: Optional[int] = None
    status: SettlementStatus = SettlementStatus.CONNECTED
    classification: str = "CONNECTED"
    nearest_destination_id: Optional[str] = None
    nearest_destination_name: Optional[str] = None
    destination_type: Optional[str] = None
    baseline_distance_m: Optional[float] = None
    baseline_time_s: Optional[float] = None
    post_disruption_distance_m: Optional[float] = None
    post_disruption_time_s: Optional[float] = None
    alternate_route_distance_m: Optional[float] = None
    detour_distance_m: Optional[float] = None
    connectivity_lost: bool = False
    disrupting_road_ids: list[str] = Field(default_factory=list)
    disrupting_evidence_tiers: list[str] = Field(default_factory=list)
    isolation_reason: Optional[str] = None
    snapped_distance_to_road_m: Optional[float] = None
    nearest_hospital_id: Optional[str] = None
    nearest_hospital_name: Optional[str] = None
    hospital_baseline_distance_m: Optional[float] = None
    hospital_post_disruption_distance_m: Optional[float] = None
    hospital_status: Optional[str] = None
    component_id: Optional[int] = None


class ConnectivitySummary(_Model):
    """Machine-readable summary of Stage 5 road network connectivity & cut-off settlement analysis."""
    analysis_id: str
    data_mode: DataMode
    event_date: date
    osm_snapshot_date: date
    aoi_name: Optional[str] = None
    disruption_mode: str = "remove"
    total_settlements: int = 0
    connected_count: int = 0
    disconnected_count: int = 0
    no_baseline_route_count: int = 0
    outside_network_count: int = 0
    insufficient_evidence_count: int = 0
    alternate_route_count: int = 0
    disconnected_components_count: int = 0
    isolated_components_count: int = 0
    affected_road_segments_count: int = 0
    destination_types: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    interpretation_notes: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ stage 6 downstream path

class DownstreamSettlementRecord(_Model):
    order: int
    settlement_id: str
    settlement_name: Optional[str] = None
    place_type: str = "settlement"
    distance_to_path_m: float
    distance_along_path_m: float
    distance_along_path_km: float


class DownstreamPathSummary(_Model):
    """Machine-readable summary of Stage 6 DEM downstream flood path tracing."""
    analysis_id: str
    data_mode: DataMode
    event_date: date
    aoi_name: Optional[str] = None
    upstream_input_coords: tuple[float, float] = Field(description="(lon, lat)")
    upstream_snapped_coords: tuple[float, float] = Field(description="(lon, lat)")
    snapping_distance_m: float = 0.0
    start_elevation_m: float = 0.0
    end_elevation_m: float = 0.0
    total_elevation_drop_m: float = 0.0
    path_length_km: float = 0.0
    path_vertices_count: int = 0
    termination_reason: str = "max_length_reached"
    settlements_count: int = 0
    settlements: list[DownstreamSettlementRecord] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    methodology_and_limitations: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ request

class AnalysisRequest(_Model):
    """What a user/judge submits: area + date (+ optionally explicit scene files)."""
    aoi: AOI
    event_date: date
    before_scene: Optional[Path] = None
    after_scene: Optional[Path] = None
    s2_before_scene: Optional[Path] = None
    s2_after_scene: Optional[Path] = None
    dem: Optional[Path] = None
    osm_dir: Optional[Path] = None
    name: Optional[str] = None
    options: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ API & Orchestration Contracts

class DataAvailabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_PROVIDED = "NOT_PROVIDED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    SYNTHETIC = "SYNTHETIC"


class DataAvailabilitySummary(_Model):
    """Explicitly documents data availability per source, preventing false zeros."""
    sentinel1: DataAvailabilityStatus = DataAvailabilityStatus.AVAILABLE
    sentinel2: DataAvailabilityStatus = DataAvailabilityStatus.NOT_PROVIDED
    copernicus_dem: DataAvailabilityStatus = DataAvailabilityStatus.AVAILABLE
    osm_preevent: DataAvailabilityStatus = DataAvailabilityStatus.UNAVAILABLE
    ai_segmentation: DataAvailabilityStatus = DataAvailabilityStatus.AVAILABLE
    ai_model_name: str = "s1_mini_unet_v1"
    ai_model_role: str = "Baseline development model (uncalibrated, not teammate's Model A)"
    osm_unavailable_reason: Optional[str] = None
    ai_unavailable_reason: Optional[str] = None


class AnalysisRunConfig(_Model):
    """Optional per-run configuration overrides."""
    ai_model: Optional[str] = None
    ai_threshold: Optional[float] = None
    change_threshold_db: Optional[float] = None
    disruption_mode: Optional[str] = "remove"
    sink_handling: Optional[str] = "fill"
    downstream_max_length_km: Optional[float] = 50.0
    downstream_proximity_radius_m: Optional[float] = 1000.0


class AnalysisSubmitRequest(_Model):
    """Payload accepted by POST /api/v1/analyses."""
    aoi: Any = Field(description="GeoJSON geometry/Feature, bbox [min_lon, min_lat, max_lon, max_lat], or path")
    event_date: date = Field(description="Disaster event date (YYYY-MM-DD)")
    upstream_point: Optional[Any] = Field(default=None, description="Optional upstream point: [lon, lat], GeoJSON Point, or 'lon,lat'")
    before_scene: Optional[str] = Field(default=None, description="Path or identifier for pre-event S1 raster")
    after_scene: Optional[str] = Field(default=None, description="Path or identifier for post-event S1 raster")
    dem: Optional[str] = Field(default=None, description="Path to Copernicus DEM GeoTIFF")
    osm_dir: Optional[str] = Field(default=None, description="Path to pre-event OSM directory")
    name: Optional[str] = Field(default=None, description="Optional human-readable analysis name")
    config: Optional[AnalysisRunConfig] = Field(default=None, description="Optional pipeline overrides")
    sync: bool = Field(default=False, description="Run synchronously and wait for full completion")


class LayerMetadata(_Model):
    """Describes an output layer available for map display or download."""
    layer_id: str
    name: str
    layer_type: str  # raster_geotiff, vector_geojson, quicklook_png
    file_path: str
    url: str
    description: str
    data_available: bool = True


class FullAnalysisResult(_Model):
    """Unified machine-readable response aggregating Stages 1 through 6."""
    analysis_id: str
    status: AnalysisStatus
    data_mode: DataMode
    event_date: date
    aoi_name: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    data_availability: DataAvailabilitySummary
    stages_executed: dict[str, str] = Field(default_factory=dict)
    stage3_summary: Optional[Stage3Summary] = None
    stage4_summary: Optional[InfrastructureSummary] = None
    stage5_summary: Optional[ConnectivitySummary] = None
    stage6_summary: Optional[DownstreamPathSummary] = None
    available_layers: dict[str, LayerMetadata] = Field(default_factory=dict)
    provenance: Optional[dict[str, Any]] = None
    uncertainty_and_limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SituationReport(_Model):
    """Operational situation report synthesizing all disaster evidence."""
    analysis_id: str
    event_date: date
    data_mode: DataMode
    generated_at: datetime
    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)
    flood_findings: dict[str, Any] = Field(default_factory=dict)
    infrastructure_findings: dict[str, Any] = Field(default_factory=dict)
    connectivity_findings: dict[str, Any] = Field(default_factory=dict)
    downstream_findings: Optional[dict[str, Any]] = None
    data_provenance_highlights: dict[str, Any] = Field(default_factory=dict)
    uncertainty_and_limitations: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ frontend assess adapter contracts

class FrontendAssessRequest(_Model):
    """Payload accepted by POST /api/assess from floodintel frontend."""
    area: str = Field(description="Area name, case identifier, or bounding box: west,south,east,north")
    date: str = Field(description="Flood / disaster event date (YYYY-MM-DD)")
    case_id: Optional[str] = Field(default=None, description="Optional explicit case study identifier")


class SettlementStatusItem(_Model):
    """Settlement status record for frontend table and map display."""
    id: str
    n: str
    cut: bool
    place_type: str = "settlement"
    nearest_destination: Optional[str] = None
    nearest_hospital: Optional[str] = None
    status_label: str = "Connected"
    x: Optional[float] = None
    y: Optional[float] = None


class ImpactCountsSummary(_Model):
    """Pre-computed impact summary counts avoiding large GeoJSON transmission for counts."""
    buildings_affected: int = 0
    buildings_total: int = 0
    roads_affected_segments: int = 0
    roads_total_segments: int = 0
    roads_affected_km: float = 0.0
    roads_total_km: float = 0.0
    bridges_affected: int = 0
    bridges_total: int = 0
    settlements_cut_off: int = 0
    settlements_connected: int = 0
    settlements_total: int = 0


class HazardMetricsSummary(_Model):
    """Granular multi-sensor hazard breakdown."""
    dual_corroborated_valley_water_km2: float = 0.0
    ai_only_candidate_water_km2: float = 0.0
    sar_new_open_water_km2: float = 0.0
    permanent_water_km2: float = 0.0
    sar_valley_bottom_change_km2: float = 0.0
    sar_hillside_change_km2: float = 0.0
    sar_total_non_water_change_km2: float = 0.0
    exploratory_terrain_gated_ai_water_km2: float = 0.0


class FrontendAssessResponse(_Model):
    """Stable response contract tailored for floodintel frontend."""
    area: str
    date: str
    analysis_id: str
    status: str = "completed"
    validation_status: str = "not_validated_on_Trishuli"
    data_mode: str = "REAL"
    case_id: Optional[str] = None
    bbox: Optional[list[float]] = None
    center: Optional[list[float]] = None

    # Compatibility fields for frontend findings cards
    floodAreaKm2: Optional[float] = None
    debrisAreaKm2: Optional[float] = None

    # Granular summary counts and metrics (None if data not prepared)
    hazard_metrics: Optional[HazardMetricsSummary] = None
    summary: Optional[ImpactCountsSummary] = None
    downstream_summary: Optional[dict[str, Any]] = None
    settlements: list[SettlementStatusItem] = Field(default_factory=list)
    buildings: list[Any] = Field(default_factory=list)
    roads: list[Any] = Field(default_factory=list)
    bridges: list[Any] = Field(default_factory=list)
    layers: dict[str, str] = Field(default_factory=dict)

    # Status explanations and missing data requirements
    message: Optional[str] = None
    data_requirements: Optional[dict[str, str]] = None

    # Scientific limitations and caveats
    scientific_disclaimers: list[str] = Field(default_factory=list)

    # Defensive SVG mock coordinates for unadapted frontend map drawing
    river: Optional[str] = None
    floodW: Optional[int] = None
    debris: list[Any] = Field(default_factory=list)
    town: Optional[list[int]] = None
    hospital: Optional[list[int]] = None


