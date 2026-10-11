"""Registry of disaster case studies for multi-event flood intelligence.

Ensures real, non-cosmetic multi-event handling:
- Each case study maps to an explicit disaster event, geographic extent, and sensor baseline.
- 'ready' cases have verified outputs generated from genuine local satellite, DEM, and OSM data.
- 'data_not_prepared' cases clearly state missing inputs (Sentinel-1 pairs, DEM, OSM) and
  NEVER return or leak cached data from another AOI.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CaseStudy(BaseModel):
    """Specification of a disaster event case study."""
    id: str = Field(description="Unique machine-readable case identifier")
    title: str = Field(description="Display title of the case study")
    region: str = Field(description="Region or administrative district")
    country: str = Field(description="Country")
    event_date: str = Field(description="Central event date (YYYY-MM-DD)")
    status: str = Field(description="'ready' or 'data_not_prepared'")
    bbox: list[float] = Field(description="[west, south, east, north] in WGS84")
    center: list[float] = Field(description="[latitude, longitude] for map centering")
    default_zoom: int = Field(default=11, description="Initial Leaflet map zoom level")
    validation_status: str = Field(description="Rigorous AI model validation status for this case")
    sensors_summary: str = Field(description="Sensors and baseline inputs summary")
    run_id: Optional[str] = Field(default=None, description="Active run ID in data/runs if processed")
    description: str = Field(description="Contextual event summary")
    data_requirements: Optional[dict[str, str]] = Field(default=None, description="Required inputs if data not prepared")
    independent_reference: Optional[str] = Field(default=None, description="Independent ground truth / reference dataset")


# Official Registry of Disaster Case Studies
_CASE_STUDIES: dict[str, CaseStudy] = {
    "trishuli_basin": CaseStudy(
        id="trishuli_basin",
        title="Trishuli River Basin & Upper Gorge",
        region="Rasuwa District & Trishuli Corridor",
        country="Nepal",
        event_date="2026-08-26",
        status="ready",
        bbox=[85.17, 27.95, 85.36, 28.18],
        center=[27.97, 85.20],
        default_zoom=11,
        validation_status="not_validated_on_Trishuli",
        sensors_summary="S1 (Aug 16/28) · S2 (Aug 27) · GLO-30 DEM · OSM",
        run_id="smoke_stage3_real",
        description=(
            "Catastrophic monsoon flood and debris flow corridor in high-relief Himalayan terrain. "
            "Multi-evidence pipeline evaluated with Sentinel-1 SAR change detection and exploratory Attention U-Net."
        ),
        independent_reference="EMSR927 (mass movement grading post-hoc comparison) and HOT Observed Flood Extent.",
    ),
    "emilia_romagna": CaseStudy(
        id="emilia_romagna",
        title="Emilia-Romagna River Basins",
        region="Ravenna / Forlì / Faenza Corridor",
        country="Italy",
        event_date="2023-05-17",
        status="data_not_prepared",
        bbox=[11.70, 44.15, 12.35, 44.55],
        center=[44.35, 12.02],
        default_zoom=11,
        validation_status="uncalibrated_external_aoi",
        sensors_summary="S1 IW (Track 44) · S2B (Tile 32TPQ) · GLO-30 DEM · OSM",
        run_id=None,
        description=(
            "Severe May 2023 riverine flooding with 23 river breaches across Emilia-Romagna agricultural plains "
            "and Apennine foothills."
        ),
        data_requirements={
            "sentinel1": "Missing Sentinel-1 IW GRD before/after pair (Track 44: 2023-05-09 / 2023-05-21) from CDSE.",
            "dem": "Missing Copernicus GLO-30 DEM tiles (N44_E011, N44_E012) from AWS S3.",
            "osm": "Missing pre-event historical OSM snapshot (2023-05-01) from ohsome API.",
            "pipeline_step": "Run 'python scripts/run_stage3.py' with the acquired inputs.",
        },
        independent_reference="Copernicus EMS Rapid Mapping Activation EMSR659 (Flood Delineation DEL packages).",
    ),
    "valencia_dana": CaseStudy(
        id="valencia_dana",
        title="Valencia DANA Flash Flooding",
        region="Turia River Basin & L'Horta Sud",
        country="Spain",
        event_date="2024-10-29",
        status="data_not_prepared",
        bbox=[-0.60, 39.25, -0.20, 39.60],
        center=[39.42, -0.40],
        default_zoom=11,
        validation_status="uncalibrated_external_aoi",
        sensors_summary="S1 IW (Track 8) · S2 (Oct 31) · GLO-30 DEM · OSM",
        run_id=None,
        description=(
            "Catastrophic October 2024 flash flooding and ravine inundation across Rambla del Poyo "
            "and the Valencia metropolitan periphery."
        ),
        data_requirements={
            "sentinel1": "Missing Sentinel-1 IW GRD before/after pair (Track 8: 2024-10-23 / 2024-10-30) from CDSE.",
            "dem": "Missing Copernicus GLO-30 DEM tile (N39_W001) from AWS S3.",
            "osm": "Missing pre-event historical OSM snapshot (2024-10-20) from ohsome API.",
            "pipeline_step": "Run 'python scripts/run_stage3.py' with the acquired inputs.",
        },
        independent_reference="Copernicus EMS Rapid Mapping Activation EMSR773 (Flood Delineation DEL packages).",
    ),
}


def list_case_studies() -> list[CaseStudy]:
    """Return all registered case studies."""
    return list(_CASE_STUDIES.values())


def get_case_study(case_id: str) -> Optional[CaseStudy]:
    """Look up a case study by its unique ID."""
    return _CASE_STUDIES.get(case_id.strip().lower())


def resolve_case_study(query: str, default_to_trishuli_if_matched: bool = True) -> Optional[CaseStudy]:
    """Resolve a case study by ID, location keyword, or spatial bounding box."""
    q = query.strip().lower()

    # 1. Exact or normalized ID match
    if q in _CASE_STUDIES:
        return _CASE_STUDIES[q]
    if "trishuli" in q or "bhote" in q or "koshi" in q or "rasuwa" in q or "langtang" in q or "nepal" in q:
        return _CASE_STUDIES["trishuli_basin"]
    if "emilia" in q or "romagna" in q or "italy" in q or "faenza" in q or "forli" in q or "ravenna" in q:
        return _CASE_STUDIES["emilia_romagna"]
    if "valencia" in q or "spain" in q or "dana" in q or "poyo" in q:
        return _CASE_STUDIES["valencia_dana"]

    # 2. Bounding box coordinates: west,south,east,north
    parts = [p.strip() for p in q.replace(";", ",").split(",") if p.strip()]
    if len(parts) == 4:
        try:
            w, s, e, n = [float(p) for p in parts]
            # Check overlap against registered case study bboxes
            for case in _CASE_STUDIES.values():
                cw, cs, ce, cn = case.bbox
                # Check bounding box intersection
                if not (w > ce or e < cw or s > cn or n < cs):
                    return case
        except ValueError:
            pass

    return _CASE_STUDIES["trishuli_basin"] if default_to_trishuli_if_matched and q in ("default", "dev", "smoke", "aoi") else None
