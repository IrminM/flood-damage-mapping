import json
from datetime import date

import pytest
from pydantic import ValidationError

from floodmap.core.contracts import (AOI, AnalysisRequest, EvidenceSource, OutputClass, SceneMetadata,
                                     SourceType)

SQUARE = {"type": "Polygon", "coordinates": [[[85.0, 27.9], [85.1, 27.9], [85.1, 28.0], [85.0, 28.0], [85.0, 27.9]]]}


def test_aoi_valid_and_roundtrip():
    aoi = AOI(geometry=SQUARE, name="test")
    again = AOI.model_validate_json(aoi.model_dump_json())
    assert again == aoi


@pytest.mark.parametrize("geom", [
    {"type": "Point", "coordinates": [85, 27]},
    {"type": "Polygon", "coordinates": []},
    {"type": "Polygon", "coordinates": [[[85, 27], [86, 27], [85, 27]]]},          # too few positions
    {"type": "Polygon", "coordinates": [[[185, 27], [186, 27], [186, 28], [185, 27]]]},  # lon out of range
])
def test_aoi_invalid(geom):
    with pytest.raises(ValidationError):
        AOI(geometry=geom)


def test_request_roundtrip():
    req = AnalysisRequest(aoi=AOI(geometry=SQUARE), event_date=date(2026, 8, 26), before_scene="a.tif")
    data = json.loads(req.model_dump_json())
    assert data["event_date"] == "2026-08-26"
    assert AnalysisRequest.model_validate(data).before_scene.name == "a.tif"


def test_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        AnalysisRequest(aoi=AOI(geometry=SQUARE), event_date="2026-08-26", flood_area=18.4)


def test_scene_metadata_requires_bands(s1_meta_factory):
    with pytest.raises(ValidationError):
        s1_meta_factory("2026-08-20", bands=[])


def test_canonical_codes_are_stable():
    # Downstream modules, saved rasters and the frontend legend rely on these values.
    assert [c.value for c in OutputClass] == [0, 1, 2, 3, 255]
    assert EvidenceSource.CHANGE_DETECTION | EvidenceSource.AI_SEGMENTATION == EvidenceSource.BOTH
    assert SourceType("osm-pre-event") is SourceType.OSM_PRE_EVENT


def test_frontend_assess_contracts():
    from floodmap.core.contracts import (
        FrontendAssessRequest,
        FrontendAssessResponse,
        HazardMetricsSummary,
        ImpactCountsSummary,
        SettlementStatusItem,
    )

    # 1. Request contract
    req = FrontendAssessRequest(area="Trishuli", date="2026-08-26")
    assert req.area == "Trishuli"
    assert req.date == "2026-08-26"
    assert "area" in req.model_dump()

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        FrontendAssessRequest(area="Trishuli", date="2026-08-26", extra="invalid")

    # 2. Settlement item
    settl = SettlementStatusItem(
        id="node/123",
        n="Test Settlement",
        cut=True,
        place_type="hamlet",
        nearest_destination="Town A",
        nearest_hospital="Hospital B",
        status_label="Cut off",
        x=100.5,
        y=200.5,
    )
    assert settl.cut is True
    assert settl.n == "Test Settlement"

    # 3. Counts summary
    counts = ImpactCountsSummary(
        buildings_affected=873,
        buildings_total=35117,
        roads_affected_segments=62,
        roads_total_segments=7188,
        roads_affected_km=12.65,
        roads_total_km=5180.52,
        bridges_affected=29,
        bridges_total=458,
        settlements_cut_off=49,
        settlements_connected=295,
        settlements_total=505,
    )
    assert counts.buildings_affected == 873

    # 4. Hazard metrics
    hazard = HazardMetricsSummary(
        dual_corroborated_valley_water_km2=0.456,
        ai_only_candidate_water_km2=2.8931,
    )
    assert hazard.dual_corroborated_valley_water_km2 == 0.456

    # 5. Full response roundtrip
    resp = FrontendAssessResponse(
        area="Trishuli",
        date="2026-08-26",
        analysis_id="test_run",
        status="completed",
        validation_status="not_validated_on_Trishuli",
        data_mode="REAL",
        floodAreaKm2=0.456,
        debrisAreaKm2=1.0856,
        hazard_metrics=hazard,
        summary=counts,
        settlements=[settl],
        layers={"affected_roads": "/api/layers/affected_roads"},
        scientific_disclaimers=["Exploratory AI model."],
    )
    serialized = resp.model_dump_json()
    deserialized = FrontendAssessResponse.model_validate_json(serialized)
    assert deserialized.area == "Trishuli"
    assert deserialized.summary.buildings_affected == 873
    assert len(deserialized.settlements) == 1
    assert deserialized.settlements[0].cut is True
