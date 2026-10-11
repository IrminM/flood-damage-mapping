"""Unit tests for multi-event disaster case study switching and data isolation."""
import pytest
from fastapi.testclient import TestClient

from floodmap.api.app import app
from floodmap.core.case_studies import (
    get_case_study,
    list_case_studies,
    resolve_case_study,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_case_study_registry_metadata():
    """Verify registered disaster case studies conform to specification."""
    cases = list_case_studies()
    case_ids = [c.id for c in cases]

    assert "trishuli_basin" in case_ids
    assert "emilia_romagna" in case_ids
    assert "valencia_dana" in case_ids

    # 1. Trishuli Basin (Ready baseline)
    trishuli = get_case_study("trishuli_basin")
    assert trishuli is not None
    assert trishuli.status == "ready"
    assert trishuli.validation_status == "not_validated_on_Trishuli"
    assert trishuli.country == "Nepal"
    assert len(trishuli.bbox) == 4
    assert len(trishuli.center) == 2

    # 2. Emilia-Romagna (Data Not Prepared)
    emilia = get_case_study("emilia_romagna")
    assert emilia is not None
    assert emilia.status == "data_not_prepared"
    assert emilia.validation_status == "uncalibrated_external_aoi"
    assert emilia.data_requirements is not None
    assert "sentinel1" in emilia.data_requirements
    assert "dem" in emilia.data_requirements
    assert "osm" in emilia.data_requirements
    assert "pipeline_step" in emilia.data_requirements

    # 3. Valencia DANA (Data Not Prepared)
    valencia = get_case_study("valencia_dana")
    assert valencia is not None
    assert valencia.status == "data_not_prepared"
    assert valencia.validation_status == "uncalibrated_external_aoi"
    assert valencia.data_requirements is not None


def test_case_study_resolution():
    """Verify case study resolution by exact ID, keyword, and spatial bbox."""
    # By ID
    assert resolve_case_study("trishuli_basin").id == "trishuli_basin"
    assert resolve_case_study("emilia_romagna").id == "emilia_romagna"
    assert resolve_case_study("valencia_dana").id == "valencia_dana"

    # By keyword
    assert resolve_case_study("Trishuli River Corridor").id == "trishuli_basin"
    assert resolve_case_study("Rasuwa, Nepal").id == "trishuli_basin"
    assert resolve_case_study("Faenza, Italy").id == "emilia_romagna"
    assert resolve_case_study("Valencia DANA").id == "valencia_dana"

    # By bounding box overlap
    # Trishuli area bbox
    assert resolve_case_study("85.18, 27.96, 85.35, 28.17").id == "trishuli_basin"
    # Emilia-Romagna bbox
    assert resolve_case_study("11.80, 44.20, 12.20, 44.50").id == "emilia_romagna"

    # Unrecognized returns None
    assert resolve_case_study("Miami, Florida", default_to_trishuli_if_matched=False) is None


def test_api_cases_endpoints(client):
    """Verify GET /cases and GET /cases/{case_id} endpoints."""
    # List all cases
    resp = client.get("/api/cases")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 3

    # Specific ready case
    t_resp = client.get("/api/cases/trishuli_basin")
    assert t_resp.status_code == 200
    t_data = t_resp.json()
    assert t_data["id"] == "trishuli_basin"
    assert t_data["status"] == "ready"
    assert t_data["validation_status"] == "not_validated_on_Trishuli"

    # Specific unprepared case
    e_resp = client.get("/api/cases/emilia_romagna")
    assert e_resp.status_code == 200
    e_data = e_resp.json()
    assert e_data["id"] == "emilia_romagna"
    assert e_data["status"] == "data_not_prepared"
    assert "Missing Sentinel-1" in e_data["data_requirements"]["sentinel1"]

    # 404 for unknown case
    bad_resp = client.get("/api/cases/unknown_basin")
    assert bad_resp.status_code == 404


def test_post_assess_case_trishuli_basin(client):
    """Verify POST /api/assess with trishuli_basin returns real verified outputs."""
    payload = {
        "case_id": "trishuli_basin",
        "area": "trishuli_basin",
        "date": "2026-08-26",
    }
    resp = client.post("/api/assess", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["case_id"] == "trishuli_basin"
    assert data["status"] == "completed"
    assert data["validation_status"] == "not_validated_on_Trishuli"
    assert data["data_mode"] == "REAL"
    assert data["bbox"] == [85.17, 27.95, 85.36, 28.18]
    assert data["center"] == [27.97, 85.20]

    # Verified metrics
    assert data["floodAreaKm2"] == 0.4560
    assert data["summary"]["buildings_total"] == 35117
    assert data["summary"]["settlements_cut_off"] == 49
    assert len(data["settlements"]) > 0
    assert "flood_evidence" in data["layers"]


def test_post_assess_case_data_not_prepared_isolation(client):
    """Verify unprepared case NEVER leaks cached Trishuli data or layers."""
    payload = {
        "case_id": "emilia_romagna",
        "area": "emilia_romagna",
        "date": "2023-05-17",
    }
    resp = client.post("/api/assess", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Case metadata
    assert data["case_id"] == "emilia_romagna"
    assert data["status"] == "data_not_prepared"
    assert data["validation_status"] == "uncalibrated_external_aoi"
    assert data["bbox"] == [11.70, 44.15, 12.35, 44.55]
    assert data["center"] == [44.35, 12.02]

    # Data isolation: NO Trishuli metrics or layers leaked
    assert data["floodAreaKm2"] is None
    assert data["debrisAreaKm2"] is None
    assert data["hazard_metrics"] is None
    assert data["summary"] is None
    assert data["settlements"] == []
    assert data["layers"] == {}

    # Clear instructions returned
    assert data["message"] is not None
    assert "has not been prepared" in data["message"]
    assert data["data_requirements"] is not None
    assert "sentinel1" in data["data_requirements"]
    assert "dem" in data["data_requirements"]
    assert "osm" in data["data_requirements"]


def test_cross_case_layer_isolation_returns_404(client):
    """Verify accessing layer endpoints with an unprepared case ID returns 404 to avoid leaking cached rasters."""
    # When requesting layers for an unprepared case, direct layer route must reject with 404
    resp = client.get("/api/layers/flood_evidence?case_id=emilia_romagna")
    assert resp.status_code == 404
    assert "No map layers exist for case 'emilia_romagna'" in resp.json()["detail"]

    resp_roads = client.get("/api/layers/affected_roads?case_id=valencia_dana")
    assert resp_roads.status_code == 404
    assert "No map layers exist for case 'valencia_dana'" in resp_roads.json()["detail"]
