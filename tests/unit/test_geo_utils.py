import json

import pytest

from floodmap.core.geo_utils import (aoi_bounds, aoi_from_bounds, aoi_from_geojson, analysis_crs_for_aoi,
                                     buffer_lonlat_bounds_km, load_aoi, utm_epsg_for_lonlat)


@pytest.mark.parametrize("lon,lat,epsg", [
    (85.3, 27.7, 32645),    # Kathmandu
    (83.9, 28.2, 32644),    # Pokhara (zone 44)
    (84.0, 28.0, 32645),    # exactly on the 44/45 boundary -> 45
    (-70.0, -33.0, 32719),  # southern hemisphere
])
def test_utm_zone(lon, lat, epsg):
    assert utm_epsg_for_lonlat(lon, lat) == epsg


def test_analysis_crs():
    aoi = aoi_from_bounds((85.1, 27.9, 85.3, 28.1))
    assert analysis_crs_for_aoi(aoi) == "EPSG:32645"
    assert analysis_crs_for_aoi(aoi, "EPSG:3857") == "EPSG:3857"


def test_buffer_km_is_roughly_right():
    b = buffer_lonlat_bounds_km((85.0, 28.0, 85.0, 28.0), 10)
    dlat = b[3] - 28.0
    dlon = b[2] - 85.0
    assert dlat == pytest.approx(10 / 111.32, rel=1e-6)
    assert dlon > dlat  # degrees of longitude are shorter at 28°N, so more of them are needed


def test_aoi_from_featurecollection_and_bbox_string(tmp_path):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": aoi_from_bounds((85, 27, 85.1, 27.1)).geometry},
        {"type": "Feature", "properties": {}, "geometry": aoi_from_bounds((86, 27, 86.1, 27.1)).geometry},
    ]}
    aoi = aoi_from_geojson(fc)
    assert aoi.geometry["type"] == "MultiPolygon"
    assert aoi_bounds(aoi) == (85, 27, 86.1, 27.1)
    p = tmp_path / "x.geojson"
    p.write_text(json.dumps(fc))
    assert load_aoi(str(p)).name == "x"
    assert aoi_bounds(load_aoi("85.1,27.9,85.3,28.1")) == (85.1, 27.9, 85.3, 28.1)
    with pytest.raises(ValueError):
        load_aoi("1,2,3")


def test_missing_aoi_file_is_reported_not_parsed_as_coordinates(tmp_path, monkeypatch):
    # regression: "data\raw\dev\aoi_from_search.geojson" that did not exist raised
    # "could not convert string to float" instead of saying the file is missing
    monkeypatch.chdir(tmp_path)
    for arg in ["data\\raw\\dev\\aoi_from_search.geojson", "data/raw/dev/aoi_from_search.geojson", "aoi.json"]:
        with pytest.raises(ValueError, match="AOI file not found") as e:
            load_aoi(arg)
        assert "could not convert" not in str(e.value) and str(tmp_path) in str(e.value)


def test_aoi_from_search_candidates_json(tmp_path):
    aoi = aoi_from_bounds((85.17, 27.95, 85.36, 28.18), name="trishuli_corridor_dev")
    cand = tmp_path / "s1_candidates.json"
    cand.write_text(json.dumps({"aoi": aoi.model_dump(), "event_date": "2026-08-26", "scenes": [], "pairs": []}))
    got = load_aoi(str(cand))
    assert got == aoi                                      # identical AOI, no coordinates copied by hand
    exported = tmp_path / "aoi_from_search.geojson"        # the file the user exported from it
    exported.write_text(json.dumps({"type": "Feature", "properties": {"name": aoi.name}, "geometry": aoi.geometry}))
    assert load_aoi(str(exported)).geometry == aoi.geometry
    assert load_aoi(f'"{exported}"').geometry == aoi.geometry   # stray quotes from a shell are tolerated


def test_bbox_validation(tmp_path):
    assert aoi_bounds(load_aoi(" 85.1, 27.9, 85.3, 28.1 ")) == (85.1, 27.9, 85.3, 28.1)
    with pytest.raises(ValueError, match="min < max"):
        load_aoi("85.3,27.9,85.1,28.1")
    bad = tmp_path / "bad.geojson"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_aoi(str(bad))
