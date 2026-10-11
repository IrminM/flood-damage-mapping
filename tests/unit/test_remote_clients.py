"""Offline tests for the ohsome and CDSE catalogue clients (network is mocked)."""
import json
from datetime import date

import pytest

from floodmap.core.errors import DataRuleViolation
from floodmap.core.geo_utils import aoi_from_bounds
from floodmap.data_sources import cdse_catalogue, ohsome

AOI = aoi_from_bounds((85.15, 27.95, 85.35, 28.15), name="test")


class FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def test_ohsome_requests(settings):
    reqs = {r.layer: r for r in ohsome.build_requests(AOI, date(2026, 7, 27), settings.osm)}
    assert set(reqs) == set(settings.osm.layers)
    assert reqs["roads"].url.endswith("/v1/elements/geometry")
    assert reqs["roads"].params["time"] == "2026-07-27"
    assert reqs["roads"].params["properties"] == "tags"
    assert "highway in" in reqs["roads"].params["filter"] and "\n" not in reqs["roads"].params["filter"]
    # roads use the 20 km network buffer, buildings only 1 km -> roads bbox is larger
    rb, bb = reqs["roads"].bbox, reqs["buildings"].bbox
    assert rb[0] < bb[0] < AOI.geometry["coordinates"][0][0][0]
    assert rb[3] > bb[3] > 28.15


def test_ohsome_fetch_snapshot_writes_files(tmp_path, settings, guard):
    calls = []

    def fake_post(url, data, timeout):
        calls.append(data)
        return FakeResp({"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [85.2, 28.0]},
             "properties": {"@osmId": "node/1", "place": "village"}},
            {"type": "Feature", "geometry": None, "properties": {"@osmId": "node/2"}},
        ]})

    out = ohsome.fetch_snapshot(AOI, date(2026, 8, 26), date(2026, 7, 27), tmp_path / "osm",
                                settings.osm, guard, post=fake_post, log=lambda m: None)
    assert len(calls) == len(settings.osm.layers)
    meta = json.loads((out / "snapshot_meta.json").read_text())
    assert meta["snapshot_date"] == "2026-07-27" and meta["source_type"] == "osm-pre-event"
    fc = json.loads((out / "places.geojson").read_text())
    assert [f["id"] for f in fc["features"]] == ["node/1"]   # geometry-less feature dropped


def test_ohsome_refuses_post_event_snapshot(tmp_path, settings, guard):
    with pytest.raises(DataRuleViolation):
        ohsome.fetch_snapshot(AOI, date(2026, 8, 26), date(2026, 9, 1), tmp_path, settings.osm, guard,
                              post=lambda *a, **k: pytest.fail("must not call the API"))


def _product(name, start, orbit, direction, serial="A"):
    return {"Id": "x", "Name": name, "Online": True, "ContentDate": {"Start": start},
            "Attributes": [
                {"Name": "relativeOrbitNumber", "Value": orbit},
                {"Name": "orbitDirection", "Value": direction},
                {"Name": "platformSerialIdentifier", "Value": serial},
                {"Name": "polarisationChannels", "Value": "VV&VH"},
            ]}


def test_cdse_query_and_parse():
    q = cdse_catalogue.build_s1_grd_query(AOI, date(2026, 8, 26))
    assert "SENTINEL-1" in q["$filter"] and "IW_GRDH" in q["$filter"] and "POLYGON((85.15 27.95" in q["$filter"]
    payload = {"value": [
        _product("S1A_IW_GRDH_1SDV_20260820T001500_X.SAFE", "2026-08-20T00:15:00.000Z", 85, "ASCENDING"),
        _product("S1A_IW_GRDH_1SDV_20260820T001500_X_COG.SAFE", "2026-08-20T00:15:00.000Z", 85, "ASCENDING"),
        _product("S1C_IW_GRDH_1SDV_20260901T001500_Y.SAFE", "2026-09-01T00:15:00.000Z", 85, "ASCENDING", "C"),
    ]}
    scenes = cdse_catalogue.parse_products(payload)
    assert len(scenes) == 2                                  # COG duplicate removed
    assert scenes[0].relative_orbit == 85 and scenes[0].orbit_direction == "ascending"
    assert scenes[0].bands == ["VV", "VH"] and scenes[1].platform == "sentinel-1c"
    got = cdse_catalogue.search_s1_grd(AOI, date(2026, 8, 26), get=lambda *a, **k: FakeResp(payload))
    assert len(got) == 2
