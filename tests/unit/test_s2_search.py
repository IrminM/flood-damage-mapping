from datetime import date
from floodmap.core.geo_utils import aoi_from_bounds
from floodmap.data_sources.cdse_catalogue import (
    build_s2_l1c_query,
    parse_s2_products,
    search_s2_l1c,
    S2_L1C_BANDS,
)


def test_build_s2_l1c_query():
    aoi = aoi_from_bounds((85.17, 27.95, 85.36, 28.18))
    params = build_s2_l1c_query(aoi, date(2026, 8, 26), days_before=14, days_after=7)
    assert "$filter" in params
    assert "Collection/Name eq 'SENTINEL-2'" in params["$filter"]
    assert "contains(Name,'MSIL1C')" in params["$filter"]
    assert "2026-08-12" in params["$filter"]


def test_parse_s2_products():
    fake_payload = {
        "value": [
            {
                "Id": "abc-123",
                "Name": "S2C_MSIL1C_20260812T045701_N0512_R119_T45RUM_20260812T083153.SAFE",
                "Online": True,
                "ContentDate": {"Start": "2026-08-12T04:57:01.025Z"},
                "Attributes": [
                    {"Name": "platformSerialIdentifier", "Value": "C"},
                    {"Name": "cloudCover", "Value": 11.2},
                    {"Name": "relativeOrbitNumber", "Value": 119},
                ],
            }
        ]
    }
    scenes = parse_s2_products(fake_payload)
    assert len(scenes) == 1
    s = scenes[0]
    assert s.sensor == "S2"
    assert s.platform == "sentinel-2c"
    assert s.product_type == "L1C"
    assert s.cloud_cover_pct == 11.2
    assert s.relative_orbit == 119
    assert s.bands == S2_L1C_BANDS
    assert "B10" in s.bands
    assert s.extra["cdse_id"] == "abc-123"


def test_search_s2_l1c_injectable():
    aoi = aoi_from_bounds((85.17, 27.95, 85.36, 28.18))

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "value": [
                    {
                        "Id": "mock-s2-id",
                        "Name": "S2B_MSIL1C_20260827T045659_N0512_R119_T45RUM_20260827T083139.SAFE",
                        "Online": True,
                        "ContentDate": {"Start": "2026-08-27T04:56:59.024Z"},
                        "Attributes": [
                            {"Name": "platformSerialIdentifier", "Value": "B"},
                            {"Name": "cloudCover", "Value": 65.6},
                            {"Name": "relativeOrbitNumber", "Value": 119},
                        ],
                    }
                ]
            }

    def fake_get(url, params=None, timeout=None):
        return FakeResponse()

    scenes = search_s2_l1c(aoi, date(2026, 8, 26), get=fake_get)
    assert len(scenes) == 1
    assert scenes[0].platform == "sentinel-2b"
    assert scenes[0].product_type == "L1C"
