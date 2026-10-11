from datetime import date

import pytest

from floodmap.core.errors import DataRuleViolation, InputDataError
from floodmap.data_sources.osm_local import LocalOSMProvider

EVENT = date(2026, 8, 26)


def test_loads_layers_validates_types_and_ids(tmp_path, guard, settings, osm_dir_factory):
    d = osm_dir_factory(tmp_path / "osm")
    data = LocalOSMProvider(guard, settings.osm).load(d, EVENT)
    assert data.meta.snapshot_date == date(2026, 7, 27)
    assert data.count("buildings") == 1
    assert data.count("roads") == 1                     # the Point was dropped
    assert any("roads: dropped 1" in w for w in data.warnings)
    assert set(data.missing_layers) == {"bridges", "health"}
    assert data.layers["bridges"]["features"] == []
    assert data.layers["roads"]["features"][0]["id"] == "way/1"


def test_post_event_snapshot_rejected(tmp_path, guard, settings, osm_dir_factory):
    d = osm_dir_factory(tmp_path / "osm", snapshot_date="2026-08-30")
    with pytest.raises(DataRuleViolation):
        LocalOSMProvider(guard, settings.osm).load(d, EVENT)


def test_missing_snapshot_meta_rejected(tmp_path, guard, settings, osm_dir_factory):
    d = osm_dir_factory(tmp_path / "osm")
    (d / "snapshot_meta.json").unlink()
    with pytest.raises(InputDataError):
        LocalOSMProvider(guard, settings.osm).load(d, EVENT)


def test_prohibited_directory_name_rejected(tmp_path, guard, settings, osm_dir_factory):
    d = osm_dir_factory(tmp_path / "EMSR927_reference")
    with pytest.raises(DataRuleViolation):
        LocalOSMProvider(guard, settings.osm).load(d, EVENT)


def test_duplicate_ids_made_unique(tmp_path, guard, settings, osm_dir_factory):
    line = {"type": "LineString", "coordinates": [[85.0, 27.9], [85.01, 27.91]]}
    d = osm_dir_factory(tmp_path / "osm", layers={"roads": [line, line]})
    # both features get @osmId way/1, way/2 from the fixture; force a duplicate:
    import json
    p = d / "roads.geojson"
    fc = json.loads(p.read_text())
    for f in fc["features"]:
        f["properties"]["@osmId"] = "way/7"
    p.write_text(json.dumps(fc))
    data = LocalOSMProvider(guard, settings.osm).load(d, EVENT)
    ids = [f["id"] for f in data.layers["roads"]["features"]]
    assert len(set(ids)) == 2 and ids[0] == "way/7"
