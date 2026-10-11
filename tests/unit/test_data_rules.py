from datetime import date
from pathlib import Path

import pytest

from floodmap.core.contracts import SceneRef, SourceType
from floodmap.core.errors import DataRuleViolation


@pytest.mark.parametrize("name", [
    "data/validation/EMSR927_AOI01_GRA_PRODUCT_builtUpA_v1.shp",
    "downloads/copernicus_ems_delineation.geojson",
    "UNOSAT_FL20260830NPL_damage.geojson",
    "my_damage_assessment.tif",
])
def test_prohibited_names_rejected(guard, name):
    with pytest.raises(DataRuleViolation):
        guard.check_input(name, SourceType.SENTINEL_1)


def test_prohibited_metadata_text_rejected(guard):
    with pytest.raises(DataRuleViolation):
        guard.check_input("scene.tif", SourceType.SENTINEL_1, extra_text=["derived from EMSR927 grading"])


def test_unknown_source_type_rejected(guard):
    with pytest.raises(DataRuleViolation):
        guard.check_input("map.tif", "published-damage-map")


def test_allowed_inputs_pass(guard):
    guard.check_input("data/raw/aoi1/s1_before_20260820.tif", SourceType.SENTINEL_1)
    guard.check_input("data/dem/Copernicus_DSM_COG_10_N27_00_E085_00_DEM.tif", "copernicus-dem")
    guard.check_input("data/osm/aoi1", SourceType.OSM_PRE_EVENT)


@pytest.mark.parametrize("snap,ok", [("2026-07-27", True), ("2026-08-25", True),
                                     ("2026-08-26", False), ("2026-09-01", False)])
def test_osm_snapshot_must_precede_event(guard, snap, ok):
    if ok:
        guard.check_osm_snapshot(snap, date(2026, 8, 26))
    else:
        with pytest.raises(DataRuleViolation):
            guard.check_osm_snapshot(snap, date(2026, 8, 26))


def test_osm_snapshot_accepts_datetime_strings(guard):
    guard.check_osm_snapshot("2026-07-27T00:00:00Z", "2026-08-26")


def test_scene_sensor_source_mismatch(guard, s1_meta_factory):
    meta = s1_meta_factory("2026-08-20", source_type=SourceType.SENTINEL_2)
    with pytest.raises(DataRuleViolation):
        guard.check_scene(SceneRef(path=Path("x.tif"), metadata=meta))


def test_scene_product_id_checked(guard, s1_meta_factory):
    meta = s1_meta_factory("2026-08-20", product_ids=["EMSR927_something"])
    with pytest.raises(DataRuleViolation):
        guard.check_scene(SceneRef(path=Path("x.tif"), metadata=meta))
