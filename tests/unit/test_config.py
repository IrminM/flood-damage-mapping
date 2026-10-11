import ntpath
import posixpath
from pathlib import Path

import pytest

from floodmap.core.config import (REPO_ROOT, ConfigError, deep_merge, env_overrides, is_absolute_path,
                                  load_dotenv, load_settings, resolve_against)


def test_defaults_load(settings):
    assert settings.crs.analysis == "auto_utm"
    assert settings.satellite.s1.require_same_relative_orbit is True
    assert settings.osm.default_snapshot_date == "2026-07-27"
    assert set(settings.osm.layers) == {"buildings", "roads", "bridges", "places", "health"}
    for layer in settings.osm.layers:
        assert layer in settings.osm.filters, f"no ohsome filter for {layer}"


def test_env_override_types():
    s = load_settings(environ={
        "FLOODMAP__OSM__NETWORK_BUFFER_KM": "5",
        "FLOODMAP__SATELLITE__S1__REQUIRE_SAME_RELATIVE_ORBIT": "false",
        "FLOODMAP__AI__MODEL": "teammate_unet",
        "UNRELATED": "x",
    })
    assert s.osm.network_buffer_km == 5.0
    assert s.satellite.s1.require_same_relative_orbit is False
    assert s.ai.model == "teammate_unet"


def test_override_yaml(tmp_path: Path):
    # A real absolute path on whatever OS runs the test (C:/... on Windows, /tmp/... on Linux),
    # written with forward slashes, which YAML and both platforms accept.
    target = tmp_path / "x"
    p = tmp_path / "local.yaml"
    p.write_text(f"osm:\n  network_buffer_km: 3\npaths:\n  data_dir: '{target.as_posix()}'\n")
    s = load_settings(override_path=p, environ={})
    assert s.osm.network_buffer_km == 3
    assert s.paths.data_dir == target            # Path comparison: separator-agnostic
    assert s.resolve_path(s.paths.data_dir) == target
    assert s.osm.layers  # untouched keys survive the merge


@pytest.mark.parametrize("quoted,why", [
    ('"C:\\new\\temp"', "control characters"),   # \n and \t are valid YAML escapes -> silently corrupted path
    ('"C:\\new\\data"', "Cannot parse YAML"),    # \d is an invalid YAML escape -> parse error
])
def test_windows_path_in_double_quoted_yaml_is_rejected(tmp_path: Path, quoted, why):
    p = tmp_path / "local.yaml"
    p.write_text(f"paths:\n  data_dir: {quoted}\n")
    with pytest.raises(ConfigError, match=why):
        load_settings(override_path=p, environ={})


@pytest.mark.parametrize("yaml_value", ["C:/data/x", "'C:\\data\\x'", "C:\\data\\x"])
def test_windows_paths_safe_yaml_forms(tmp_path: Path, yaml_value):
    p = tmp_path / "local.yaml"
    p.write_text(f"paths:\n  data_dir: {yaml_value}\n")
    s = load_settings(override_path=p, environ={})
    assert ntpath.normpath(str(s.paths.data_dir)) == "C:\\data\\x"


def test_windows_path_env_override_stays_a_string():
    s = load_settings(environ={"FLOODMAP__PATHS__DATA_DIR": "C:\\new\\data", "FLOODMAP__PATHS__RUNS_DIR": "D:/runs"})
    assert ntpath.normpath(str(s.paths.data_dir)) == "C:\\new\\data"
    assert ntpath.normpath(str(s.paths.runs_dir)) == "D:\\runs"


def test_invalid_value_raises():
    with pytest.raises(ConfigError):
        load_settings(environ={"FLOODMAP__SATELLITE__S1__MAX_PAIR_GAP_DAYS": "not-a-number"})


def test_missing_override_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(override_path=tmp_path / "nope.yaml", environ={})


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}} and base == {"a": {"b": 1, "c": 2}}


def test_env_overrides_nesting():
    assert env_overrides({"FLOODMAP__A__B__C": "[1, 2]"}) == {"a": {"b": {"c": [1, 2]}}}


def test_dotenv_does_not_override(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# comment\nFOO=from_file\nBAR='quoted'\n")
    env = {"FOO": "already"}
    load_dotenv(f, env)
    assert env == {"FOO": "already", "BAR": "quoted"}


def test_resolve_path_native(settings, tmp_path: Path):
    # Native behaviour on the OS running the tests (Windows or POSIX).
    rel = settings.resolve_path("data")
    assert rel.is_absolute() and rel == REPO_ROOT / "data"
    assert settings.resolve_path(tmp_path / "x") == tmp_path / "x"          # absolute kept
    assert settings.resolve_path(str(tmp_path / "a" / ".." / "x")) == tmp_path / "x"   # normalised
    assert settings.resolve_path("data/runs") == REPO_ROOT / "data" / "runs"


@pytest.mark.parametrize("p,root,expected", [
    ("/abs/x", "/repo", "/abs/x"),
    ("data", "/repo", "/repo/data"),
    ("./data/runs", "/repo", "/repo/data/runs"),
    ("../x", "/repo/sub", "/repo/x"),
    ("/abs//y/./z", "/repo", "/abs/y/z"),
])
def test_resolve_against_posix_rules(p, root, expected):
    assert resolve_against(p, root, posixpath) == expected


@pytest.mark.parametrize("p,root,expected", [
    ("C:\\abs\\x", "C:\\repo", "C:\\abs\\x"),
    ("C:/abs/x", "C:\\repo", "C:\\abs\\x"),               # forward slashes accepted
    ("D:\\other", "C:\\repo", "D:\\other"),                # another drive stays put
    ("data", "C:\\repo", "C:\\repo\\data"),
    ("data/runs", "C:\\repo", "C:\\repo\\data\\runs"),
    ("./data", "C:\\repo", "C:\\repo\\data"),
    ("/abs/x", "D:\\repo", "D:\\abs\\x"),                  # rooted, no drive -> drive of root
    ("\\abs\\x", "D:\\repo", "D:\\abs\\x"),
    ("\\\\server\\share\\d", "C:\\repo", "\\\\server\\share\\d"),   # UNC is absolute
])
def test_resolve_against_windows_rules(p, root, expected):
    assert resolve_against(p, root, ntpath) == expected


@pytest.mark.parametrize("p,abs_nt", [("C:\\x", True), ("C:/x", True), ("\\\\srv\\sh\\x", True),
                                       ("\\x", False), ("/x", False), ("C:x", False), ("x", False)])
def test_is_absolute_windows(p, abs_nt):
    assert is_absolute_path(p, ntpath) is abs_nt


def test_is_absolute_posix():
    assert is_absolute_path("/x", posixpath) and not is_absolute_path("x", posixpath)
    assert not is_absolute_path("C:\\x", posixpath)        # a Windows path is relative on POSIX
