"""Typed configuration loading.

Order of precedence (later wins):
    1. config/default.yaml (shipped with the repo)
    2. an optional override YAML given by FLOODMAP_CONFIG or the `override_path` argument
    3. environment variables FLOODMAP__<SECTION>__<KEY>[__<SUBKEY>...]=value
       (values are parsed as YAML scalars, so "true", "20", "[VV, VH]" work)

A `.env` file in the working directory is read (without overriding variables that are
already set) so teammates can keep local paths/credentials out of git.

Only sections used by implemented modules are strictly typed; unknown keys are kept
(extra="allow") so later stages can add config without breaking older code.
"""
from __future__ import annotations

import copy
import ntpath
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
ENV_PREFIX = "FLOODMAP__"


class _Section(BaseModel):
    model_config = ConfigDict(extra="allow")


class PathsConfig(_Section):
    data_dir: Path = Path("./data")
    runs_dir: Path = Path("./data/runs")

    @field_validator("data_dir", "runs_dir", mode="before")
    @classmethod
    def _no_control_chars(cls, v: Any) -> Any:
        # Catches Windows paths written in YAML double quotes, where "C:\new" becomes
        # "C:<newline>ew". Use forward slashes ("C:/new"), single quotes, or no quotes.
        if isinstance(v, str) and any(ord(ch) < 32 for ch in v):
            raise ValueError(
                f"path {v!r} contains control characters - on Windows, write paths in YAML with "
                "forward slashes (C:/data/x), in single quotes ('C:\\data\\x') or unquoted"
            )
        return v


class LoggingConfig(_Section):
    level: str = "INFO"


class CRSConfig(_Section):
    analysis: str = "auto_utm"
    output: str = "EPSG:4326"


class S1Config(_Section):
    require_same_relative_orbit: bool = True
    require_same_orbit_direction: bool = True
    max_pair_gap_days: int = 36
    warn_days_from_event: int = 14
    expected_bands: list[str] = Field(default_factory=lambda: ["VV", "VH"])


class SatelliteConfig(_Section):
    s1: S1Config = Field(default_factory=S1Config)


class GridConfig(_Section):
    resolution_m: float = 10.0
    margin_m: float = 200.0


class SpeckleFilterConfig(_Section):
    enabled: bool = True
    method: str = "lee"
    window: int = 5
    enl: float = 4.4


class S1PreprocessConfig(_Section):
    speckle_filter: SpeckleFilterConfig = Field(default_factory=SpeckleFilterConfig)
    linear_floor: float = 1e-5
    valid_db_range: tuple[float, float] = (-40.0, 15.0)
    min_valid_fraction: float = 0.5


class PreprocessingConfig(_Section):
    s1: S1PreprocessConfig = Field(default_factory=S1PreprocessConfig)


class TerrainConfig(_Section):
    enabled: bool = True
    dem_resolution_m: float = 30.0
    margin_m: float = 2000.0
    nominal_incidence_deg: float = 39.0
    nominal_heading_deg: dict[str, float] = Field(default_factory=lambda: {"ascending": 348.0, "descending": 192.0})
    hand_stream_threshold_km2: float = 1.0
    fill_epsilon_m: float = 1e-4


class WaterThresholdConfig(_Section):
    mode: str = "auto_otsu"
    fixed_db: float = -18.0
    otsu_bounds_db: tuple[float, float] = (-24.0, -14.0)
    otsu_min_samples: int = 1000
    otsu_out_of_bounds: str = "fixed"
    min_ashman_d: float = 3.5


class ChangeThresholdConfig(_Section):
    mode: str = "noise_adaptive"
    k_sigma: float = 3.0
    min_samples: int = 10000


class ChangeContextConfig(_Section):
    enabled: bool = True
    valley_max_hand_m: float = 25.0
    valley_max_slope_deg: float = 15.0


class LogRatioV1Config(_Section):
    water_threshold: WaterThresholdConfig = Field(default_factory=WaterThresholdConfig)
    min_vv_drop_db: float = 3.0
    change_abs_db: float = 3.0
    change_threshold: ChangeThresholdConfig = Field(default_factory=ChangeThresholdConfig)
    change_context: ChangeContextConfig = Field(default_factory=ChangeContextConfig)
    water_max_slope_deg: float = 15.0
    water_max_hand_m: float = 25.0
    change_max_slope_deg: float = 35.0
    min_area_m2: float = 2000.0
    opening_radius_px: int = 1


class ChangeDetectionConfig(_Section):
    method: str = "s1_logratio_v1"
    s1_logratio_v1: LogRatioV1Config = Field(default_factory=LogRatioV1Config)


class DataRulesConfig(_Section):
    osm_snapshot_must_precede_event: bool = True
    prohibited_source_patterns: list[str] = Field(default_factory=list)
    allowed_source_types: list[str] = Field(default_factory=list)


class OSMConfig(_Section):
    ohsome_url: str = "https://api.ohsome.org/v1"
    default_snapshot_date: str = "2026-07-27"
    network_buffer_km: float = 20.0
    buildings_buffer_km: float = 1.0
    layers: list[str] = Field(default_factory=lambda: ["buildings", "roads", "bridges", "places", "health"])
    filters: dict[str, str] = Field(default_factory=dict)
    geometry_types: dict[str, list[str]] = Field(default_factory=dict)


class S1MiniUNetConfig(_Section):
    channels: list[str] = Field(default_factory=lambda: ["VV", "VH"])
    db_range: list[float] = Field(default_factory=lambda: [-35.0, 0.0])
    normalization: str = "min_max"
    tile_size: int = 256
    overlap: int = 32
    blend_mode: str = "ramp"
    threshold: float = 0.50
    water_max_slope_deg: float = 15.0
    water_max_hand_m: float = 25.0
    min_area_m2: float = 2000.0


class FusionConfig(_Section):
    mode: str = "union"
    preserve_permanent_water: bool = True
    preserve_debris: bool = True


class AttentionUNetS1S2Config(_Section):
    threshold: float = 0.40
    tile_size: int = 256
    checkpoint_filename: str = "best_flood_attention_unet.keras"
    product_level: Optional[str] = "UNRESOLVED"
    b10_source: Optional[str] = "UNRESOLVED"
    target_resolution_m: Optional[float] = None
    resampling_method: Optional[str] = "UNRESOLVED"
    co_registration_method: Optional[str] = "UNRESOLVED"
    temporal_pairing_rule: Optional[str] = "UNRESOLVED"


class AIConfig(_Section):
    model: str = "s1_mini_unet_v1"
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)
    s1_mini_unet_v1: S1MiniUNetConfig = Field(default_factory=S1MiniUNetConfig)
    attention_unet_s1_s2: AttentionUNetS1S2Config = Field(default_factory=AttentionUNetS1S2Config)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class ImpactConfig(_Section):
    building_buffer_m: float = 0.0
    min_affected_building_area_m2: float = 1.0
    road_buffer_m: float = 5.0
    min_affected_road_length_m: float = 5.0
    bridge_buffer_m: float = 15.0
    waterway_buffer_m: float = 50.0
    include_flood_classes: list[int] = Field(default_factory=lambda: [1, 3])


class ConnectivityConfig(_Section):
    disruption_mode: str = "remove"             # "remove" | "penalty"
    penalty_multiplier: float = 1000.0          # multiplier on edge length in penalty mode
    max_snapping_distance_m: float = 1000.0     # max distance to snap place to road network
    settlement_place_types: list[str] = Field(
        default_factory=lambda: ["village", "hamlet", "isolated_dwelling", "settlement"]
    )
    destination_place_types: list[str] = Field(
        default_factory=lambda: ["city", "town"]
    )
    include_health_destinations: bool = True
    speeds_kmh: dict[str, float] = Field(
        default_factory=lambda: {
            "motorway": 80.0,
            "trunk": 60.0,
            "primary": 50.0,
            "secondary": 40.0,
            "tertiary": 30.0,
            "unclassified": 25.0,
            "residential": 20.0,
            "track": 10.0,
            "default": 25.0,
        }
    )


class DownstreamConfig(_Section):
    max_length_m: float = 50000.0                 # 50 km max path length
    max_steps: int = 2000                         # max number of raster cells
    sink_handling: str = "fill"                   # "fill" (depression-filled routing) | "stop" (terminate at pit)
    min_elevation_drop_m: float = 0.0             # minimum elevation drop per step
    settlement_proximity_radius_m: float = 1000.0 # buffer (m) to detect nearby settlements along path
    settlement_place_types: list[str] = Field(
        default_factory=lambda: ["village", "hamlet", "isolated_dwelling", "settlement", "town", "city"]
    )


class Settings(_Section):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    crs: CRSConfig = Field(default_factory=CRSConfig)
    satellite: SatelliteConfig = Field(default_factory=SatelliteConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    terrain: TerrainConfig = Field(default_factory=TerrainConfig)
    change_detection: ChangeDetectionConfig = Field(default_factory=ChangeDetectionConfig)
    data_rules: DataRulesConfig = Field(default_factory=DataRulesConfig)
    osm: OSMConfig = Field(default_factory=OSMConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    impact: ImpactConfig = Field(default_factory=ImpactConfig)
    connectivity: ConnectivityConfig = Field(default_factory=ConnectivityConfig)
    downstream: DownstreamConfig = Field(default_factory=DownstreamConfig)


    def resolve_path(self, p: Path | str) -> Path:
        """Resolve a configured (possibly relative) path against the repo root (platform-aware).

        See `resolve_against` for the exact rules on POSIX and Windows.
        """
        return Path(resolve_against(p, REPO_ROOT))


# ---------------------------------------------------------------- helpers

def is_absolute_path(p: str, pathmod: Any = os.path) -> bool:
    """True if `p` is fully absolute under the given path flavour.

    POSIX  : starts with "/".
    Windows: needs a drive AND a root, e.g. "C:\\data", "C:/data" or a UNC path
             "\\\\server\\share\\data". A rooted path without a drive ("\\data", "/data")
             and a drive-relative path ("C:data") are NOT absolute. This rule is fixed here
             rather than delegated to os.path.isabs, whose answer for "\\data" changed in
             Python 3.13.
    """
    p = str(p)
    if pathmod is ntpath:
        drive, rest = ntpath.splitdrive(p)
        return bool(drive) and rest[:1] in ("\\", "/")
    return pathmod.isabs(p)


def resolve_against(p: Path | str, root: Path | str, pathmod: Any = os.path) -> str:
    """Absolute, normalised path for `p`, interpreting relative paths against `root`.

    * "~" is expanded.
    * Absolute paths are kept (only normalised: separators, "..", ".").
    * Relative paths are joined onto `root`.
    * Windows only: a rooted path without a drive ("/data", "\\data") takes the drive of
      `root` (same as pathlib/ntpath.join), and "C:data" is joined onto `root` when it
      is on the same drive.
    Pure string operations (no filesystem access, symlinks are not followed), so the
    result is deterministic and `pathmod` can be ntpath/posixpath for testing either
    platform's rules on any OS.
    """
    s = pathmod.expanduser(str(p))
    if not is_absolute_path(s, pathmod):
        s = pathmod.join(str(root), s)
    return pathmod.normpath(s)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(dict(base))
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Cannot parse YAML {path}: {e}\nHint: Windows paths inside double quotes are read as escape "
            "sequences; use forward slashes (C:/data/x), single quotes ('C:\\data\\x') or no quotes."
        ) from e
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")
    return data


def load_dotenv(path: Path | str = ".env", environ: dict[str, str] | None = None) -> None:
    """Minimal .env reader: KEY=VALUE lines, '#' comments; never overrides existing vars."""
    environ = os.environ if environ is None else environ
    path = Path(path)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in environ:
            environ[key] = value


def env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    """Turn FLOODMAP__A__B=value variables into {"a": {"b": value}}."""
    out: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX):].split("__") if p]
        if not parts:
            continue
        try:
            value = yaml.safe_load(raw) if raw != "" else ""
        except yaml.YAMLError:
            value = raw
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Env override {key} conflicts with a scalar value")
        node[parts[-1]] = value
    return out


def load_settings(
    override_path: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    default_path: Path | str = DEFAULT_CONFIG_PATH,
    read_dotenv: bool = True,
) -> Settings:
    """Load and validate settings (defaults → override YAML → environment)."""
    if environ is None:
        if read_dotenv:
            load_dotenv()
        environ = os.environ
    data = _read_yaml(Path(default_path))
    override_path = override_path or environ.get("FLOODMAP_CONFIG")
    if override_path:
        data = deep_merge(data, _read_yaml(Path(override_path)))
    data = deep_merge(data, env_overrides(environ))
    try:
        return Settings.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration:\n{e}") from e
