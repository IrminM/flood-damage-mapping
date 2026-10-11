"""Data-rules guard — enforces the hackathon input restrictions in code.

Rules (Track B brief, "Technical Considerations"):
  * Allowed inputs: Sentinel-1, Sentinel-2, Copernicus DEM, OpenStreetMap as it was
    before the event, and the listed training datasets.
  * NOT allowed as inputs: Copernicus EMS, UNOSAT or any other published damage maps,
    and OpenStreetMap edits made after the event. These may only be used to check results.

The guard is called by every data loader and by the orchestrator. A violation raises
DataRuleViolation; it is never downgraded to a warning.

Validation tools (e.g. a future EMSR927 comparison script) must live outside the
pipeline input path and must not call these loaders.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional, Union

from .config import DataRulesConfig
from .contracts import SceneRef, SourceType
from .errors import DataRuleViolation

PathLike = Union[str, Path]


class DataRulesGuard:
    def __init__(self, cfg: DataRulesConfig):
        self.cfg = cfg
        self._patterns = [re.compile(p, re.IGNORECASE) for p in cfg.prohibited_source_patterns]
        self._allowed = {s.lower() for s in cfg.allowed_source_types}

    # ------------------------------------------------------------ generic
    def _matching_pattern(self, text: str) -> Optional[str]:
        for p in self._patterns:
            if p.search(text):
                return p.pattern
        return None

    def check_input(
        self,
        name_or_path: PathLike,
        declared_source: Union[SourceType, str],
        extra_text: Iterable[str] = (),
    ) -> None:
        """Reject inputs whose declared source isn't allowed or whose name/metadata
        matches a prohibited-product pattern (EMS, UNOSAT, damage assessment...)."""
        src = declared_source.value if isinstance(declared_source, SourceType) else str(declared_source)
        if src.lower() not in self._allowed:
            raise DataRuleViolation(
                f"Input {name_or_path!s}: declared source {src!r} is not an allowed input type "
                f"(allowed: {sorted(self._allowed)})"
            )
        for text in [str(name_or_path), *extra_text]:
            hit = self._matching_pattern(text)
            if hit:
                raise DataRuleViolation(
                    f"Input {name_or_path!s} matches prohibited-source pattern {hit!r} ({text!r}). "
                    "Published damage maps (Copernicus EMS, UNOSAT, ...) may only be used for "
                    "validation, never as system inputs."
                )

    # ------------------------------------------------------------ specific
    def check_scene(self, scene: SceneRef) -> None:
        m = scene.metadata
        expected = {"S1": SourceType.SENTINEL_1, "S2": SourceType.SENTINEL_2}[m.sensor]
        if m.source_type not in (expected, SourceType.SYNTHETIC_TEST):
            raise DataRuleViolation(
                f"Scene {scene.path}: sensor {m.sensor} but declared source {m.source_type.value}"
            )
        self.check_input(scene.path, m.source_type, extra_text=[m.processing, *m.product_ids])

    def check_osm_snapshot(self, snapshot_date: Union[date, datetime, str], event_date: Union[date, str]) -> None:
        """OSM snapshot must be strictly before the event date (no post-event edits)."""
        if not self.cfg.osm_snapshot_must_precede_event:
            return
        snap = to_date(snapshot_date)
        event = to_date(event_date)
        if snap >= event:
            raise DataRuleViolation(
                f"OSM snapshot date {snap} is not before the event date {event}. "
                "Post-event OSM edits are prohibited as inputs; use a pre-event snapshot."
            )


def to_date(v: Union[date, datetime, str]) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()
    if "T" in s:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    return date.fromisoformat(s)
