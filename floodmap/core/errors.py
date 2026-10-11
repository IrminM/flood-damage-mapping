"""Project-wide exception types.

Distinct types let the orchestrator/API tell "bad user input" apart from
"data rule violated" apart from "a component failed".
"""


class FloodmapError(Exception):
    """Base class for all floodmap errors."""


class ConfigError(FloodmapError):
    """Configuration file or environment override is invalid."""


class DataRuleViolation(FloodmapError):
    """An input breaks the hackathon data rules (prohibited source, post-event OSM...).

    This is never downgraded to a warning: breaking the rules means disqualification.
    """


class InputDataError(FloodmapError):
    """An input file is missing, unreadable, or does not match its declared metadata."""


class GridMismatchError(FloodmapError):
    """Two rasters were expected to share a pixel grid but do not."""


class ScenePairError(FloodmapError):
    """A before/after scene pair is not valid for change detection."""
