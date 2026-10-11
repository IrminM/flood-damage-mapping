"""Threshold selection for SAR water mapping.

Bounded Otsu: Otsu's method on the after-event VV backscatter (dB) of observable,
water-plausible pixels. The Otsu split is only accepted if it is a credible land/open-water
split:
  * it must fall inside the physically plausible range for open water (otsu_bounds_db), and
  * the two halves it separates must be clearly bimodal (Ashman's D >= min_ashman_d; a unimodal
    histogram split by Otsu already gives D~2.65, land + >=10 % open water gives D~6).
Global Otsu cannot detect water that covers only a few percent of the scene (e.g. a narrow
Himalayan river): it then falls back to the fixed threshold. Split-based (tile-wise) thresholding
is the standard upgrade for that case and can replace this function behind the same interface.
Otherwise the histogram is not land-vs-water (e.g. little open water, or terrain-dominated
backscatter in steep valleys) and the conservative fixed threshold is used instead.
Clamping an invalid Otsu value to the nearest bound is available only as an explicit
policy ("clamp"); it is not the default because it picks the most permissive bound.
The value, method and evidence are always returned for provenance and the summary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdResult:
    value_db: float
    method: str                 # "otsu" | "otsu_clamped" | "fixed" | "fixed_fallback"
    raw_otsu_db: float | None
    n_samples: int
    note: str = ""
    ashman_d: float | None = None        # separation of the two Otsu halves (unimodal ~2.65, land+water ~6)
    dark_fraction: float | None = None   # share of samples below the raw Otsu split

    def as_dict(self) -> dict:
        r = lambda x: None if x is None else round(float(x), 3)  # noqa: E731
        return {"value_db": r(self.value_db), "method": self.method, "raw_otsu_db": r(self.raw_otsu_db),
                "n_samples": self.n_samples, "ashman_d": r(self.ashman_d), "dark_fraction": r(self.dark_fraction),
                "note": self.note}


def otsu(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold of a 1-D sample (maximises between-class variance)."""
    v = values[np.isfinite(values)]
    if v.size < 2 or np.ptp(v) == 0:
        raise ValueError("Otsu needs at least two distinct finite values")
    hist, edges = np.histogram(v, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist).astype(float)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between[:-1]))])


def ashman_d(values: np.ndarray, split: float) -> float:
    """Ashman's D between the two classes on either side of `split` (D > 2: clean separation)."""
    a, b = values[values <= split], values[values > split]
    if a.size < 2 or b.size < 2:
        return 0.0
    var = a.var() + b.var()
    return float(np.sqrt(2.0) * abs(a.mean() - b.mean()) / np.sqrt(var)) if var > 0 else float("inf")


def water_threshold(values_db: np.ndarray, mode: str, fixed_db: float,
                    bounds_db: tuple[float, float], min_samples: int,
                    out_of_bounds: str = "fixed", min_ashman_d: float = 3.5) -> ThresholdResult:
    v = values_db[np.isfinite(values_db)]
    if mode == "fixed":
        return ThresholdResult(fixed_db, "fixed", None, int(v.size))
    if mode != "auto_otsu":
        raise ValueError(f"Unknown water threshold mode {mode!r}")
    if out_of_bounds not in ("fixed", "clamp"):
        raise ValueError(f"Unknown otsu_out_of_bounds policy {out_of_bounds!r}")
    if v.size < min_samples:
        return ThresholdResult(fixed_db, "fixed_fallback", None, int(v.size),
                               f"only {v.size} samples (< {min_samples}); used fixed {fixed_db} dB")
    raw = otsu(v)
    d = ashman_d(v, raw)
    dark = float((v <= raw).mean())
    lo, hi = bounds_db
    if raw < lo or raw > hi:
        if out_of_bounds == "clamp":
            val = float(np.clip(raw, lo, hi))
            return ThresholdResult(val, "otsu_clamped", raw, int(v.size),
                                   f"Otsu {raw:.2f} dB outside [{lo}, {hi}] dB; clamped to {val} dB "
                                   "(explicit 'clamp' policy)", d, dark)
        return ThresholdResult(fixed_db, "fixed_fallback", raw, int(v.size),
                               f"Otsu {raw:.2f} dB is outside the open-water range [{lo}, {hi}] dB, so the after-event "
                               "VV histogram is not a land/open-water split (little open water and/or terrain-"
                               f"dominated backscatter); used conservative fixed {fixed_db} dB", d, dark)
    if d < min_ashman_d:
        return ThresholdResult(fixed_db, "fixed_fallback", raw, int(v.size),
                               f"Otsu {raw:.2f} dB is in range but the histogram is not clearly bimodal "
                               f"(Ashman D={d:.2f} < {min_ashman_d}); used conservative fixed {fixed_db} dB", d, dark)
    return ThresholdResult(raw, "otsu", raw, int(v.size), "", d, dark)


# ------------------------------------------------------------------ change (log-ratio) threshold

@dataclass
class ChangeThresholdResult:
    value_db: float
    method: str                  # "fixed" | "noise_adaptive" | "noise_adaptive_floor" | "fixed_fallback"
    robust_sigma_db: float | None
    n_samples: int
    note: str = ""

    def as_dict(self) -> dict:
        return {"value_db": round(float(self.value_db), 3), "method": self.method,
                "robust_sigma_db": None if self.robust_sigma_db is None else round(float(self.robust_sigma_db), 3),
                "n_samples": self.n_samples, "note": self.note}


def robust_sigma(values: np.ndarray) -> float:
    """1.4826 * median absolute deviation (Gaussian-consistent, tolerant of a minority of real change).

    A one-sided fraction of real change biases it upward somewhat (~+14 % with 10 % strong change),
    which makes a k*sigma threshold slightly conservative - never permissive.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def change_threshold(log_ratio_db: np.ndarray, mode: str, floor_db: float, k_sigma: float,
                     min_samples: int) -> ChangeThresholdResult:
    """|log-ratio| threshold for one polarisation.

    fixed          -> floor_db.
    noise_adaptive -> max(floor_db, k_sigma * robust_sigma(log_ratio)); the floor keeps the
                      threshold from dropping below a physically meaningful change on very
                      clean scenes. Falls back to floor_db when there are too few samples.
    """
    v = log_ratio_db[np.isfinite(log_ratio_db)]
    if mode == "fixed":
        return ChangeThresholdResult(floor_db, "fixed", None, int(v.size))
    if mode != "noise_adaptive":
        raise ValueError(f"Unknown change_threshold mode {mode!r}")
    if k_sigma <= 0:
        raise ValueError("k_sigma must be > 0")
    if v.size < min_samples:
        return ChangeThresholdResult(floor_db, "fixed_fallback", None, int(v.size),
                                     f"only {v.size} samples (< {min_samples}); used {floor_db} dB")
    sigma = robust_sigma(v)
    adaptive = k_sigma * sigma
    if adaptive <= floor_db:
        return ChangeThresholdResult(floor_db, "noise_adaptive_floor", sigma, int(v.size),
                                     f"{k_sigma:g} x sigma = {adaptive:.2f} dB <= floor; used floor {floor_db} dB")
    return ChangeThresholdResult(adaptive, "noise_adaptive", sigma, int(v.size),
                                 f"{k_sigma:g} x robust sigma {sigma:.2f} dB = {adaptive:.2f} dB")
