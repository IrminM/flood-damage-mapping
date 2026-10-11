"""Baseline Sentinel-1 change detection: `s1_logratio_v1`.

Inputs : PreprocessedPair (VV[/VH] dB, before/after, same grid), optional TerrainProducts.
Outputs: ChangeResult with canonical classes
           0 no detected change
           1 new open water        after VV <= T_water, before VV > T_water, VV drop >= min_vv_drop_db
           2 water in both dates   before and after VV <= T_water
           3 non-water change      |log-ratio| >= change_abs_db in VV or VH, and not water after
         255 not observable        outside AOI, nodata, radar shadow/layover
         plus per-pixel exclusion flags explaining every rejection.

Rules of interpretation (kept in all labels):
  * class 1 detects OPEN water only (specular, dark). Flooded vegetation and flooded
    built-up areas brighten (double bounce) and are NOT captured as water here.
  * class 3 is NOT debris. It is any strong backscatter change that is not water:
    debris/mud deposits, damaged structures, landslides, but also vegetation/crop/
    moisture change and residual noise.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

from ..core.config import LogRatioV1Config, Settings
from ..core.contracts import ChangeContext, ExclusionFlag, OutputClass
from ..satellite.preprocess import PreprocessedPair
from ..satellite.terrain import TerrainProducts
from .interface import ChangeResult, register
from .thresholds import change_threshold, robust_sigma, water_threshold

U8 = np.uint8


HAND_BANDS_M = [(0.0, 10.0), (10.0, 25.0), (25.0, 100.0), (100.0, float("inf"))]
SLOPE_BANDS_DEG = [(0.0, 5.0), (5.0, 15.0), (15.0, 35.0), (35.0, 90.1)]


def _band_breakdown(values: np.ndarray, bands, change: np.ndarray, base: np.ndarray, px_area: float) -> list[dict]:
    """Share of class-3 pixels per band vs share of the observable non-water area per band.

    enrichment = share_of_change / share_of_area; > 1 means change is concentrated in that band.
    """
    n_change, n_base = max(int(change.sum()), 1), max(int(base.sum()), 1)
    out = []
    with np.errstate(invalid="ignore"):
        for lo, hi in bands:
            m = (values >= lo) & (values < hi)
            sc, sa = (change & m).sum() / n_change, (base & m).sum() / n_base
            out.append({"band": [lo, None if np.isinf(hi) else hi], "class3_km2": round(float((change & m).sum()) * px_area / 1e6, 4),
                        "share_of_class3": round(float(sc), 4), "share_of_area": round(float(sa), 4),
                        "enrichment": round(float(sc / sa), 3) if sa > 0 else None})
        unknown = ~np.isfinite(values)
        if (base & unknown).any():
            out.append({"band": "unknown", "class3_km2": round(float((change & unknown).sum()) * px_area / 1e6, 4),
                        "share_of_class3": round(float((change & unknown).sum() / n_change), 4),
                        "share_of_area": round(float((base & unknown).sum() / n_base), 4), "enrichment": None})
    return out


def change_diagnostics(lr_vv: np.ndarray, lr_vh: Optional[np.ndarray], nonwater: np.ndarray,
                       change_raw: np.ndarray, change_final: np.ndarray, terrain: Optional[TerrainProducts],
                       thresholds_db: dict[str, float], px_area: float) -> dict:
    """Evidence for judging the non-water change class. Does NOT alter any class.

    * noise: robust sigma (1.4826*MAD) of the log-ratio over observable non-water pixels, the
      exceedance of +/-threshold a Gaussian with that sigma would give, and the observed exceedance.
      The Gaussian figure is a LOWER BOUND: speckle log-ratios are heavier-tailed (in a synthetic
      no-change pair it predicts ~0 % while ~0.35 % of pixels exceed 3 dB before clean-up). The
      real false-alarm rate must come from a pre-event "null" pair processed the same way.
    * sign: brightening vs darkening share of the final class-3 pixels.
    * terrain: class-3 share per HAND / slope band vs that band's share of the area (enrichment).
      Valley-bottom processes (flood, debris deposition) should be enriched at low HAND.
    """
    from scipy.special import ndtr

    d: dict = {"threshold_db": {k: round(float(v), 3) for k, v in thresholds_db.items()}, "noise": {}}
    expected_any = 1.0
    for name, lr in (("VV", lr_vv), ("VH", lr_vh)):
        if lr is None:
            continue
        v = lr[nonwater & np.isfinite(lr)]
        if v.size < 100:
            continue
        threshold_db = thresholds_db[name]
        med = float(np.median(v))
        sigma = robust_sigma(v)
        p_noise = float(2.0 * ndtr(-threshold_db / sigma)) if sigma > 0 else 0.0
        expected_any *= (1.0 - p_noise)
        d["noise"][name] = {"threshold_db": round(float(threshold_db), 3),
                            "median_db": round(med, 3), "robust_sigma_db": round(sigma, 3),
                            "gaussian_exceedance_lower_bound": round(p_noise, 5),
                            "observed_exceedance": round(float((np.abs(v) >= threshold_db).mean()), 5)}
    n_base = max(int(nonwater.sum()), 1)
    d["gaussian_exceedance_any_pol_lower_bound"] = round(1.0 - expected_any, 5)
    d["observed_candidates_fraction"] = round(float(change_raw.sum()) / n_base, 5)
    d["final_class3_fraction"] = round(float(change_final.sum()) / n_base, 5)
    if lr_vh is not None:
        dominant = np.where(np.abs(lr_vh) > np.abs(lr_vv), lr_vh, lr_vv)
    else:
        dominant = lr_vv
    n3 = max(int(change_final.sum()), 1)
    d["class3_sign"] = {"brightening_share": round(float((change_final & (dominant > 0)).sum()) / n3, 4),
                        "darkening_share": round(float((change_final & (dominant < 0)).sum()) / n3, 4)}
    if terrain is not None:
        d["class3_by_hand_m"] = _band_breakdown(terrain.hand_m.data[0], HAND_BANDS_M, change_final, nonwater, px_area)
        d["class3_by_slope_deg"] = _band_breakdown(terrain.slope_deg.data[0], SLOPE_BANDS_DEG, change_final,
                                                   nonwater, px_area)
    return d


def classify_change_context(change: np.ndarray, observable: np.ndarray, hand: Optional[np.ndarray],
                            slope: Optional[np.ndarray], valley_max_hand_m: float,
                            valley_max_slope_deg: float) -> np.ndarray:
    """uint8 ChangeContext raster: label class-3 pixels as valley bottom / hillside / unknown.

    Valley bottom = HAND <= valley_max_hand_m AND slope <= valley_max_slope_deg (where floods and
    debris flows deposit). Labels only - the change classes themselves are not modified.
    """
    out = np.full(change.shape, U8(ChangeContext.NODATA), U8)
    out[observable] = U8(ChangeContext.NOT_CHANGE)
    if hand is None or slope is None:
        out[change] = U8(ChangeContext.UNKNOWN)
        return out
    known = np.isfinite(hand) & np.isfinite(slope)
    with np.errstate(invalid="ignore"):
        valley = known & (hand <= valley_max_hand_m) & (slope <= valley_max_slope_deg)
    out[change & valley] = U8(ChangeContext.VALLEY_BOTTOM)
    out[change & known & ~valley] = U8(ChangeContext.HILLSIDE)
    out[change & ~known] = U8(ChangeContext.UNKNOWN)
    return out


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def clean_mask(mask: np.ndarray, opening_radius_px: int, min_pixels: int) -> np.ndarray:
    """Morphological opening + removal of connected blobs smaller than min_pixels (8-connected)."""
    out = mask.copy()
    if opening_radius_px > 0:
        out = ndimage.binary_opening(out, structure=_disk(opening_radius_px))
    if min_pixels > 1 and out.any():
        lab, n = ndimage.label(out, structure=np.ones((3, 3), bool))
        sizes = np.bincount(lab.ravel())
        small = sizes < min_pixels
        small[0] = False
        out[small[lab]] = False
    return out


class S1LogRatioV1:
    name = "s1_logratio_v1"

    def __init__(self, cfg: LogRatioV1Config):
        self.cfg = cfg

    def detect(self, pair: PreprocessedPair, terrain: Optional[TerrainProducts]) -> ChangeResult:
        cfg = self.cfg
        warnings: list[str] = []
        b, a = pair.before_db, pair.after_db
        vv_b, vv_a = b.band("VV"), a.band("VV")
        has_vh = "VH" in pair.bands
        lr_vv = vv_a - vv_b
        lr_vh = a.band("VH") - b.band("VH") if has_vh else None
        valid = pair.valid.copy()

        excl = np.zeros(valid.shape, U8)
        excl[~pair.aoi_mask] |= U8(ExclusionFlag.OUTSIDE_AOI)
        excl[pair.nodata_in_aoi] |= U8(ExclusionFlag.NO_DATA)

        # --- terrain: geometry we cannot observe, and gates for physical plausibility
        if terrain is not None:
            shadow = terrain.shadow & valid
            layover = terrain.layover & valid
            excl[shadow] |= U8(ExclusionFlag.RADAR_SHADOW)
            excl[layover] |= U8(ExclusionFlag.RADAR_LAYOVER)
            observable = valid & ~shadow & ~layover
            slope, hand = terrain.slope_deg.data[0], terrain.hand_m.data[0]
            with np.errstate(invalid="ignore"):
                # unknown slope/HAND (NaN) does not block: the gate cannot be evaluated there
                water_gate_fail = (slope > cfg.water_max_slope_deg) | (hand > cfg.water_max_hand_m)
                change_gate_fail = slope > cfg.change_max_slope_deg
        else:
            warnings.append("No DEM: radar shadow/layover and slope/HAND gates were NOT applied; "
                            "expect false water on dark mountain slopes")
            observable = valid
            water_gate_fail = np.zeros(valid.shape, bool)
            change_gate_fail = np.zeros(valid.shape, bool)

        # --- dark-water threshold from observable, water-plausible pixels
        wt = cfg.water_threshold
        sample = vv_a[observable & ~water_gate_fail]
        thr = water_threshold(sample, wt.mode, wt.fixed_db, tuple(wt.otsu_bounds_db), wt.otsu_min_samples,
                              out_of_bounds=wt.otsu_out_of_bounds, min_ashman_d=wt.min_ashman_d)
        if thr.note:
            warnings.append(f"Water threshold: {thr.note}")
        T = thr.value_db

        with np.errstate(invalid="ignore"):
            water_after = observable & (vv_a <= T)
            water_before = observable & (vv_b <= T)
            new_water = water_after & ~water_before & (lr_vv <= -cfg.min_vv_drop_db)
            both_water = water_after & water_before

        # --- non-water change threshold per polarisation (fixed or noise-adaptive), estimated on
        #     observable non-water pixels of THIS pair
        ct = cfg.change_threshold
        nonwater = observable & ~water_after
        change_thr = {}
        for pol, lr in (("VV", lr_vv), ("VH", lr_vh)):
            if lr is not None:
                change_thr[pol] = change_threshold(lr[nonwater], ct.mode, cfg.change_abs_db, ct.k_sigma,
                                                   ct.min_samples)
                if change_thr[pol].method == "fixed_fallback":
                    warnings.append(f"Change threshold {pol}: {change_thr[pol].note}")
        with np.errstate(invalid="ignore"):
            big_change = np.abs(lr_vv) >= change_thr["VV"].value_db
            if lr_vh is not None:
                big_change |= np.abs(lr_vh) >= change_thr["VH"].value_db
            change = nonwater & big_change

        cand_px = {"new_water": int(new_water.sum()), "both_water": int(both_water.sum()),
                   "change": int(change.sum())}
        cand_change = change.copy()          # before terrain gate and clean-up (for diagnostics)

        # --- terrain gates
        rej_w = (new_water | both_water) & water_gate_fail
        rej_c = change & change_gate_fail
        excl[rej_w] |= U8(ExclusionFlag.TERRAIN_WATER_GATE)
        excl[rej_c] |= U8(ExclusionFlag.TERRAIN_CHANGE_GATE)
        new_water &= ~water_gate_fail
        both_water &= ~water_gate_fail
        change &= ~change_gate_fail

        # --- clean-up (speckle-sized blobs)
        px_area = pair.before_db.pixel_area
        min_px = int(np.ceil(cfg.min_area_m2 / px_area))
        cleaned = {}
        for name, m in (("new_water", new_water), ("both_water", both_water), ("change", change)):
            c = clean_mask(m, cfg.opening_radius_px, min_px)
            excl[m & ~c] |= U8(ExclusionFlag.SMALL_BLOB)
            cleaned[name] = c

        cc = cfg.change_context
        context = None
        if cc.enabled:
            context = classify_change_context(
                cleaned["change"], observable,
                terrain.hand_m.data[0] if terrain is not None else None,
                terrain.slope_deg.data[0] if terrain is not None else None,
                cc.valley_max_hand_m, cc.valley_max_slope_deg)

        classes = np.full(valid.shape, U8(OutputClass.NODATA), U8)
        classes[observable] = U8(OutputClass.BACKGROUND)
        classes[cleaned["change"]] = U8(OutputClass.DEBRIS_OR_CHANGE)
        classes[cleaned["both_water"]] = U8(OutputClass.PERMANENT_WATER)
        classes[cleaned["new_water"]] = U8(OutputClass.FLOOD_WATER)

        lr_bands = [lr_vv] + ([lr_vh] if lr_vh is not None else [])
        lr_names = ["VV_log_ratio"] + (["VH_log_ratio"] if lr_vh is not None else [])
        lr = b.with_data(np.stack(lr_bands).astype(np.float32), band_names=lr_names, nodata=np.nan,
                         metadata={**b.metadata, "units": "dB", "definition": "after_dB - before_dB"})
        cls_r = b.with_data(classes, band_names=["change_class"], nodata=255,
                            metadata={**b.metadata, "method": self.name})
        stats = {
            "observable_pixels": int(observable.sum()),
            "candidates_px": cand_px,
            "rejected_by_terrain_px": {"water": int(rej_w.sum()), "change": int(rej_c.sum())},
            "median_log_ratio_vv_db": float(np.nanmedian(lr_vv[observable])) if observable.any() else None,
            "diagnostics": change_diagnostics(lr_vv, lr_vh, nonwater, cand_change, cleaned["change"], terrain,
                                              {k: v.value_db for k, v in change_thr.items()}, px_area),
        }
        thresholds = {"water_vv": thr.as_dict(),
                      "change": {"mode": ct.mode, "k_sigma": ct.k_sigma, "floor_db": cfg.change_abs_db,
                                 **{pol: r.as_dict() for pol, r in change_thr.items()}}}
        return ChangeResult(classes=cls_r, log_ratio_db=lr, exclusion=excl, method=self.name,
                            params=cfg.model_dump(), thresholds=thresholds, stats=stats, warnings=warnings,
                            context=context)


@register("s1_logratio_v1")
def _factory(settings: Settings) -> S1LogRatioV1:
    return S1LogRatioV1(settings.change_detection.s1_logratio_v1)
