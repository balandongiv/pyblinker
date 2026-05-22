"""Shared utilities for blink morphology and kinematic metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

from pyblinker._logging import get_logger

logger = get_logger(__name__)

#: Supported segmentation methods per modality. The EEG/EOG modality exposes
#: all historical landmark strategies, while EAR (Eye Aspect Ratio) blinks do
#: not have zero-crossings and therefore omit the ``zero`` variants.
METHODS_BY_MODALITY: Dict[str, Sequence[str]] = {
    "eeg": ("base", "zero", "tent", "half_base", "half_zero"),
    "eog": ("base", "zero", "tent", "half_base", "half_zero"),
    "ear": ("base", "tent", "half_base"),
}

#: Ordered metric stems produced by :func:`compute_blink_core`.  The concrete
#: keys are created by appending ``_{method}`` where ``method`` is one of the
#: segmentation strategies above.
CANONICAL_METRIC_STEMS: Sequence[str] = (
    "area_abs_total_trapz",
    "area_abs_total_rect",
    "symmetry_trapz",
    "symmetry_rect",
    "rise_time_peak",
    "fall_time_peak",
    "rise_time_10_90",
    "fall_time_10_90",
    "half_width",
    "vel_peak_abs",
    "vel_mean_abs",
    "slope_rise_pos",
    "slope_fall_neg",
    "acc_peak_abs",
    "acc_mean_abs",
    "amp_peak_signed",
    "amp_trough_signed",
    "amp_peak_to_trough",
    "amp_peak_abs",
)

ALL_METHODS = tuple(
    sorted({method for methods in METHODS_BY_MODALITY.values() for method in methods})
)

_EPS = 1e-12


def _method_keys(method: str, stems: Sequence[str]) -> Sequence[str]:
    return tuple(f"{stem}_{method}" for stem in stems)


def core_nan_dict(keys: Iterable[str]) -> Dict[str, float]:
    """Return a dictionary that maps ``keys`` to ``NaN`` values."""

    return {key: float("nan") for key in keys}


def _symmetry(left: float, right: float) -> float:
    denom = left + right
    if np.isnan(left) or np.isnan(right) or abs(denom) <= _EPS:
        return float("nan")
    return (left - right) / denom


def _first_index_ge(values: np.ndarray, threshold: float) -> int | None:
    matches = np.flatnonzero(values >= threshold)
    return int(matches[0]) if matches.size else None


def _first_index_le(values: np.ndarray, threshold: float) -> int | None:
    matches = np.flatnonzero(values <= threshold)
    return int(matches[0]) if matches.size else None


def normalize_modality(modality: str | None) -> str:
    """Normalize modality inputs to canonical keys used by blink metrics."""

    modality_norm = str(modality or "eeg").lower()
    if modality_norm == "eog":
        modality_norm = "eeg"
    return "ear" if modality_norm == "ear" else "eeg"
