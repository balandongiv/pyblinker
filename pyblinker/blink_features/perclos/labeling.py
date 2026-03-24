"""Fatigue-label assignment from epoch-level PERCLOS values."""

from __future__ import annotations

import math
from typing import Iterable, List


def _validate_cutoff(perclos_cutoff: float) -> float:
    cutoff = float(perclos_cutoff)
    if not math.isfinite(cutoff):
        raise ValueError("perclos_cutoff must be finite")
    if cutoff < 0.0 or cutoff > 1.0:
        raise ValueError("perclos_cutoff must be between 0.0 and 1.0")
    return cutoff


def assign_fatigue_label(perclos: float, *, perclos_cutoff: float = 0.80) -> int:
    """Return the binary fatigue label for one PERCLOS value."""

    cutoff = _validate_cutoff(perclos_cutoff)
    value = float(perclos)
    if not math.isfinite(value):
        raise ValueError("perclos must be finite")
    return int(value > cutoff)


def assign_fatigue_labels(
    values: Iterable[float], *, perclos_cutoff: float = 0.80
) -> List[int]:
    """Return binary fatigue labels for multiple PERCLOS values."""

    cutoff = _validate_cutoff(perclos_cutoff)
    return [assign_fatigue_label(value, perclos_cutoff=cutoff) for value in values]


__all__ = ["assign_fatigue_label", "assign_fatigue_labels"]
