"""Time-domain energy features for 30-second segments."""

from __future__ import annotations

from typing import Dict

import numpy as np

from pyblinker._logging import get_logger

from .common import compute_energy_metrics

logger = get_logger(__name__)


def compute_time_domain_features(signal: np.ndarray, sfreq: float) -> Dict[str, float]:
    """Compute energy metrics for a signal segment.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional eyelid aperture samples for the segment.
    sfreq : float
        Sampling frequency in Hertz.

    Returns
    -------
    dict
        Dictionary with energy, Teager energy, line length and velocity integral.
    """
    signal_arr = np.asarray(signal, dtype=float)
    logger.debug(
        "Computing time-domain features for segment of length %d", signal_arr.size
    )
    metrics = compute_energy_metrics(signal_arr, sfreq)

    features = {
        "energy": float(metrics["signal_energy"]),
        "teager": float(metrics["teager_kaiser_energy"]),
        "line_length": float(metrics["line_length"]),
        "velocity_integral": float(metrics["velocity_integral"]),
    }
    logger.debug("Time-domain feature values: %s", features)
    return features
