"""Time-domain energy features for 30-second segments."""
from typing import Dict
import logging
import numpy as np

logger = logging.getLogger(__name__)


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
    logger.info("Computing time-domain features for segment of length %d", len(signal))
    dt = 1.0 / sfreq
    energy = float(np.trapezoid(signal ** 2, dx=dt))

    if signal.size > 2:
        tkeo = signal[1:-1] ** 2 - signal[:-2] * signal[2:]
        teager = float(np.sum(np.abs(tkeo)) * dt)
    else:
        teager = float("nan")

    line_length = float(np.sum(np.abs(np.diff(signal))))
    velocity = np.gradient(signal, dt)
    velocity_integral = float(np.trapezoid(np.abs(velocity), dx=dt))

    features = {
        "energy": energy,
        "teager": teager,
        "line_length": line_length,
        "velocity_integral": velocity_integral,
    }
    logger.debug("Time-domain feature values: %s", features)
    return features

