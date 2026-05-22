"""Per-blink energy feature calculations."""

from typing import Any, Dict

import numpy as np

from pyblinker._logging import get_logger

from .common import compute_energy_metrics

logger = get_logger(__name__)


def compute_blink_energy(blink: Dict[str, Any], sfreq: float) -> Dict[str, float]:
    """Compute energy-related metrics for a single blink.

    These features quantify the overall effort of a blink and the
    intricacy of eyelid motion. Reduced signal energy or line length has
    been observed in drowsy drivers, reflecting diminished neuromuscular
    activation and smoother, less forceful blinks.

    Parameters
    ----------
    blink : dict
        Blink annotation with ``refined_start_frame``, ``refined_end_frame``
        and ``epoch_signal``.
    sfreq : float
        Sampling frequency of the recording in Hertz.

    Returns
    -------
    dict
        Dictionary with signal energy, Teager–Kaiser energy, line length
        and the integral of absolute velocity for the blink.
    """
    start = int(blink["refined_start_frame"])
    end = int(blink["refined_end_frame"])
    signal = np.asarray(blink["epoch_signal"], dtype=float)

    segment = signal[start : end + 1]
    metrics = compute_energy_metrics(segment, sfreq)

    logger.debug("Blink metrics: %s", metrics)

    return {
        "blink_signal_energy": float(metrics["signal_energy"]),
        "teager_kaiser_energy": float(metrics["teager_kaiser_energy"]),
        "blink_line_length": float(metrics["line_length"]),
        "blink_velocity_integral": float(metrics["velocity_integral"]),
    }
