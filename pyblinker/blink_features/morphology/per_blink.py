"""Per-blink morphology feature calculations."""
from __future__ import annotations

from typing import Dict, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_blink_waveform_metrics(segment: np.ndarray, sfreq: float) -> Optional[Dict[str, float]]:
    """Compute morphology metrics for a single blink waveform.

    Parameters
    ----------
    segment : numpy.ndarray
        One-dimensional blink waveform. The segment is expected to start and end
        at the eyelid baseline.
    sfreq : float
        Sampling frequency in Hertz.

    Returns
    -------
    dict or None
        Dictionary with morphology metrics or ``None`` if the segment is too
        short to evaluate.

    Notes
    -----
    ``rise_time`` measures the latency from segment start to peak amplitude,
    while ``fall_time`` spans from the peak back to the segment end. The
    ``half_width`` corresponds to the full width at half of the peak-to-peak
    amplitude. ``slope_rise`` and ``slope_fall`` capture the extreme positive
    and negative derivatives of the waveform.
    """
    segment = np.asarray(segment, dtype=float)
    if segment.size < 3:
        logger.debug("Segment too short for morphology metrics: len=%d", segment.size)
        return None

    peak_idx = int(np.argmax(segment))
    trough_idx = int(np.argmin(segment))
    peak = float(segment[peak_idx])
    trough = float(segment[trough_idx])
    peak_to_peak = float(peak - trough)
    area_abs = float(np.trapezoid(np.abs(segment), dx=1.0 / sfreq))

    rise_time = peak_idx / sfreq
    fall_time = (segment.size - 1 - peak_idx) / sfreq

    half_level = trough + 0.5 * peak_to_peak
    left = next((i for i in range(peak_idx, -1, -1) if segment[i] <= half_level), None)
    right = next((i for i in range(peak_idx, segment.size) if segment[i] <= half_level), None)
    if left is None or right is None or right <= left:
        half_width = float("nan")
    else:
        half_width = (right - left) / sfreq

    deriv = np.diff(segment) * sfreq
    if deriv.size == 0:
        slope_rise = float("nan")
        slope_fall = float("nan")
    else:
        slope_rise = float(np.max(deriv))
        slope_fall = float(np.min(deriv))

    logger.debug(
        "Blink metrics: peak=%s trough=%s peak_to_peak=%s", peak, trough, peak_to_peak
    )

    return {
        "peak_amplitude": peak,
        "trough_amplitude": trough,
        "peak_to_peak": peak_to_peak,
        "area_abs": area_abs,
        "rise_time": rise_time,
        "fall_time": fall_time,
        "half_width": half_width,
        "slope_rise": slope_rise,
        "slope_fall": slope_fall,
    }


# Derive metric names for reuse elsewhere without hardcoding
WAVEFORM_METRICS = tuple(compute_blink_waveform_metrics(np.zeros(3), 1.0).keys())

