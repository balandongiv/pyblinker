"""Duration-related blink metrics based on the EAR waveform.

This module adapts ideas from the `BLINKER`_ project and its
`GitHub implementation`_ to compute simple blink durations.  The
functions measure how long the eyelid remains closed based on landmarks
derived from the eye aspect ratio (EAR) trace.

Example
-------
# >>> from pyblinker.features.waveform_features.features.duration_features import duration_base
# >>> feat = duration_base(blink, sfreq=100.0)

.. _BLINKER: https://github.com/VisLab/EEG-Blinks
.. _GitHub implementation: https://github.com/VisLab/EEG-Blinks
"""
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def duration_base(blink: Dict[str, Any], sfreq: float) -> float:
    """Compute blink duration from baseline crossings.

    This implementation is loosely adapted from the open-source
    `BLINKER`_ code base, which extracts blink timing from eyelid
    signals. It simply measures the interval between the refined
    start and end frames.

    Parameters
    ----------
    blink : dict
        Blink annotation with ``refined_start_frame`` and ``refined_end_frame``
        indices inside ``epoch_signal``.
    sfreq : float
        Sampling frequency in Hertz.
s
    Returns
    -------
    float
        Duration in seconds from blink start to end.
    """
    start = int(blink["refined_start_frame"])
    end = int(blink["refined_end_frame"])
    duration = (end - start) / sfreq
    logger.debug("duration_base=%s", duration)
    return float(duration)


def duration_zero(blink: Dict[str, Any], sfreq: float) -> float:
    """Compute blink duration using zero-slope landmarks.

    This approach is inspired by the `BLINKER`_ toolkit, which
    locates zero-crossings in the derivative of the EAR signal to
    refine blink onset and offset.

    The zero landmarks correspond to where the first derivative of the EAR
    signal crosses zero at blink onset and offset. If such points cannot be
    determined, the start and end indices are used as a fallback.

    Parameters
    ----------
    blink : dict
        Blink annotation with ``refined_start_frame``, ``refined_end_frame`` and
        ``epoch_signal``.
    sfreq : float
        Sampling frequency in Hertz.

    Returns
    -------
    float
        Duration in seconds between zero-slope landmarks.
    """
    start = int(blink["refined_start_frame"])
    end = int(blink["refined_end_frame"])
    signal = np.asarray(blink["epoch_signal"], dtype=float)
    segment = signal[start : end + 1]
    dt = 1.0 / sfreq
    if segment.size < 2:
        logger.warning("Segment too short in duration_zero. Returning NaN.")
        return float("nan")
    derivative = np.gradient(segment, dt)

    left_indices = np.where(np.diff(np.sign(derivative[: (end - start) // 2])))[0]
    right_indices = np.where(
        np.diff(np.sign(derivative[(end - start) // 2 :]))
    )[0]

    if left_indices.size > 0:
        left_zero = int(left_indices[0])
    else:
        left_zero = 0
    if right_indices.size > 0:
        right_zero = int(right_indices[-1] + (end - start) // 2)
    else:
        right_zero = len(segment) - 1

    duration = (right_zero - left_zero) / sfreq
    logger.debug("duration_zero=%s", duration)
    return float(duration)
