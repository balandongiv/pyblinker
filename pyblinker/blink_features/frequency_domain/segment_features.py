"""Frequency-domain features for arbitrary segments."""

from __future__ import annotations
from pyblinker._logging import get_logger

from typing import Any, Dict, List

import numpy as np

from .features import _compute_wavelet_energies

logger = get_logger(__name__)


def compute_frequency_domain_features(
    blinks: List[Dict[str, Any]], segment_signal: np.ndarray, sfreq: float
) -> Dict[str, float]:
    """Compute wavelet energies for a single signal segment.

    Parameters
    ----------
    blinks : list of dict
        Ignored and only kept for backward compatibility with older APIs.
    segment_signal : numpy.ndarray
        Signal samples for the segment.
    sfreq : float
        Sampling frequency of ``segment_signal`` in Hertz.

    Returns
    -------
    dict
        Mapping from ``wavelet_energy_d1`` .. ``wavelet_energy_d4`` to their
        corresponding energy values. Levels that cannot be computed are ``NaN``.
    """

    logger.debug(
        "Computing segment frequency-domain features (n=%d sfreq=%.2f)",
        len(segment_signal),
        sfreq,
    )
    energies = _compute_wavelet_energies(np.asarray(segment_signal, dtype=float), sfreq)
    return {f"wavelet_energy_d{i + 1}": val for i, val in enumerate(energies)}
