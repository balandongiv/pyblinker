"""EAR extrema metrics.

This module extracts the minimum and maximum eye aspect ratio (EAR)
observed in an epoch. Extreme values reflect maximal eye closure and
opening, characterizing blink depth. Similar measurements exist in
the `Jena Facial Palsy Tool <https://github.com/cvjena/JeFaPaTo>`_.
"""

from typing import Dict
import numpy as np
from pyblinker._logging import get_logger

logger = get_logger(__name__)


def ear_extrema_epoch(epoch_signal: np.ndarray) -> Dict[str, float]:
    """Compute minimum and maximum EAR for a single epoch.

    Parameters
    ----------
    epoch_signal : numpy.ndarray
        EAR samples for the epoch.

    Returns
    -------
    dict
        Dictionary with keys ``ear_min`` and ``ear_max``.
    """
    ear_min = float(np.min(epoch_signal))
    ear_max = float(np.max(epoch_signal))
    logger.debug("EAR min=%s, max=%s", ear_min, ear_max)
    return {"ear_min": ear_min, "ear_max": ear_max}
