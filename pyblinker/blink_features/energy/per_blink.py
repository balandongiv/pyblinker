"""Per-blink energy feature calculations."""
from typing import Any, Dict
import logging
import numpy as np

logger = logging.getLogger(__name__)


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
    dt = 1.0 / sfreq

    if segment.size < 2:
        logging.warning("Segment too short to compute energy metrics. Returning NaNs.")
        return {
            "blink_signal_energy": float("nan"),
            "teager_kaiser_energy": float("nan"),
            "blink_line_length": float("nan"),
            "blink_velocity_integral": float("nan"),
        }

    energy = float(np.trapezoid(segment ** 2, dx=dt))

    if segment.size > 2:
        tkeo = segment[1:-1] ** 2 - segment[:-2] * segment[2:]
        teager = float(np.sum(np.abs(tkeo)) * dt)
    else:
        teager = float("nan")

    line_length = float(np.sum(np.abs(np.diff(segment))))

    velocity = np.gradient(segment, dt)
    vel_integral = float(np.trapezoid(np.abs(velocity), dx=dt))

    logger.debug(
        "Blink energy: energy=%s, teager=%s, line_length=%s, vel_int=%s",
        energy,
        teager,
        line_length,
        vel_integral,
    )

    return {
        "blink_signal_energy": energy,
        "teager_kaiser_energy": teager,
        "blink_line_length": line_length,
        "blink_velocity_integral": vel_integral,
    }
