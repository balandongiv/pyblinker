"""Shared helpers for computing blink energy metrics."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from pyblinker._logging import get_logger

logger = get_logger(__name__)


def compute_energy_metrics(
    segment: Iterable[float], sfreq: float
) -> Mapping[str, float]:
    """Return energy-related metrics for a one-dimensional signal segment.

    Parameters
    ----------
    segment
        Sequence of eyelid aperture samples representing a blink or
        arbitrary signal segment.
    sfreq
        Sampling frequency of ``segment`` in Hertz.

    Returns
    -------
    Mapping[str, float]
        Dictionary containing ``signal_energy``, ``teager_kaiser_energy``,
        ``line_length`` and ``velocity_integral`` entries. ``NaN`` values are
        returned when the segment is too short to evaluate a metric.
    """

    data = np.asarray(segment, dtype=float).ravel()
    if data.size == 0:
        logger.warning("Empty segment received; returning NaN metrics.")
        return {
            "signal_energy": float("nan"),
            "teager_kaiser_energy": float("nan"),
            "line_length": float("nan"),
            "velocity_integral": float("nan"),
        }

    dt = 1.0 / float(sfreq)

    if data.size < 2:
        logger.warning(
            "Segment too short to compute velocity or energy metrics. Returning NaNs."
        )
        return {
            "signal_energy": float("nan"),
            "teager_kaiser_energy": float("nan"),
            "line_length": float("nan"),
            "velocity_integral": float("nan"),
        }

    energy = float(np.trapezoid(data**2, dx=dt))

    if data.size > 2:
        tkeo = data[1:-1] ** 2 - data[:-2] * data[2:]
        teager = float(np.sum(np.abs(tkeo)) * dt)
    else:
        teager = float("nan")

    line_length = float(np.sum(np.abs(np.diff(data))))
    velocity = np.gradient(data, dt)
    velocity_integral = float(np.trapezoid(np.abs(velocity), dx=dt))

    metrics = {
        "signal_energy": energy,
        "teager_kaiser_energy": teager,
        "line_length": line_length,
        "velocity_integral": velocity_integral,
    }
    logger.debug("Computed energy metrics: %s", metrics)
    return metrics
