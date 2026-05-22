"""Velocity computation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from pyblinker._logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class SupportsCoef(Protocol):
    """Protocol describing polynomial-like objects exposing ``coef``."""

    @property
    def coef(self) -> Sequence[float]:
        """Return polynomial coefficients ordered by degree."""


CoefficientLike = SupportsCoef | Sequence[float] | np.ndarray


def _extract_linear_slope(coefficients: CoefficientLike) -> float:
    """Return the slope component from ``coefficients``."""

    if isinstance(coefficients, SupportsCoef):
        coef_array = np.asarray(coefficients.coef, dtype=float)
        if coef_array.size < 2:
            msg = "Coefficients must include a linear term to compute velocity."
            logger.error(msg)
            raise ValueError(msg)
        return float(coef_array[1])

    coef_array = np.asarray(coefficients, dtype=float).ravel()
    if coef_array.size == 0:
        msg = "Coefficient sequence must not be empty."
        logger.error(msg)
        raise ValueError(msg)
    return float(coef_array[0])


def average_velocity(
    coefficients: CoefficientLike,
    *,
    x_values: ArrayLike | None = None,
    x_scale: float | None = None,
) -> float:
    """Compute the average velocity associated with a linear fit."""

    if x_scale is None:
        if x_values is None:
            msg = "Either 'x_values' or 'x_scale' must be provided."
            logger.error(msg)
            raise ValueError(msg)
        x_array = np.asarray(x_values, dtype=float)
        if x_array.size == 0:
            msg = "'x_values' must contain at least one element."
            logger.error(msg)
            raise ValueError(msg)
        x_scale = float(np.std(x_array))

    slope = _extract_linear_slope(coefficients)
    return float(slope / x_scale)


__all__ = ["average_velocity"]
