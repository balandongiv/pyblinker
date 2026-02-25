"""Blink feature defaults and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BlinkerConfig:
    """Configuration for blink feature extraction."""

    stat_names: tuple[str, ...] = ("mean", "std", "cv")
    base_fraction: float = 0.5
    shut_amp_fraction: float = 0.9
    p_avr_threshold: float = 3.0
    z_thresholds: np.ndarray = field(
        default_factory=lambda: np.array([[0.9, 0.98], [2.0, 5.0]], dtype=float)
    )


DEFAULT_CONFIG = BlinkerConfig()
