"""Shared blink-feature constants and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import os

import mne
import numpy as np

STATS = ("mean", "std", "cv")

_DEFAULT_BASE_FRACTION = 0.1
_DEFAULT_SHUT_AMP_FRACTION = 0.9
_DEFAULT_P_AVR_THRESHOLD = 3.0
_DEFAULT_Z_THRESHOLDS = np.array([[0.9, 0.98], [2.0, 5.0]], dtype=float)

METRICS_BY_FAMILY = {
    "energy": (
        "blink_signal_energy",
        "teager_kaiser_energy",
        "blink_line_length",
        "blink_velocity_integral",
    ),
}


@dataclass(frozen=True)
class BlinkerConfig:
    """Configuration values shared by feature extractors."""

    stat_names: tuple[str, ...] = STATS
    base_fraction: float = _DEFAULT_BASE_FRACTION
    shut_amp_fraction: float = _DEFAULT_SHUT_AMP_FRACTION
    p_avr_threshold: float = _DEFAULT_P_AVR_THRESHOLD
    z_thresholds: np.ndarray = field(default_factory=lambda: _DEFAULT_Z_THRESHOLDS.copy())


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def default_blinker_config() -> BlinkerConfig:
    """Build config defaults allowing environment variable overrides."""

    return BlinkerConfig(
        stat_names=STATS,
        base_fraction=_env_float("PYBLINKER_BASE_FRACTION", _DEFAULT_BASE_FRACTION),
        shut_amp_fraction=_env_float("PYBLINKER_SHUT_AMP_FRACTION", _DEFAULT_SHUT_AMP_FRACTION),
        p_avr_threshold=_env_float("PYBLINKER_P_AVR_THRESHOLD", _DEFAULT_P_AVR_THRESHOLD),
        z_thresholds=_DEFAULT_Z_THRESHOLDS.copy(),
    )


DEFAULT_BLINKER_CONFIG = default_blinker_config()


def infer_modality(channel_name: str, info: mne.Info) -> str:
    """Infer modality label (ear/eeg/eog) from channel metadata."""

    ch_type = info.get_channel_types(picks=[channel_name])[0]
    ch_lower = channel_name.lower()
    if "ear" in ch_lower:
        return "ear"
    if ch_type == "eog" or "eog" in ch_lower:
        return "eog"
    if ch_type == "eeg" or "eeg" in ch_lower:
        return "eeg"
    return ch_type.lower()
