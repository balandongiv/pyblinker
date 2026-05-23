"""Prepared epoch detection input dataclass and one-time preparation function."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import mne
import numpy as np

from ..blinker.legacy_eeglab_filter import legacy_blinker_bandpass


@dataclass
class PreparedEpochDetectionInput:
    """Cached, preprocessed epoch data ready for per-channel concatenation."""

    data: np.ndarray
    channel_names: tuple[str, ...]
    sfreq: float
    epoch_length_samples: int
    selection: np.ndarray


def _resample_epoch_array(
    data: np.ndarray,
    *,
    orig_sfreq: float,
    target_sfreq: float,
) -> np.ndarray:
    if np.isclose(orig_sfreq, target_sfreq):
        return data
    ratio = (
        Fraction(str(target_sfreq)).limit_denominator(1000)
        / Fraction(str(orig_sfreq)).limit_denominator(1000)
    ).limit_denominator(1000)
    return mne.filter.resample(
        data,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=-1,
        verbose="ERROR",
    )


def prepare_epoch_detection_input(
    epochs: mne.Epochs,
    *,
    pick_types_options: dict | None = None,
    filter_low: float = 1.0,
    filter_high: float = 20.0,
    resample_rate: float | None = None,
) -> PreparedEpochDetectionInput:
    """Load, pick, filter, and optionally resample epoch data once."""

    epochs.load_data()
    print("Total epochs:", len(epochs))
    pick_options = pick_types_options or {"eeg": True}
    picks = mne.pick_types(epochs.info, **pick_options)
    if picks.size == 0:
        raise ValueError("No channels matched the requested pick_types_options.")

    channel_names = tuple(epochs.ch_names[pick] for pick in picks)
    raw_data = epochs.get_data(picks=picks)
    orig_sfreq = float(epochs.info["sfreq"])
    target_sfreq = orig_sfreq if resample_rate in (None, 0) else float(resample_rate)

    processed_epochs: list[np.ndarray] = []
    for epoch_data in raw_data:
        filtered = legacy_blinker_bandpass(
            epoch_data,
            sfreq=orig_sfreq,
            low_cutoff_hz=float(filter_low),
            high_cutoff_hz=float(filter_high),
        )
        processed = _resample_epoch_array(
            filtered,
            orig_sfreq=orig_sfreq,
            target_sfreq=target_sfreq,
        )
        processed_epochs.append(np.asarray(processed, dtype=np.float64))

    prepared = np.stack(processed_epochs, axis=0) if processed_epochs else raw_data[:, :, :0]
    return PreparedEpochDetectionInput(
        data=prepared,
        channel_names=channel_names,
        sfreq=float(target_sfreq),
        epoch_length_samples=int(prepared.shape[-1]),
        selection=np.asarray(epochs.selection, dtype=int).copy(),
    )


__all__ = ["PreparedEpochDetectionInput", "prepare_epoch_detection_input"]
