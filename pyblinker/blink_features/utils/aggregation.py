"""Utility helpers for aggregating epoch-level blink features."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import mne
import numpy as np
import pandas as pd

from pyblinker._logging import get_logger

logger = get_logger(__name__)


def prepare_epoch_channel_data(
    *,
    epochs: mne.Epochs | None,
    picks: str | Sequence[str] | None,
    sfreq: float,
) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], pd.Index, int, int]:
    """Validate epochs and return channel data for aggregation."""

    if epochs is None:
        raise ValueError("self.epochs is required for feature computation")

    if sfreq < 30:
        logger.warning(
            "Frequency-domain features may be unreliable below 30 Hz",
            extra={"sfreq": sfreq},
        )

    ch_names = _normalize_picks(picks, epochs.ch_names)
    _raise_for_missing_channels(ch_names, epochs.ch_names)

    raw_channel_data: Dict[str, np.ndarray] = {
        ch: epochs.get_data(picks=[ch])[:, 0, :] for ch in ch_names
    }
    n_epochs, n_times = next(iter(raw_channel_data.values())).shape
    channel_data: Dict[str, Dict[str, np.ndarray]] = {}
    for ch, raw in raw_channel_data.items():
        dx1 = np.gradient(raw, axis=1) * sfreq
        dx2 = np.gradient(dx1, axis=1) * sfreq
        channel_data[ch] = {"raw": raw, "dx1": dx1, "dx2": dx2}
    index = (
        epochs.metadata.index
        if isinstance(epochs.metadata, pd.DataFrame)
        else pd.RangeIndex(n_epochs)
    )
    return ch_names, channel_data, index, n_epochs, n_times


def _normalize_picks(
    picks: str | Sequence[str] | None, epoch_channels: Iterable[str]
) -> List[str]:
    if picks is None:
        return list(epoch_channels)
    if isinstance(picks, str):
        return [picks]
    return list(picks)


def _raise_for_missing_channels(
    ch_names: List[str], epoch_channels: Iterable[str]
) -> None:
    missing = [ch for ch in ch_names if ch not in epoch_channels]
    if missing:
        raise ValueError(f"Channels not found: {missing}")
