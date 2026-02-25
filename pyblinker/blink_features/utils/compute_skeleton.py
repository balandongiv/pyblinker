"""Shared orchestration helpers for epoch/channel/style feature computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Set

import mne
import pandas as pd

from pyblinker.blink_features.constants import infer_modality
from pyblinker.blink_features.utils.aggregation import prepare_epoch_channel_data


@dataclass(frozen=True)
class ComputeContext:
    """Prepared orchestration inputs shared across feature families."""

    sfreq: float
    ch_names: list[str]
    channel_data: dict
    index: pd.Index
    n_epochs: int
    n_times: int
    modality_map: Dict[str, str]
    modality_channels: Dict[str, list[str]]
    styles_by_modality: Dict[str, Set[str]]


StyleGetter = Callable[[Sequence[str] | None, str], Set[str]]


def prepare_compute_context(
    *,
    epochs: mne.Epochs,
    picks: str | Sequence[str] | None,
    style_getter: StyleGetter,
) -> ComputeContext:
    """Prepare common compute inputs for feature extractor loops."""

    sfreq = float(epochs.info["sfreq"])
    ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
        epochs=epochs,
        picks=picks,
        sfreq=sfreq,
    )

    modality_map: Dict[str, str] = {ch: infer_modality(ch, epochs.info) for ch in ch_names}
    modality_channels: Dict[str, list[str]] = {}
    for ch, mod in modality_map.items():
        modality_channels.setdefault(mod, []).append(ch)

    metadata_cols: Sequence[str] | None = (
        tuple(epochs.metadata.columns) if isinstance(epochs.metadata, pd.DataFrame) else None
    )
    styles_by_modality = {
        mod: style_getter(metadata_cols, mod)
        for mod in set(modality_map.values())
    }

    return ComputeContext(
        sfreq=sfreq,
        ch_names=ch_names,
        channel_data=channel_data,
        index=index,
        n_epochs=n_epochs,
        n_times=n_times,
        modality_map=modality_map,
        modality_channels=modality_channels,
        styles_by_modality=styles_by_modality,
    )


def build_epoch_metadata_row(epochs: mne.Epochs, epoch_index: int) -> pd.Series:
    """Return a metadata row for a given epoch index."""

    return (
        epochs.metadata.iloc[epoch_index]
        if isinstance(epochs.metadata, pd.DataFrame)
        else pd.Series(dtype=float)
    )
