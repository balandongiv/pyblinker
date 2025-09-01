"""Aggregate blink event features using :class:`mne.Epochs` metadata."""
from __future__ import annotations

from typing import Iterable, Sequence

import logging

import mne
import pandas as pd

from .blink_count import blink_count
from .inter_blink_interval import inter_blink_interval_epochs
from .utils import normalize_picks, require_channels

logger = logging.getLogger(__name__)


def aggregate_blink_event_features(
    epochs: mne.Epochs,
    picks: str | Iterable[str],
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate blink-event metrics for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoch object with metadata containing blink onset and duration
        information. If modality-specific columns such as ``blink_onset_eeg``
        exist they are used; otherwise the generic ``blink_onset`` and
        ``blink_duration`` columns are expected.
    picks : str or iterable of str
        Channel name(s) used when computing inter-blink interval (IBI)
        statistics. The same IBI values are used for all channels because
        blink timing is not channel-specific in the metadata yet.
    features : sequence of str or None, optional
        Subset of feature groups to compute. Valid keys are
        ``"blink_total"``, ``"blink_rate"`` and ``"ibi"``. Passing ``None``
        (default) computes all features.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` containing one row per epoch with the
        requested features. Columns may include ``ep``, ``blink_total``,
        ``blink_rate`` and ``ibi_<channel>`` depending on ``features``.

    Raises
    ------
    ValueError
        If an unknown feature key is requested or if required channels are
        missing from ``epochs`` when ``"ibi"`` is selected.
    """

    logger.info("Aggregating blink features for %d epochs", len(epochs))

    valid = {"blink_total", "blink_rate", "ibi"}
    selected = set(features) if features is not None else valid
    invalid = selected - valid
    if invalid:
        raise ValueError(f"Unknown feature keys: {sorted(invalid)}")

    pieces: list[pd.DataFrame] = []

    if selected & {"blink_total", "blink_rate"}:
        counts_df = blink_count(epochs).rename(
            columns={"blink_count": "blink_total"}
        )
        pieces.append(counts_df)

    if "ibi" in selected:
        picks_list = normalize_picks(picks)
        require_channels(epochs, picks_list)
        ibis_df = inter_blink_interval_epochs(epochs, picks_list)
        if pieces:
            ibis_df = ibis_df.drop(columns=["ep"])
        pieces.append(ibis_df)
    elif not selected:
        # If no features selected we still need an empty index-aligned frame
        pieces.append(pd.DataFrame(index=range(len(epochs))))

    df = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=range(len(epochs)))

    if "ep" in df.columns:
        df = df[["ep"] + [c for c in df.columns if c != "ep"]]

    if "blink_rate" in selected:
        epoch_len = epochs.tmax - epochs.tmin + 1.0 / epochs.info["sfreq"]
        df["blink_rate"] = df["blink_total"] / epoch_len * 60.0

    # Reduce to requested columns if a subset was specified
    if features is not None:
        cols: list[str] = []
        if "blink_total" in selected:
            cols.append("blink_total")
        if "blink_rate" in selected:
            cols.append("blink_rate")
        if "ibi" in selected:
            cols.extend(df.columns[df.columns.str.startswith("ibi_")].tolist())
        df = df[cols]

    logger.debug("Aggregated feature DataFrame shape: %s", df.shape)
    return df


__all__ = ["aggregate_blink_event_features"]

