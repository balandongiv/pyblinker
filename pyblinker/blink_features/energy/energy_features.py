"""Blink energy feature calculations.

Features are computed **per channel**, and column names are suffixed with
``_<channel>`` to clearly indicate the source channel.


"""
from __future__ import annotations
from pyblinker.logging import get_logger

from typing import Dict, List, Mapping, Sequence, Tuple

import mne
import pandas as pd

from .common import compute_energy_metrics
from .helpers import _safe_stats
from ...utils.iter_utils import ensure_list

logger = get_logger(__name__)

_METRICS = (
    "blink_signal_energy",
    "teager_kaiser_energy",
    "blink_line_length",
    "blink_velocity_integral",
)
_STATS = ("mean", "std", "cv")


def _make_columns(ch_names: Sequence[str]) -> List[str]:
    """Generate ordered column names for all metrics and statistics."""
    columns: List[str] = []
    for ch in ch_names:
        for metric in _METRICS:
            for stat in _STATS:
                columns.append(f"{metric}_{stat}_{ch}")
    return columns


def _compute_epoch_channel_energy_stats(
    *,
    metadata_row: pd.Series,
    ch: str,
    signal_1d,
    sfreq: float,
    n_times: int,
    info: mne.Info | None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-metric summary stats for all blinks in one epoch/channel.

    Returns
    -------
    dict
        Mapping of metric name -> stats dict (mean/std/cv). Stats are NaN if
        there are no valid blink segments.
    """
    windows = _channel_windows(
        metadata_row=metadata_row,
        channel_name=ch,
        info=info,
        n_times=n_times,
    )

    energies: List[float] = []
    tkeo_vals: List[float] = []
    lengths: List[float] = []
    vel_ints: List[float] = []

    for start_idx, end_idx in windows:
        if start_idx >= n_times:
            continue
        sl = slice(max(0, start_idx), min(end_idx, n_times))
        segment = signal_1d[sl]
        if getattr(segment, "size", 0) == 0:
            continue
        metrics = compute_energy_metrics(segment, sfreq)
        energies.append(float(metrics["signal_energy"]))
        tkeo_vals.append(float(metrics["teager_kaiser_energy"]))
        lengths.append(float(metrics["line_length"]))
        vel_ints.append(float(metrics["velocity_integral"]))

    # Average the metrics over all blinks in the epoch.
    stats_energy = _safe_stats(energies)
    stats_tkeo = _safe_stats(tkeo_vals)
    stats_len = _safe_stats(lengths)
    stats_vel = _safe_stats(vel_ints)

    return {
        _METRICS[0]: stats_energy,
        _METRICS[1]: stats_tkeo,
        _METRICS[2]: stats_len,
        _METRICS[3]: stats_vel,
    }


def _infer_modality(channel_name: str, info: mne.Info | None = None) -> str:
    """Infer modality label (ear/eeg/eog) from channel naming/type hints."""

    ch_lower = channel_name.lower()
    if "ear" in ch_lower:
        return "ear"

    if info is not None and channel_name in info["ch_names"]:
        ch_type = info.get_channel_types(picks=[channel_name])[0]
        if ch_type in {"eeg", "eog"}:
            return ch_type

    if "eog" in ch_lower:
        return "eog"
    return "eeg"


def _style_windows(
    metadata_row: Mapping[str, object],
    modality: str,
    style: str,
    n_times: int,
) -> List[Tuple[int, int]]:
    """Extract frame-aligned blink windows for a modality/style pair."""

    style_landmarks = {
        "zero_base": ("start__left_zero", "end__right_zero"),
        "tent": ("start__left_x_intercept", "end__right_x_intercept"),
        "half_peak": ("start__left_base_half_height", "end__right_base_half_height"),
        # canonical aliases used in morphology/kinematics metadata
        "zero": ("start__left_zero", "end__right_zero"),
        "base": ("start__left_base", "end__right_base"),
        "half_base": ("start__left_base_half_height", "end__right_base_half_height"),
        "half_zero": ("start__left_zero_half_height", "end__right_zero_half_height"),
    }

    if style in style_landmarks:
        start_prefix, end_prefix = style_landmarks[style]
        starts = ensure_list(metadata_row.get(f"{start_prefix}__{modality}"))
        ends = ensure_list(metadata_row.get(f"{end_prefix}__{modality}"))
    else:
        starts = ensure_list(metadata_row.get(f"start__{style}__{modality}"))
        ends = ensure_list(metadata_row.get(f"end__{style}__{modality}"))

    windows: List[Tuple[int, int]] = []
    for start_frame, end_frame in zip(starts, ends):
        if start_frame is None or end_frame is None:
            continue
        if pd.isna(start_frame) or pd.isna(end_frame):
            continue
        start_idx = max(int(round(float(start_frame))), 0)
        end_idx = min(int(round(float(end_frame))), n_times)
        if end_idx <= start_idx:
            continue
        windows.append((start_idx, end_idx))
    return windows


def _channel_windows(
    *,
    metadata_row: Mapping[str, object],
    channel_name: str,
    info: mne.Info | None,
    n_times: int,
) -> List[Tuple[int, int]]:
    """Resolve segmentation windows using channel-type-aware style defaults."""

    modality = _infer_modality(channel_name, info)
    if modality == "ear":
        candidate_styles = ("th_interpolation", "th_point")
    else:
        candidate_styles = ("zero_base", "tent", "half_peak", "zero", "base", "half_base")

    for style in candidate_styles:
        windows = _style_windows(metadata_row, modality, style, n_times)
        if windows:
            return windows

    return []


def compute_energy_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute energy features for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs with metadata containing ``blink_onset`` and
        ``blink_duration`` columns.
    picks : str | list of str | None, optional
        Channel name or list of channel names to use. If ``None``, all
        channels are processed.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` with one row per epoch and
        statistics for each metric per channel.

    Raises
    ------
    ValueError
        If any requested channels are missing from ``epochs``.

    Notes
    -----
    For epochs containing no blinks the returned statistics are ``NaN``.
    Features are computed per channel and the resulting columns are
    suffixed with ``_<channel>`` for clarity.
    """
    if picks is None:
        ch_names = epochs.ch_names
    elif isinstance(picks, str):
        ch_names = [picks]
    else:
        ch_names = list(picks)

    missing = [ch for ch in ch_names if ch not in epochs.ch_names]
    if missing:
        raise ValueError(f"Channels not found: {missing}")

    sfreq = float(epochs.info["sfreq"])
    n_epochs = len(epochs)
    n_times = epochs.get_data(picks=[ch_names[0]]).shape[-1] if n_epochs else 0
    columns = _make_columns(ch_names)
    index = (
        epochs.metadata.index
        if isinstance(epochs.metadata, pd.DataFrame)
        else pd.RangeIndex(n_epochs)
    )
    if n_epochs == 0:
        return pd.DataFrame(index=index, columns=columns, dtype=float)

    data = epochs.get_data(picks=ch_names)
    logger.info("Computing energy features for %d epochs", n_epochs)
    records: List[Dict[str, float]] = []

    for ei in range(n_epochs):
        metadata_row = (
            epochs.metadata.iloc[ei]
            if isinstance(epochs.metadata, pd.DataFrame)
            else pd.Series(dtype=float)
        )
        record: Dict[str, float] = {}
        for ci, ch in enumerate(ch_names):
            stats_by_metric = _compute_epoch_channel_energy_stats(
                metadata_row=metadata_row,
                ch=ch,
                signal_1d=data[ei, ci, :],
                sfreq=sfreq,
                n_times=n_times,
                info=epochs.info,
            )
            for metric, stats in stats_by_metric.items():
                for stat_name, value in stats.items():
                    record[f"{metric}_{stat_name}_{ch}"] = value
        records.append(record)

    df = pd.DataFrame.from_records(records, index=index, columns=columns)
    logger.debug("Energy feature DataFrame shape: %s", df.shape)
    return df
