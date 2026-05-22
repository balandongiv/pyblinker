"""Blink energy feature calculations."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Set

import mne
import pandas as pd

from pyblinker._logging import get_logger

from .._epoch_context import (
    available_styles_by_modality,
    build_epoch_context,
    empty_feature_frame,
    get_metadata_row,
)
from .._style_windows import style_windows_from_metadata
from .column_headers import METRICS, build_output_columns, make_stat_column
from .common import compute_energy_metrics
from .helpers import compute_basic_statistics

logger = get_logger(__name__)


def _normalize_styles_for_modality(styles: Set[str], modality: str) -> Set[str]:
    if modality in {"eeg", "eog"}:
        normalized_styles: Set[str] = set()
        if "zero" in styles:
            normalized_styles.add("zero")
        if "base" in styles:
            normalized_styles.add("base")
        if "tent" in styles:
            normalized_styles.add("tent")
        if "half_base" in styles or "half_zero" in styles:
            normalized_styles.add("half")
        if "tent" in styles or "base" in styles:
            normalized_styles.add("peak")
        return normalized_styles

    if modality == "ear":
        if "th_point" in styles:
            return {"th_point"}
        if "th_interpolation" in styles:
            return {"th_interpolation"}
        return set()

    return styles


def _group_channels_by_modality(
    modality_by_channel: Mapping[str, str],
    ch_names: Sequence[str],
) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for channel_name in ch_names:
        modality = modality_by_channel[channel_name]
        grouped.setdefault(modality, []).append(channel_name)
    return grouped


def _compute_style_windows_for_modality(
    *,
    metadata_row: Mapping[str, object],
    modality: str,
    available_raw_styles: Set[str],
    normalized_styles: Set[str],
    n_times: int,
) -> Dict[str, List[tuple[int, int]]]:
    windows_by_style = style_windows_from_metadata(
        metadata_row=metadata_row,
        modality=modality,
        available_styles=available_raw_styles,
        n_times=n_times,
        include_half=True,
        include_peak=True,
        ear_mode="keep",
    )
    return {
        style: windows
        for style, windows in windows_by_style.items()
        if style in normalized_styles
    }


def _compute_epoch_channel_energy_stats(
    *,
    style_windows: Dict[str, List[tuple[int, int]]],
    signal_1d,
    sfreq: float,
    n_times: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute per-metric summary stats for all style windows in one epoch/channel."""

    style_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for style, windows in style_windows.items():
        energies: List[float] = []
        tkeo_vals: List[float] = []
        lengths: List[float] = []
        vel_ints: List[float] = []

        for start_idx, end_idx in windows:
            if start_idx >= n_times:
                continue
            sl = slice(max(0, start_idx), min(end_idx, n_times))
            if sl.stop <= sl.start:
                continue
            segment = signal_1d[sl]
            if getattr(segment, "size", 0) == 0:
                continue
            metrics = compute_energy_metrics(segment, sfreq)
            energies.append(float(metrics["signal_energy"]))
            tkeo_vals.append(float(metrics["teager_kaiser_energy"]))
            lengths.append(float(metrics["line_length"]))
            vel_ints.append(float(metrics["velocity_integral"]))

        style_stats[style] = {
            METRICS[0]: compute_basic_statistics(energies),
            METRICS[1]: compute_basic_statistics(tkeo_vals),
            METRICS[2]: compute_basic_statistics(lengths),
            METRICS[3]: compute_basic_statistics(vel_ints),
        }

    return style_stats


def _write_energy_style_stats_into_record(
    *,
    record: Dict[str, float],
    modality: str,
    style: str,
    channel_name: str,
    style_metrics: Dict[str, Dict[str, float]],
) -> None:
    for metric, stats in style_metrics.items():
        for stat_name, value in stats.items():
            record[
                make_stat_column(
                    modality=modality,
                    style=style,
                    metric=metric,
                    stat=stat_name,
                    channel=channel_name if modality == "eog" else channel_name.upper(),
                )
            ] = value


def _compute_channel_record(
    *,
    record: Dict[str, float],
    epoch_index: int,
    modality: str,
    channel_name: str,
    signal_1d,
    windows_by_style: Dict[str, List[tuple[int, int]]],
    sfreq: float,
    n_times: int,
) -> None:
    stats_by_style = _compute_epoch_channel_energy_stats(
        style_windows=windows_by_style,
        signal_1d=signal_1d,
        sfreq=sfreq,
        n_times=n_times,
    )
    for style, style_metrics in stats_by_style.items():
        _write_energy_style_stats_into_record(
            record=record,
            modality=modality,
            style=style,
            channel_name=channel_name,
            style_metrics=style_metrics,
        )


def _compute_epoch_record(
    *,
    epoch_index: int,
    metadata_row: Mapping[str, object],
    modality_channels: Dict[str, List[str]],
    styles_by_modality: Dict[str, Set[str]],
    available_raw_styles: Dict[str, Set[str]],
    data,
    ch_to_ci: Dict[str, int],
    sfreq: float,
    n_times: int,
) -> Dict[str, float]:
    record: Dict[str, float] = {}
    for modality, channels in modality_channels.items():
        windows_by_style = _compute_style_windows_for_modality(
            metadata_row=metadata_row,
            modality=modality,
            available_raw_styles=available_raw_styles.get(modality, set()),
            normalized_styles=styles_by_modality.get(modality, set()),
            n_times=n_times,
        )
        for channel_name in channels:
            ci = ch_to_ci[channel_name]
            _compute_channel_record(
                record=record,
                epoch_index=epoch_index,
                modality=modality,
                channel_name=channel_name,
                signal_1d=data[epoch_index, ci, :],
                windows_by_style=windows_by_style,
                sfreq=sfreq,
                n_times=n_times,
            )
    return record


def compute_energy_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute style-aware energy features for each epoch/channel."""

    ctx = build_epoch_context(epochs, picks)
    available = available_styles_by_modality(
        ctx.metadata_cols,
        set(ctx.modality_by_channel.values()),
        include_eeg_for_eog=True,
    )
    styles_by_modality: Dict[str, Set[str]] = {
        modality: _normalize_styles_for_modality(raw_styles, modality)
        for modality, raw_styles in available.items()
    }

    columns = build_output_columns(ctx.modality_by_channel, styles_by_modality)
    if ctx.n_epochs == 0 or not columns:
        return empty_feature_frame(index=ctx.index, columns=columns)

    modality_channels = _group_channels_by_modality(
        ctx.modality_by_channel, ctx.ch_names
    )
    data = epochs.get_data(picks=ctx.ch_names)
    ch_to_ci = {ch: i for i, ch in enumerate(ctx.ch_names)}

    logger.info("Computing energy features for %d epochs", ctx.n_epochs)
    records: List[Dict[str, float]] = []

    for ei in range(ctx.n_epochs):
        metadata_row = get_metadata_row(epochs, ei)
        record = _compute_epoch_record(
            epoch_index=ei,
            metadata_row=metadata_row,
            modality_channels=modality_channels,
            styles_by_modality=styles_by_modality,
            available_raw_styles=available,
            data=data,
            ch_to_ci=ch_to_ci,
            sfreq=ctx.sfreq,
            n_times=ctx.n_times,
        )
        records.append(record)
    df = pd.DataFrame.from_records(records, index=ctx.index)
    # df = frame_from_records(records, index=ctx.index, columns=columns)
    logger.debug("Energy feature DataFrame shape: %s", df.shape)
    return df
