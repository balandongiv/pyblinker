"""Shared compute skeleton for epoch/channel/style feature extraction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

import mne
import numpy as np
import pandas as pd

from .constants import cast_columns_to_object, infer_modality
from .default_setting import BlinkerConfig, DEFAULT_CONFIG
from .utils.aggregation import prepare_epoch_channel_data
from .utils.style_windows import available_styles, extract_windows


ComputeFunc = Callable[[str, object, float, str], Mapping[str, float]]


def _group_channels_by_modality(modality_map: Mapping[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for ch, mod in modality_map.items():
        grouped[mod].append(ch)
    return dict(grouped)


def _summarize(values: list[float], stat_names: Sequence[str]) -> dict[str, float]:
    if not values:
        return {name: float("nan") for name in stat_names}

    arr = np.asarray(values, dtype=float)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    cv = float(std / mean) if mean and not np.isnan(mean) else float("nan")
    base = {"mean": mean, "std": std, "cv": cv}
    return {name: float(base.get(name, np.nan)) for name in stat_names}


def compute_features(
    epochs: mne.Epochs,
    picks: Sequence[str] | None,
    metrics_by_style: Mapping[str, Sequence[str]],
    compute_func: ComputeFunc,
    config: BlinkerConfig = DEFAULT_CONFIG,
    *,
    family_name: str,
    resolve_styles: Callable[[set[str], str], set[str]] | None = None,
    channel_label: Callable[[str, str], str] | None = None,
) -> pd.DataFrame:
    """Shared orchestration logic for blink feature extraction."""

    sfreq = float(epochs.info["sfreq"])
    ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
        epochs=epochs,
        picks=picks,
        sfreq=sfreq,
    )

    modality_map = {ch: infer_modality(ch, epochs.info) for ch in ch_names}
    modalities = set(modality_map.values())
    metadata_cols = tuple(epochs.metadata.columns) if isinstance(epochs.metadata, pd.DataFrame) else None

    styles_by_modality: dict[str, set[str]] = {}
    for mod in modalities:
        styles = available_styles(metadata_cols, mod)
        if resolve_styles is not None:
            styles = resolve_styles(styles, mod)
        styles_by_modality[mod] = styles

    grouped = _group_channels_by_modality(modality_map)
    columns: list[str] = []
    for mod, channels in grouped.items():
        for style in sorted(styles_by_modality.get(mod, set())):
            metrics = metrics_by_style.get(style, metrics_by_style.get("default", ()))
            for metric in metrics:
                for stat in config.stat_names:
                    for ch in channels:
                        col_ch = channel_label(ch, mod) if channel_label else ch
                        columns.append(f"{mod}__{style}__{family_name}__{metric}_{stat}__{col_ch}")

    records: list[dict[str, float]] = []
    for ei in range(n_epochs):
        metadata_row = epochs.metadata.iloc[ei] if isinstance(epochs.metadata, pd.DataFrame) else pd.Series(dtype=float)
        record: dict[str, float] = {}
        for ch, mod in modality_map.items():
            signal = channel_data[ch]["raw"][ei]
            for style in sorted(styles_by_modality.get(mod, set())):
                windows = extract_windows(metadata_row, mod, style, n_times)
                if style == "half" and not windows:
                    windows = extract_windows(metadata_row, mod, "half_base", n_times)
                    if not windows:
                        windows = extract_windows(metadata_row, mod, "half_zero", n_times)
                if style == "peak" and not windows:
                    windows = extract_windows(metadata_row, mod, "tent", n_times)
                    if not windows:
                        windows = extract_windows(metadata_row, mod, "base", n_times)
                metrics = metrics_by_style.get(style, metrics_by_style.get("default", ()))
                collected: dict[str, list[float]] = {metric: [] for metric in metrics}
                for start_idx, end_idx in windows:
                    segment = signal[start_idx:end_idx]
                    values = compute_func(style, segment, sfreq, mod)
                    for metric, value in values.items():
                        if metric in collected:
                            collected[metric].append(float(value))
                for metric, vals in collected.items():
                    stats = _summarize(vals, config.stat_names)
                    for stat_name, value in stats.items():
                        col_ch = channel_label(ch, mod) if channel_label else ch
                        col = f"{mod}__{style}__{family_name}__{metric}_{stat_name}__{col_ch}"
                        record[col] = value
        records.append(record)

    df = pd.DataFrame.from_records(records, index=index, columns=columns)
    return cast_columns_to_object(df)
