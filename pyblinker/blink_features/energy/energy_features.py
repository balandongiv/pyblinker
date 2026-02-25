"""Blink energy feature calculations."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Set

import mne
import pandas as pd

from pyblinker.logging import get_logger

from ..constants import DEFAULT_BLINKER_CONFIG, BlinkerConfig, METRICS_BY_FAMILY
from .common import compute_energy_metrics
from .helpers import _safe_stats
from ..utils.compute_skeleton import build_epoch_metadata_row, prepare_compute_context
from ..utils.style_windows import available_styles, extract_windows

logger = get_logger(__name__)

_METRICS = METRICS_BY_FAMILY["energy"]


def _available_styles(metadata_columns: Sequence[str] | None, modality: str) -> Set[str]:
    """Return frame-based segmentation styles present in metadata for a modality."""

    return available_styles(metadata_columns, modality, onset_prefix=None, duration_prefix=None)


def _style_windows(
    metadata_row: Mapping[str, object],
    modality: str,
    style: str,
    n_times: int,
) -> List[tuple[int, int]]:
    """Extract frame-aligned blink windows as ``(start_sample, end_sample)`` tuples."""

    return extract_windows(metadata_row, modality, style, n_times)


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


def _channel_style_windows(
    *,
    metadata_row: Mapping[str, object],
    modality: str,
    available_styles: Set[str],
    n_times: int,
) -> Dict[str, List[tuple[int, int]]]:
    """Resolve output energy styles to frame windows by modality."""

    style_windows: Dict[str, List[tuple[int, int]]] = {}
    if modality in {"eeg", "eog"}:
        if "zero" in available_styles:
            style_windows["zero"] = _style_windows(metadata_row, modality, "zero", n_times)
        if "base" in available_styles:
            style_windows["base"] = _style_windows(metadata_row, modality, "base", n_times)
        if "tent" in available_styles:
            style_windows["tent"] = _style_windows(metadata_row, modality, "tent", n_times)

        if "half_base" in available_styles:
            style_windows["half"] = _style_windows(metadata_row, modality, "half_base", n_times)
        elif "half_zero" in available_styles:
            style_windows["half"] = _style_windows(metadata_row, modality, "half_zero", n_times)

        if "tent" in style_windows:
            style_windows["peak"] = style_windows["tent"]
        elif "base" in style_windows:
            style_windows["peak"] = style_windows["base"]

    elif modality == "ear":
        if "th_point" in available_styles:
            style_windows["th_point"] = _style_windows(metadata_row, modality, "th_point", n_times)
        elif "th_interpolation" in available_styles:
            style_windows["th_interpolation"] = _style_windows(metadata_row, modality, "th_interpolation", n_times)

    return style_windows




def _feature_channel_name(channel_name: str, modality: str) -> str:
    """Return output-channel label for feature columns by modality."""

    return channel_name if modality == "eog" else channel_name.upper()
def _make_columns(
    modality_by_channel: Dict[str, str],
    styles_by_modality: Dict[str, Set[str]],
    *,
    config: BlinkerConfig,
) -> List[str]:
    """Generate ordered output columns for modality/style/metric/stat combinations."""

    columns: List[str] = []
    for ch, modality in modality_by_channel.items():
        for style in sorted(styles_by_modality.get(modality, set())):
            for metric in _METRICS:
                for stat in config.stat_names:
                    columns.append(f"{modality}__{style}__energy__{metric}_{stat}__{_feature_channel_name(ch, modality)}")
    return columns


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
            _METRICS[0]: _safe_stats(energies),
            _METRICS[1]: _safe_stats(tkeo_vals),
            _METRICS[2]: _safe_stats(lengths),
            _METRICS[3]: _safe_stats(vel_ints),
        }

    return style_stats


def compute_energy_features(
    epochs: mne.Epochs,
    picks: str | Sequence[str] | None = None,
    config: BlinkerConfig = DEFAULT_BLINKER_CONFIG,
) -> pd.DataFrame:
    """Compute style-aware energy features for each epoch/channel."""

    if picks is None:
        ch_names = epochs.ch_names
    elif isinstance(picks, str):
        ch_names = [picks]
    else:
        ch_names = list(picks)

    missing = [ch for ch in ch_names if ch not in epochs.ch_names]
    if missing:
        raise ValueError(f"Channels not found: {missing}")

    context = prepare_compute_context(
        epochs=epochs,
        picks=ch_names,
        style_getter=_available_styles,
    )
    sfreq = context.sfreq
    n_epochs = context.n_epochs
    n_times = context.n_times
    modality_by_channel = context.modality_map

    eeg_styles = context.styles_by_modality.get("eeg", set())
    available_styles_by_modality: Dict[str, Set[str]] = {}
    styles_by_modality: Dict[str, Set[str]] = {}
    for modality in set(modality_by_channel.values()):
        raw_styles = context.styles_by_modality.get(modality, set())
        if modality == "eog" and eeg_styles:
            raw_styles = raw_styles | eeg_styles
        available_styles_by_modality[modality] = raw_styles
        styles_by_modality[modality] = _normalize_styles_for_modality(raw_styles, modality)

    columns = _make_columns(modality_by_channel, styles_by_modality, config=config)
    index = context.index
    if n_epochs == 0:
        return pd.DataFrame(index=index, columns=columns, dtype=float)

    data = epochs.get_data(picks=ch_names)
    logger.info("Computing energy features for %d epochs", n_epochs)
    records: List[Dict[str, float]] = []

    for ei in range(n_epochs):
        metadata_row = (
            build_epoch_metadata_row(epochs, ei)
        )
        record: Dict[str, float] = {}
        for ci, ch in enumerate(ch_names):
            modality = modality_by_channel[ch]
            stats_by_style = _compute_epoch_channel_energy_stats(
                style_windows=_channel_style_windows(
                    metadata_row=metadata_row,
                    modality=modality,
                    available_styles=available_styles_by_modality.get(modality, set()),
                    n_times=n_times,
                ),
                signal_1d=data[ei, ci, :],
                sfreq=sfreq,
                n_times=n_times,
            )
            for style, style_metrics in stats_by_style.items():
                for metric, stats in style_metrics.items():
                    for stat_name, value in stats.items():
                        record[
                            f"{modality}__{style}__energy__{metric}_{stat_name}__{_feature_channel_name(ch, modality)}"
                        ] = value
        records.append(record)

    df = pd.DataFrame.from_records(records, index=index, columns=columns)
    df.columns = pd.Index([str(col) for col in df.columns], dtype=object)
    logger.debug("Energy feature DataFrame shape: %s", df.shape)
    return df
