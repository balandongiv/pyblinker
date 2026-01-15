"""Aggregate blink morphology features from :class:`mne.Epochs`."""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence, Set

import mne
import numpy as np
import pandas as pd
from pyblinker.logging import get_logger

from .core_metrics import (
    MORPHOLOGY_METRIC_STEMS,
    compute_blink_durations,
    compute_blink_peak_times,
    compute_time_base_shut,
    compute_time_zero_shut,
)
from .per_blink import compute_blink_waveform_metrics
from ..utils.aggregation import prepare_epoch_channel_data
logger = get_logger(__name__)

_STATS = ("mean", "std", "cv")
_DEFAULT_SHUT_AMP_FRACTION = 0.9
_LEGACY_METRICS = (
    "duration_zero",
    "duration_base",
    "duration_tent",
    "duration_half_base",
    "duration_half_zero",
    "closing_time_zero",
    "reopening_time_zero",
    "time_shut_zero",
    "time_shut_base",
    "closing_time_tent",
    "reopening_time_tent",
    "time_shut_tent",
    "inter_blink_max_amp",
)
_STYLE_LANDMARK_COLUMNS = {
    "base": ("start__left_base__{modality}", "end__right_base__{modality}"),
    "zero": ("start__left_zero__{modality}", "end__right_zero__{modality}"),
    "tent": ("start__left_x_intercept__{modality}", "end__right_x_intercept__{modality}"),
    "half_base": (
        "start__left_base_half_height__{modality}",
        "end__right_base_half_height__{modality}",
    ),
    "half_zero": (
        "start__left_zero_half_height__{modality}",
        "end__right_zero_half_height__{modality}",
    ),
}
_STYLE_DURATION_COLUMNS = {
    "base": "duration_base",
    "zero": "duration_zero",
    "tent": "duration_tent",
    "half_base": "duration_half_base",
    "half_zero": "duration_half_zero",
}


def segment_to_samples(onset_s: float, duration_s: float, sfreq: float, n_times: int) -> slice:
    """Convert blink onset and duration in seconds to a sample slice."""

    start = int(round(onset_s * sfreq))
    stop = start + int(round(duration_s * sfreq))
    start = max(start, 0)
    stop = min(stop, n_times)
    return slice(start, stop)


def _safe_stats(values: Sequence[float]) -> Dict[str, float]:
    """Compute basic statistics while handling empty input safely."""

    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return {"mean": np.nan, "std": np.nan, "cv": np.nan}

    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=0))
    cv = float(std / mean) if mean != 0 else float("nan")
    return {"mean": mean, "std": std, "cv": cv}


def ensure_list(value: object) -> List[object]:
    """Return value as a list."""

    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _resolve_channels(
    epochs: mne.Epochs,
    picks: str | Sequence[str] | None,
    *,
    default: Callable[[mne.Epochs], Sequence[str]] | None = None,
) -> List[str]:
    """Resolve channel picks without importing shared utils."""

    if picks is None:
        ch_names = list(epochs.ch_names) if default is None else list(default(epochs))
    else:
        ch_names = [picks] if isinstance(picks, str) else list(picks)
    missing = [ch for ch in ch_names if ch not in epochs.ch_names]
    if missing:
        raise ValueError(f"Unknown channel(s) in picks: {missing}")
    return ch_names


def _infer_modality(channel_name: str, info: mne.Info) -> str:
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


def _default_morphology_channels(epochs: mne.Epochs) -> List[str]:
    """Select default EAR/EOG channels when ``picks`` are unspecified."""

    ch_names = [
        ch for ch in epochs.ch_names if "EOG" in ch.upper() or "EAR" in ch.upper()
    ]
    if not ch_names:
        raise ValueError("No default EAR/EOG channels found")
    return ch_names

def _available_styles(metadata_columns: Sequence[str] | None, modality: str) -> Set[str]:
    """Return segmentation styles present in metadata for a modality."""

    if metadata_columns is None:
        return set()

    styles: Set[str] = set()
    suffix = f"__{modality}"
    for col in metadata_columns:
        if not col.startswith("onset__") or not col.endswith(suffix):
            continue
        style = col[len("onset__") : -len(suffix)]
        if "sample" in style.lower():
            continue
        duration_key = f"duration__{style}__{modality}"
        if duration_key in metadata_columns:
            styles.add(style)

    for style, (start_key, end_key) in _STYLE_LANDMARK_COLUMNS.items():
        start_col = start_key.format(modality=modality)
        end_col = end_key.format(modality=modality)
        if start_col in metadata_columns and end_col in metadata_columns:
            styles.add(style)

    return styles

def _list_from_metadata(metadata_row: Mapping[str, object], key: str) -> List[float]:
    """Return a list of floats (with NaNs) for a metadata key."""

    if key not in metadata_row:
        return []
    value = metadata_row.get(key)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    values = ensure_list(value)
    cleaned: List[float] = []
    for entry in values:
        if entry is None or (isinstance(entry, float) and pd.isna(entry)):
            cleaned.append(float("nan"))
        else:
            cleaned.append(float(entry))
    return cleaned


def _style_sample_windows(
    metadata_row: Mapping[str, object],
    modality: str,
    style: str,
    *,
    sfreq: float,
    n_times: int,
) -> List[slice]:
    """Extract blink sample windows for a modality/style pair."""

    windows: List[slice] = []
    onset_key = f"onset__{style}__{modality}"
    duration_key = f"duration__{style}__{modality}"
    onsets = _list_from_metadata(metadata_row, onset_key)
    durations = _list_from_metadata(metadata_row, duration_key)
    if onsets and durations:
        for onset, duration in zip(onsets, durations):
            if pd.isna(onset) or pd.isna(duration):
                continue
            sl = segment_to_samples(float(onset), float(duration), sfreq, n_times)
            if sl.stop > sl.start:
                windows.append(sl)

    if windows:
        return windows

    landmark_cols = _STYLE_LANDMARK_COLUMNS.get(style)
    if not landmark_cols:
        return windows

    start_key, end_key = landmark_cols
    start_key = start_key.format(modality=modality)
    end_key = end_key.format(modality=modality)
    starts = _list_from_metadata(metadata_row, start_key)
    ends = _list_from_metadata(metadata_row, end_key)
    for start, end in zip(starts, ends):
        if pd.isna(start) or pd.isna(end):
            continue
        start_idx = int(round(start))
        end_idx = int(round(end)) + 1
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, n_times)
        if end_idx > start_idx:
            windows.append(slice(start_idx, end_idx))

    return windows


def _blink_count(values: Sequence[List[float]]) -> int:
    return max((len(entry) for entry in values if entry), default=0)


def _value_at(values: Sequence[float], idx: int) -> float:
    if idx < len(values):
        return values[idx]
    return float("nan")


def _extract_peak_samples(
    metadata_row: Mapping[str, object],
    modality: str,
    *,
    sfreq: float,
    n_times: int,
) -> List[float]:
    key_options = [
        f"onset__refine_extremum__{modality}",
        f"blink_onset_extremum_{modality}",
    ]
    values: List[float] = []
    for key in key_options:
        values = _list_from_metadata(metadata_row, key)
        if values:
            break
    if not values:
        return []

    max_time = n_times / sfreq
    samples: List[float] = []
    for value in values:
        if pd.isna(value):
            samples.append(float("nan"))
        elif value > max_time:
            samples.append(float(value))
        else:
            samples.append(float(int(round(value * sfreq))))
    return samples


def _build_blink_properties_frame(
    metadata_row: Mapping[str, object],
    modality: str,
    *,
    sfreq: float,
    signal: np.ndarray,
) -> pd.DataFrame:
    """Build a per-blink DataFrame from epoch metadata."""

    landmark_map = {
        "left_base": f"start__left_base__{modality}",
        "right_base": f"end__right_base__{modality}",
        "left_zero": f"start__left_zero__{modality}",
        "right_zero": f"end__right_zero__{modality}",
        "left_x_intercept": f"start__left_x_intercept__{modality}",
        "right_x_intercept": f"end__right_x_intercept__{modality}",
        "left_base_half_height": f"start__left_base_half_height__{modality}",
        "right_base_half_height": f"end__right_base_half_height__{modality}",
        "left_zero_half_height": f"start__left_zero_half_height__{modality}",
        "right_zero_half_height": f"end__right_zero_half_height__{modality}",
        "x_intersect": f"x_intersect__{modality}",
        "y_intersect": f"y_intersect__{modality}",
    }

    lists: Dict[str, List[float]] = {
        name: _list_from_metadata(metadata_row, key) for name, key in landmark_map.items()
    }
    peaks = _extract_peak_samples(
        metadata_row,
        modality,
        sfreq=sfreq,
        n_times=signal.size,
    )
    lists["max_blink"] = peaks

    n_blinks = _blink_count(list(lists.values()))
    columns = list(landmark_map.keys()) + ["max_blink", "max_value"]
    if n_blinks == 0:
        return pd.DataFrame(columns=columns)

    data: Dict[str, List[float]] = {col: [] for col in columns}
    for idx in range(n_blinks):
        max_blink = _value_at(lists["max_blink"], idx)
        max_value = float("nan")
        if not pd.isna(max_blink):
            max_idx = int(max(0, min(int(round(max_blink)), signal.size - 1)))
            max_value = float(signal[max_idx])
        for key in landmark_map.keys():
            data[key].append(_value_at(lists[key], idx))
        data["max_blink"].append(max_blink)
        data["max_value"].append(max_value)

    return pd.DataFrame(data)


def _compute_legacy_metrics(
    blink_df: pd.DataFrame,
    candidate_signal: np.ndarray,
    *,
    sfreq: float,
    modality: str,
) -> pd.DataFrame:
    """Compute legacy blink properties while tolerating missing landmarks."""

    if blink_df.empty:
        for metric in _LEGACY_METRICS:
            blink_df[metric] = []
        return blink_df

    df = blink_df.copy()
    for metric in _LEGACY_METRICS:
        if metric not in df.columns:
            df[metric] = np.nan

    compute_blink_durations(df, sfreq, modality=modality, fitted=True)
    shut_amp_fraction = _DEFAULT_SHUT_AMP_FRACTION

    if modality != "ear":
        zero_mask = (
            df["left_zero"].notna()
            & df["right_zero"].notna()
            & df["max_blink"].notna()
            & df["max_value"].notna()
        )
        if zero_mask.any():
            df_zero = df.loc[zero_mask].copy()
            compute_time_zero_shut(
                df_zero,
                candidate_signal,
                sfreq,
                modality=modality,
                shut_amp_fraction=shut_amp_fraction,
            )
            df.loc[zero_mask, ["closing_time_zero", "reopening_time_zero", "time_shut_zero"]] = df_zero[
                ["closing_time_zero", "reopening_time_zero", "time_shut_zero"]
            ]

    base_mask = df["left_base"].notna() & df["right_base"].notna() & df["max_value"].notna()
    if base_mask.any():
        df_base = df.loc[base_mask].copy()
        compute_time_base_shut(
            df_base,
            candidate_signal,
            sfreq,
            shut_amp_fraction=shut_amp_fraction,
            fitted=False,
        )
        df.loc[base_mask, "time_shut_base"] = df_base["time_shut_base"]

    tent_mask = (
        base_mask
        & df["left_x_intercept"].notna()
        & df["right_x_intercept"].notna()
        & df["x_intersect"].notna()
    )
    if tent_mask.any():
        df_tent = df.loc[tent_mask].copy()
        compute_time_base_shut(
            df_tent,
            candidate_signal,
            sfreq,
            shut_amp_fraction=shut_amp_fraction,
            fitted=True,
        )
        df.loc[
            tent_mask, ["closing_time_tent", "reopening_time_tent", "time_shut_tent"]
        ] = df_tent[["closing_time_tent", "reopening_time_tent", "time_shut_tent"]]

    peak_mask = df["max_blink"].notna() & df["max_value"].notna()
    if peak_mask.any():
        df_peak = df.loc[peak_mask].copy()
        compute_blink_peak_times(df_peak, candidate_signal, sfreq, fitted=False)
        df.loc[peak_mask, "inter_blink_max_amp"] = df_peak["inter_blink_max_amp"]

    return df


class MorphologyBlinkFeatureExtractor:
    """Compute blink morphology features from MNE objects."""

    def __init__(self, epochs: mne.Epochs | None = None, raw: mne.io.BaseRaw | None = None):
        self.epochs = epochs
        self.raw = raw

    def _sampling_frequency(self) -> float:
        """Return sampling frequency from available MNE object."""
        if hasattr(self, "epochs") and self.epochs is not None:
            return float(self.epochs.info["sfreq"])
        if hasattr(self, "raw") and self.raw is not None:
            return float(self.raw.info["sfreq"])
        raise ValueError("Neither self.epochs nor self.raw defined (need MNE object).")

    def compute(self, picks: str | Sequence[str] | None = None) -> pd.DataFrame:
        """Compute blink morphology statistics for each epoch.

        Parameters
        ----------
        picks : str | list of str | None, optional
            Channel name(s) to include. ``None`` selects channels containing
            ``"EOG"`` or ``"EAR"``. If any requested channel is missing a
            :class:`ValueError` is raised.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed like ``epochs`` containing ``mean``, ``std``, and
            ``cv`` aggregates for each morphology metric per channel.

        Raises
        ------
        ValueError
            If required metadata columns are absent or ``picks`` contain unknown
            channels.

        Notes
        -----
        If an epoch contains no blinks, all morphology statistics for that epoch
        are ``NaN``.
        """
        logger.info("Computing morphology features for epochs")

        if self.epochs is None:
            raise ValueError("self.epochs is required for feature computation")
        if self.epochs.metadata is None:
            raise ValueError("epochs.metadata must be provided")

        sfreq = self._sampling_frequency()
        ch_names = _resolve_channels(self.epochs, picks, default=_default_morphology_channels)
        ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
            epochs=self.epochs,
            picks=ch_names,
            sfreq=sfreq,
        )

        modality_map: Dict[str, str] = {ch: _infer_modality(ch, self.epochs.info) for ch in ch_names}
        modality_channels: Dict[str, List[str]] = {}
        for ch, mod in modality_map.items():
            modality_channels.setdefault(mod, []).append(ch)
        styles_by_modality: Dict[str, Set[str]] = {
            modality: {"base"} for modality in modality_channels
        }

        metadata_cols: Sequence[str] | None = (
                tuple(self.epochs.metadata.columns) if isinstance(self.epochs.metadata, pd.DataFrame) else None
        )

        for mod in set(modality_map.values()):
            styles = _available_styles(metadata_cols, mod)
            styles_by_modality[mod] = styles if styles else {"base"}

        column_set: Set[str] = set()
        for mod, channels in modality_channels.items():
            include_legacy = mod == "eeg"
            for style in sorted(styles_by_modality.get(mod, {"base"})):
                metrics_for_style = [f"{stem}_{style}" for stem in MORPHOLOGY_METRIC_STEMS]
                if style in _STYLE_DURATION_COLUMNS:
                    metrics_for_style.append("duration")
                for metric in metrics_for_style:
                    for stat in _STATS:
                        for ch in channels:
                            column_set.add(
                                f"{mod}__{style}__morphology__{metric}_{stat}__{ch}"
                            )
            if include_legacy:
                column_set.update(_LEGACY_METRICS)

        columns = sorted(column_set)
        if n_epochs == 0:
            return pd.DataFrame(index=index, columns=columns, dtype=float)

        records: List[Dict[str, float]] = []
        logger.info("Computing morphology features for %d epochs", n_epochs)

        for ei in range(n_epochs):
            metadata_row = (
                self.epochs.metadata.iloc[ei]
                if isinstance(self.epochs.metadata, pd.DataFrame)
                else pd.Series(dtype=float)
            )
            record: Dict[str, float] = {}
            for modality, channels in modality_channels.items():
                styles = styles_by_modality.get(modality, {"base"})
                blink_frames: Dict[str, pd.DataFrame] = {}
                for ch in channels:
                    blink_df = _build_blink_properties_frame(
                        metadata_row,
                        modality,
                        sfreq=sfreq,
                        signal=channel_data[ch]["raw"][ei],
                    )
                    blink_df = _compute_legacy_metrics(
                        blink_df,
                        channel_data[ch]["raw"][ei],
                        sfreq=sfreq,
                        modality=modality,
                    )
                    blink_frames[ch] = blink_df

                    if modality == "eeg" and ch == channels[0]:
                        for metric in _LEGACY_METRICS:
                            stats = _safe_stats(blink_df.get(metric, []))
                            record[metric] = stats["mean"]
                for style in sorted(styles):
                    metrics_for_style = [f"{stem}_{style}" for stem in MORPHOLOGY_METRIC_STEMS]
                    if style in _STYLE_DURATION_COLUMNS:
                        metrics_for_style.append("duration")
                    for ch in channels:
                        per_metric: Dict[str, List[float]] = {m: [] for m in metrics_for_style}
                        windows = _style_sample_windows(
                            metadata_row,
                            modality,
                            style,
                            sfreq=sfreq,
                            n_times=n_times,
                        )
                        for sl in windows:
                            segment = channel_data[ch]["raw"][ei, sl]
                            metrics = compute_blink_waveform_metrics(
                                segment,
                                sfreq,
                                method=style,
                                modality=modality,
                            )
                            for metric_name in metrics_for_style:
                                if metric_name == "duration":
                                    continue
                                per_metric[metric_name].append(
                                    metrics.get(metric_name, float("nan"))
                                )

                        blink_df = blink_frames[ch]
                        duration_col = _STYLE_DURATION_COLUMNS.get(style)
                        if duration_col and duration_col in blink_df.columns:
                            per_metric["duration"] = blink_df[duration_col].tolist()

                        for metric, values in per_metric.items():
                            stats = _safe_stats(values)
                            for stat_name, value in stats.items():
                                column = (
                                    f"{modality}__{style}__morphology__{metric}_{stat_name}__{ch}"
                                )
                                record[column] = value

            records.append(record)

        df = pd.DataFrame.from_records(records, index=index, columns=columns)
        logger.debug("Morphology feature DataFrame shape: %s", df.shape)
        return df


def compute_morphology_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute blink morphology features for each epoch and channel."""

    extractor = MorphologyBlinkFeatureExtractor(epochs=epochs)
    return extractor.compute(picks=picks)


def compute_epoch_morphology_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute blink morphology statistics for each epoch."""

    return compute_morphology_features(epochs, picks=picks)
