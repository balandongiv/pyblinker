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

_LEGACY_MORPHOLOGY_METRICS = (
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

_DURATION_STYLE_MAP = {
    "base": "duration_base",
    "zero": "duration_zero",
    "tent": "duration_tent",
    "half_base": "duration_half_base",
    "half_zero": "duration_half_zero",
}


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

    return styles


def _coerce_list(metadata_row: Mapping[str, object], key: str) -> List[float]:
    """Return a list of float values for a metadata key."""

    values = (
        _ensure_list(metadata_row.get(key))
        if metadata_row.get(key) is not None
        else []
    )
    cleaned: List[float] = []
    for value in values:
        if value is None or pd.isna(value):
            cleaned.append(float("nan"))
        else:
            cleaned.append(float(value))
    return cleaned


def _pad_list(values: List[float], length: int) -> List[float]:
    if len(values) >= length:
        return values
    return values + [float("nan")] * (length - len(values))


def _get_max_blink(
    candidate_signal: np.ndarray, start: int, end: int
) -> tuple[float, float]:
    segment = np.asarray(candidate_signal)[start : end + 1]
    if segment.size == 0:
        return float("nan"), float("nan")
    max_idx = int(np.argmax(segment))
    return float(segment[max_idx]), float(start + max_idx)


def segment_to_samples(
    onset_s: float, duration_s: float, sfreq: float, n_times: int
) -> slice:
    """Convert blink onset/duration to a bounded sample slice."""

    start = int(round(onset_s * sfreq))
    stop = start + int(round(duration_s * sfreq))
    start = max(start, 0)
    stop = min(stop, n_times)
    return slice(start, stop)


def _safe_stats(values: Sequence[float]) -> Dict[str, float]:
    """Compute mean/std/cv while handling empty input."""

    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return {"mean": np.nan, "std": np.nan, "cv": np.nan}

    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=0))
    cv = float(std / mean) if mean != 0 else float("nan")
    return {"mean": mean, "std": std, "cv": cv}


def _ensure_list(value: object) -> List[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    return [value]


def _normalize_picks(picks: str | Sequence[str]) -> List[str]:
    if isinstance(picks, str):
        return [picks]
    return list(picks)


def _require_channels(epochs: mne.Epochs, ch_names: Sequence[str]) -> None:
    missing = [ch for ch in ch_names if ch not in epochs.ch_names]
    if missing:
        raise ValueError(f"Missing channels in epochs: {missing}")


def _resolve_channels(
    epochs: mne.Epochs,
    picks: str | Sequence[str] | None,
    *,
    default: Callable[[mne.Epochs], Sequence[str]] | None = None,
) -> List[str]:
    if picks is None:
        ch_names = list(epochs.ch_names) if default is None else list(default(epochs))
    else:
        ch_names = _normalize_picks(picks)
    _require_channels(epochs, ch_names)
    return ch_names


def _duration_metric_for_style(style: str) -> str | None:
    return _DURATION_STYLE_MAP.get(style.lower())


def _resolve_blink_count(metadata_row: Mapping[str, object], modality: str) -> int:
    candidate_keys = [
        f"blink_onset_{modality}",
        "blink_onset",
        f"onset__refine__{modality}",
    ]
    lengths = [len(_coerce_list(metadata_row, key)) for key in candidate_keys]
    return max(lengths) if lengths else 0


def _extract_landmarks(
    metadata_row: Mapping[str, object], modality: str, n_blinks: int
) -> Dict[str, List[float]]:
    mapping = {
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
    landmarks: Dict[str, List[float]] = {}
    for column, key in mapping.items():
        values = _pad_list(_coerce_list(metadata_row, key), n_blinks)
        landmarks[column] = values
    return landmarks


def _find_blink_bounds(
    blink_index: int,
    *,
    metadata_row: Mapping[str, object],
    modality: str,
    landmarks: Dict[str, List[float]],
    sfreq: float,
    n_times: int,
) -> tuple[int, int] | None:
    candidates = [
        ("left_base", "right_base"),
        ("left_zero", "right_zero"),
        ("left_x_intercept", "right_x_intercept"),
    ]
    for left_key, right_key in candidates:
        left_val = landmarks[left_key][blink_index]
        right_val = landmarks[right_key][blink_index]
        if np.isfinite(left_val) and np.isfinite(right_val):
            left = int(round(left_val))
            right = int(round(right_val))
            if 0 <= left <= right < n_times:
                return left, right

    onset_refine = _pad_list(
        _coerce_list(metadata_row, f"onset__refine__{modality}"), blink_index + 1
    )
    duration_refine = _pad_list(
        _coerce_list(metadata_row, f"duration__refine__{modality}"), blink_index + 1
    )
    onset_s = onset_refine[blink_index]
    duration_s = duration_refine[blink_index]
    if np.isfinite(onset_s) and np.isfinite(duration_s):
        start = int(round(onset_s * sfreq))
        end = start + int(round(duration_s * sfreq))
        start = max(0, min(start, n_times - 1))
        end = max(start, min(end, n_times - 1))
        return start, end

    return None


def _compute_blink_properties(
    metadata_row: Mapping[str, object],
    modality: str,
    candidate_signal: np.ndarray,
    sfreq: float,
    n_times: int,
) -> pd.DataFrame:
    n_blinks = _resolve_blink_count(metadata_row, modality)
    if n_blinks == 0:
        return pd.DataFrame(columns=list(_LEGACY_MORPHOLOGY_METRICS))

    landmarks = _extract_landmarks(metadata_row, modality, n_blinks)
    data: Dict[str, List[float]] = dict(landmarks)

    max_values: List[float] = []
    max_blinks: List[float] = []
    for blink_index in range(n_blinks):
        bounds = _find_blink_bounds(
            blink_index,
            metadata_row=metadata_row,
            modality=modality,
            landmarks=landmarks,
            sfreq=sfreq,
            n_times=n_times,
        )
        if bounds is None:
            max_values.append(float("nan"))
            max_blinks.append(float("nan"))
            continue
        start, end = bounds
        max_value, max_blink = _get_max_blink(candidate_signal, start, end)
        max_values.append(float(max_value))
        max_blinks.append(float(max_blink))

    data["max_value"] = max_values
    data["max_blink"] = max_blinks

    df = pd.DataFrame(data)

    compute_blink_durations(df, sfreq, modality=modality, fitted=True)

    df["closing_time_zero"] = np.nan
    df["reopening_time_zero"] = np.nan
    df["time_shut_zero"] = np.nan
    zero_mask = (
        np.isfinite(df["left_zero"])
        & np.isfinite(df["right_zero"])
        & np.isfinite(df["max_blink"])
    )
    if zero_mask.any():
        zero_df = df.loc[zero_mask].copy()
        compute_time_zero_shut(
            zero_df,
            candidate_signal,
            sfreq,
            modality=modality,
            shut_amp_fraction=_DEFAULT_SHUT_AMP_FRACTION,
        )
        df.loc[zero_mask, ["closing_time_zero", "reopening_time_zero", "time_shut_zero"]] = (
            zero_df[["closing_time_zero", "reopening_time_zero", "time_shut_zero"]].values
        )

    df["time_shut_base"] = np.nan
    df["closing_time_tent"] = np.nan
    df["reopening_time_tent"] = np.nan
    df["time_shut_tent"] = np.nan

    base_mask = (
        np.isfinite(df["left_base"])
        & np.isfinite(df["right_base"])
        & np.isfinite(df["max_value"])
    )
    if base_mask.any():
        base_df = df.loc[base_mask].copy()
        compute_time_base_shut(
            base_df,
            candidate_signal,
            sfreq,
            shut_amp_fraction=_DEFAULT_SHUT_AMP_FRACTION,
            fitted=False,
        )
        df.loc[base_mask, ["time_shut_base"]] = base_df[["time_shut_base"]].values

    tent_mask = (
        base_mask
        & np.isfinite(df["left_x_intercept"])
        & np.isfinite(df["right_x_intercept"])
        & np.isfinite(df["x_intersect"])
    )
    if tent_mask.any():
        tent_df = df.loc[tent_mask].copy()
        compute_time_base_shut(
            tent_df,
            candidate_signal,
            sfreq,
            shut_amp_fraction=_DEFAULT_SHUT_AMP_FRACTION,
            fitted=True,
        )
        df.loc[tent_mask, ["time_shut_base"]] = tent_df[["time_shut_base"]].values
        df.loc[tent_mask, ["closing_time_tent", "reopening_time_tent", "time_shut_tent"]] = (
            tent_df[["closing_time_tent", "reopening_time_tent", "time_shut_tent"]].values
        )

    if df["max_blink"].notna().all():
        compute_blink_peak_times(df, candidate_signal, sfreq, fitted=True)
    else:
        df["inter_blink_max_amp"] = np.nan

    return df

def _style_windows(
    metadata_row: Mapping[str, object],
    modality: str,
    style: str,
) -> List[tuple[float, float]]:
    """Extract blink windows for a modality/style pair."""

    onset_key = f"onset__{style}__{modality}"
    duration_key = f"duration__{style}__{modality}"
    onsets = (
        _ensure_list(metadata_row.get(onset_key))
        if metadata_row.get(onset_key) is not None
        else []
    )
    durations = (
        _ensure_list(metadata_row.get(duration_key))
        if metadata_row.get(duration_key) is not None
        else []
    )
    windows: List[tuple[float, float]] = []
    for onset, duration in zip(onsets, durations):
        if onset is None or duration is None:
            continue
        if pd.isna(onset) or pd.isna(duration):
            continue
        windows.append((float(onset), float(duration)))
    return windows


def _fallback_windows(
    metadata_row: Mapping[str, object], modality: str
) -> List[tuple[float, float]]:
    onset_key = f"blink_onset_{modality}"
    duration_key = f"blink_duration_{modality}"
    onsets = (
        _ensure_list(metadata_row.get(onset_key))
        if metadata_row.get(onset_key) is not None
        else []
    )
    durations = (
        _ensure_list(metadata_row.get(duration_key))
        if metadata_row.get(duration_key) is not None
        else []
    )
    windows: List[tuple[float, float]] = []
    for onset, duration in zip(onsets, durations):
        if onset is None or duration is None:
            continue
        if pd.isna(onset) or pd.isna(duration):
            continue
        windows.append((float(onset), float(duration)))
    return windows


def _resolve_style_windows(
    metadata_row: Mapping[str, object], modality: str, style: str
) -> List[tuple[float, float]]:
    windows = _style_windows(metadata_row, modality, style)
    if windows:
        return windows
    if style.lower() == "base":
        fallback = _style_windows(metadata_row, modality, "refine")
        if fallback:
            return fallback
    return _fallback_windows(metadata_row, modality)


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
        ch_names = _resolve_channels(
            self.epochs, picks, default=_default_morphology_channels
        )
        ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
            epochs=self.epochs,
            picks=ch_names,
            sfreq=sfreq,
        )

        modality_map: Dict[str, str] = {
            ch: _infer_modality(ch, self.epochs.info) for ch in ch_names
        }
        modality_channels: Dict[str, List[str]] = {}
        for ch, mod in modality_map.items():
            modality_channels.setdefault(mod, []).append(ch)
        styles_by_modality: Dict[str, Set[str]] = {
            modality: {"base"} for modality in modality_channels
        }

        metadata_cols: Sequence[str] | None = (
            tuple(self.epochs.metadata.columns)
            if isinstance(self.epochs.metadata, pd.DataFrame)
            else None
        )

        for mod in set(modality_map.values()):
            styles = _available_styles(metadata_cols, mod)
            styles_by_modality[mod] = styles or {"base"}

        column_set: Set[str] = set()
        for mod, channels in modality_channels.items():
            for style in sorted(styles_by_modality.get(mod, {"base"})):
                metrics_for_style = [
                    f"{stem}_{style}" for stem in MORPHOLOGY_METRIC_STEMS
                ] + ["duration"]
                for metric in metrics_for_style:
                    for stat in _STATS:
                        for ch in channels:
                            column_set.add(
                                f"{mod}__{style}__morphology__{metric}_{stat}__{ch}"
                            )
        column_set.update(_LEGACY_MORPHOLOGY_METRICS)

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
            legacy_values: Dict[str, List[float]] = {
                metric: [] for metric in _LEGACY_MORPHOLOGY_METRICS
            }
            for modality, channels in modality_channels.items():
                styles = styles_by_modality.get(modality, {"base"})
                for ch in channels:
                    per_blink_df = _compute_blink_properties(
                        metadata_row,
                        modality,
                        channel_data[ch]["raw"][ei],
                        sfreq,
                        n_times,
                    )
                    if modality == "eeg":
                        for metric in _LEGACY_MORPHOLOGY_METRICS:
                            if metric in per_blink_df.columns:
                                legacy_values[metric].extend(
                                    per_blink_df[metric].tolist()
                                )
                    for style in sorted(styles):
                        metrics_for_style = [
                            f"{stem}_{style}" for stem in MORPHOLOGY_METRIC_STEMS
                        ] + ["duration"]
                        windows = _resolve_style_windows(metadata_row, modality, style)
                        per_metric: Dict[str, List[float]] = {
                            m: [] for m in metrics_for_style
                        }
                        duration_col = _duration_metric_for_style(style)
                        for blink_index, (onset_s, duration_s) in enumerate(windows):
                            sl = segment_to_samples(
                                onset_s, duration_s, sfreq, n_times
                            )
                            segment = channel_data[ch]["raw"][ei, sl]
                            metrics = compute_blink_waveform_metrics(
                                segment,
                                sfreq,
                                method=style,
                                modality=modality,
                            )
                            for metric_name in metrics_for_style:
                                if metric_name == "duration":
                                    value = duration_s
                                    if (
                                        duration_col is not None
                                        and duration_col in per_blink_df.columns
                                        and blink_index < len(per_blink_df)
                                    ):
                                        value = per_blink_df.loc[
                                            blink_index, duration_col
                                        ]
                                    per_metric[metric_name].append(value)
                                else:
                                    per_metric[metric_name].append(
                                        metrics.get(metric_name, float("nan"))
                                    )
                        for metric, values in per_metric.items():
                            stats = _safe_stats(values)
                            for stat_name, value in stats.items():
                                column = (
                                    f"{modality}__{style}__morphology__"
                                    f"{metric}_{stat_name}__{ch}"
                                )
                                record[column] = value
            for metric, values in legacy_values.items():
                record[metric] = _safe_stats(values)["mean"]
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
