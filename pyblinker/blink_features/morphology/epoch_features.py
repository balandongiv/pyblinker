"""Aggregate blink morphology features from :class:`mne.Epochs`."""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Set, Tuple

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
from ..energy.helpers import _safe_stats, segment_to_samples
from ..utils.aggregation import prepare_epoch_channel_data
from ...utils.epoch_utils import resolve_channels
from ...utils.iter_utils import ensure_list

logger = get_logger(__name__)

_STATS = ("mean", "std", "cv")
_SHUT_AMP_FRACTION = 0.9
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
    "peak_time_blink",
    "peak_time_tent",
    "peak_max_blink",
    "peak_max_tent",
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

    landmark_styles = {
        "base": (
            f"start__left_base__{modality}",
            f"end__right_base__{modality}",
        ),
        "zero": (
            f"start__left_zero__{modality}",
            f"end__right_zero__{modality}",
        ),
        "tent": (
            f"start__left_x_intercept__{modality}",
            f"end__right_x_intercept__{modality}",
        ),
        "half_base": (
            f"start__left_base_half_height__{modality}",
            f"end__right_base_half_height__{modality}",
        ),
        "half_zero": (
            f"start__left_zero_half_height__{modality}",
            f"end__right_zero_half_height__{modality}",
        ),
    }
    for style, (start_key, end_key) in landmark_styles.items():
        if start_key in metadata_columns and end_key in metadata_columns:
            styles.add(style)

    return styles


def _style_windows(
    metadata_row: Mapping[str, object],
    modality: str,
    style: str,
) -> List[tuple[float, float]]:
    """Extract blink windows for a modality/style pair."""

    onset_key = f"onset__{style}__{modality}"
    duration_key = f"duration__{style}__{modality}"
    onsets = (
        ensure_list(metadata_row.get(onset_key))
        if metadata_row.get(onset_key) is not None
        else []
    )
    durations = (
        ensure_list(metadata_row.get(duration_key))
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


def _coerce_list(value: object) -> List[float]:
    values = ensure_list(value) if value is not None else []
    cleaned: List[float] = []
    for item in values:
        if item is None or pd.isna(item):
            cleaned.append(float("nan"))
        else:
            cleaned.append(float(item))
    return cleaned


def _pad_to_length(values: List[float], length: int) -> List[float]:
    if len(values) >= length:
        return values[:length]
    return values + [float("nan")] * (length - len(values))


def _peak_times_from_windows(
    signal: np.ndarray,
    windows: Sequence[tuple[float, float]],
    sfreq: float,
    n_times: int,
) -> tuple[List[float], List[float]]:
    peak_indices: List[float] = []
    peak_values: List[float] = []
    for onset_s, duration_s in windows:
        sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
        segment = signal[sl]
        if segment.size == 0:
            peak_indices.append(float("nan"))
            peak_values.append(float("nan"))
            continue
        peak_idx = int(np.argmax(segment))
        peak_indices.append(float(sl.start + peak_idx))
        peak_values.append(float(segment[peak_idx]))
    return peak_indices, peak_values


def _build_blink_landmark_frame(
    metadata_row: Mapping[str, object],
    signal: np.ndarray,
    sfreq: float,
    n_times: int,
    *,
    modality: str,
    styles: Sequence[str],
) -> pd.DataFrame:
    landmark_keys = {
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
    landmark_lists = {key: _coerce_list(metadata_row.get(col)) for key, col in landmark_keys.items()}

    peak_key_candidates = (
        f"onset__refine_extremum__{modality}",
        f"blink_onset_extremum_{modality}",
    )
    peak_times_sec: List[float] = []
    for peak_key in peak_key_candidates:
        if metadata_row.get(peak_key) is not None:
            peak_times_sec = _coerce_list(metadata_row.get(peak_key))
            if peak_times_sec:
                break

    window_style = None
    for candidate in ("refine", "outer"):
        if candidate in styles:
            window_style = candidate
            break
    if window_style is None and styles:
        window_style = styles[0]
    windows = (
        _style_windows(metadata_row, modality, window_style)
        if window_style is not None
        else []
    )

    lengths = [len(values) for values in landmark_lists.values()]
    lengths.append(len(peak_times_sec))
    lengths.append(len(windows))
    n_blinks = max(lengths) if lengths else 0
    if n_blinks == 0:
        return pd.DataFrame()

    data: Dict[str, List[float]] = {
        key: _pad_to_length(values, n_blinks)
        for key, values in landmark_lists.items()
    }

    if peak_times_sec:
        peak_idx = [
            float(round(time_s * sfreq)) if not pd.isna(time_s) else float("nan")
            for time_s in peak_times_sec
        ]
        peak_idx = _pad_to_length(peak_idx, n_blinks)
        peak_values = []
        for idx in peak_idx:
            if pd.isna(idx):
                peak_values.append(float("nan"))
                continue
            int_idx = int(idx)
            if int_idx < 0 or int_idx >= n_times:
                peak_values.append(float("nan"))
                continue
            peak_values.append(float(signal[int_idx]))
        data["max_blink"] = peak_idx
        data["max_value"] = _pad_to_length(peak_values, n_blinks)
    else:
        peak_idx, peak_values = _peak_times_from_windows(signal, windows, sfreq, n_times)
        data["max_blink"] = _pad_to_length(peak_idx, n_blinks)
        data["max_value"] = _pad_to_length(peak_values, n_blinks)

    return pd.DataFrame(data)


def _apply_morphology_properties(
    blink_df: pd.DataFrame,
    signal: np.ndarray,
    sfreq: float,
    *,
    modality: str,
) -> pd.DataFrame:
    """Populate morphology timing metrics on a per-blink DataFrame.

    - compute_blink_durations: duration_base, duration_zero, duration_tent,
      duration_half_base, duration_half_zero.
    - compute_time_zero_shut: closing_time_zero, reopening_time_zero, time_shut_zero.
    - compute_time_base_shut: time_shut_base, closing_time_tent, reopening_time_tent,
      time_shut_tent.
    - compute_blink_peak_times: inter_blink_max_amp (inter-blink timing).
    """
    if blink_df.empty:
        return blink_df

    compute_blink_durations(blink_df, sfreq, modality=modality, fitted=True)

    for col in (
        "closing_time_zero",
        "reopening_time_zero",
        "time_shut_zero",
        "time_shut_base",
        "closing_time_tent",
        "reopening_time_tent",
        "time_shut_tent",
        "inter_blink_max_amp",
        "peak_time_blink",
        "peak_time_tent",
        "peak_max_blink",
        "peak_max_tent",
    ):
        if col not in blink_df.columns:
            blink_df[col] = np.nan

    zero_valid = blink_df[
        ["left_zero", "right_zero", "max_blink", "max_value"]
    ].notna().all(axis=1)
    if zero_valid.any():
        zero_df = blink_df.loc[zero_valid, :].copy()
        compute_time_zero_shut(
            zero_df,
            signal,
            sfreq,
            modality=modality,
            shut_amp_fraction=_SHUT_AMP_FRACTION,
        )
        blink_df.loc[zero_valid, ["closing_time_zero", "reopening_time_zero", "time_shut_zero"]] = (
            zero_df[["closing_time_zero", "reopening_time_zero", "time_shut_zero"]]
        )

    base_valid = blink_df[["left_base", "right_base", "max_value"]].notna().all(axis=1)
    fitted_valid = base_valid & blink_df[
        ["left_x_intercept", "right_x_intercept", "x_intersect"]
    ].notna().all(axis=1)
    if fitted_valid.any():
        fitted_df = blink_df.loc[fitted_valid, :].copy()
        compute_time_base_shut(
            fitted_df,
            signal,
            sfreq,
            shut_amp_fraction=_SHUT_AMP_FRACTION,
            fitted=True,
        )
        blink_df.loc[
            fitted_valid,
            ["time_shut_base", "closing_time_tent", "reopening_time_tent", "time_shut_tent"],
        ] = fitted_df[
            ["time_shut_base", "closing_time_tent", "reopening_time_tent", "time_shut_tent"]
        ]

    base_only = base_valid & ~fitted_valid
    if base_only.any():
        base_df = blink_df.loc[base_only, :].copy()
        compute_time_base_shut(
            base_df,
            signal,
            sfreq,
            shut_amp_fraction=_SHUT_AMP_FRACTION,
            fitted=False,
        )
        blink_df.loc[base_only, ["time_shut_base"]] = base_df[["time_shut_base"]]

    peak_valid = blink_df[["max_blink"]].notna().all(axis=1)
    if peak_valid.any():
        peak_df = blink_df.loc[peak_valid, :].copy()
        compute_blink_peak_times(peak_df, signal, sfreq, fitted=True)
        copy_cols = [
            "inter_blink_max_amp",
            "peak_time_blink",
            "peak_time_tent",
            "peak_max_blink",
            "peak_max_tent",
        ]
        copy_cols = [c for c in copy_cols if c in peak_df.columns]
        blink_df.loc[peak_valid, copy_cols] = peak_df[copy_cols]

    return blink_df


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
        """
        Compute blink morphology statistics for each epoch.

        Pipeline (functions called)
        ---------------------------
        This function orchestrates the full morphology extraction pipeline and
        internally calls, in order:

        1) _validate_inputs()
        2) _sampling_frequency()
        3) _prepare_inputs() -> resolve_channels(), prepare_epoch_channel_data()
        4) _build_modality_map() -> _infer_modality()
        5) _group_channels_by_modality()
        6) _get_metadata_cols()
        7) _build_styles_by_modality() -> _available_styles()
        8) _build_output_columns()
        9) For each epoch -> _compute_epoch_record()
           - which calls _compute_channel_record() and deeper steps

        Parameters
        ----------
        picks : str | list[str] | None
            Channels to include. If None, uses default channels such as those
            containing "EOG" or "EAR".

        Returns
        -------
        df : pandas.DataFrame
            A wide DataFrame indexed like `epochs` (same index returned by
            prepare_epoch_channel_data). Columns contain per-epoch summary
            statistics for each morphology metric per channel.

            Each column follows the pattern:
                "{modality}__{style}__morphology__{metric}_{stat}__{channel}"

            Examples of returned stats include:
                mean, std, cv (depending on your `_STATS` constant)

        Raises
        ------
        ValueError
            If `self.epochs` is missing or `epochs.metadata` is not provided.

        Notes
        -----
        If an epoch contains no valid blink windows, all values for that epoch
        will be NaN for the affected metrics.
        """
        self._validate_inputs()
        logger.info("Computing morphology features for epochs")

        sfreq = self._sampling_frequency()

        ch_names, channel_data, index, n_epochs, n_times = self._prepare_inputs(
            picks=picks,
            sfreq=sfreq,
        )

        modality_map = self._build_modality_map(ch_names)
        modality_channels = self._group_channels_by_modality(modality_map)
        metadata_cols = self._get_metadata_cols()

        styles_by_modality = self._build_styles_by_modality(
            modalities=set(modality_channels.keys()),
            metadata_cols=metadata_cols,
        )

        columns = self._build_output_columns(modality_channels, styles_by_modality)
        logger.debug("Morphology output columns: %d", len(columns))

        if n_epochs == 0:
            logger.info("No epochs available. Returning empty DataFrame.")
            return pd.DataFrame(index=index, columns=columns, dtype=float)

        logger.info("Computing morphology features for %d epochs", n_epochs)

        records: List[Dict[str, float]] = []
        for ei in range(n_epochs):
            metadata_row = self._get_metadata_row(ei)
            record = self._compute_epoch_record(
                epoch_index=ei,
                metadata_row=metadata_row,
                modality_channels=modality_channels,
                styles_by_modality=styles_by_modality,
                channel_data=channel_data,
                sfreq=sfreq,
                n_times=n_times,
                n_epochs=n_epochs,
            )
            records.append(record)

        df = pd.DataFrame.from_records(records, index=index, columns=columns)
        logger.debug("Morphology feature DataFrame shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # Input preparation & validation
    # ------------------------------------------------------------------
    def _validate_inputs(self) -> None:
        """Validate that the required MNE objects and metadata exist."""
        if self.epochs is None:
            raise ValueError("self.epochs is required for feature computation")
        if self.epochs.metadata is None:
            raise ValueError("epochs.metadata must be provided")

    def _prepare_inputs(
        self,
        *,
        picks: str | Sequence[str] | None,
        sfreq: float,
    ) -> Tuple[List[str], dict, pd.Index, int, int]:
        """
        Resolve target channels and extract epoch-wise signal arrays.

        Pipeline (functions called)
        ---------------------------
        1) resolve_channels(self.epochs, picks, default=_default_morphology_channels)
        2) prepare_epoch_channel_data(epochs=self.epochs, picks=..., sfreq=sfreq)

        Parameters
        ----------
        picks : str | list[str] | None
            Channel selection rule (passed to resolve_channels()).
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        ch_names : list[str]
            Final resolved channel names to compute morphology on.
        channel_data : dict
            Channel dictionary holding per-channel extracted arrays.
            Expected access pattern later:
                channel_data[ch]["raw"][epoch_index]
        index : pandas.Index
            Index aligned with epochs (used as output DataFrame index).
        n_epochs : int
            Number of epochs present.
        n_times : int
            Number of samples per epoch.
        """
        ch_names = resolve_channels(self.epochs, picks, default=_default_morphology_channels)
        return prepare_epoch_channel_data(
            epochs=self.epochs,
            picks=ch_names,
            sfreq=sfreq,
        )

    def _get_metadata_cols(self) -> Sequence[str] | None:
        """Return metadata column names if epochs.metadata is a DataFrame."""
        if isinstance(self.epochs.metadata, pd.DataFrame):
            return tuple(self.epochs.metadata.columns)
        return None

    def _get_metadata_row(self, epoch_index: int) -> pd.Series:
        """Fetch metadata row for an epoch (or an empty Series if unavailable)."""
        if isinstance(self.epochs.metadata, pd.DataFrame):
            return self.epochs.metadata.iloc[epoch_index]
        return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Modality/style/column planning
    # ------------------------------------------------------------------
    def _build_modality_map(self, ch_names: Sequence[str]) -> Dict[str, str]:
        """Infer modality for each channel (e.g., eeg/eog/ear)."""
        return {ch: _infer_modality(ch, self.epochs.info) for ch in ch_names}

    def _group_channels_by_modality(self, modality_map: Dict[str, str]) -> Dict[str, List[str]]:
        """Group channels by modality."""
        grouped: Dict[str, List[str]] = {}
        for ch, mod in modality_map.items():
            grouped.setdefault(mod, []).append(ch)
        return grouped


    def _build_styles_by_modality(
        self,
        modalities: Set[str],
        metadata_cols: Sequence[str] | None,
    ) -> Dict[str, Set[str]]:
        """Determine available waveform styles per modality based on metadata."""
        styles_by_modality: Dict[str, Set[str]] = {}
        for mod in modalities:
            styles = _available_styles(metadata_cols, mod)
            styles_by_modality[mod] = styles
        return styles_by_modality

    def _build_output_columns(
        self,
        modality_channels: Dict[str, List[str]],
        styles_by_modality: Dict[str, Set[str]],
    ) -> List[str]:
        """
        Build the full list of output column names upfront.

        Why this exists
        ---------------
        Building all columns first ensures the output DataFrame has a stable shape
        even when some epochs have no blinks/windows.

        Pipeline (functions called)
        ---------------------------
        - _metrics_for_style(style)
        - uses constants: MORPHOLOGY_METRIC_STEMS, _STATS
        - adds legacy metrics: _LEGACY_MORPHOLOGY_METRICS (EEG only)

        Parameters
        ----------
        modality_channels : dict[str, list[str]]
            Mapping of modality -> list of channel names.
        styles_by_modality : dict[str, set[str]]
            Mapping of modality -> waveform metric styles to compute.

        Returns
        -------
        columns : list[str]
            Sorted list of unique output column names.
        """
        column_set: Set[str] = set()

        for mod, channels in modality_channels.items():
            styles = sorted(styles_by_modality.get(mod, {"base"}))
            for style in styles:
                metric_names = self._metrics_for_style(style)
                for metric in metric_names:
                    for stat in _STATS:
                        for ch in channels:
                            column_set.add(
                                f"{mod}__{style}__morphology__{metric}_{stat}__{ch}"
                            )

            if mod == "eeg" and channels:
                column_set.update(_LEGACY_MORPHOLOGY_METRICS)

        return sorted(column_set)

    def _metrics_for_style(self, style: str) -> List[str]:
        """Return list of metrics for a given style (including duration)."""
        metric_names = [f"{stem}_{style}" for stem in MORPHOLOGY_METRIC_STEMS]
        metric_names.append("duration")
        return metric_names

    # ------------------------------------------------------------------
    # Core computation (epoch/modality/channel/style)
    # ------------------------------------------------------------------
    def _compute_epoch_record(
        self,
        epoch_index: int,
        metadata_row: pd.Series,
        modality_channels: Dict[str, List[str]],
        styles_by_modality: Dict[str, Set[str]],
        channel_data: dict,
        sfreq: float,
        n_times: int,
        n_epochs: int,
    ) -> Dict[str, float]:
        """
        Compute all morphology stats for a single epoch.

        Pipeline (functions called)
        ---------------------------
        For each modality and channel, this function calls:

        - _compute_channel_record(...)
            -> _build_blink_df(...)
            -> _compute_style_stats_into_record(...)
            -> _add_legacy_metrics_if_available(...) (EEG only)

        Parameters
        ----------
        epoch_index : int
            Current epoch index (0-based).
        metadata_row : pandas.Series
            One row of epochs.metadata corresponding to the epoch.
        modality_channels : dict[str, list[str]]
            Modality -> channel names to process.
        styles_by_modality : dict[str, set[str]]
            Modality -> styles available for that modality.
        channel_data : dict
            Data prepared by prepare_epoch_channel_data().
        sfreq : float
            Sampling frequency (Hz).
        n_times : int
            Number of samples in this epoch.
        n_epochs : int
            Total number of epochs (used for logging only).

        Returns
        -------
        record : dict[str, float]
            Dictionary of computed feature values for this epoch.
            Keys are column names, values are floats (possibly NaN).
        """
        logger.debug("Morphology epoch %d/%d", epoch_index + 1, n_epochs)

        record: Dict[str, float] = {}

        for modality, channels in modality_channels.items():
            styles = sorted(styles_by_modality.get(modality, {"base"}))
            logger.debug(
                "Epoch %d: modality=%s styles=%s channels=%s",
                epoch_index + 1,
                modality,
                styles,
                channels,
            )

            for ch_index, ch in enumerate(channels):
                signal = channel_data[ch]["raw"][epoch_index]
                self._compute_channel_record(
                    record=record,
                    epoch_index=epoch_index,
                    modality=modality,
                    channel_name=ch,
                    channel_index_in_modality=ch_index,
                    metadata_row=metadata_row,
                    signal=signal,
                    sfreq=sfreq,
                    n_times=n_times,
                    styles=styles,
                )

        return record

    def _compute_channel_record(
        self,
        record: Dict[str, float],
        epoch_index: int,
        modality: str,
        channel_name: str,
        channel_index_in_modality: int,
        metadata_row: pd.Series,
        signal: np.ndarray,
        sfreq: float,
        n_times: int,
        styles: Sequence[str],
    ) -> None:
        """
        Compute morphology features for one (epoch, modality, channel).

        Pipeline (functions called)
        ---------------------------
        1) _build_blink_df(...)
            -> _build_blink_landmark_frame(...)
            -> _apply_morphology_properties(...)

        2) For each style in `styles`:
            _compute_style_stats_into_record(...)

        3) Legacy EEG metrics:
            _add_legacy_metrics_if_available(...) (only if modality == "eeg" and
            channel_index_in_modality == 0)

        Parameters
        ----------
        record : dict[str, float]
            Output feature dictionary mutated in-place.
        epoch_index : int
            Current epoch index.
        modality : str
            Modality name (e.g., "eeg", "eog", "ear").
        channel_name : str
            Channel name being processed.
        channel_index_in_modality : int
            Index of the channel inside modality channel list (used for legacy metrics).
        metadata_row : pandas.Series
            Epoch metadata row.
        signal : array-like
            1D signal waveform of shape (n_times,).
        sfreq : float
            Sampling frequency in Hz.
        n_times : int
            Number of samples per epoch.
        styles : list[str]
            Styles to compute for this modality.

        Returns
        -------
        None
            This function does not return anything. It updates `record` in-place.
        """
        logger.debug(
            "Epoch %d: channel=%s modality=%s",
            epoch_index + 1,
            channel_name,
            modality,
        )

        blink_df = self._build_blink_df(
            metadata_row=metadata_row,
            signal=signal,
            sfreq=sfreq,
            n_times=n_times,
            modality=modality,
            styles=styles,
        )

        for style in styles:
            self._compute_style_stats_into_record(
                record=record,
                metadata_row=metadata_row,
                signal=signal,
                sfreq=sfreq,
                n_times=n_times,
                modality=modality,
                style=style,
                channel_name=channel_name,
                blink_df=blink_df,
            )

        if modality == "eeg" and channel_index_in_modality == 0:
            self._add_legacy_metrics_if_available(record, blink_df)

    def _build_blink_df(
        self,
        metadata_row: pd.Series,
        signal: np.ndarray,
        sfreq: float,
        n_times: int,
        modality: str,
        styles: Sequence[str],
    ) -> pd.DataFrame:
        """Build and enrich the blink landmark dataframe."""
        blink_df = _build_blink_landmark_frame(
            metadata_row,
            signal,
            sfreq,
            n_times,
            modality=modality,
            styles=styles,
        )
        blink_df = _apply_morphology_properties(
            blink_df,
            signal,
            sfreq,
            modality=modality,
        )
        return blink_df

    def _compute_style_stats_into_record(
        self,
        record: Dict[str, float],
        metadata_row: pd.Series,
        signal: np.ndarray,
        sfreq: float,
        n_times: int,
        modality: str,
        style: str,
        channel_name: str,
        blink_df: pd.DataFrame,
    ) -> None:
        """
        Compute per-window waveform metrics for one style and aggregate into stats.

        Pipeline (functions called)
        ---------------------------
        1) windows = _style_windows(metadata_row, modality, style)
        2) per_metric_values = _compute_metrics_over_windows(...)
            -> segment_to_samples(...)
            -> compute_blink_waveform_metrics(...)
        3) per_metric_values["duration"] = _duration_values_from_blink_df(...)
        4) _write_metric_stats_to_record(...)
            -> _safe_stats(...)

        Parameters
        ----------
        record : dict[str, float]
            Feature dict updated in-place.
        metadata_row : pandas.Series
            Metadata row for the epoch.
        signal : array-like
            1D channel waveform for the epoch.
        sfreq : float
            Sampling frequency (Hz).
        n_times : int
            Number of samples in epoch.
        modality : str
            Modality name.
        style : str
            Style/method name ("base", "blink", etc.).
        channel_name : str
            Channel name used to build output keys.
        blink_df : pandas.DataFrame
            Blink landmark and morphology properties frame.

        Returns
        -------
        None
            This function does not return anything. It updates `record` in-place.
        """
        windows = _style_windows(metadata_row, modality, style)
        metric_names = [f"{stem}_{style}" for stem in MORPHOLOGY_METRIC_STEMS]

        logger.debug(
            "Style compute: modality=%s style=%s channel=%s windows=%d",
            modality,
            style,
            channel_name,
            len(windows),
        )

        per_metric_values = self._compute_metrics_over_windows(
            signal=signal,
            windows=windows,
            metric_names=metric_names,
            sfreq=sfreq,
            n_times=n_times,
            modality=modality,
            style=style,
        )

        per_metric_values["duration"] = self._duration_values_from_blink_df(
            blink_df,
            style,
        )

        self._write_metric_stats_to_record(
            record=record,
            modality=modality,
            style=style,
            channel_name=channel_name,
            per_metric_values=per_metric_values,
        )

    def _compute_metrics_over_windows(
        self,
        signal: np.ndarray,
        windows: Sequence[Tuple[float, float]],
        metric_names: Sequence[str],
        sfreq: float,
        n_times: int,
        modality: str,
        style: str,
    ) -> Dict[str, List[float]]:
        """
        Compute waveform metrics for each window and collect values per metric.

        Pipeline (functions called)
        ---------------------------
        For every window (onset_s, duration_s):
        1) sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
        2) segment = signal[sl]
        3) metrics = compute_blink_waveform_metrics(segment, sfreq, method=style, modality=modality)

        Parameters
        ----------
        signal : array-like
            1D waveform array of shape (n_times,).
        windows : list[tuple[float, float]]
            List of windows as (onset_seconds, duration_seconds).
        metric_names : list[str]
            Metric keys expected from compute_blink_waveform_metrics().
        sfreq : float
            Sampling frequency in Hz.
        n_times : int
            Total number of samples for bounds checking.
        modality : str
            Modality name.
        style : str
            Metric computation method/style.

        Returns
        -------
        per_metric_values : dict[str, list[float]]
            Dictionary mapping each metric name to a list of computed values
            (one value per window). Missing metrics are filled with NaN.
        """
        out: Dict[str, List[float]] = {m: [] for m in metric_names}

        for onset_s, duration_s in windows:
            logger.debug(
                "Window compute: modality=%s style=%s onset=%s duration=%s",
                modality,
                style,
                onset_s,
                duration_s,
            )
            sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
            segment = signal[sl]

            metrics = compute_blink_waveform_metrics(
                segment,
                sfreq,
                method=style,
                modality=modality,
            )

            for metric_name in metric_names:
                out[metric_name].append(metrics.get(metric_name, float("nan")))

        return out

    def _duration_values_from_blink_df(
        self,
        blink_df: pd.DataFrame,
        style: str,
    ) -> List[float]:
        """Extract duration list from blink_df for the given style."""
        duration_key = _DURATION_STYLE_MAP.get(style)
        if duration_key and duration_key in blink_df.columns:
            return blink_df[duration_key].tolist()
        return []

    def _write_metric_stats_to_record(
        self,
        record: Dict[str, float],
        modality: str,
        style: str,
        channel_name: str,
        per_metric_values: Dict[str, List[float]],
    ) -> None:
        """Aggregate each metric list and write into the feature record."""
        for metric, values in per_metric_values.items():
            stats = _safe_stats(values)
            for stat_name, value in stats.items():
                col = (
                    f"{modality}__{style}__morphology__{metric}_{stat_name}__{channel_name}"
                )
                record[col] = value

    def _add_legacy_metrics_if_available(
        self,
        record: Dict[str, float],
        blink_df: pd.DataFrame,
    ) -> None:
        """Preserve legacy EEG morphology metrics behavior."""
        if blink_df.empty:
            return

        for legacy_metric in _LEGACY_MORPHOLOGY_METRICS:
            if legacy_metric in blink_df.columns:
                record[legacy_metric] = _safe_stats(
                    blink_df[legacy_metric].tolist()
                )["mean"]


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
