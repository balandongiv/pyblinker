"""Blink kinematic feature calculations based on epoch metadata.
All feature calculations rely only on blink onset and blink duration stored in the metadata.
This design intentionally decouples feature extraction from how blink boundaries are defined.

As a result, users should have full flexibility to define blink onset and duration according to their needs.
See pyblinker/segmentation/refinement.py
"""

from __future__ import annotations
from pyblinker.logging import get_logger

from typing import Dict, List, Mapping, Sequence, Set

import mne
import pandas as pd

from .core_metrics import (
    KINEMATIC_METRIC_STEMS,
    KINEMATIC_METRICS_NO_STYLE,
    compute_amp_vel_ratio_base,
    compute_amp_vel_ratio_tent,
    compute_amp_vel_ratio_zero_to_max,
    compute_blink_velocity,
    compute_inter_blink_max_vel,
)
from .per_blink import compute_segment_kinematics
from ..energy.helpers import segment_to_samples, _safe_stats
from ...utils.iter_utils import ensure_list
from ..utils.aggregation import prepare_epoch_channel_data

logger = get_logger(__name__)

# Base statistic names (kinematics defaults to base per modality)
_STATS = ("mean", "std", "cv")
_EXTENDED_KINEMATIC_METRICS = (
    "aver_left_velocity",
    "aver_right_velocity",
    "neg_amp_vel_ratio_base",
    "pos_amp_vel_ratio_base",
    "neg_amp_vel_ratio_zero",
    "pos_amp_vel_ratio_zero",
    "neg_amp_vel_ratio_tent",
    "pos_amp_vel_ratio_tent",
    "inter_blink_max_vel_base",
    "inter_blink_max_vel_zero",
)


def _coerce_numeric_list(value: object) -> List[float]:
    values = ensure_list(value) if value is not None else []
    out: List[float] = []
    for item in values:
        if item is None or pd.isna(item):
            out.append(float("nan"))
        else:
            out.append(float(item))
    return out


def _pad(values: List[float], length: int) -> List[float]:
    if len(values) >= length:
        return values[:length]
    return values + [float("nan")] * (length - len(values))


def _build_kinematic_blink_frame(
    metadata_row: Mapping[str, object],
    *,
    modality: str,
    sfreq: float,
) -> pd.DataFrame:
    landmark_keys = {
        "left_base": f"start__left_base__{modality}",
        "right_base": f"end__right_base__{modality}",
        "left_zero": f"start__left_zero__{modality}",
        "right_zero": f"end__right_zero__{modality}",
        "left_x_intercept": f"start__left_x_intercept__{modality}",
        "right_x_intercept": f"end__right_x_intercept__{modality}",
    }
    data = {k: _coerce_numeric_list(metadata_row.get(col)) for k, col in landmark_keys.items()}

    peak_key_candidates = (
        f"onset__refine_extremum__{modality}",
        f"blink_onset_extremum_{modality}",
    )
    peak_times_sec: List[float] = []
    for peak_key in peak_key_candidates:
        if metadata_row.get(peak_key) is not None:
            peak_times_sec = _coerce_numeric_list(metadata_row.get(peak_key))
            if peak_times_sec:
                break

    lengths = [len(v) for v in data.values()]
    lengths.append(len(peak_times_sec))
    n_blinks = max(lengths) if lengths else 0
    if n_blinks == 0:
        return pd.DataFrame()

    for key, values in data.items():
        data[key] = _pad(values, n_blinks)

    max_blink = [float("nan")] * n_blinks
    for i, peak_time in enumerate(_pad(peak_times_sec, n_blinks)):
        if not pd.isna(peak_time):
            max_blink[i] = float(round(peak_time * sfreq))
    data["max_blink"] = max_blink
    return pd.DataFrame(data)


def _compute_extended_kinematic_metrics(
    blink_df: pd.DataFrame,
    signal: pd.Series | List[float] | object,
    sfreq: float,
    *,
    modality: str,
) -> pd.DataFrame:
    if blink_df.empty:
        return blink_df

    candidate_signal = pd.Series(signal, copy=False).to_numpy(dtype=float)
    blink_df = blink_df.copy()
    blink_velocity = compute_blink_velocity(candidate_signal)

    blink_df["aver_left_velocity"] = float("nan")
    blink_df["aver_right_velocity"] = float("nan")

    for col in (
        "pos_amp_vel_ratio_base",
        "neg_amp_vel_ratio_base",
        "peaks_pos_vel_base",
        "pos_amp_vel_ratio_zero",
        "neg_amp_vel_ratio_zero",
        "peaks_pos_vel_zero",
        "pos_amp_vel_ratio_tent",
        "neg_amp_vel_ratio_tent",
        "inter_blink_max_vel_base",
        "inter_blink_max_vel_zero",
    ):
        if col not in blink_df.columns:
            blink_df[col] = float("nan")

    velocity_valid = blink_df[["left_base", "right_base", "max_blink"]].notna().all(axis=1)
    for idx, row in blink_df.loc[velocity_valid].iterrows():
        left_base = int(row["left_base"])
        max_blink = int(row["max_blink"])
        right_base = int(row["right_base"])

        left_base = max(0, min(left_base, blink_velocity.size))
        max_blink = max(0, min(max_blink, blink_velocity.size))
        right_base = max(0, min(right_base, blink_velocity.size))

        left_segment = blink_velocity[left_base:max_blink]
        right_segment = blink_velocity[max_blink:right_base]

        blink_df.at[idx, "aver_left_velocity"] = (
            float(left_segment.mean()) if left_segment.size > 0 else float("nan")
        )
        blink_df.at[idx, "aver_right_velocity"] = (
            float(right_segment.mean()) if right_segment.size > 0 else float("nan")
        )

    base_valid = blink_df[["left_base", "right_base", "max_blink"]].notna().all(axis=1)
    if base_valid.any():
        base_df = blink_df.loc[base_valid].copy()
        compute_amp_vel_ratio_base(base_df, candidate_signal, blink_velocity, sfreq)
        blink_df.loc[
            base_valid,
            ["pos_amp_vel_ratio_base", "neg_amp_vel_ratio_base", "peaks_pos_vel_base"],
        ] = base_df[["pos_amp_vel_ratio_base", "neg_amp_vel_ratio_base", "peaks_pos_vel_base"]]

    zero_valid = blink_df[["left_zero", "right_zero", "max_blink"]].notna().all(axis=1)
    if zero_valid.any():
        zero_df = blink_df.loc[zero_valid].copy()
        compute_amp_vel_ratio_zero_to_max(
            zero_df,
            candidate_signal,
            blink_velocity,
            sfreq,
            modality=modality,
        )
        blink_df.loc[
            zero_valid,
            ["pos_amp_vel_ratio_zero", "neg_amp_vel_ratio_zero", "peaks_pos_vel_zero"],
        ] = zero_df[["pos_amp_vel_ratio_zero", "neg_amp_vel_ratio_zero", "peaks_pos_vel_zero"]]

    tent_valid = blink_df[["max_blink", "aver_left_velocity", "aver_right_velocity"]].notna().all(axis=1)
    if tent_valid.any():
        tent_df = blink_df.loc[tent_valid].copy()
        compute_amp_vel_ratio_tent(tent_df, candidate_signal, sfreq)
        blink_df.loc[
            tent_valid,
            ["pos_amp_vel_ratio_tent", "neg_amp_vel_ratio_tent"],
        ] = tent_df[["pos_amp_vel_ratio_tent", "neg_amp_vel_ratio_tent"]]

    inter_valid = blink_df[["peaks_pos_vel_base"]].notna().all(axis=1)
    if inter_valid.any():
        inter_df = blink_df.loc[inter_valid].copy()
        compute_inter_blink_max_vel(inter_df, sfreq, modality=modality, signal_len=len(candidate_signal))
        cols = ["inter_blink_max_vel_base"]
        if modality != "ear":
            cols.append("inter_blink_max_vel_zero")
        blink_df.loc[inter_valid, cols] = inter_df[cols]

    blink_df["amp_vel_ratio_base"] = blink_df[["pos_amp_vel_ratio_base", "neg_amp_vel_ratio_base"]].mean(axis=1)
    blink_df["amp_vel_ratio_zero_to_max"] = blink_df[["pos_amp_vel_ratio_zero", "neg_amp_vel_ratio_zero"]].mean(axis=1)
    blink_df["amp_vel_ratio_tent"] = blink_df[["pos_amp_vel_ratio_tent", "neg_amp_vel_ratio_tent"]].mean(axis=1)
    blink_df["blink_velocity"] = blink_df[["aver_left_velocity", "aver_right_velocity"]].abs().mean(axis=1)
    blink_df["inter_blink_max_vel"] = blink_df.get("inter_blink_max_vel_base", float("nan"))

    return blink_df


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


def _style_windows(
    metadata_row: Mapping[str, object],
    modality: str,
    style: str,
) -> List[tuple[float, float]]:
    """Extract blink windows for a modality/style pair."""

    onset_key = f"onset__{style}__{modality}"
    duration_key = f"duration__{style}__{modality}"
    onsets = ensure_list(metadata_row.get(onset_key)) if metadata_row.get(onset_key) is not None else []
    durations = (
        ensure_list(metadata_row.get(duration_key)) if metadata_row.get(duration_key) is not None else []
    )
    windows: List[tuple[float, float]] = []
    for onset, duration in zip(onsets, durations):
        if onset is None or duration is None:
            continue
        if pd.isna(onset) or pd.isna(duration):
            continue
        windows.append((float(onset), float(duration)))
    return windows


class KinematicBlinkFeatureExtractor:
    """Compute blink kinematic features from MNE objects."""

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

    def _metrics_for_style(self, style: str) -> List[str]:
        """Return kinematic metric names for a segmentation style.

        The returned names follow existing extractor conventions:
        - style-aware stems include a ``_{style}`` suffix,
        - style-agnostic stems remain unsuffixed,
        - extended parity metrics are appended as plain names.
        """

        metrics_for_style = [
            stem if stem in KINEMATIC_METRICS_NO_STYLE else f"{stem}_{style}"
            for stem in KINEMATIC_METRIC_STEMS
        ]
        metrics_for_style.extend(_EXTENDED_KINEMATIC_METRICS)
        return metrics_for_style

    def _build_output_columns(
        self,
        modality_channels: Dict[str, List[str]],
        styles_by_modality: Dict[str, Set[str]],
    ) -> List[str]:
        """Build the complete sorted output column set for all styles/channels."""

        column_set: Set[str] = set()
        for mod, channels in modality_channels.items():
            for style in sorted(styles_by_modality.get(mod, {"base"})):
                for metric in self._metrics_for_style(style):
                    for stat in _STATS:
                        for ch in channels:
                            column_set.add(f"{mod}__{style}__kinematic__{metric}_{stat}__{ch}")
        return sorted(column_set)

    def _compute_channel_style_metric_values(
        self,
        *,
        epoch_index: int,
        modality: str,
        style: str,
        channel_name: str,
        metadata_row: Mapping[str, object],
        channel_data: Dict[str, Dict[str, object]],
        sfreq: float,
        n_times: int,
    ) -> Dict[str, List[float]]:
        """Compute per-blink metric lists for one epoch/modality/style/channel."""

        metrics_for_style = self._metrics_for_style(style)
        windows = _style_windows(metadata_row, modality, style)

        blink_df = _compute_extended_kinematic_metrics(
            _build_kinematic_blink_frame(metadata_row, modality=modality, sfreq=sfreq),
            channel_data[channel_name]["raw"][epoch_index],
            sfreq,
            modality=modality,
        )

        per_metric: Dict[str, List[float]] = {m: [] for m in metrics_for_style}
        for onset_s, duration_s in windows:
            sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
            segment = {
                "raw": channel_data[channel_name]["raw"][epoch_index, sl],
                "dx1": channel_data[channel_name]["dx1"][epoch_index, sl],
                "dx2": channel_data[channel_name]["dx2"][epoch_index, sl],
            }
            metrics = compute_segment_kinematics(
                segment,
                sfreq,
                method=style,
                modality=modality,
            )
            for metric_name in metrics_for_style:
                if metric_name in _EXTENDED_KINEMATIC_METRICS:
                    continue
                per_metric[metric_name].append(metrics[metric_name])

        for metric_name in _EXTENDED_KINEMATIC_METRICS:
            if metric_name in blink_df.columns:
                per_metric[metric_name] = blink_df[metric_name].tolist()

        return per_metric

    def _write_channel_style_stats(
        self,
        *,
        record: Dict[str, float],
        modality: str,
        style: str,
        channel_name: str,
        per_metric: Dict[str, List[float]],
    ) -> None:
        """Aggregate and write one channel/style metric map into ``record``."""

        for metric, values in per_metric.items():
            stats = _safe_stats(values)
            for stat_name, value in stats.items():
                column = f"{modality}__{style}__kinematic__{metric}_{stat_name}__{channel_name}"
                record[column] = value

    def _compute_epoch_record(
        self,
        *,
        epoch_index: int,
        modality_channels: Dict[str, List[str]],
        styles_by_modality: Dict[str, Set[str]],
        metadata_row: Mapping[str, object],
        channel_data: Dict[str, Dict[str, object]],
        sfreq: float,
        n_times: int,
    ) -> Dict[str, float]:
        """Compute all channel/style kinematic aggregates for one epoch."""

        record: Dict[str, float] = {}
        for modality, channels in modality_channels.items():
            styles = styles_by_modality.get(modality, {"base"})
            for style in sorted(styles):
                for channel_name in channels:
                    per_metric = self._compute_channel_style_metric_values(
                        epoch_index=epoch_index,
                        modality=modality,
                        style=style,
                        channel_name=channel_name,
                        metadata_row=metadata_row,
                        channel_data=channel_data,
                        sfreq=sfreq,
                        n_times=n_times,
                    )
                    self._write_channel_style_stats(
                        record=record,
                        modality=modality,
                        style=style,
                        channel_name=channel_name,
                        per_metric=per_metric,
                    )
        return record

    def compute(self, picks: str | Sequence[str] | None = None) -> pd.DataFrame:
        """Compute kinematic blink features for each epoch and channel.

        Parameters
        ----------
        picks : str | sequence of str | None, optional
            Channel name or list of channel names to process. ``None`` uses all
            available channels.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed like ``epochs`` containing aggregated statistics of
            kinematic metrics for each channel.

        Notes
        -----
        If an epoch contains no blinks, all kinematic statistics for that epoch
        are ``NaN``.
        """

        sfreq = self._sampling_frequency()
        ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
            epochs=self.epochs,
            picks=picks,
            sfreq=sfreq,
        )

        modality_map: Dict[str, str] = {ch: _infer_modality(ch, self.epochs.info) for ch in ch_names}
        modality_channels: Dict[str, List[str]] = {}
        for ch, mod in modality_map.items():
            modality_channels.setdefault(mod, []).append(ch)

        metadata_cols: Sequence[str] | None = (
            tuple(self.epochs.metadata.columns) if isinstance(self.epochs.metadata, pd.DataFrame) else None
        )
        styles_by_modality: Dict[str, Set[str]] = {}
        for mod in set(modality_map.values()):
            styles_by_modality[mod] = _available_styles(metadata_cols, mod)

        columns = self._build_output_columns(modality_channels, styles_by_modality)
        if n_epochs == 0:
            return pd.DataFrame(index=index, columns=columns, dtype=float)

        records: List[Dict[str, float]] = []
        logger.info("Computing kinematic features for %d epochs", n_epochs)

        for ei in range(n_epochs):
            metadata_row = (
                self.epochs.metadata.iloc[ei]
                if isinstance(self.epochs.metadata, pd.DataFrame)
                else pd.Series(dtype=float)
            )
            records.append(
                self._compute_epoch_record(
                    epoch_index=ei,
                    modality_channels=modality_channels,
                    styles_by_modality=styles_by_modality,
                    metadata_row=metadata_row,
                    channel_data=channel_data,
                    sfreq=sfreq,
                    n_times=n_times,
                )
            )

        df = pd.DataFrame.from_records(records, index=index, columns=columns)
        logger.debug("Kinematic feature DataFrame shape: %s", df.shape)
        return df


def compute_kinematic_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute kinematic blink features for each epoch and channel."""

    extractor = KinematicBlinkFeatureExtractor(epochs=epochs)
    return extractor.compute(picks=picks)
