"""Blink kinematic feature calculations based on epoch metadata.

Blink windows are resolved from start/end frame metadata so signal segments can be
indexed directly in sample space without onset+duration conversions.
"""

from __future__ import annotations
from pyblinker._logging import get_logger

from typing import Dict, List, Mapping, Sequence, Set

import mne
import pandas as pd

from .column_headers import (
    EXTENDED_METRICS,
    build_output_columns,
    make_stat_column,
    metrics_for_style,
)
from .core_metrics import (
    compute_amp_vel_ratio_base,
    compute_amp_vel_ratio_tent,
    compute_amp_vel_ratio_zero_to_max,
    compute_blink_velocity,
    compute_inter_blink_max_vel,
)
from .per_blink import compute_segment_kinematics
from . import helpers as kin_helpers
from ..energy.helpers import compute_basic_statistics
from ...utils.iter_utils import ensure_list
from ..utils.aggregation import prepare_epoch_channel_data
from .._epoch_context import build_epoch_context, empty_feature_frame, get_metadata_row
from ..constants import cast_columns_to_object

logger = get_logger(__name__)


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
    data = {
        k: kin_helpers.coerce_numeric_list(metadata_row.get(col), ensure_list)
        for k, col in landmark_keys.items()
    }

    peak_key_candidates = (
        f"onset__refine_extremum__{modality}",
        f"blink_onset_extremum_{modality}",
    )
    peak_times_sec: List[float] = []
    for peak_key in peak_key_candidates:
        if metadata_row.get(peak_key) is not None:
            peak_times_sec = kin_helpers.coerce_numeric_list(
                metadata_row.get(peak_key), ensure_list
            )
            if peak_times_sec:
                break

    lengths = [len(v) for v in data.values()]
    lengths.append(len(peak_times_sec))
    n_blinks = max(lengths) if lengths else 0
    if n_blinks == 0:
        return pd.DataFrame()

    for key, values in data.items():
        data[key] = kin_helpers.pad(values, n_blinks)

    max_blink = [float("nan")] * n_blinks
    for i, peak_time in enumerate(kin_helpers.pad(peak_times_sec, n_blinks)):
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

    _initialize_extended_columns(blink_df)
    _populate_average_velocities(blink_df, blink_velocity)
    _populate_amp_velocity_ratios(
        blink_df, candidate_signal, blink_velocity, sfreq, modality
    )
    _populate_inter_blink_velocity(blink_df, candidate_signal, sfreq, modality)

    blink_df["amp_vel_ratio_base"] = blink_df[
        ["pos_amp_vel_ratio_base", "neg_amp_vel_ratio_base"]
    ].mean(axis=1)
    blink_df["amp_vel_ratio_zero_to_max"] = blink_df[
        ["pos_amp_vel_ratio_zero", "neg_amp_vel_ratio_zero"]
    ].mean(axis=1)
    blink_df["amp_vel_ratio_tent"] = blink_df[
        ["pos_amp_vel_ratio_tent", "neg_amp_vel_ratio_tent"]
    ].mean(axis=1)
    blink_df["blink_velocity"] = (
        blink_df[["aver_left_velocity", "aver_right_velocity"]].abs().mean(axis=1)
    )
    blink_df["inter_blink_max_vel"] = blink_df.get(
        "inter_blink_max_vel_base", float("nan")
    )

    return blink_df


def _initialize_extended_columns(blink_df: pd.DataFrame) -> None:
    """Ensure all intermediate extended-kinematic columns exist before filling values."""

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


def _populate_average_velocities(
    blink_df: pd.DataFrame, blink_velocity: object
) -> None:
    """Populate mean opening/closing velocities for each blink from frame-aligned bounds."""

    velocity = pd.Series(blink_velocity, copy=False).to_numpy(dtype=float)
    velocity_valid = (
        blink_df[["left_base", "right_base", "max_blink"]].notna().all(axis=1)
    )
    for idx, row in blink_df.loc[velocity_valid].iterrows():
        left_base = max(0, min(int(row["left_base"]), velocity.size))
        max_blink = max(0, min(int(row["max_blink"]), velocity.size))
        right_base = max(0, min(int(row["right_base"]), velocity.size))

        left_segment = velocity[left_base:max_blink]
        right_segment = velocity[max_blink:right_base]

        blink_df.at[idx, "aver_left_velocity"] = (
            float(left_segment.mean()) if left_segment.size > 0 else float("nan")
        )
        blink_df.at[idx, "aver_right_velocity"] = (
            float(right_segment.mean()) if right_segment.size > 0 else float("nan")
        )


def _populate_amp_velocity_ratios(
    blink_df: pd.DataFrame,
    candidate_signal: object,
    blink_velocity: object,
    sfreq: float,
    modality: str,
) -> None:
    """Compute base/zero/tent amplitude-velocity ratio variants on valid blink subsets."""

    base_valid = blink_df[["left_base", "right_base", "max_blink"]].notna().all(axis=1)
    if base_valid.any():
        base_df = blink_df.loc[base_valid].copy()
        compute_amp_vel_ratio_base(base_df, candidate_signal, blink_velocity, sfreq)
        blink_df.loc[
            base_valid,
            ["pos_amp_vel_ratio_base", "neg_amp_vel_ratio_base", "peaks_pos_vel_base"],
        ] = base_df[
            ["pos_amp_vel_ratio_base", "neg_amp_vel_ratio_base", "peaks_pos_vel_base"]
        ]

    zero_valid = blink_df[["left_zero", "right_zero", "max_blink"]].notna().all(axis=1)
    if zero_valid.any():
        zero_df = blink_df.loc[zero_valid].copy()
        compute_amp_vel_ratio_zero_to_max(
            zero_df, candidate_signal, blink_velocity, sfreq, modality=modality
        )
        blink_df.loc[
            zero_valid,
            ["pos_amp_vel_ratio_zero", "neg_amp_vel_ratio_zero", "peaks_pos_vel_zero"],
        ] = zero_df[
            ["pos_amp_vel_ratio_zero", "neg_amp_vel_ratio_zero", "peaks_pos_vel_zero"]
        ]

    tent_valid = (
        blink_df[["max_blink", "aver_left_velocity", "aver_right_velocity"]]
        .notna()
        .all(axis=1)
    )
    if tent_valid.any():
        tent_df = blink_df.loc[tent_valid].copy()
        compute_amp_vel_ratio_tent(tent_df, candidate_signal, sfreq)
        blink_df.loc[
            tent_valid, ["pos_amp_vel_ratio_tent", "neg_amp_vel_ratio_tent"]
        ] = tent_df[["pos_amp_vel_ratio_tent", "neg_amp_vel_ratio_tent"]]


def _populate_inter_blink_velocity(
    blink_df: pd.DataFrame,
    candidate_signal: object,
    sfreq: float,
    modality: str,
) -> None:
    """Compute inter-blink max velocity values using previously estimated positive peaks."""

    inter_valid = blink_df[["peaks_pos_vel_base"]].notna().all(axis=1)
    if not inter_valid.any():
        return

    inter_df = blink_df.loc[inter_valid].copy()
    compute_inter_blink_max_vel(
        inter_df, sfreq, modality=modality, signal_len=len(candidate_signal)
    )
    cols = ["inter_blink_max_vel_base"]
    if modality != "ear":
        cols.append("inter_blink_max_vel_zero")
    blink_df.loc[inter_valid, cols] = inter_df[cols]


def _compute_metrics_over_windows(
    *,
    windows: Sequence[tuple[int, int]],
    n_times: int,
    channel_data: Mapping[str, Mapping[str, object]],
    channel_name: str,
    epoch_index: int,
    sfreq: float,
    style: str,
    modality: str,
    metrics_for_style: Sequence[str],
) -> Dict[str, List[float]]:
    """Compute per-window kinematic metrics for one epoch/channel/style."""

    per_metric: Dict[str, List[float]] = {m: [] for m in metrics_for_style}
    for start_idx, end_idx in windows:
        if start_idx >= n_times:
            continue
        sl = slice(max(0, start_idx), min(end_idx, n_times))
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
            if metric_name in EXTENDED_METRICS:
                continue
            metric_value = metrics.get(metric_name)
            if (
                metric_value is None
                and style not in {"base", "zero", "tent"}
                and metric_name.endswith("_base")
            ):
                style_metric = metric_name[: -len("_base")] + f"_{style}"
                metric_value = metrics.get(style_metric)
            if metric_value is None:
                metric_value = float("nan")
            per_metric[metric_name].append(float(metric_value))

    return per_metric


def _write_style_stats_into_record(
    *,
    record: Dict[str, float],
    per_metric: Dict[str, List[float]],
    blink_df: pd.DataFrame,
    modality: str,
    style: str,
    channel_name: str,
) -> None:
    """Merge legacy extended metrics and write style statistics into an epoch record."""

    for metric_name in EXTENDED_METRICS:
        if metric_name in blink_df.columns:
            per_metric[metric_name] = blink_df[metric_name].tolist()

    for metric_name, values in per_metric.items():
        stats = compute_basic_statistics(values)
        for stat_name, value in stats.items():
            column = make_stat_column(
                modality=modality,
                style=style,
                metric=metric_name,
                stat=stat_name,
                channel=channel_name,
            )
            record[column] = value


def _available_styles(
    metadata_columns: Sequence[str] | None, modality: str
) -> Set[str]:
    """Return frame-based segmentation styles present in metadata for a modality."""

    if metadata_columns is None:
        return set()

    styles: Set[str] = set()

    # Canonical styles keep existing EEG/EOG behavior unchanged.
    landmark_styles = {
        "base": ("start__left_base", "end__right_base"),
        "zero": ("start__left_zero", "end__right_zero"),
        "tent": ("start__left_x_intercept", "end__right_x_intercept"),
    }
    for style, (start_key, end_key) in landmark_styles.items():
        start_col = f"{start_key}__{modality}"
        end_col = f"{end_key}__{modality}"
        if start_col in metadata_columns and end_col in metadata_columns:
            styles.add(style)

    start_prefix = "start__"
    modality_suffix = f"__{modality}"
    metadata_set = set(metadata_columns)
    for col in metadata_columns:
        if not col.startswith(start_prefix) or not col.endswith(modality_suffix):
            continue
        style = col[len(start_prefix) : -len(modality_suffix)]
        if not style:
            continue
        end_col = f"end__{style}__{modality}"
        if end_col in metadata_set:
            styles.add(style)

    return styles


def _style_windows(
    metadata_row: Mapping[str, object],
    modality: str,
    style: str,
) -> List[tuple[int, int]]:
    """Extract frame-aligned blink windows as ``(start_sample, end_sample)`` tuples."""

    landmark_style_keys = {
        "base": ("start__left_base", "end__right_base"),
        "zero": ("start__left_zero", "end__right_zero"),
        "tent": ("start__left_x_intercept", "end__right_x_intercept"),
    }
    if style in landmark_style_keys:
        start_prefix, end_prefix = landmark_style_keys[style]
        start_key = f"{start_prefix}__{modality}"
        end_key = f"{end_prefix}__{modality}"
    else:
        start_key = f"start__{style}__{modality}"
        end_key = f"end__{style}__{modality}"

    starts = ensure_list(metadata_row.get(start_key))
    ends = ensure_list(metadata_row.get(end_key))

    windows: List[tuple[int, int]] = []
    for start_frame, end_frame in zip(starts, ends):
        if start_frame is None or end_frame is None:
            continue
        if pd.isna(start_frame) or pd.isna(end_frame):
            continue
        start_idx = int(round(float(start_frame)))
        end_idx = int(round(float(end_frame)))
        if end_idx <= start_idx:
            continue
        windows.append((start_idx, end_idx))
    return windows


class KinematicBlinkFeatureExtractor:
    """Compute blink kinematic features from MNE objects."""

    def __init__(
        self, epochs: mne.Epochs | None = None, raw: mne.io.BaseRaw | None = None
    ):
        self.epochs = epochs
        self.raw = raw

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

        ctx = build_epoch_context(self.epochs, picks)
        ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
            epochs=self.epochs,
            picks=ctx.ch_names,
            sfreq=ctx.sfreq,
        )

        modality_map: Dict[str, str] = ctx.modality_by_channel
        modality_channels = self._group_channels_by_modality(modality_map)
        styles_by_modality = self._build_styles_by_modality(
            set(modality_channels), ctx.metadata_cols
        )

        columns = build_output_columns(modality_channels, styles_by_modality)

        if n_epochs == 0 or not columns:
            return cast_columns_to_object(
                empty_feature_frame(index=index, columns=columns)
            )

        records: List[Dict[str, float]] = []

        for ei in range(n_epochs):
            metadata_row = get_metadata_row(self.epochs, ei)
            record = self._compute_epoch_record(
                epoch_index=ei,
                metadata_row=metadata_row,
                modality_channels=modality_channels,
                styles_by_modality=styles_by_modality,
                channel_data=channel_data,
                sfreq=ctx.sfreq,
                n_times=n_times,
                n_epochs=n_epochs,
            )
            records.append(record)
        df = pd.DataFrame.from_records(records, index=index)
        logger.debug("Kinematic feature DataFrame shape: %s", df.shape)
        return cast_columns_to_object(df)

    def _group_channels_by_modality(
        self, modality_map: Dict[str, str]
    ) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for channel_name, modality in modality_map.items():
            grouped.setdefault(modality, []).append(channel_name)
        return grouped

    def _build_styles_by_modality(
        self,
        modalities: Set[str],
        metadata_cols: Sequence[str] | None,
    ) -> Dict[str, Set[str]]:
        styles_by_modality: Dict[str, Set[str]] = {}
        for modality in modalities:
            styles_by_modality[modality] = _available_styles(metadata_cols, modality)
        return styles_by_modality

    def _compute_epoch_record(
        self,
        epoch_index: int,
        metadata_row: pd.Series,
        modality_channels: Dict[str, List[str]],
        styles_by_modality: Dict[str, Set[str]],
        channel_data: Mapping[str, Mapping[str, object]],
        sfreq: float,
        n_times: int,
        n_epochs: int,
    ) -> Dict[str, float]:
        logger.debug("Kinematic epoch %d/%d", epoch_index + 1, n_epochs)
        record: Dict[str, float] = {}
        for modality, channels in modality_channels.items():
            styles = sorted(styles_by_modality.get(modality) or {"base"})
            for channel_name in channels:
                self._compute_channel_record(
                    record=record,
                    metadata_row=metadata_row,
                    channel_data=channel_data,
                    channel_name=channel_name,
                    epoch_index=epoch_index,
                    sfreq=sfreq,
                    n_times=n_times,
                    modality=modality,
                    styles=styles,
                )
        return record

    def _compute_channel_record(
        self,
        *,
        record: Dict[str, float],
        metadata_row: pd.Series,
        channel_data: Mapping[str, Mapping[str, object]],
        channel_name: str,
        epoch_index: int,
        sfreq: float,
        n_times: int,
        modality: str,
        styles: Sequence[str],
    ) -> None:
        signal = channel_data[channel_name]["raw"][epoch_index]
        blink_df = self._build_blink_df(metadata_row, signal, sfreq, modality)
        for style in styles:
            self._compute_style_stats_into_record(
                record=record,
                metadata_row=metadata_row,
                channel_data=channel_data,
                channel_name=channel_name,
                epoch_index=epoch_index,
                sfreq=sfreq,
                n_times=n_times,
                modality=modality,
                style=style,
                blink_df=blink_df,
            )

    def _build_blink_df(
        self,
        metadata_row: Mapping[str, object],
        signal: pd.Series | List[float] | object,
        sfreq: float,
        modality: str,
    ) -> pd.DataFrame:
        blink_df = _build_kinematic_blink_frame(
            metadata_row, modality=modality, sfreq=sfreq
        )
        blink_df = _compute_extended_kinematic_metrics(
            blink_df, signal, sfreq, modality=modality
        )
        return blink_df

    def _compute_style_stats_into_record(
        self,
        *,
        record: Dict[str, float],
        metadata_row: Mapping[str, object],
        channel_data: Mapping[str, Mapping[str, object]],
        channel_name: str,
        epoch_index: int,
        sfreq: float,
        n_times: int,
        modality: str,
        style: str,
        blink_df: pd.DataFrame,
    ) -> None:
        windows = _style_windows(metadata_row, modality, style)
        style_metrics = list(metrics_for_style(style)) + list(EXTENDED_METRICS)
        per_metric = _compute_metrics_over_windows(
            windows=windows,
            n_times=n_times,
            channel_data=channel_data,
            channel_name=channel_name,
            epoch_index=epoch_index,
            sfreq=sfreq,
            style=style,
            modality=modality,
            metrics_for_style=style_metrics,
        )
        _write_style_stats_into_record(
            record=record,
            per_metric=per_metric,
            blink_df=blink_df,
            modality=modality,
            style=style,
            channel_name=channel_name,
        )


def compute_kinematic_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute kinematic blink features for each epoch and channel."""

    extractor = KinematicBlinkFeatureExtractor(epochs=epochs)
    return extractor.compute(picks=picks)
