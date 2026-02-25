"""Blink kinematic feature calculations based on epoch metadata.

Blink windows are resolved from start/end frame metadata so signal segments can be
indexed directly in sample space without onset+duration conversions.
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
from ..energy.helpers import _safe_stats
from ..constants import DEFAULT_BLINKER_CONFIG, BlinkerConfig
from ..utils.compute_skeleton import build_epoch_metadata_row, prepare_compute_context
from ..utils.style_windows import available_styles, extract_windows
from .helpers import (
    _build_kinematic_blink_frame,
    _initialize_extended_columns,
)

logger = get_logger(__name__)

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
    _populate_amp_velocity_ratios(blink_df, candidate_signal, blink_velocity, sfreq, modality)
    _populate_inter_blink_velocity(blink_df, candidate_signal, sfreq, modality)

    blink_df["amp_vel_ratio_base"] = blink_df[["pos_amp_vel_ratio_base", "neg_amp_vel_ratio_base"]].mean(axis=1)
    blink_df["amp_vel_ratio_zero_to_max"] = blink_df[["pos_amp_vel_ratio_zero", "neg_amp_vel_ratio_zero"]].mean(axis=1)
    blink_df["amp_vel_ratio_tent"] = blink_df[["pos_amp_vel_ratio_tent", "neg_amp_vel_ratio_tent"]].mean(axis=1)
    blink_df["blink_velocity"] = blink_df[["aver_left_velocity", "aver_right_velocity"]].abs().mean(axis=1)
    blink_df["inter_blink_max_vel"] = blink_df.get("inter_blink_max_vel_base", float("nan"))

    return blink_df


def _populate_average_velocities(blink_df: pd.DataFrame, blink_velocity: object) -> None:
    """Populate mean opening/closing velocities for each blink from frame-aligned bounds."""

    velocity = pd.Series(blink_velocity, copy=False).to_numpy(dtype=float)
    velocity_valid = blink_df[["left_base", "right_base", "max_blink"]].notna().all(axis=1)
    for idx, row in blink_df.loc[velocity_valid].iterrows():
        left_base = max(0, min(int(row["left_base"]), velocity.size))
        max_blink = max(0, min(int(row["max_blink"]), velocity.size))
        right_base = max(0, min(int(row["right_base"]), velocity.size))

        left_segment = velocity[left_base:max_blink]
        right_segment = velocity[max_blink:right_base]

        blink_df.at[idx, "aver_left_velocity"] = float(left_segment.mean()) if left_segment.size > 0 else float("nan")
        blink_df.at[idx, "aver_right_velocity"] = float(right_segment.mean()) if right_segment.size > 0 else float("nan")


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
        blink_df.loc[base_valid, ["pos_amp_vel_ratio_base", "neg_amp_vel_ratio_base", "peaks_pos_vel_base"]] = base_df[
            ["pos_amp_vel_ratio_base", "neg_amp_vel_ratio_base", "peaks_pos_vel_base"]
        ]

    zero_valid = blink_df[["left_zero", "right_zero", "max_blink"]].notna().all(axis=1)
    if zero_valid.any():
        zero_df = blink_df.loc[zero_valid].copy()
        compute_amp_vel_ratio_zero_to_max(zero_df, candidate_signal, blink_velocity, sfreq, modality=modality)
        blink_df.loc[zero_valid, ["pos_amp_vel_ratio_zero", "neg_amp_vel_ratio_zero", "peaks_pos_vel_zero"]] = zero_df[
            ["pos_amp_vel_ratio_zero", "neg_amp_vel_ratio_zero", "peaks_pos_vel_zero"]
        ]

    tent_valid = blink_df[["max_blink", "aver_left_velocity", "aver_right_velocity"]].notna().all(axis=1)
    if tent_valid.any():
        tent_df = blink_df.loc[tent_valid].copy()
        compute_amp_vel_ratio_tent(tent_df, candidate_signal, sfreq)
        blink_df.loc[tent_valid, ["pos_amp_vel_ratio_tent", "neg_amp_vel_ratio_tent"]] = tent_df[
            ["pos_amp_vel_ratio_tent", "neg_amp_vel_ratio_tent"]
        ]


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
    compute_inter_blink_max_vel(inter_df, sfreq, modality=modality, signal_len=len(candidate_signal))
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
            if metric_name in _EXTENDED_KINEMATIC_METRICS:
                continue
            metric_value = metrics.get(metric_name)
            if metric_value is None and style not in {"base", "zero", "tent"} and metric_name.endswith("_base"):
                style_metric = metric_name[:-len("_base")] + f"_{style}"
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

    for metric_name in _EXTENDED_KINEMATIC_METRICS:
        if metric_name in blink_df.columns:
            per_metric[metric_name] = blink_df[metric_name].tolist()

    for metric_name, values in per_metric.items():
        stats = _safe_stats(values)
        for stat_name, value in stats.items():
            column = f"{modality}__{style}__kinematic__{metric_name}_{stat_name}__{channel_name}"
            record[column] = value



def _normalize_styles_for_modality(styles: Set[str], modality: str) -> Set[str]:
    if modality in {"eeg", "eog"}:
        normalized: Set[str] = set()
        if "zero" in styles:
            normalized.add("zero")
        if "base" in styles:
            normalized.add("base")
        if "tent" in styles:
            normalized.add("tent")
        return normalized

    if modality == "ear":
        normalized: Set[str] = set()
        if "th_interpolation" in styles:
            normalized.add("th_interpolation")
        if "th_point" in styles:
            normalized.add("th_point")
        return normalized

    return styles

def _metrics_for_style(style: str) -> List[str]:
    """Return output metric names for a segmentation style."""

    metric_suffix = style if style in {"base", "zero", "tent"} else "base"
    return [
        stem if stem in KINEMATIC_METRICS_NO_STYLE else f"{stem}_{metric_suffix}"
        for stem in KINEMATIC_METRIC_STEMS
    ]

class KinematicBlinkFeatureExtractor:
    """Compute blink kinematic features from MNE objects."""

    def __init__(
        self,
        epochs: mne.Epochs | None = None,
        raw: mne.io.BaseRaw | None = None,
        config: BlinkerConfig = DEFAULT_BLINKER_CONFIG,
    ):
        self.epochs = epochs
        self.raw = raw
        self.config = config

    def _sampling_frequency(self) -> float:
        """Return sampling frequency from available MNE object."""
        if hasattr(self, "epochs") and self.epochs is not None:
            return float(self.epochs.info["sfreq"])
        if hasattr(self, "raw") and self.raw is not None:
            return float(self.raw.info["sfreq"])
        raise ValueError("Neither self.epochs nor self.raw defined (need MNE object).")

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

        context = prepare_compute_context(
            epochs=self.epochs,
            picks=picks,
            style_getter=lambda metadata_cols, mod: _normalize_styles_for_modality(
                available_styles(metadata_cols, mod, onset_prefix=None, duration_prefix=None),
                mod,
            ),
        )
        sfreq = context.sfreq
        channel_data = context.channel_data
        index = context.index
        n_epochs = context.n_epochs
        n_times = context.n_times
        modality_channels = context.modality_channels
        styles_by_modality = context.styles_by_modality

        column_set: Set[str] = set()
        for mod, channels in modality_channels.items():
            for style in sorted(styles_by_modality.get(mod) or {"base"}):
                metrics_for_style = _metrics_for_style(style)
                metrics_for_style.extend(_EXTENDED_KINEMATIC_METRICS)
                for metric in metrics_for_style:
                    for stat in self.config.stat_names:
                        for ch in channels:
                            column_set.add(f"{mod}__{style}__kinematic__{metric}_{stat}__{ch}")
        columns = sorted(column_set)
        if n_epochs == 0:
            return pd.DataFrame(index=index, columns=columns, dtype=float)

        records: List[Dict[str, float]] = []
        logger.info("Computing kinematic features for %d epochs", n_epochs)

        for ei in range(n_epochs):
            metadata_row = build_epoch_metadata_row(self.epochs, ei)
            record: Dict[str, float] = {}
            for modality, channels in modality_channels.items():
                styles = styles_by_modality.get(modality) or {"base"}
                # use_fallback = fallback_styles.get(modality, False)
                for style in sorted(styles):
                    metrics_for_style = _metrics_for_style(style)
                    metrics_for_style.extend(_EXTENDED_KINEMATIC_METRICS)
                    windows = extract_windows(metadata_row, modality, style, n_times)

                    for ch in channels:
                        # calculate the legacy kinematics features
                        blink_df = _compute_extended_kinematic_metrics(
                            _build_kinematic_blink_frame(metadata_row, modality=modality, sfreq=sfreq),
                            channel_data[ch]["raw"][ei],
                            sfreq,
                            modality=modality,
                        )
                        per_metric = _compute_metrics_over_windows(
                            windows=windows,
                            n_times=n_times,
                            channel_data=channel_data,
                            channel_name=ch,
                            epoch_index=ei,
                            sfreq=sfreq,
                            style=style,
                            modality=modality,
                            metrics_for_style=metrics_for_style,
                        )
                        _write_style_stats_into_record(
                            record=record,
                            per_metric=per_metric,
                            blink_df=blink_df,
                            modality=modality,
                            style=style,
                            channel_name=ch,
                        )
            records.append(record)

        df = pd.DataFrame.from_records(records, index=index, columns=columns)
        # df = _add_legacy_ear_interpolation_aliases(df) # If there is error, this is the place to check for the column names in the test and make sure they match the expected format.
        df.columns = pd.Index([str(col) for col in df.columns], dtype=object)
        logger.debug("Kinematic feature DataFrame shape: %s", df.shape)
        return df


def _add_legacy_ear_interpolation_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Expose historical EAR interpolation column aliases used by old tests."""

    if df.empty:
        return df

    alias_updates: Dict[str, pd.Series] = {}
    for col in df.columns:
        if "ear__th_interpolation__kinematic__" not in col:
            continue
        alias_col = col.replace("ear__th_interpolation__", "ear__ interpolated_threshold__")
        if "__" in alias_col:
            head, tail = alias_col.rsplit("__", 1)
            alias_col = f"{head}____{tail}"
        alias_updates[alias_col] = df[col]

    if not alias_updates:
        return df

    return df.assign(**alias_updates)


def compute_kinematic_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute kinematic blink features for each epoch and channel."""

    extractor = KinematicBlinkFeatureExtractor(epochs=epochs)
    return extractor.compute(picks=picks)
