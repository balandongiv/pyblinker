"""Internal helper utilities for kinematic feature extraction."""

from __future__ import annotations

from typing import List, Mapping

import pandas as pd

from pyblinker.utils.iter_utils import ensure_list


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
