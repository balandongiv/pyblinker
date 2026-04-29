"""Blink statistics helper functions.
This module is similar to as in
extractBlinkProperties.m"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from pyblinker.blinker.default_setting import SCALING_FACTOR
from pyblinker.fitutils.forking import mad
from pyblinker.segmentation.geometry import get_max_blink


def _amplitude_gate_tolerance(
    amplitude_tolerance: float,
) -> float:
    """Return a small absolute tolerance for MATLAB-vs-Python drift at the gate."""
    return max(0.0, float(amplitude_tolerance))


def calculate_within_range(
    all_values: np.ndarray, best_median: float, best_robust_std: float
) -> int:
    """Return the count of values within two robust standard deviations."""

    lower_bound = best_median - 2 * best_robust_std
    upper_bound = best_median + 2 * best_robust_std
    within_mask = (all_values >= lower_bound) & (all_values <= upper_bound)
    return int(np.sum(within_mask))


def calculate_good_ratio(
    all_values: np.ndarray,
    best_median: float,
    best_robust_std: float,
    all_x: int,
) -> float:
    """Return the fraction of ``all_values`` within the robust range."""

    lower_bound = best_median - 2 * best_robust_std
    upper_bound = best_median + 2 * best_robust_std
    within_mask = (all_values >= lower_bound) & (all_values <= upper_bound)
    return float(np.sum(within_mask) / all_x)


def get_blink_statistic(
    df: pd.DataFrame, z_thresholds: np.ndarray, signal: np.ndarray | None = None
) -> dict:
    """Compute blink statistics for a DataFrame of blink fits.
    This is same as in extractBlinks.m under the for loop
    Calculate an amplitude criterion (frames in blink to those out) and Now calculate the cutoff ratios -- use default for the values
    %% Calculate an amplitude criterion (frames in blink to those out)
        % below is the same as in get_blink_statistic under pyblinker/utils/statistics_utils.py


    """
    correlation_threshold_bottom, correlation_threshold_top = z_thresholds[0]
    number_blinks = int(len(df))
    df_data = df[["left_zero", "right_zero", "leftR2", "rightR2", "max_value"]].copy()

    # MATLAB removes records with NaN for any of these fields before all later
    # computations (blink mask, good/worst values, and robust statistics).
    df_data = df_data.dropna(
        subset=["left_zero", "right_zero", "leftR2", "rightR2", "max_value"]
    )

    # The Python pipeline uses 0-based frame indices in blink fit outputs.
    # Keep this indexing (do not shift) and mirror MATLAB's inclusive range test.
    signal_values = (
        np.asarray(signal, dtype=float) if signal is not None else np.array([])
    )
    blink_mask = np.zeros(signal_values.shape[0], dtype=bool)
    for lz, rz in zip(
        df_data["left_zero"].to_numpy(), df_data["right_zero"].to_numpy()
    ):
        # MATLAB indexing accepts integer frame indices; use nearest integer
        # to mirror float-to-index semantics for values stored as doubles.
        left = int(np.rint(lz))
        right = int(np.rint(rz))
        if right <= left:
            continue
        left = max(0, left)
        right = min(signal_values.shape[0] - 1, right)
        blink_mask[left : right + 1] = True

    inside_blink = (signal_values > 0) & blink_mask
    outside_blink = (signal_values > 0) & ~blink_mask
    inside_mean = (
        np.mean(signal_values[inside_blink]) if np.any(inside_blink) else np.nan
    )
    outside_mean = (
        np.mean(signal_values[outside_blink]) if np.any(outside_blink) else np.nan
    )
    blink_amp_ratio = inside_mean / outside_mean

    df_data = df_data[["leftR2", "rightR2", "max_value"]]

    good_mask_top = (df_data["leftR2"] >= correlation_threshold_top) & (
        df_data["rightR2"] >= correlation_threshold_top
    )
    good_mask_bottom = (df_data["leftR2"] >= correlation_threshold_bottom) & (
        df_data["rightR2"] >= correlation_threshold_bottom
    )

    best_values = df_data.loc[good_mask_top, "max_value"].to_numpy()
    worst_values = df_data.loc[~good_mask_bottom, "max_value"].to_numpy()
    good_values = df_data.loc[good_mask_bottom, "max_value"].to_numpy()

    # MATLAB exits this candidate when less than two top-quality blink fits are
    # available. Keep deterministic, NaN outputs for missing statistics.
    if np.sum(good_mask_top) < 2:
        return {
            "number_blinks": number_blinks,
            "number_good_blinks": int(np.sum(good_mask_bottom)),
            "blink_amp_ratio": blink_amp_ratio,
            "cutoff": np.nan,
            "best_median": np.nan,
            "best_robust_std": np.nan,
            "good_ratio": np.nan,
        }

    best_median = np.nanmedian(best_values)
    best_robust_std = SCALING_FACTOR * mad(best_values)
    worst_median = np.nanmedian(worst_values)
    worst_robust_std = SCALING_FACTOR * mad(worst_values)

    denom = best_robust_std + worst_robust_std
    cutoff = (
        np.nan
        if np.isnan(denom) or np.isclose(denom, 0.0)
        else (best_median * worst_robust_std + worst_median * best_robust_std) / denom
    )

    all_x = calculate_within_range(
        df_data["max_value"].to_numpy(), best_median, best_robust_std
    )
    good_ratio = (
        np.nan
        if all_x <= 0
        else calculate_good_ratio(good_values, best_median, best_robust_std, all_x)
    )

    number_good_blinks = int(np.sum(good_mask_bottom))

    return {
        "number_blinks": number_blinks,
        "number_good_blinks": number_good_blinks,
        "blink_amp_ratio": blink_amp_ratio,
        "cutoff": cutoff,
        "best_median": best_median,
        "best_robust_std": best_robust_std,
        "good_ratio": good_ratio,
    }


def get_good_blink_mask(
    blink_fits: pd.DataFrame,
    specified_median: float,
    specified_std: float,
    z_thresholds: np.ndarray,
    *,
    amplitude_tolerance: float = 0.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return mask of good blinks and subset DataFrame based on thresholds."""
    # Calculate an amplitude criterion (frames in blink to those out)

    blink_fits = blink_fits.dropna(subset=["leftR2", "rightR2", "max_value"])

    left_r2 = blink_fits["leftR2"].to_numpy()
    right_r2 = blink_fits["rightR2"].to_numpy()
    max_value = blink_fits["max_value"].to_numpy()

    correlation_thresholds = z_thresholds[0]
    z_score_thresholds = z_thresholds[1]
    amp_tol = _amplitude_gate_tolerance(amplitude_tolerance)

    lower_bounds = np.maximum(0, specified_median - z_score_thresholds * specified_std)
    upper_bounds = specified_median + z_score_thresholds * specified_std

    left_r2 = left_r2[:, None]
    right_r2 = right_r2[:, None]
    max_value = max_value[:, None]
    correlation_thresholds = correlation_thresholds[None, :]
    lower_bounds = lower_bounds[None, :]
    upper_bounds = upper_bounds[None, :]

    masks = (
        (left_r2 >= correlation_thresholds)
        & (right_r2 >= correlation_thresholds)
        & (max_value >= (lower_bounds - amp_tol))
        & (max_value <= (upper_bounds + amp_tol))
    )
    good_blink_mask = np.any(masks, axis=1)
    selected_rows = blink_fits[good_blink_mask]
    return good_blink_mask, selected_rows


__all__ = [
    "get_max_blink",
    "calculate_within_range",
    "calculate_good_ratio",
    "get_blink_statistic",
    "get_good_blink_mask",
]
