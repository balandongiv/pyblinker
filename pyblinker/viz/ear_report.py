"""Helpers for preparing EAR threshold refinement results for visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["prepare_threshold_report_dataframe"]


def prepare_threshold_report_dataframe(
    features: pd.DataFrame, sfreq: float, threshold_value: float
) -> pd.DataFrame:
    """Return a report-ready DataFrame for a single EAR threshold value."""

    report_df = features.loc[features["threshold_value"] == threshold_value].copy()
    report_df["ear_threshold_left_sample"] = pd.to_numeric(
        report_df["refined_left_threshold"], errors="coerce"
    )
    report_df["ear_threshold_right_sample"] = pd.to_numeric(
        report_df["refined_right_threshold"], errors="coerce"
    )
    report_df["ear_threshold_min_sample"] = pd.to_numeric(
        report_df["refined_lowest_point_sample"], errors="coerce"
    )

    missing_left = report_df["refined_left_threshold"].isna()
    missing_right = report_df["refined_right_threshold"].isna()
    missing_min = report_df["refined_lowest_point_sample"].isna()

    report_df.loc[missing_left, "ear_threshold_left_sample"] = report_df.loc[
        missing_left, "refined_start_sample"
    ]
    report_df.loc[missing_right, "ear_threshold_right_sample"] = report_df.loc[
        missing_right, "refined_end_sample"
    ]
    report_df.loc[missing_min, "ear_threshold_min_sample"] = report_df.loc[
        missing_min, "refined_start_sample"
    ]

    report_df["ear_threshold_left_sample"] = (
        report_df["ear_threshold_left_sample"].fillna(report_df["refined_start_sample"]).astype(int)
    )
    report_df["ear_threshold_right_sample"] = (
        report_df["ear_threshold_right_sample"].fillna(report_df["refined_end_sample"]).astype(int)
    )
    report_df["ear_threshold_min_sample"] = (
        report_df["ear_threshold_min_sample"].fillna(report_df["refined_start_sample"]).astype(int)
    )
    report_df["ear_threshold_left_time"] = report_df["ear_threshold_left_sample"] / float(sfreq)
    report_df["ear_threshold_right_time"] = report_df["ear_threshold_right_sample"] / float(sfreq)
    report_df["ear_threshold_min_time"] = report_df["ear_threshold_min_sample"] / float(sfreq)
    report_df["threshold_crossing_found"] = report_df["refinement_succeeded"].astype(bool)

    left_interp_time = report_df.get(
        "left_interpolated_threshold", pd.Series(np.nan, index=report_df.index)
    )
    right_interp_time = report_df.get(
        "right_interpolated_threshold", pd.Series(np.nan, index=report_df.index)
    )
    left_interp_sample = report_df.get(
        "left_interpolated_threshold_sample", pd.Series(np.nan, index=report_df.index)
    )
    right_interp_sample = report_df.get(
        "right_interpolated_threshold_sample", pd.Series(np.nan, index=report_df.index)
    )

    report_df["ear_interpolated_left_time"] = pd.to_numeric(left_interp_time, errors="coerce")
    report_df["ear_interpolated_right_time"] = pd.to_numeric(right_interp_time, errors="coerce")
    report_df["ear_interpolated_left_sample"] = pd.to_numeric(left_interp_sample, errors="coerce")
    report_df["ear_interpolated_right_sample"] = pd.to_numeric(right_interp_sample, errors="coerce")

    missing_left_sample = report_df["ear_interpolated_left_sample"].isna()
    missing_right_sample = report_df["ear_interpolated_right_sample"].isna()

    report_df.loc[missing_left_sample, "ear_interpolated_left_sample"] = (
        report_df.loc[missing_left_sample, "ear_interpolated_left_time"] * float(sfreq)
    )
    report_df.loc[missing_right_sample, "ear_interpolated_right_sample"] = (
        report_df.loc[missing_right_sample, "ear_interpolated_right_time"] * float(sfreq)
    )

    return report_df
