"""Helpers for preparing EAR threshold refinement results for visualization."""

from __future__ import annotations

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
    report_df["threshold_crossing_found"] = report_df["refinement_succeeded"].astype(bool)

    return report_df
