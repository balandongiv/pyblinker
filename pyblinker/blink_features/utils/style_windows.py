"""Shared style/window discovery helpers across feature families."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from ...utils.iter_utils import ensure_list


def available_styles(
    metadata_columns,
    modality: str,
    *,
    onset_prefix: str = "onset__",
    duration_prefix: str = "duration__",
) -> set[str]:
    """Return detected styles for a modality from metadata columns."""

    if metadata_columns is None:
        return set()

    styles: set[str] = set()
    suffix = f"__{modality}"
    metadata_set = set(metadata_columns)

    for col in metadata_columns:
        if not col.startswith(onset_prefix) or not col.endswith(suffix):
            continue
        style = col[len(onset_prefix) : -len(suffix)]
        if not style or "sample" in style.lower():
            continue
        duration_col = f"{duration_prefix}{style}__{modality}"
        if duration_col in metadata_set:
            styles.add(style)

    landmark_styles = {
        "base": (f"start__left_base__{modality}", f"end__right_base__{modality}"),
        "zero": (f"start__left_zero__{modality}", f"end__right_zero__{modality}"),
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
    for style, (start_col, end_col) in landmark_styles.items():
        if start_col in metadata_set and end_col in metadata_set:
            styles.add(style)

    for col in metadata_columns:
        if not col.startswith("start__") or not col.endswith(suffix):
            continue
        style = col[len("start__") : -len(suffix)]
        if style and f"end__{style}__{modality}" in metadata_set:
            styles.add(style)

    return styles


def extract_windows(
    metadata_row: Mapping[str, object],
    modality: str,
    style: str,
    n_times: int,
    *,
    start_prefix: str = "start__",
    end_prefix: str = "end__",
) -> list[tuple[int, int]]:
    """Extract frame-index windows for a modality/style from metadata row."""

    landmark_style_keys = {
        "base": ("left_base", "right_base"),
        "zero": ("left_zero", "right_zero"),
        "tent": ("left_x_intercept", "right_x_intercept"),
        "half_base": ("left_base_half_height", "right_base_half_height"),
        "half_zero": ("left_zero_half_height", "right_zero_half_height"),
    }
    if style in landmark_style_keys:
        start_key, end_key = landmark_style_keys[style]
        start_col = f"{start_prefix}{start_key}__{modality}"
        end_col = f"{end_prefix}{end_key}__{modality}"
    else:
        start_col = f"{start_prefix}{style}__{modality}"
        end_col = f"{end_prefix}{style}__{modality}"

    starts = ensure_list(metadata_row.get(start_col))
    ends = ensure_list(metadata_row.get(end_col))

    windows: list[tuple[int, int]] = []
    for start_frame, end_frame in zip(starts, ends):
        if start_frame is None or end_frame is None:
            continue
        if pd.isna(start_frame) or pd.isna(end_frame):
            continue
        start_idx = max(0, int(round(float(start_frame))))
        end_idx = min(n_times, int(round(float(end_frame))))
        if end_idx > start_idx:
            windows.append((start_idx, end_idx))
    return windows
