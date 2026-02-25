"""Shared style discovery and window extraction helpers."""

from __future__ import annotations

from typing import List, Mapping, Sequence, Set

import pandas as pd

from pyblinker.utils.iter_utils import ensure_list
from pyblinker.utils.metadata_utils import segment_to_samples


def available_styles(
    metadata_columns: Sequence[str] | None,
    modality: str,
    *,
    onset_prefix: str | None = "onset__",
    duration_prefix: str | None = "duration__",
) -> Set[str]:
    """Return segmentation styles present in metadata for a modality."""

    if metadata_columns is None:
        return set()

    styles: Set[str] = set()
    metadata_set = set(metadata_columns)
    suffix = f"__{modality}"

    if onset_prefix and duration_prefix:
        for col in metadata_columns:
            if not col.startswith(onset_prefix) or not col.endswith(suffix):
                continue
            style = col[len(onset_prefix) : -len(suffix)]
            if not style or "sample" in style.lower():
                continue
            if f"{duration_prefix}{style}__{modality}" in metadata_set:
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
    for style, (start_key, end_key) in landmark_styles.items():
        if start_key in metadata_set and end_key in metadata_set:
            styles.add(style)

    start_prefix = "start__"
    for col in metadata_columns:
        if not col.startswith(start_prefix) or not col.endswith(suffix):
            continue
        style = col[len(start_prefix) : -len(suffix)]
        if style and f"end__{style}__{modality}" in metadata_set:
            styles.add(style)

    return styles


def _frame_windows(
    metadata_row: Mapping[str, object],
    start_key: str,
    end_key: str,
    n_times: int,
) -> List[tuple[int, int]]:
    windows: List[tuple[int, int]] = []
    starts = ensure_list(metadata_row.get(start_key))
    ends = ensure_list(metadata_row.get(end_key))
    for start_frame, end_frame in zip(starts, ends):
        if start_frame is None or end_frame is None:
            continue
        if pd.isna(start_frame) or pd.isna(end_frame):
            continue
        start_idx = max(0, int(round(float(start_frame))))
        end_idx = min(n_times, int(round(float(end_frame))))
        if end_idx <= start_idx:
            continue
        windows.append((start_idx, end_idx))
    return windows


def extract_windows(
    metadata_row: Mapping[str, object],
    modality: str,
    style: str,
    n_times: int,
    *,
    start_prefix: str = "start__",
    end_prefix: str = "end__",
    sfreq: float | None = None,
) -> List[tuple[int, int]]:
    """Extract blink windows for a modality/style as sample-frame bounds."""

    landmark_style_keys = {
        "base": ("start__left_base", "end__right_base"),
        "zero": ("start__left_zero", "end__right_zero"),
        "tent": ("start__left_x_intercept", "end__right_x_intercept"),
        "half_base": ("start__left_base_half_height", "end__right_base_half_height"),
        "half_zero": ("start__left_zero_half_height", "end__right_zero_half_height"),
    }
    if style in landmark_style_keys:
        start_key = f"{landmark_style_keys[style][0]}__{modality}"
        end_key = f"{landmark_style_keys[style][1]}__{modality}"
    else:
        start_key = f"{start_prefix}{style}__{modality}"
        end_key = f"{end_prefix}{style}__{modality}"

    windows = _frame_windows(metadata_row, start_key, end_key, n_times)
    if windows:
        return windows

    if sfreq is None:
        return []

    onset_key = f"onset__{style}__{modality}"
    duration_key = f"duration__{style}__{modality}"
    windows = []
    onsets = ensure_list(metadata_row.get(onset_key)) if metadata_row.get(onset_key) is not None else []
    durations = ensure_list(metadata_row.get(duration_key)) if metadata_row.get(duration_key) is not None else []
    for onset, duration in zip(onsets, durations):
        if onset is None or duration is None:
            continue
        if pd.isna(onset) or pd.isna(duration):
            continue
        sl = segment_to_samples(float(onset), float(duration), sfreq, n_times)
        if sl.stop <= sl.start:
            continue
        windows.append((int(sl.start), int(sl.stop)))
    return windows
