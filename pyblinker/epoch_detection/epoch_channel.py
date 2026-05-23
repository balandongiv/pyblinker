"""Concatenated-signal epoch mapping for blink candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def map_concatenated_blinks_to_epochs(
    blink_df: pd.DataFrame,
    *,
    channel: str,
    valid_epoch_indices: list[int],
    epoch_boundaries: list[tuple[int, int]],
    sfreq: float,
) -> pd.DataFrame:
    """Project concatenated-signal blink rows back into epoch-local timing."""

    if blink_df.empty or not valid_epoch_indices:
        return pd.DataFrame(
            columns=[
                "epoch_index",
                "channel",
                "blink_onset",
                "blink_duration",
                "start_blink",
                "end_blink",
            ]
        )

    boundary_starts = np.asarray([start for start, _ in epoch_boundaries], dtype=int)
    boundary_stops = np.asarray([stop for _, stop in epoch_boundaries], dtype=int)
    blink_rows = blink_df.copy().reset_index(drop=True)
    blink_rows["channel"] = channel

    start_samples = blink_rows["start_blink"].to_numpy(dtype=int)
    epoch_offsets = np.searchsorted(boundary_stops, start_samples, side="right")
    valid_mask = (
        (epoch_offsets >= 0)
        & (epoch_offsets < len(boundary_starts))
        & (start_samples >= boundary_starts[epoch_offsets])
        & (start_samples < boundary_stops[epoch_offsets])
    )
    if not np.any(valid_mask):
        return pd.DataFrame(
            columns=[
                "epoch_index",
                "channel",
                "blink_onset",
                "blink_duration",
                "start_blink",
                "end_blink",
            ]
        )

    mapped = blink_rows.loc[valid_mask].copy().reset_index(drop=True)
    mapped_epoch_offsets = epoch_offsets[valid_mask]
    mapped["epoch_index"] = [valid_epoch_indices[idx] for idx in mapped_epoch_offsets]
    mapped["blink_onset"] = (
        mapped["start_blink"].to_numpy(dtype=float) - boundary_starts[mapped_epoch_offsets]
    ) / float(sfreq)
    mapped["blink_duration"] = (
        mapped["end_blink"].to_numpy(dtype=float)
        - mapped["start_blink"].to_numpy(dtype=float)
    ) / float(sfreq)
    return mapped.sort_values(["epoch_index", "blink_onset"]).reset_index(drop=True)


__all__ = ["map_concatenated_blinks_to_epochs"]
