"""Utilities for assigning baseline epoch health status to differently-sized epochs."""

from __future__ import annotations

import pandas as pd


def assign_epoch_health(
    health_df: pd.DataFrame,
    epoch_duration_s: float,
    n_epochs: int,
) -> list[int | None]:
    """Assign health status to each new epoch from a baseline health table.

    For each new epoch [i*dur, (i+1)*dur], the minimum health of all overlapping
    baseline epochs is returned.  Returns None when no baseline epoch overlaps.

    Args:
        health_df: DataFrame with columns ``epoch_start_s``, ``epoch_end_s``, ``health``.
        epoch_duration_s: Duration of the new epochs in seconds.
        n_epochs: Number of new epochs to produce.

    Returns:
        List of length ``n_epochs`` with integer health values (or None).
    """
    result: list[int | None] = []
    for i in range(n_epochs):
        epoch_start = i * epoch_duration_s
        epoch_end = epoch_start + epoch_duration_s
        overlapping = health_df[
            (health_df["epoch_start_s"] < epoch_end)
            & (health_df["epoch_end_s"] > epoch_start)
        ]
        if overlapping.empty:
            result.append(None)
        else:
            result.append(int(overlapping["health"].min()))
    return result


def get_valid_epoch_indices_by_health(
    health_values: list[int | None],
    min_health: int = 3,
) -> list[int]:
    """Return epoch indices whose health meets the minimum threshold.

    Epochs with ``None`` health (no baseline coverage) are excluded.

    Args:
        health_values: Health status per epoch, as returned by ``assign_epoch_health``.
        min_health: Minimum acceptable health level (inclusive).

    Returns:
        List of epoch indices with health >= min_health.
    """
    return [
        idx
        for idx, h in enumerate(health_values)
        if h is not None and h >= min_health
    ]


__all__ = ["assign_epoch_health", "get_valid_epoch_indices_by_health"]
