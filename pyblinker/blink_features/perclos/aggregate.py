"""Aggregate PERCLOS features from refined MNE epochs."""

from __future__ import annotations

import mne
import pandas as pd

from .perclos import compute_perclos_features


def aggregate_perclos_features(
    epochs: mne.Epochs,
    *,
    perclos_cutoff: float = 0.80,
    requested_picks: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Return epoch-level PERCLOS and fatigue labels for refined epochs."""

    return compute_perclos_features(
        epochs,
        perclos_cutoff=perclos_cutoff,
        requested_picks=requested_picks,
    )


__all__ = ["aggregate_perclos_features"]
