"""Helpers for centrally controlling which epochs are valid."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

BAD_FLAG_COLUMNS = ("is_bad_epoch", "bad_epoch", "epoch_is_bad")
GOOD_FLAG_COLUMNS = ("is_good_epoch", "good_epoch", "epoch_is_good")


def _coerce_boolean_series(values: pd.Series) -> pd.Series:
    def _normalize_one(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "t", "yes", "y", "bad", "drop", "dropped"}:
                return True
            if lowered in {"0", "false", "f", "no", "n", "good", "keep"}:
                return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        return False

    return values.apply(_normalize_one)


def _metadata_with_epoch_index(epochs) -> pd.DataFrame | None:
    metadata = epochs.metadata
    if not isinstance(metadata, pd.DataFrame):
        return None
    frame = metadata.copy().reset_index(drop=True)
    return frame.reindex(range(len(epochs))).reset_index(drop=True)


def get_valid_epoch_indices(epochs) -> list[int]:
    """Return the loaded epoch indices allowed to contribute to detection."""

    n_epochs = len(epochs)
    if n_epochs == 0:
        return []

    metadata = _metadata_with_epoch_index(epochs)
    if metadata is None:
        return list(range(n_epochs))

    bad_mask = np.zeros(n_epochs, dtype=bool)
    found_explicit_flags = False

    for column in BAD_FLAG_COLUMNS:
        if column in metadata.columns:
            bad_mask |= _coerce_boolean_series(metadata[column]).to_numpy(dtype=bool)
            found_explicit_flags = True

    for column in GOOD_FLAG_COLUMNS:
        if column in metadata.columns:
            good_mask = _coerce_boolean_series(metadata[column]).to_numpy(dtype=bool)
            bad_mask |= ~good_mask
            found_explicit_flags = True

    if not found_explicit_flags:
        return list(range(n_epochs))

    return [idx for idx in range(n_epochs) if not bad_mask[idx]]


def simulate_bad_epochs(
    epochs,
    drop_ratio: float,
    random_state: int,
) -> tuple[object, list[int]]:
    """Return a copy with reproducible bad-epoch flags stored in metadata."""

    if not 0.0 <= float(drop_ratio) <= 1.0:
        raise ValueError("drop_ratio must be between 0 and 1.")

    simulated = epochs.copy()
    n_epochs = len(simulated)
    if n_epochs == 0:
        return simulated, []

    metadata = _metadata_with_epoch_index(simulated)
    if metadata is None:
        metadata = pd.DataFrame(index=range(n_epochs))

    rng = np.random.default_rng(random_state)
    n_bad = int(np.floor(n_epochs * float(drop_ratio)))
    if drop_ratio > 0.0 and n_bad == 0:
        n_bad = 1
    if np.isclose(drop_ratio, 1.0):
        n_bad = n_epochs

    bad_indices = (
        sorted(rng.choice(n_epochs, size=n_bad, replace=False).tolist())
        if n_bad > 0
        else []
    )

    metadata["is_bad_epoch"] = False
    if bad_indices:
        metadata.loc[bad_indices, "is_bad_epoch"] = True
    simulated.metadata = metadata
    return simulated, bad_indices


__all__ = ["get_valid_epoch_indices", "simulate_bad_epochs"]
