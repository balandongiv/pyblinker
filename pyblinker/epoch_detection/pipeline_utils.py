"""Epoch boundary and per-channel signal helpers."""

from __future__ import annotations

import numpy as np

from .epoch_input import PreparedEpochDetectionInput


def build_epoch_boundaries(
    valid_epoch_count: int,
    epoch_length_samples: int,
) -> list[tuple[int, int]]:
    """Return concatenated-signal sample boundaries for each valid epoch."""
    return [
        (
            idx * epoch_length_samples,
            (idx + 1) * epoch_length_samples,
        )
        for idx in range(valid_epoch_count)
    ]


def build_signal_by_epoch(
    prepared: PreparedEpochDetectionInput,
    ch_idx: int,
) -> dict[int, np.ndarray]:
    """Build the signal_by_epoch dict for one channel (all epoch indices)."""
    return {
        epoch_idx: prepared.data[epoch_idx, ch_idx, :].astype(float)
        for epoch_idx in range(prepared.data.shape[0])
    }


__all__ = ["build_epoch_boundaries", "build_signal_by_epoch"]
