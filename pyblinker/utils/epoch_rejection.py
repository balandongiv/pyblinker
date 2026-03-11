"""Automatic epoch-quality rejection for blink preprocessing.

This module implements an autoreject-inspired threshold selection workflow using
peak-to-peak amplitudes on fixed-length epochs. It is intentionally scoped to
reject whole epochs (trials) and does not perform interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mne


@dataclass(frozen=True)
class EpochRejectionResult:
    """Container for epoch-level quality screening outputs."""

    threshold: float
    scores: np.ndarray
    good_epoch_indices: np.ndarray
    bad_epoch_indices: np.ndarray
    epoch_bounds_samples: list[tuple[int, int]]
    cv_errors: np.ndarray
    good_epochs: mne.Epochs | None = None
    bad_epochs: mne.Epochs | None = None


def _split_fixed_length_epochs(
    signal: np.ndarray,
    sfreq: float,
    epoch_duration_s: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Split a 1D signal into consecutive fixed-length epochs."""

    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")
    if epoch_duration_s <= 0:
        raise ValueError("epoch_duration_s must be positive")

    n_times_per_epoch = int(round(epoch_duration_s * sfreq))
    if n_times_per_epoch <= 0:
        raise ValueError("epoch_duration_s * sfreq must be at least one sample")

    n_epochs = signal.size // n_times_per_epoch
    if n_epochs < 2:
        raise ValueError("Need at least 2 full epochs for CV-based rejection")

    used = n_epochs * n_times_per_epoch
    trimmed = signal[:used]
    epochs = trimmed.reshape(n_epochs, n_times_per_epoch)

    bounds = [
        (idx * n_times_per_epoch, (idx + 1) * n_times_per_epoch)
        for idx in range(n_epochs)
    ]
    return epochs, bounds


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2, dtype=np.float64)))


def _compute_candidate_thresholds(scores: np.ndarray, n_candidates: int) -> np.ndarray:
    """Build monotonic threshold candidates from score quantiles."""

    if n_candidates < 3:
        raise ValueError("n_candidates must be at least 3")

    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if np.isclose(lo, hi):
        return np.array([hi], dtype=np.float64)

    quantile_grid = np.linspace(0.1, 0.99, n_candidates)
    candidates = np.quantile(scores, quantile_grid)
    return np.unique(candidates.astype(np.float64))


def _kfold_indices(n_epochs: int, n_splits: int, random_state: int) -> list[np.ndarray]:
    """Create deterministic shuffled folds without sklearn dependency."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > n_epochs:
        raise ValueError("n_splits must not exceed number of epochs")

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n_epochs)
    return [arr.astype(int) for arr in np.array_split(perm, n_splits)]


def _run_cv_threshold_selection(
    epochs: np.ndarray,
    *,
    n_splits: int,
    n_candidates: int,
    random_state: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return best threshold, epoch scores, and per-candidate CV errors."""

    scores = np.ptp(epochs, axis=1).astype(np.float64)
    candidates = _compute_candidate_thresholds(scores, n_candidates)

    folds = _kfold_indices(scores.size, n_splits=n_splits, random_state=random_state)
    cv_errors = np.full(candidates.size, np.inf, dtype=np.float64)

    all_idx = np.arange(scores.size)
    for c_idx, threshold in enumerate(candidates):
        fold_errors: list[float] = []
        for val_idx in folds:
            train_mask = np.ones(scores.size, dtype=bool)
            train_mask[val_idx] = False
            train_idx = all_idx[train_mask]

            retained_train_idx = train_idx[scores[train_idx] <= threshold]
            if retained_train_idx.size == 0:
                fold_errors.append(np.inf)
                continue

            train_mean = np.mean(epochs[retained_train_idx], axis=0)
            val_median = np.median(epochs[val_idx], axis=0)
            fold_errors.append(_rmse(train_mean, val_median))

        cv_errors[c_idx] = float(np.mean(fold_errors, dtype=np.float64))

    best_idx = int(np.argmin(cv_errors))
    best_threshold = float(candidates[best_idx])
    return best_threshold, scores, cv_errors


def detect_bad_epochs_peak_to_peak(
    signal: np.ndarray,
    sfreq: float,
    *,
    epoch_duration_s: float = 30.0,
    n_splits: int = 5,
    n_candidates: int = 31,
    random_state: int = 7,
) -> EpochRejectionResult:
    """Detect bad fixed-length epochs using CV-selected peak-to-peak threshold."""

    epochs, bounds = _split_fixed_length_epochs(signal, sfreq, epoch_duration_s)
    best_threshold, scores, cv_errors = _run_cv_threshold_selection(
        epochs,
        n_splits=n_splits,
        n_candidates=n_candidates,
        random_state=random_state,
    )

    bad_idx = np.flatnonzero(scores > best_threshold).astype(int)
    good_idx = np.flatnonzero(scores <= best_threshold).astype(int)

    return EpochRejectionResult(
        threshold=best_threshold,
        scores=scores,
        good_epoch_indices=good_idx,
        bad_epoch_indices=bad_idx,
        epoch_bounds_samples=bounds,
        cv_errors=cv_errors,
    )


def detect_bad_epochs_peak_to_peak_mne(
    epochs: mne.Epochs,
    *,
    picks: str | list[str] = "EEG-E8",
    n_splits: int = 5,
    n_candidates: int = 31,
    random_state: int = 7,
) -> EpochRejectionResult:
    """Detect bad epochs directly from an ``mne.Epochs`` object.

    Returns bad/good epoch indices and the corresponding ``mne.Epochs`` subsets.
    """

    picked = epochs.copy().pick(picks)
    data = picked.get_data(copy=True)
    if data.shape[1] != 1:
        raise ValueError("detect_bad_epochs_peak_to_peak_mne requires one picked channel")

    one_channel_epochs = data[:, 0, :]
    best_threshold, scores, cv_errors = _run_cv_threshold_selection(
        one_channel_epochs,
        n_splits=n_splits,
        n_candidates=n_candidates,
        random_state=random_state,
    )

    bad_idx = np.flatnonzero(scores > best_threshold).astype(int)
    good_idx = np.flatnonzero(scores <= best_threshold).astype(int)

    n_times = one_channel_epochs.shape[1]
    bounds = [(idx * n_times, (idx + 1) * n_times) for idx in range(one_channel_epochs.shape[0])]

    cleaned_epochs = epochs.copy()
    if bad_idx.size:
        cleaned_epochs.drop(bad_idx.tolist(), reason="BAD_peak_to_peak")

    return EpochRejectionResult(
        threshold=best_threshold,
        scores=scores,
        good_epoch_indices=good_idx,
        bad_epoch_indices=bad_idx,
        epoch_bounds_samples=bounds,
        cv_errors=cv_errors,
        good_epochs=cleaned_epochs,
        bad_epochs=epochs[bad_idx],
    )


def select_signal_samples_from_good_epochs(
    signal: np.ndarray,
    rejection_result: EpochRejectionResult,
) -> np.ndarray:
    """Return a concatenated 1D signal containing only good epochs."""

    if signal.ndim != 1:
        raise ValueError("signal must be 1D")

    good_idx_set = set(rejection_result.good_epoch_indices.tolist())
    kept_segments = [
        signal[start:stop]
        for idx, (start, stop) in enumerate(rejection_result.epoch_bounds_samples)
        if idx in good_idx_set
    ]
    if not kept_segments:
        return np.array([], dtype=signal.dtype)
    return np.concatenate(kept_segments)


__all__ = [
    "EpochRejectionResult",
    "detect_bad_epochs_peak_to_peak",
    "detect_bad_epochs_peak_to_peak_mne",
    "select_signal_samples_from_good_epochs",
]
