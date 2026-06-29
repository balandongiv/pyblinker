"""Stage A: Epoch-level screening using autoreject thresholds.

Uses ``autoreject.compute_thresholds`` to learn per-channel peak-to-peak (PTP)
thresholds, then flags epochs where at least one channel exceeds its threshold.
This avoids the channel-position requirement of ``AutoReject.fit()`` while still
leveraging the autoreject threshold-learning machinery.

``autoreject`` is an optional dependency.  Install it with::

    pip install autoreject

(or ``pip install pyblinker[double-thresholding]``).
"""

from __future__ import annotations

from types import SimpleNamespace

import mne
import numpy as np

from ..epoch_detection.epoch_input import PreparedEpochDetectionInput


def _build_stage1_epochs(
    stage1_data: np.ndarray,
    *,
    channel_names: tuple[str, ...],
    sfreq: float,
) -> mne.Epochs:
    """Create an MNE EpochsArray from stage-1 array data.

    Parameters
    ----------
    stage1_data:
        3-D array of shape ``(n_epochs, n_channels, n_times)``.
    channel_names:
        Channel labels matching the second axis of ``stage1_data``.
    sfreq:
        Sampling frequency in Hz.
    """
    info = mne.create_info(
        list(channel_names),
        sfreq=float(sfreq),
        ch_types=["eeg"] * len(channel_names),
    )
    return mne.EpochsArray(stage1_data, info, verbose="ERROR")


def screen_epochs_with_autoreject(
    prepared: PreparedEpochDetectionInput,
    valid_epoch_indices: list[int],
    *,
    random_state: int = 42,
    autoreject_method: str = "bayesian_optimization",
    min_flagged_epochs: int = 1,
    verbose: bool = False,
) -> SimpleNamespace:
    """Identify suspicious epochs using autoreject PTP thresholds (Stage A).

    Per-channel PTP thresholds are learned via ``compute_thresholds``.  An epoch
    is flagged as suspicious when the peak-to-peak amplitude of *any* channel
    exceeds that channel's threshold.

    Parameters
    ----------
    prepared:
        Pre-processed epoch data.
    valid_epoch_indices:
        Indices of epochs to screen.
    random_state:
        Random seed forwarded to ``compute_thresholds``.
    autoreject_method:
        Threshold estimation method (``"bayesian_optimization"`` or
        ``"random_search"``).
    min_flagged_epochs:
        Minimum number of flagged epochs required.  If fewer are found the
        returned ``flagged_valid_epoch_indices`` list will be empty and the
        caller should fall back to all valid epochs.

    Returns
    -------
    SimpleNamespace with fields:
        - ``flagged_epoch_mask``: boolean mask of shape (n_valid_epochs,)
        - ``flagged_valid_epoch_indices``: original epoch indices that were flagged
        - ``channel_thresholds``: dict mapping channel_name -> PTP threshold
        - ``n_flagged``: number of flagged epochs
    """
    try:
        from autoreject import compute_thresholds
    except ImportError as exc:  # pragma: no cover - exercised only without autoreject
        raise ImportError(
            "screen_epochs_with_autoreject requires the optional 'autoreject' "
            "package. Install it with `pip install autoreject` "
            "(or `pip install pyblinker[double-thresholding]`)."
        ) from exc

    channel_names = tuple(prepared.channel_names)
    valid_indices = np.asarray(valid_epoch_indices, dtype=int)

    if verbose:
        print(
            f"[Stage A] screening {len(valid_indices)} valid epoch(s) with "
            f"autoreject ({autoreject_method})"
        )

    data = prepared.data[valid_indices, :, :]

    stage1_epochs = _build_stage1_epochs(
        data,
        channel_names=channel_names,
        sfreq=prepared.sfreq,
    )

    # Learn per-channel PTP thresholds (augment=False avoids the location check).
    # We use the autoreject implementation.
    raw_thresholds = compute_thresholds(
        stage1_epochs,
        method=autoreject_method,
        random_state=int(random_state),
        augment=False,
        verbose=False,
        n_jobs=1,
    )
    channel_thresholds = {ch: float(raw_thresholds[ch]) for ch in channel_names}

    # Flag epochs where any channel's PTP exceeds its threshold
    threshold_array = np.array([channel_thresholds[ch] for ch in channel_names])  # (n_channels,)
    # data shape: (n_valid_epochs, n_channels, n_times)
    ptp = data.max(axis=-1) - data.min(axis=-1)  # (n_valid_epochs, n_channels)

    # We consider the contribution of all channels
    flagged_epoch_mask = np.any(ptp > threshold_array[np.newaxis, :], axis=1)  # (n_valid_epochs,)

    flagged_local_indices = np.where(flagged_epoch_mask)[0]
    flagged_valid_epoch_indices = [int(valid_indices[i]) for i in flagged_local_indices]

    if verbose:
        print(
            f"[Stage A] {len(flagged_valid_epoch_indices)} / {len(valid_indices)} epoch(s) "
            f"exceeded PTP threshold (epoch indices: {flagged_valid_epoch_indices})"
        )

    if len(flagged_valid_epoch_indices) < min_flagged_epochs:
        if verbose:
            print(
                f"[Stage A] flagged count {len(flagged_valid_epoch_indices)} "
                f"< min_flagged_epochs {min_flagged_epochs} — clearing flagged list"
            )
        flagged_valid_epoch_indices = []
        flagged_epoch_mask = np.zeros(len(valid_indices), dtype=bool)

    return SimpleNamespace(
        flagged_epoch_mask=flagged_epoch_mask,
        flagged_valid_epoch_indices=flagged_valid_epoch_indices,
        channel_thresholds=channel_thresholds,
        n_flagged=len(flagged_valid_epoch_indices),
    )


__all__ = ["screen_epochs_with_autoreject"]
