"""Blink detection directly on an :class:`mne.Epochs` object.

This module exposes a small pipeline to locate blink-like events in a
single channel or component, attach them to their respective epochs and
summarise the results in ``epochs.metadata``.  The overall flow of the
functions is illustrated below.

Flowchart
---------

::

       +--------------------+                 
       |  find_blinks_epoch |   High-level API
       +--------------------+
                  |
                  v
       +----------------------------+
       | _get_blink_position_epoching |  Blink detection
       +----------------------------+
                  |
                  v
       +---------------------+
       |  map_blinks_to_epochs |  Map events to epochs
       +---------------------+
                  |
                  v
       +--------------------+
       |   add_blink_counts |  Count blinks per epoch
       +--------------------+
                  |
                  v
       +----------------------+
       |    Epochs.metadata   |
       +----------------------+

``find_blinks_epoch`` orchestrates the steps above and returns the input
`Epochs` instance enriched with blink onset times, durations and counts.
"""

from typing import Optional, List, Tuple, Literal

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from .default_setting import SCALING_FACTOR
from ..fitutils import mad
from ..utils._logging import logger
# from ..ear.blink_epoch_mapper import _get_blink_position_epoching_ear


def _infer_signal_type(
    ch_name: Optional[str] = None, ch_type: Optional[str] = None
) -> Literal["EEG", "EAR"]:
    """Infer whether a channel represents an EAR or EEG signal.

    Parameters
    ----------
    ch_name : str | None
        Channel name.
    ch_type : str | None
        MNE channel type, if available.

    Returns
    -------
    {"EEG", "EAR"}
        Inferred signal type.
    """
    if ch_type and ch_type.lower() == "ear":
        return "EAR"
    if ch_name and "ear" in ch_name.lower():
        return "EAR"
    return "EEG"


def _get_blink_position_epoching(
    signal: np.ndarray,
    params: dict,
    ch: Optional[str] = None,
    *,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """Detect blinks in a 1D EEG/EOG signal using robust thresholding.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional array containing the blink-related signal.
    params : dict
        Parameter dictionary with keys ``sfreq``, ``min_event_len`` and
        ``std_threshold``.
    ch : str | None
        Channel name used only for log messages.
    progress_bar : bool
        Display a progress bar during processing.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``start_blink`` and ``end_blink`` giving
        sample indices for each detected blink.
    """
    logger.info("Detecting blinks in signal for channel %s", ch)
    assert signal.ndim == 1, "Signal must be 1D"

    mu = np.mean(signal)
    mad_val = mad(signal)
    robust_std = SCALING_FACTOR * mad_val
    threshold = mu + params["std_threshold"] * robust_std
    min_blink_frames = int(params["min_event_len"] * params["sfreq"])

    in_blink = False
    start_blinks: List[int] = []
    end_blinks: List[int] = []

    for idx in tqdm(
        range(signal.size),
        desc=f"Detect blinks - {ch}",
        disable=not progress_bar,
    ):
        val = signal[idx]
        if not in_blink and val > threshold:
            start = idx
            in_blink = True
        elif in_blink and val < threshold:
            if (idx - start) > min_blink_frames:
                start_blinks.append(start)
                end_blinks.append(idx)
            in_blink = False

    arr_start = np.array(start_blinks)
    arr_end = np.array(end_blinks)

    if arr_start.size == 0:
        return pd.DataFrame({"start_blink": [], "end_blink": []})

    pos_mask = np.ones(arr_end.size, dtype=bool)
    durations = (arr_start[1:] - arr_end[:-1]) / params["sfreq"]
    close = np.where(durations <= params["min_event_len"])[0]
    pos_mask[close] = False
    pos_mask[close + 1] = False

    return pd.DataFrame({
        "start_blink": arr_start[pos_mask],
        "end_blink": arr_end[pos_mask],
    })


def _epoch_boundaries_in_samples(epochs: mne.Epochs) -> Tuple[np.ndarray, np.ndarray]:
    """Return the absolute start and end sample of each epoch."""
    sfreq = epochs.info["sfreq"]
    starts = epochs.events[:, 0] + int(np.round(epochs.tmin * sfreq))
    n_samp = epochs.get_data().shape[-1]
    ends = starts + n_samp - 1
    return starts, ends


def _assign_single_blink(
    blink_start: int,
    blink_end: int,
    ep_starts: np.ndarray,
    ep_ends: np.ndarray,
    *,
    policy: str = "majority",
    majority_threshold: float = 0.5,
) -> List[int]:
    """Assign a blink to epochs based on the chosen policy."""
    overlaps = np.nonzero((blink_start <= ep_ends) & (blink_end >= ep_starts))[0]
    if overlaps.size == 0:
        return []

    if policy == "strict":
        return overlaps.tolist()
    if policy == "majority":
        blink_len = blink_end - blink_start + 1
        frac_inside = [
            (min(blink_end, ep_ends[idx]) - max(blink_start, ep_starts[idx]) + 1)
            / blink_len
            for idx in overlaps
        ]
        winner = overlaps[int(np.argmax(frac_inside))]
        if max(frac_inside) >= majority_threshold:
            return [winner]
        return []
    raise ValueError(f"Unknown policy: {policy}")


def map_blinks_to_epochs(
    epochs: mne.Epochs,
    blink_positions: pd.DataFrame,
    *,
    boundary_policy: str = "majority",
    majority_threshold: float = 0.5,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """Map blink onsets back to each epoch."""
    logger.info("Entering map_blinks_to_epochs")
    sfreq = epochs.info["sfreq"]
    ep_starts, ep_ends = _epoch_boundaries_in_samples(epochs)

    onsets: List[List[float]] = [[] for _ in range(len(epochs))]
    durations: List[List[float]] = [[] for _ in range(len(epochs))]

    for b_start, b_end in tqdm(
        zip(
            blink_positions["start_blink"].to_numpy(dtype=int),
            blink_positions["end_blink"].to_numpy(dtype=int),
        ),
        total=len(blink_positions),
        desc="Assigning blinks to epochs",
        disable=not progress_bar,
    ):
        recipients = _assign_single_blink(
            b_start,
            b_end,
            ep_starts,
            ep_ends,
            policy=boundary_policy,
            majority_threshold=majority_threshold,
        )
        for idx in recipients:
            rel_onset = (max(b_start, ep_starts[idx]) - ep_starts[idx]) / sfreq
            onsets[idx].append(rel_onset)
            durations[idx].append((b_end - b_start) / sfreq)

    meta = pd.DataFrame({
        "blink_onsets": onsets,
        "blink_durations": durations,
    }, index=epochs.selection)

    logger.info("Leaving map_blinks_to_epochs")
    return meta


def add_blink_counts(epochs: mne.Epochs) -> mne.Epochs:
    """Compute blink counts from metadata and store them in-place.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object with metadata from :func:`map_blinks_to_epochs`.

    Returns
    -------
    mne.Epochs
        The same object with a new ``n_blinks`` column in ``metadata``.
    """
    logger.info("Adding blink counts to metadata")
    if epochs.metadata is None:
        raise ValueError("Epochs.metadata is missing blink information")

    meta = epochs.metadata.copy()
    meta["n_blinks"] = meta["blink_onsets"].apply(len)
    epochs.metadata = meta
    logger.info("Blink counts added")
    return epochs


def find_blinks_epoch(
    epochs: mne.Epochs,
    *,
    ch_name: Optional[str] = None,
    params: dict,
    boundary_policy: str = "majority",
    majority_threshold: float = 0.5,
) -> mne.Epochs:
    """Detect blinks and assign metadata to each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object containing the blink-related signal.
    ch_name : str | None
        Channel name to process. Defaults to the first channel in ``epochs``.
    params : dict
        Must include ``"sfreq"`` along with algorithm-specific parameters.
        The function supports EEG/EOG and EAR signals via their respective
        detection pipelines.
    boundary_policy : str
        Policy for assigning blinks crossing epoch boundaries: ``"strict"`` or
        ``"majority"``.
    majority_threshold : float
        Fraction used with the ``"majority"`` policy.

    Returns
    -------
    mne.Epochs
        The input epochs updated with blink metadata attached to
        ``epochs.metadata``.
    """
    logger.info("Entering find_blinks_epoch")

    if ch_name is None:
        ch_idx = 0
        ch_name = epochs.ch_names[0]
    else:
        ch_idx = epochs.ch_names.index(ch_name)

    logger.debug("Using channel %s at index %d", ch_name, ch_idx)
    ch_type = epochs.get_channel_types(picks=[ch_idx])[0]
    signal = epochs.get_data()[:, ch_idx, :]
    flat_signal = signal.flatten()

    params = dict(params or {})
    params.setdefault("sfreq", float(epochs.info["sfreq"]))

    signal_type = _infer_signal_type(ch_name, ch_type)
    if signal_type == "EAR":
        pass
        # We disable EAR support, but this is where it would go
        # blink_positions = _get_blink_position_epoching_ear(
        #     flat_signal, params, ch=ch_name, progress_bar=False
        # )
    else:
        blink_positions = _get_blink_position_epoching(
            flat_signal,
            params,
            ch=ch_name,
            progress_bar=False,
        )
    meta = map_blinks_to_epochs(
        epochs,
        blink_positions,
        boundary_policy=boundary_policy,
        majority_threshold=majority_threshold,
        progress_bar=False,
    )
    epochs.metadata = meta
    add_blink_counts(epochs)

    logger.info("Assigned blink metadata to epochs")
    logger.info("Leaving find_blinks_epoch")
    return epochs

