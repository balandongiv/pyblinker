import numpy as np
import pandas as pd
from tqdm import tqdm

from .default_setting import SCALING_FACTOR
from ..fitutils import mad


def _compute_detection_threshold(
    blink_component: np.ndarray, params: dict
) -> tuple[float, float]:
    mu = np.mean(blink_component, dtype=np.float64)
    mad_val = mad(blink_component)
    robust_std = SCALING_FACTOR * mad_val
    min_blink_frames = params["min_event_len"] * params["sfreq"]
    threshold = mu + params["std_threshold"] * robust_std
    return threshold, min_blink_frames


def _find_blink_candidates(
    blink_component: np.ndarray, threshold: float, min_blink_frames: float
) -> tuple[np.ndarray, np.ndarray]:
    above_or_equal = blink_component >= threshold
    if not np.any(above_or_equal):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    segment_starts = np.flatnonzero(
        np.logical_and(~above_or_equal[:-1], above_or_equal[1:])
    ) + 1
    segment_ends = np.flatnonzero(
        np.logical_and(above_or_equal[:-1], ~above_or_equal[1:])
    ) + 1

    if above_or_equal[0]:
        segment_starts = np.insert(segment_starts, 0, 0)

    if segment_starts.size and segment_ends.size and segment_ends[0] < segment_starts[0]:
        segment_ends = segment_ends[1:]

    pair_count = min(segment_starts.size, segment_ends.size)
    if pair_count == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    segment_starts = segment_starts[:pair_count]
    segment_ends = segment_ends[:pair_count]

    starts: list[int] = []
    ends: list[int] = []
    for seg_start, seg_end in zip(segment_starts, segment_ends, strict=False):
        segment = blink_component[seg_start:seg_end]
        above_strict = np.flatnonzero(segment > threshold)
        if above_strict.size == 0:
            continue
        start_idx = int(seg_start + above_strict[0])
        duration = seg_end - start_idx
        if duration > min_blink_frames:
            starts.append(start_idx)
            ends.append(int(seg_end))

    if not starts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def _remove_close_blinks(
    starts: np.ndarray, ends: np.ndarray, *, sfreq: float, min_event_sep: float
) -> tuple[np.ndarray, np.ndarray]:
    if ends.size == 0:
        return starts, ends

    pos_mask = np.ones(ends.size, dtype=bool)
    blink_durations = (starts[1:] - ends[:-1]) / sfreq
    close_indices = np.argwhere(blink_durations <= min_event_sep).ravel()

    pos_mask[close_indices] = False
    pos_mask[close_indices + 1] = False
    return starts[pos_mask], ends[pos_mask]


def get_blink_position(
    params, blink_component=None, ch=None, *, progress_bar: bool = True
):
    """Detect blink start and end frames using the legacy MATLAB Blinker approach.
    
    Parameters
    ----------
    params : dict
        A dictionary containing processing parameters, which must include:
        - 'sfreq' (float): Sampling frequency of the candidate_signal in Hz.
        - 'min_event_len' (float): Minimum blink length in seconds.
        - 'std_threshold' (float): Standard deviation threshold for blink detection.
    blink_component : numpy.ndarray
        A 1D array representing the blink component (e.g., an independent component related to eye blinks).
    ch : str, optional
        The name of the channel for logging purposes. Default is None.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing two columns:
        - 'start_blink' (numpy.ndarray): Indices of the start frames of detected blinks.
        - 'end_blink' (numpy.ndarray): Indices of the end frames of detected blinks.
        If no blinks are detected, an empty DataFrame with the same column names is returned.
    """

    # Ensure 1D array
    assert blink_component.ndim == 1, "blink_component must be a 1D array"

    threshold, min_blink_frames = _compute_detection_threshold(
        blink_component, params
    )

    if progress_bar:
        with tqdm(
            total=blink_component.size,
            desc=f"Get blink start and end for channel {ch}",
            disable=not progress_bar,
        ) as bar:
            bar.update(blink_component.size)

    arr_start, arr_end = _find_blink_candidates(
        blink_component, threshold, min_blink_frames
    )

    if arr_end.size == 0:
        return pd.DataFrame({'start_blink': [], 'end_blink': []})

    min_event_sep = params.get("min_event_sep", params["min_event_len"])
    arr_start, arr_end = _remove_close_blinks(
        arr_start, arr_end, sfreq=params["sfreq"], min_event_sep=min_event_sep
    )

    blink_position = {
        "start_blink": arr_start,
        "end_blink": arr_end,
    }
    return pd.DataFrame(blink_position)
