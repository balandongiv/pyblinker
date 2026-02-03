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


def _filter_close_pairs_from_signal(
    arr: np.ndarray, *, blink_component: np.ndarray, params: dict
) -> np.ndarray:
    """Filter blink start/end pairs that violate the minimum separation rule.
    This helper reproduces the MATLAB ``getBlinkPositions`` post-processing step
    that removes blink candidates whose inter-blink separation is less than or
    equal to ``min_event_sep``. The filtering is *signal-aware*: it recomputes
    candidate starts/ends from the provided signal (using the same thresholding
    logic as the main detection pipeline) and then drops any pairs from ``arr``
    that belong to a close-pair cluster.

    Origin and significance:
    - Origin: Ported from the legacy BLINKER MATLAB code path that performs the
      close-pair removal after initial candidate detection.
    - Significance: Ensures blink counts and timings align with MATLAB outputs
      during migration comparisons, and keeps the pipeline deterministic by
      applying the same removal logic everywhere the blink positions are used.

    Parameters
    ----------
    arr : numpy.ndarray
        A (2, N) array of blink start/end indices to filter.
    blink_component : numpy.ndarray
        The 1D candidate signal used for blink detection.
    params : dict
        Blink detection parameters containing ``sfreq``, ``min_event_len``, and
        optionally ``min_event_sep``.

    Returns
    -------
    numpy.ndarray
        The filtered (2, M) array with close-pair blink candidates removed.
    """
    if arr.size == 0:
        return arr

    threshold, min_blink_frames = _compute_detection_threshold(
        blink_component, params
    )
    starts, ends = _find_blink_candidates(
        blink_component, threshold, min_blink_frames
    )
    if ends.size == 0:
        return arr[:, :0]

    min_event_sep = params.get("min_event_sep", params["min_event_len"])
    blink_durations = (starts[1:] - ends[:-1]) / params["sfreq"]
    close_indices = np.argwhere(blink_durations <= min_event_sep).ravel()
    close_pairs = {(starts[idx], ends[idx]) for idx in close_indices}
    close_pairs.update((starts[idx + 1], ends[idx + 1]) for idx in close_indices)
    pairs = np.column_stack((arr[0], arr[1]))
    mask = np.array([tuple(row) not in close_pairs for row in pairs], dtype=bool)
    return arr[:, mask]


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
        return pd.DataFrame({"start_blink": [], "end_blink": []})

    arr = np.vstack((arr_start, arr_end))
    arr = _filter_close_pairs_from_signal(
        arr, blink_component=blink_component, params=params
    )

    blink_position = {
        "start_blink": arr[0],
        "end_blink": arr[1],
    }
    return pd.DataFrame(blink_position)
