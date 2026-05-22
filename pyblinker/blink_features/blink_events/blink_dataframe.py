from typing import Dict, List, Sequence, Tuple
from pyblinker._logging import get_logger

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from ...blinker.zero_crossing import left_right_zero_crossing
from ...segmentation.refinement.eeg import compute_outer_bounds

logger = get_logger(__name__)


def _normalize_zero_crossing(value: int | float | None) -> int | float | None:
    """Convert raw zero-crossing outputs into canonical scalar values."""

    if value is None:
        return None

    try:
        if np.isnan(value):
            return np.nan
    except TypeError:
        pass

    return int(value)


def _filter_blink_annotations(
    raw: mne.io.BaseRaw, blink_label: str | None
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract blink start and end sample indices from ``raw``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw segment containing blink annotations.
    blink_label : str | None
        Annotation label that denotes blinks. ``None`` keeps all annotations.

    Returns
    -------
    tuple of ndarray
        Arrays of start and end sample indices for each blink.
    """

    sfreq = raw.info["sfreq"]
    ann = raw.annotations
    mask = np.ones(len(ann), dtype=bool)
    if blink_label is not None:
        mask &= ann.description == blink_label
    mask &= ann.onset > raw.first_time

    onsets = ann.onset[mask]
    durations = ann.duration[mask]
    starts = ((onsets - raw.first_time) * sfreq).astype(int)
    ends = ((onsets + durations - raw.first_time) * sfreq).astype(int)
    return starts, ends


def _get_channel_type(raw: mne.io.BaseRaw, channel: str, provided: str | None) -> str:
    """Determine ``channel`` type, optionally using the caller-provided value.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw segment from which to infer the channel type.
    channel : str
        Name of the channel to query.
    provided : str | None
        Channel type supplied by the caller. When not ``None`` this value is
        returned directly.

    Returns
    -------
    str
        Detected channel type. If detection fails a warning is logged and
        ``"eeg"`` is returned.
    """

    if provided is not None:
        return provided

    try:
        ch_type = raw.get_channel_types(picks=channel)[0]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Unable to determine channel type for %s: %s. Assuming 'eeg'",
            channel,
            exc,
        )
        ch_type = "eeg"
    return ch_type


def _detect_peaks(
    signal: np.ndarray, starts: np.ndarray, ends: np.ndarray, ch_type: str
) -> List[int]:
    """Detect the peak sample within each blink interval.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional blink signal.
    starts, ends : np.ndarray
        Start and end sample indices for each blink annotation.
    ch_type : str
        Type of the underlying channel (e.g., ``"eeg"``).

    Returns
    -------
    list of int
        Peak index for every blink interval.
    """

    peaks: List[int] = []
    for start, end in zip(starts, ends):
        segment = signal[start : end + 1]
        if ch_type == "eeg":
            peak = int(np.argmax(segment) + start)
        else:
            logger.warning(
                "Peak detection tuned for EEG; using absolute max for %s", ch_type
            )
            peak = int(np.argmax(np.abs(segment)) + start)
        peaks.append(peak)
    return peaks


def _process_segment_blinks(
    seg_id: int,
    raw: mne.io.BaseRaw,
    channel: str,
    blink_label: str | None,
    channel_type: str | None,
    *,
    progress_bar: bool = True,
) -> List[Dict[str, int | float | None]]:
    """Extract blink information from one raw segment.

    Parameters
    ----------
    seg_id : int
        Index of the segment within ``segments``.
    raw : mne.io.BaseRaw
        Segment containing the blink annotations and data.
    channel : str
        Channel name used for blink detection.
    blink_label : str | None
        Annotation label that marks blinks. ``None`` keeps all annotations.
    channel_type : str | None
        Optional override for the channel type. If ``None`` the type is
        determined from ``raw``.

    Returns
    -------
    list of dict
        Rows describing each detected blink with sample indices.
    """

    signal = raw.get_data(picks=channel)[0]
    ch_type = _get_channel_type(raw, channel, channel_type)

    starts, ends = _filter_blink_annotations(raw, blink_label)
    peaks = _detect_peaks(signal, starts, ends, ch_type)
    bounds = compute_outer_bounds(peaks, len(signal))

    rows: List[Dict[str, int | float | None]] = []
    for blink_id, (start, end, peak), (outer_start, outer_end) in tqdm(
        zip(range(len(peaks)), zip(starts, ends, peaks), bounds),
        desc=f"Seg {seg_id} blinks",
        leave=False,
        disable=not progress_bar,
    ):
        left_zero, right_zero = left_right_zero_crossing(
            signal,
            peak,
            outer_start,
            outer_end,
            signal_type=ch_type,
        )
        left_zero_norm = _normalize_zero_crossing(left_zero)
        right_zero_norm = _normalize_zero_crossing(right_zero)
        rows.append(
            {
                "seg_id": seg_id,
                "blink_id": blink_id,
                "start_blink": int(start),
                "max_blink": int(peak),
                "end_blink": int(end),
                "outer_start": int(outer_start),
                "outer_end": int(outer_end),
                "left_zero": left_zero_norm,
                "right_zero": right_zero_norm,
            }
        )
    return rows


def extract_blink_events_dataframe(
    segments: Sequence[mne.io.BaseRaw],
    *,
    channel: str = "EEG-E8",
    blink_label: str | None = "blink",
    channel_type: str | None = None,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """Create a blink event summary for the provided raw segments.

    Parameters
    ----------
    segments : sequence of mne.io.BaseRaw
        Iterable of equally sized raw segments with blink annotations.
    channel : str, optional
        Channel used for blink detection. Defaults to ``"EEG-E8"``.
    blink_label : str | None, optional
        Annotation label that denotes blinks. ``None`` uses all annotations.
    channel_type : str | None, optional
        Explicit channel type. When ``None`` the type is obtained from each
        segment and a warning is emitted if it cannot be determined.

    Returns
    -------
    pandas.DataFrame
        One row per detected blink with sample index information.
    """

    logger.info("Extracting blink events from %d segments", len(segments))
    rows: List[Dict[str, int | None]] = []

    for seg_id, raw in enumerate(
        tqdm(segments, desc="Processing segments", disable=not progress_bar)
    ):
        rows.extend(
            _process_segment_blinks(
                seg_id,
                raw,
                channel,
                blink_label,
                channel_type,
                progress_bar=progress_bar,
            )
        )

    df = pd.DataFrame(rows)
    logger.info("Extracted %d blink events", len(df))
    logger.debug("Blink events preview:\n%s", df.head())
    return df
