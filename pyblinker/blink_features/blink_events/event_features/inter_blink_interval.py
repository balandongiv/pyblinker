"""Inter-blink interval based features."""

from typing import Dict, List, Sequence, Iterable, Tuple
from pyblinker._logging import get_logger

import numpy as np
import pandas as pd
import mne

from pyblinker.utils.metadata_utils import extract_blink_windows

from pyblinker.utils import normalize_picks, require_channels
from pyblinker.utils.modality import infer_modality

logger = get_logger(__name__)


def _permutation_entropy(
    series: Sequence[float], *, order: int = 3, delay: int = 1
) -> float:
    """Calculate permutation entropy of a numeric sequence.

    Parameters
    ----------
    series : Sequence[float]
        Input data sequence representing inter-blink intervals.
    order : int, optional
        Embedding dimension for ordinal pattern creation, by default ``3``.
    delay : int, optional
        Sample delay between points in each pattern, by default ``1``.

    Returns
    -------
    float
        Estimated permutation entropy. Returns ``NaN`` if the sequence is too
        short for the specified parameters.
    """
    data = np.asarray(series)
    n = len(data)
    if n < order * delay:
        return float("nan")
    patterns: List[tuple[int, ...]] = []
    for i in range(n - delay * (order - 1)):
        subseq = data[i : i + order * delay : delay]
        patterns.append(tuple(np.argsort(subseq)))
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / counts.sum()
    pe = -np.sum(probs * np.log(probs))
    return float(pe)


def _hurst_exponent(series: Sequence[float]) -> float:
    """Estimate the Hurst exponent from a sequence using R/S analysis.

    Parameters
    ----------
    series : Sequence[float]
        Sequence of inter-blink intervals.

    Returns
    -------
    float
        Hurst exponent estimating long-range dependence. ``NaN`` is returned for
        very short or constant sequences.
    """
    data = np.asarray(series, dtype=float)
    n = len(data)
    if n < 20:
        return float("nan")
    mean = np.mean(data)
    dev = data - mean
    cumdev = np.cumsum(dev)
    r = np.max(cumdev) - np.min(cumdev)
    s = np.std(data, ddof=1)
    if s == 0 or r == 0:
        return float("nan")
    return float(np.log(r / s) / np.log(n))


def compute_ibi_features(
    blinks: List[Dict[str, int]], sfreq: float
) -> Dict[str, float]:
    """Compute inter-blink interval statistics for a given epoch.

    Parameters
    ----------
    blinks : list of dict
        Blink annotations belonging to one epoch.
    sfreq : float
        Sampling frequency of the original recording in Hertz.

    Returns
    -------
    dict
        Dictionary with summary metrics of inter-blink intervals including mean,
        standard deviation and nonlinear measures.
    """
    starts = np.array([b["refined_start_frame"] for b in blinks], dtype=float)
    ends = np.array([b["refined_end_frame"] for b in blinks], dtype=float)
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]
    ibis: np.ndarray | None = None
    if len(starts) >= 2:
        ibis = (starts[1:] - ends[:-1]) / sfreq
    if ibis is None or len(ibis) == 0:
        return {
            "ibi_mean": float("nan"),
            "ibi_std": float("nan"),
            "ibi_median": float("nan"),
            "ibi_min": float("nan"),
            "ibi_max": float("nan"),
            "ibi_cv": float("nan"),
            "ibi_rmssd": float("nan"),
            "poincare_sd1": float("nan"),
            "poincare_sd2": float("nan"),
            "poincare_ratio": float("nan"),
            "ibi_permutation_entropy": float("nan"),
            "ibi_hurst_exponent": float("nan"),
        }

    ibi_mean = float(np.mean(ibis))
    ibi_std = float(np.std(ibis, ddof=1)) if len(ibis) > 1 else float("nan")
    ibi_median = float(np.median(ibis))
    ibi_min = float(np.min(ibis))
    ibi_max = float(np.max(ibis))
    ibi_cv = float(ibi_std / ibi_mean) if ibi_mean != 0 else float("nan")
    diff = np.diff(ibis)
    rmssd = float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else float("nan")
    if len(ibis) > 2:
        x1 = ibis[:-1]
        x2 = ibis[1:]
        sd1 = float(np.sqrt(np.var(x2 - x1, ddof=1) / 2.0))
        sd2 = float(np.sqrt(np.var(x1 + x2, ddof=1) / 2.0))
        sd_ratio = float(sd1 / sd2) if sd2 != 0 else float("nan")
    else:
        sd1 = sd2 = sd_ratio = float("nan")
    pe = _permutation_entropy(ibis)
    hurst = _hurst_exponent(ibis)
    return {
        "ibi_mean": ibi_mean,
        "ibi_std": ibi_std,
        "ibi_median": ibi_median,
        "ibi_min": ibi_min,
        "ibi_max": ibi_max,
        "ibi_cv": ibi_cv,
        "ibi_rmssd": rmssd,
        "poincare_sd1": sd1,
        "poincare_sd2": sd2,
        "poincare_ratio": sd_ratio,
        "ibi_permutation_entropy": pe,
        "ibi_hurst_exponent": hurst,
    }


def _mean_inter_blink_interval(windows: Sequence[Tuple[float, float]]) -> float:
    """Return the mean interval separating successive blink windows."""

    if len(windows) < 2:
        return float("nan")

    starts = np.asarray([onset for onset, _ in windows], dtype=float)
    durations = np.asarray([duration for _, duration in windows], dtype=float)
    order = np.argsort(starts)
    starts = starts[order]
    durations = durations[order]
    ends = starts + durations
    intervals = starts[1:] - ends[:-1]
    if intervals.size == 0:
        return float("nan")
    return float(np.mean(intervals))


def inter_blink_interval_epochs(
    epochs: mne.Epochs, picks: str | Iterable[str] | None = None
) -> pd.DataFrame:
    """Compute mean inter-blink interval per channel for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoch object whose metadata contains blink onset and duration
        information by default. If modality-specific columns such as
        ``blink_onset_eeg`` are present they are used but this depends on the
        picks; otherwise (if ``picks`` is ``None``) the generic ``blink_onset``
        and ``blink_duration`` columns are expected.
    picks : str or iterable of str, optional
        Channel name(s) for which IBI columns are created. The modality of each
        channel determines which blink onset/duration columns are used when
        available. However, if ``picks`` is ``None`` we use the generic
        ``blink_onset`` and ``blink_duration`` columns for computation.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` with a leading ``ep`` column. When
        ``picks`` is ``None`` a single ``ibi`` column holds the per-epoch mean
        interval derived from the generic blink metadata. If channel names are
        provided, one ``ibi_<channel>`` column is produced for each requested
        channel. Epochs with fewer than two blinks receive ``NaN``.

    Raises
    ------
    ValueError
        If required metadata columns are missing or a requested channel does
        not exist in ``epochs``.

    Notes
    -----
    When multiple blinks occur within an epoch, the mean interval between
    consecutive blinks is returned. If a single blink or no blink is present,
    ``NaN`` is assigned.
    """

    picks_list = normalize_picks(picks) if picks is not None else []
    if picks_list:
        require_channels(epochs, picks_list)
    metadata = epochs.metadata
    if metadata is None:
        raise ValueError("Epochs.metadata must contain blink information")

    if isinstance(metadata, pd.DataFrame):
        metadata_df = metadata
        index = metadata_df.index
    else:
        metadata_df = pd.DataFrame(metadata)
        index = pd.RangeIndex(len(epochs))

    df = pd.DataFrame(index=index)
    df.insert(0, "ep", index.to_numpy())

    rows = [row for _, row in metadata_df.iterrows()]

    logger.info(
        "Computing inter-blink intervals for %d epochs and channels %s",
        len(epochs),
        picks_list if picks_list else "[generic]",
    )

    if not picks_list:
        ibis = [
            _mean_inter_blink_interval(extract_blink_windows(row, None, epoch_idx))
            for epoch_idx, row in enumerate(rows)
        ]
        df["ibi"] = ibis
    else:
        for ch in picks_list:
            modality = infer_modality(ch)
            ibis = [
                _mean_inter_blink_interval(extract_blink_windows(row, ch, epoch_idx))
                for epoch_idx, row in enumerate(rows)
            ]
            df[f"ibi_{ch}"] = ibis
            logger.debug(
                "IBI values for channel '%s' (modality '%s'): %s", ch, modality, ibis
            )

    logger.debug("Computed channel-wise IBI DataFrame shape: %s", df.shape)
    logger.info("Finished computing IBI DataFrame")
    return df
