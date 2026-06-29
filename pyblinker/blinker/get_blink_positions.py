import numpy as np
import pandas as pd
from tqdm import tqdm

from ..fitutils import mad
from .default_setting import SCALING_FACTOR


_SUPPORTED_CENTER_METHODS = ("median", "mean")


def compute_robust_threshold(
    samples: np.ndarray,
    std_threshold: float,
    *,
    center_method: str = "mean",
    scaling_factor: float = SCALING_FACTOR,
) -> tuple[float, float, float]:
    """Compute a robust detection threshold from a 1-D sample array.

    The threshold follows the BLINKER convention::

        dispersion = scaling_factor * MAD(samples)   # a.k.a. robust_std
        threshold  = center + std_threshold * dispersion

    where ``center`` is either the median (rank-based, robust to large blink
    peaks) or the mean (pulled upward by peaks, more conservative).

    Parameters
    ----------
    samples:
        1-D array of signal amplitude samples for a single channel.
    std_threshold:
        Multiplier ``k`` applied to the MAD-based dispersion term.
    center_method:
        ``"mean"`` (default, the legacy ``_compute_basic_statistics`` behaviour)
        or ``"median"`` (robust centre used by the double-thresholding strategy).
    scaling_factor:
        Factor normalising MAD to the standard-deviation scale (``1.4826`` for a
        normal distribution, matching MATLAB BLINKER).

    Returns
    -------
    center : float
        Central tendency of ``samples`` (median or mean).
    dispersion : float
        Robust standard-deviation estimate ``scaling_factor * MAD(samples)``.
    threshold : float
        ``center + std_threshold * dispersion``.

    Raises
    ------
    ValueError
        If ``center_method`` is not one of ``("median", "mean")``. ok check
    """

    if center_method not in _SUPPORTED_CENTER_METHODS:
        raise ValueError(
            f"center_method={center_method!r} is not supported. "
            f"Choose one of {_SUPPORTED_CENTER_METHODS}."
        )

    if center_method == "median":
        center = float(np.median(samples))
    else:  # "mean"
        center = float(np.mean(samples, dtype=np.float64))

    dispersion = float(scaling_factor * mad(samples))  # a.k.a. robust_std
    threshold = center + float(std_threshold) * dispersion
    return center, dispersion, threshold


def _compute_basic_statistics(
    params: dict,
    blink_component: np.ndarray,
) -> tuple[float, float]:
    """Return MATLAB-equivalent thresholding statistics."""

    _center, _dispersion, threshold = compute_robust_threshold(
        blink_component,
        params["std_threshold"],
        center_method="mean",
    )
    min_blink_frames = float(params["min_event_len"] * params["sfreq"])
    return min_blink_frames, threshold


def _scan_threshold_crossings(
    blink_component: np.ndarray,
    threshold: float,
    min_blink_frames: float,
    *,
    progress_bar: bool,
    channel_name: str | None, #ok
) -> tuple[np.ndarray, np.ndarray]:
    """Collect candidate blink onsets/offsets using MATLAB loop semantics."""

    in_blink = False
    start = 0
    start_blinks: list[int] = []
    end_blinks: list[int] = []

    for idx in tqdm(
        range(blink_component.size),
        desc=f"Get blink start and end for channel {channel_name}",
        disable=not progress_bar,
    ):
        value = blink_component[idx]
        if (not in_blink) and (value > threshold):
            start = idx
            in_blink = True

        if in_blink and (value < threshold):
            if (idx - start) > min_blink_frames:
                start_blinks.append(start)
                end_blinks.append(idx)
            in_blink = False

    return np.asarray(start_blinks, dtype=int), np.asarray(end_blinks, dtype=int)


def _apply_minimum_separation(
    start_blinks: np.ndarray,
    end_blinks: np.ndarray,
    *,
    sfreq: float,
    min_event_sep: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove adjacent blinks that are closer than MATLAB's minEventSep."""

    if end_blinks.size == 0:
        return start_blinks, end_blinks

    position_mask = np.ones(end_blinks.size, dtype=bool)
    delta = (start_blinks[1:] - end_blinks[:-1]) / sfreq
    too_close = np.flatnonzero(delta <= min_event_sep)
    position_mask[too_close] = False
    position_mask[too_close + 1] = False
    return start_blinks[position_mask], end_blinks[position_mask]


def get_blink_position(
    params,
    blink_component=None,
    ch=None,
    *,
    progress_bar: bool = True,
):
    """Detect blink start and end frames using the legacy MATLAB Blinker approach."""

    assert blink_component.ndim == 1, "blink_component must be a 1D array"

    min_blink_frames, threshold = _compute_basic_statistics(params, blink_component)
    start_blinks, end_blinks = _scan_threshold_crossings(
        blink_component,
        threshold,
        min_blink_frames,
        progress_bar=progress_bar,
        channel_name=ch,
    )

    if start_blinks.size == 0:
        return pd.DataFrame({"start_blink": [], "end_blink": []})

    min_event_sep = float(params.get("min_event_sep", params["min_event_len"]))
    start_blinks, end_blinks = _apply_minimum_separation(
        start_blinks,
        end_blinks,
        sfreq=params["sfreq"],
        min_event_sep=min_event_sep,
    )

    return pd.DataFrame(
        {
            "start_blink": start_blinks,
            "end_blink": end_blinks,
        }
    )


def compute_basic_statistics(
    params: dict,
    blink_component: np.ndarray,
) -> tuple[float, float]:
    """Return MATLAB-equivalent thresholding statistics (public alias).

    Returns (min_blink_frames, threshold).
    """
    return _compute_basic_statistics(params, blink_component)


def scan_threshold_crossings_kleifges(
    blink_component: np.ndarray,
    threshold: float,
    min_blink_frames: float,
    *,
    progress_bar: bool,
    channel_name: str | None,# ok
) -> tuple[np.ndarray, np.ndarray]:
    """Kleifges 2017 threshold-crossing scan — no minimum-separation filtering."""
    return _scan_threshold_crossings(
        blink_component,
        threshold,
        min_blink_frames,
        progress_bar=progress_bar,
        channel_name=channel_name,
    )
