"""Geometry helpers for blink segmentation.

These functions support both continuous raw signals and short epoch-local
segments by operating purely on the provided signal and indices.
"""

from __future__ import annotations

import numpy as np

from pyblinker._logging import get_logger
from pyblinker.blinker.stroke_utils import max_pos_vel_frame
from pyblinker.fitutils.forking import corr, get_intersection, polyfit, polyval


logger = get_logger(__name__)


def get_max_blink(
    candidate_signal: np.ndarray,
    start_idx: int | float,
    end_idx: int | float,
) -> tuple[float, int]:
    """Return the maximum value and index within ``start_idx`` and ``end_idx``."""

    start = int(start_idx)
    end = int(end_idx)

    signal = np.asarray(candidate_signal)
    blink_frame = signal[start : end + 1]
    max_idx = int(np.argmax(blink_frame))
    max_val = float(blink_frame[max_idx])
    return max_val, start + max_idx


def _find_left_zero_crossing(
    candidate_signal: np.ndarray,
    start_idx: int,
    m_frame: int,
) -> int | float:
    """Match MATLAB's left-zero logic using the local minimum and <= 0 search."""

    left_range = np.arange(start_idx, m_frame + 1, dtype=int)
    if left_range.size == 0:
        return np.nan

    min_index = int(np.argmin(candidate_signal[left_range]))
    min_frame = int(left_range[min_index])

    search_range = np.arange(min_frame, m_frame + 1, dtype=int)
    s_ind_left_zero = np.flatnonzero(candidate_signal[search_range] <= 0)

    if s_ind_left_zero.size > 0:
        return int(search_range[s_ind_left_zero[-1]])

    min_search_index = int(np.argmin(candidate_signal[search_range]))
    return int(search_range[min_search_index])


def _find_right_zero_crossing(
    candidate_signal: np.ndarray,
    m_frame: int,
    end_idx: int,
) -> int | float:
    """Match MATLAB's right-zero logic using the local minimum and <= 0 search."""

    right_range = np.arange(m_frame, end_idx + 1, dtype=int)
    if right_range.size == 0:
        return np.nan

    min_index = int(np.argmin(candidate_signal[right_range]))
    min_frame = int(right_range[min_index])

    search_range = np.arange(m_frame, min_frame + 1, dtype=int)
    s_ind_right_zero = np.flatnonzero(candidate_signal[search_range] <= 0)

    if s_ind_right_zero.size > 0:
        return int(search_range[s_ind_right_zero[0]])

    min_search_index = int(np.argmin(candidate_signal[search_range]))
    return int(search_range[min_search_index])


def left_right_zero_crossing(
    candidate_signal: np.ndarray,
    max_blink: float,
    outer_start: float,
    outer_end: float,
    *,
    signal_type: str | None = None,
) -> tuple[int | float, int | float]:
    """Locate zero-crossing indices immediately surrounding a blink peak."""

    if signal_type is not None and signal_type.lower() != "eeg":
        logger.warning(
            "left_right_zero_crossing tuned for EEG signals; results may be inaccurate for %s",
            signal_type,
        )

    start_idx = int(outer_start)
    m_frame = int(max_blink)
    end_idx = int(outer_end)

    left_zero = _find_left_zero_crossing(candidate_signal, start_idx, m_frame)
    right_zero = _find_right_zero_crossing(candidate_signal, m_frame, end_idx)

    if not np.isnan(left_zero) and left_zero > m_frame:
        raise ValueError(
            f"Validation error: left_zero = {left_zero}, max_blink = {m_frame}. Ensure left_zero <= max_blink."
        )

    if not np.isnan(right_zero) and m_frame > right_zero:
        raise ValueError(
            f"Validation error: max_blink = {m_frame}, right_zero = {right_zero}. Ensure max_blink <= right_zero."
        )

    return left_zero, right_zero


def get_left_base(blink_velocity, left_outer, max_pos_vel_frame):
    """Determine the left base frame index."""

    l_outer = int(left_outer)
    m_pos_vel = int(max_pos_vel_frame)

    left_range = np.arange(l_outer, m_pos_vel + 1)
    reversed_velocity = np.flip(blink_velocity[left_range])

    mask = reversed_velocity <= 0
    if not np.any(mask):
        return m_pos_vel

    # MATLAB: leftBaseIndex = find(..., 'first'); leftBase = maxPosVelFrame - leftBaseIndex
    # ``find`` is 1-based, so convert argmax result to 1-based before subtraction.
    left_base_index = int(np.argmax(mask)) + 1
    left_base = m_pos_vel - left_base_index
    return left_base


def get_right_base(
    candidate_signal: np.ndarray,
    blink_velocity: np.ndarray,
    right_outer: int,
    max_neg_vel_frame: float | int,
) -> int | float | None:
    """Compute the right base frame index."""

    r_outer = int(right_outer)

    if np.isnan(max_neg_vel_frame):
        return np.nan
    m_neg_vel = int(max_neg_vel_frame)

    if m_neg_vel > r_outer:
        return None

    max_size = candidate_signal.size
    max_velocity_index = max_size - 2
    end_idx = min(r_outer, max_velocity_index)
    right_range = np.arange(m_neg_vel, end_idx + 1)

    if right_range.size == 0:
        return None

    if right_range[-1] >= blink_velocity.size:
        right_range = right_range[:-1]
        if right_range.size == 0 or right_range[-1] >= blink_velocity.size:
            logger.warning(
                "Unable to compute right base: right_range %s exceeds blink_velocity length %d",
                right_range,
                blink_velocity.size,
            )
            return None

    right_base_velocity = blink_velocity[right_range]
    mask = right_base_velocity >= 0
    if not np.any(mask):
        return m_neg_vel

    # MATLAB: rightBaseIndex = find(..., 'first'); rightBase = maxNegVelFrame + rightBaseIndex
    right_base_index = int(np.argmax(mask)) + 1
    right_base = m_neg_vel + right_base_index
    return right_base


def create_left_right_base(candidate_signal, df):
    """Compute left/right baselines for each blink in ``df``."""

    df = df.copy()
    blink_velocity = np.diff(candidate_signal, axis=0)

    df[["max_pos_vel_frame", "max_neg_vel_frame"]] = df.apply(
        lambda row: max_pos_vel_frame(
            blink_velocity=blink_velocity,
            max_blink=row["max_blink"],
            left_zero=row["left_zero"],
            right_zero=row["right_zero"],
        ),
        axis=1,
        result_type="expand",
    )

    df = df.assign(
        left_base=df.apply(
            lambda row: get_left_base(
                blink_velocity=blink_velocity,
                left_outer=row["outer_start"],
                max_pos_vel_frame=row["max_pos_vel_frame"],
            ),
            axis=1,
        )
    )

    df = df.assign(
        right_base=df.apply(
            lambda row: get_right_base(
                candidate_signal=candidate_signal,
                blink_velocity=blink_velocity,
                right_outer=row["outer_end"],
                max_neg_vel_frame=row["max_neg_vel_frame"],
            ),
            axis=1,
        )
    )

    return df


def get_half_height(
    candidate_signal: np.ndarray,
    max_blink,
    left_zero,
    right_zero,
    left_base,
    right_outer,
):
    """Locate half-height landmarks around a blink peak."""

    m_frame = int(max_blink)
    l_zero = int(left_zero)
    r_zero = int(right_zero)
    l_base = int(left_base)
    r_outer = int(right_outer)

    max_val = candidate_signal[m_frame]
    left_base_val = candidate_signal[l_base]
    half_height_val = max_val - 0.5 * (max_val - left_base_val)

    left_range = np.arange(l_base, m_frame + 1)
    left_vals = candidate_signal[left_range]
    left_mask = left_vals >= half_height_val
    if np.any(left_mask):
        left_index = np.argmax(left_mask)
        left_base_half_height = l_base + left_index + 1
    else:
        left_base_half_height = np.nan

    right_range = np.arange(m_frame, r_outer + 1)
    right_vals = candidate_signal[right_range]
    right_mask = right_vals <= half_height_val
    if np.any(right_mask):
        right_base_half_height = min(
            r_outer,
            np.argmax(right_mask) + m_frame,
        )
    else:
        right_base_half_height = np.nan

    zero_half_val = 0.5 * max_val
    left_zero_range = np.arange(l_zero, m_frame + 1)
    left_zero_vals = candidate_signal[left_zero_range]
    left_zero_mask = left_zero_vals >= zero_half_val
    if np.any(left_zero_mask):
        left_zero_index = np.argmax(left_zero_mask)
        left_zero_half_height = l_zero + left_zero_index + 1
    else:
        left_zero_half_height = np.nan

    right_zero_range = np.arange(m_frame, r_zero + 1)
    right_zero_vals = candidate_signal[right_zero_range]
    right_zero_mask = right_zero_vals <= zero_half_val
    if np.any(right_zero_mask):
        right_zero_index = np.argmax(right_zero_mask)
        right_zero_half_height = min(r_outer, m_frame + right_zero_index)
    else:
        right_zero_half_height = np.nan

    return (
        left_zero_half_height,
        right_zero_half_height,
        left_base_half_height,
        right_base_half_height,
    )


def get_left_range(left_zero, max_blink, candidate_signal, blink_top, blink_bottom):
    """Identify the left blink range using the provided thresholds."""

    l_zero = int(left_zero)
    m_frame = int(max_blink)

    blink_range = np.arange(l_zero, m_frame + 1, dtype=int)
    cand_slice = candidate_signal[blink_range]

    top_idx = np.where(cand_slice < blink_top)[0]
    blink_top_point_idx = top_idx[-1]

    bottom_idx = np.flatnonzero(cand_slice > blink_bottom)
    blink_bottom_point_idx = bottom_idx[0]

    blink_top_point_l_x = blink_range[blink_top_point_idx]
    blink_top_point_l_y = candidate_signal[blink_top_point_l_x]

    blink_bottom_point_l_x = blink_range[blink_bottom_point_idx]
    blink_bottom_point_l_y = candidate_signal[blink_bottom_point_l_x]

    left_range = [blink_bottom_point_l_x, blink_top_point_l_x]

    return (
        left_range,
        blink_top_point_l_x,
        blink_top_point_l_y,
        blink_bottom_point_l_x,
        blink_bottom_point_l_y,
    )


def get_right_range(max_blink, right_zero, candidate_signal, blink_top, blink_bottom):
    """Identify the right blink range using the provided thresholds."""

    m_frame = int(max_blink)
    r_zero = int(right_zero)

    blink_range = np.arange(m_frame, r_zero + 1, dtype=int)
    cand_slice = candidate_signal[blink_range]

    top_mask = cand_slice < blink_top
    blink_top_point_r = np.argmax(top_mask)

    bottom_mask = cand_slice > blink_bottom
    bottom_idx = np.where(bottom_mask)[0]
    blink_bottom_point_r = bottom_idx[-1]

    blink_top_point_r_x = blink_range[blink_top_point_r]
    blink_top_point_r_y = candidate_signal[blink_top_point_r_x]

    blink_bottom_point_r_x = blink_range[blink_bottom_point_r]
    blink_bottom_point_r_y = candidate_signal[blink_bottom_point_r_x]

    right_range = [blink_range[blink_top_point_r], blink_range[blink_bottom_point_r]]

    return (
        right_range,
        blink_top_point_r_x,
        blink_top_point_r_y,
        blink_bottom_point_r_x,
        blink_bottom_point_r_y,
    )


def compute_fit_range(
    candidate_signal, max_blink, left_zero, right_zero, base_fraction, top_bottom=None
):
    """Compute the blink fitting ranges around a blink event."""

    m_frame = int(max_blink)
    l_zero = int(left_zero)
    r_zero = int(right_zero)

    blink_height = candidate_signal[m_frame] - candidate_signal[l_zero]
    blink_top = candidate_signal[m_frame] - base_fraction * blink_height
    blink_bottom = candidate_signal[l_zero] + base_fraction * blink_height

    (
        left_range,
        blink_top_point_l_x,
        blink_top_point_l_y,
        blink_bottom_point_l_x,
        blink_bottom_point_l_y,
    ) = get_left_range(l_zero, m_frame, candidate_signal, blink_top, blink_bottom)

    (
        right_range,
        blink_top_point_r_x,
        blink_top_point_r_y,
        blink_bottom_point_r_x,
        blink_bottom_point_r_y,
    ) = get_right_range(m_frame, r_zero, candidate_signal, blink_top, blink_bottom)

    x_left = np.arange(left_range[0], left_range[1] + 1, dtype=int)
    x_right = np.arange(right_range[0], right_range[1] + 1, dtype=int)

    if x_left.size == 0:
        x_left = np.nan
    if x_right.size == 0:
        x_right = np.nan

    if top_bottom is None:
        logger.warning(
            "compute_fit_range called without top_bottom flag; returning minimal output",
            extra={"top_bottom": top_bottom},
        )
        return x_left, x_right, left_range, right_range

    return (
        x_left,
        x_right,
        left_range,
        right_range,
        blink_bottom_point_l_y,
        blink_bottom_point_l_x,
        blink_top_point_l_y,
        blink_top_point_l_x,
        blink_bottom_point_r_x,
        blink_bottom_point_r_y,
        blink_top_point_r_x,
        blink_top_point_r_y,
    )


def get_line_intersection_slope(
    x_intersect, y_intersect, left_x_intersect, right_x_intersect
):
    """Compute slopes at the intersection point."""

    left_slope = y_intersect / (x_intersect - left_x_intersect)
    right_slope = y_intersect / (x_intersect - right_x_intersect)
    return left_slope, right_slope


def lines_intersection(
    *,
    signal: np.ndarray | None = None,
    x_right: np.ndarray | None = None,
    x_left: np.ndarray | None = None,
) -> tuple[float, ...]:
    """Return intersection metrics for the given signal segments."""

    from pyblinker.utils.velocity_utils import average_velocity

    y_right = signal[x_right]
    y_left = signal[x_left]

    degree = 1
    p_left, s_left, mu_left = polyfit(x_left, y_left, degree)
    y_pred_left, _ = polyval(p_left, x_left, S=s_left, mu=mu_left)
    left_r2, _ = corr(y_left, y_pred_left)

    p_right, s_right, mu_right = polyfit(x_right, y_right, 1)
    y_pred_right, _ = polyval(p_right, x_right, S=s_right, mu=mu_right)
    right_r2, _ = corr(y_right, y_pred_right)

    (
        x_intersect,
        y_intersect,
        left_x_intercept,
        right_x_intercept,
    ) = get_intersection(p_left, p_right, mu_left, mu_right)

    if x_intersect == left_x_intercept or x_intersect == right_x_intercept:
        left_slope = np.nan
        right_slope = np.nan
        aver_left_velocity = np.nan
        aver_right_velocity = np.nan
    else:
        left_slope, right_slope = get_line_intersection_slope(
            x_intersect, y_intersect, left_x_intercept, right_x_intercept
        )
        aver_left_velocity = average_velocity(p_left, x_scale=mu_left[1])
        aver_right_velocity = average_velocity(p_right, x_scale=mu_right[1])

    # MATLAB fixture exports preserve single-precision-looking R values; casting
    # here stabilizes bitwise equality in strict DataFrame comparisons.
    right_r2_scalar = float(np.float32(right_r2[0][0]))
    left_r2_scalar = float(np.float32(left_r2[0][0]))

    return (
        left_slope,
        right_slope,
        aver_left_velocity,
        aver_right_velocity,
        right_r2_scalar,
        left_r2_scalar,
        x_intersect,
        y_intersect,
        left_x_intercept,
        right_x_intercept,
    )


__all__ = [
    "compute_fit_range",
    "create_left_right_base",
    "get_half_height",
    "get_left_base",
    "get_left_range",
    "get_line_intersection_slope",
    "get_max_blink",
    "get_right_base",
    "get_right_range",
    "left_right_zero_crossing",
    "lines_intersection",
]
