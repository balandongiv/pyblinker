"""Utility helpers for blink stroke analysis."""

from __future__ import annotations

import numpy as np

from pyblinker.logging import get_logger


logger = get_logger(__name__)


def _coerce_velocity_window(
    blink_velocity: np.ndarray,
    *,
    max_blink,
    left_zero,
    right_zero,
):
    """Clamp frame indices into the valid blink-velocity index range.

    ``blink_velocity`` is derived with ``np.diff(candidate_signal)`` and is
    therefore one sample shorter than the original blink signal. A zero-crossing
    or max-blink frame that lands on the final raw sample is still valid in the
    source signal, but it must be clamped before indexing the velocity array.
    """

    velocity = np.asarray(blink_velocity)
    if velocity.ndim == 0:
        velocity = velocity.reshape(1)
    if velocity.size == 0:
        return None

    last_index = int(velocity.size - 1)
    m_frame = int(np.clip(int(max_blink), 0, last_index))
    l_zero = int(np.clip(int(left_zero), 0, last_index))
    r_zero = int(np.clip(int(right_zero), 0, last_index))

    if l_zero > m_frame:
        l_zero = m_frame
    if r_zero < m_frame:
        r_zero = m_frame

    return l_zero, m_frame, r_zero


def get_up_down_stroke(max_blink, left_zero, right_zero):
    """Compute index ranges for the upward and downward blink strokes."""
    m_frame = int(max_blink)
    l_zero = int(left_zero)
    r_zero = int(right_zero)

    up_stroke = np.arange(l_zero, m_frame + 1)
    down_stroke = np.arange(m_frame, r_zero + 1)
    return up_stroke, down_stroke


def max_pos_vel_frame(blink_velocity, max_blink, left_zero, right_zero):
    """Locate frames with maximum positive and negative blink velocities."""
    resolved_window = _coerce_velocity_window(
        blink_velocity,
        max_blink=max_blink,
        left_zero=left_zero,
        right_zero=right_zero,
    )
    if resolved_window is None:
        logger.debug(
            "Blink velocity was empty; forcing NaN velocity landmarks",
            extra={"velocity_size": int(np.asarray(blink_velocity).size)},
        )
        return np.nan, np.nan
    l_zero, m_frame, r_zero = resolved_window

    up_stroke, down_stroke = get_up_down_stroke(m_frame, l_zero, r_zero)

    if up_stroke.size == 0:
        logger.debug(
            "Up-stroke segment empty after clamping; forcing NaN landmarks",
            extra={
                "left_zero": int(l_zero),
                "max_blink": int(m_frame),
                "right_zero": int(r_zero),
            },
        )
        return np.nan, np.nan

    # Maximum positive velocity in the up_stroke region
    max_pos_vel_idx = np.argmax(blink_velocity[up_stroke])
    max_pos_vel_frame = up_stroke[max_pos_vel_idx]

    # Maximum negative velocity in the down_stroke region, if it exists
    if down_stroke.size > 0:
        try:
            max_neg_vel_idx = np.argmin(blink_velocity[down_stroke])
            max_neg_vel_frame = down_stroke[max_neg_vel_idx]
        except IndexError:
            logger.debug(
                "Down-stroke segment indexing failed; forcing NaN for max negative velocity",
                extra={"down_stroke_size": int(down_stroke.size)},
            )
            max_neg_vel_frame = np.nan
    else:
        logger.warning(
            "Down-stroke segment empty; forcing NaN for max negative velocity",
            extra={"down_stroke_size": int(down_stroke.size)},
        )
        max_neg_vel_frame = np.nan

    return max_pos_vel_frame, max_neg_vel_frame
