import numpy as np

from pyblinker.blinker.stroke_utils import max_pos_vel_frame


def test_max_pos_vel_frame_clamps_terminal_signal_indices():
    """Terminal raw-signal frames should not overflow the diff-based velocity array."""

    blink_velocity = np.array([0.1, 0.2, 0.4, -0.1, -0.3], dtype=float)

    max_pos_frame, max_neg_frame = max_pos_vel_frame(
        blink_velocity=blink_velocity,
        max_blink=5,
        left_zero=4,
        right_zero=5,
    )

    assert int(max_pos_frame) == 4
    assert int(max_neg_frame) == 4


def test_max_pos_vel_frame_returns_nan_for_empty_velocity():
    """Empty blink-velocity inputs should degrade gracefully instead of crashing."""

    max_pos_frame, max_neg_frame = max_pos_vel_frame(
        blink_velocity=np.array([], dtype=float),
        max_blink=0,
        left_zero=0,
        right_zero=0,
    )

    assert np.isnan(max_pos_frame)
    assert np.isnan(max_neg_frame)
