"""Unit tests for assign_epoch_health and get_valid_epoch_indices_by_health."""

from __future__ import annotations

import unittest

import pandas as pd

from pyblinker.epoch_detection import assign_epoch_health, get_valid_epoch_indices_by_health

# ---------------------------------------------------------------------------
# Shared baseline fixture (7 × 30-second windows, 210 s total coverage)
# ---------------------------------------------------------------------------

BASELINE_ROWS = [
    {"epoch_start_s": 0,   "epoch_end_s": 30,  "health": 5},
    {"epoch_start_s": 30,  "epoch_end_s": 60,  "health": 5},
    {"epoch_start_s": 60,  "epoch_end_s": 90,  "health": 5},
    {"epoch_start_s": 90,  "epoch_end_s": 120, "health": 1},
    {"epoch_start_s": 120, "epoch_end_s": 150, "health": 5},
    {"epoch_start_s": 150, "epoch_end_s": 180, "health": 5},
    {"epoch_start_s": 180, "epoch_end_s": 210, "health": 2},
]

SIGNAL_DURATION_S = 240.0


def _health_df() -> pd.DataFrame:
    return pd.DataFrame(BASELINE_ROWS)


def _n_epochs(duration_s: float) -> int:
    import math
    return math.floor(SIGNAL_DURATION_S / duration_s)


# ---------------------------------------------------------------------------
# 30-second epochs (matches baseline window size)
# ---------------------------------------------------------------------------

class TestAssignHealth30s(unittest.TestCase):
    DURATION = 30.0

    def test_health_values(self) -> None:
        health = assign_epoch_health(_health_df(), self.DURATION, _n_epochs(self.DURATION))
        self.assertEqual(health, [5, 5, 5, 1, 5, 5, 2, None])

    def test_exact_alignment_no_mixing(self) -> None:
        health = assign_epoch_health(_health_df(), self.DURATION, 7)
        self.assertEqual(health, [5, 5, 5, 1, 5, 5, 2])

    def test_bad_epoch_at_index3(self) -> None:
        health = assign_epoch_health(_health_df(), self.DURATION, 4)
        self.assertEqual(health[3], 1)

    def test_valid_indices_min3(self) -> None:
        health = assign_epoch_health(_health_df(), self.DURATION, _n_epochs(self.DURATION))
        self.assertEqual(get_valid_epoch_indices_by_health(health, min_health=3), [0, 1, 2, 4, 5])


# ---------------------------------------------------------------------------
# 40-second epochs
# ---------------------------------------------------------------------------

class TestAssignHealth40s(unittest.TestCase):
    DURATION = 40.0

    def test_health_values(self) -> None:
        health = assign_epoch_health(_health_df(), self.DURATION, _n_epochs(self.DURATION))
        self.assertEqual(health, [5, 5, 1, 5, 2, 2])

    def test_boundary_overlap_min_taken(self) -> None:
        # epoch [80, 120] overlaps [60,90]=5 AND [90,120]=1 → min=1
        health = assign_epoch_health(_health_df(), self.DURATION, 3)
        self.assertEqual(health[2], 1)

    def test_valid_indices_min3(self) -> None:
        health = assign_epoch_health(_health_df(), self.DURATION, _n_epochs(self.DURATION))
        self.assertEqual(get_valid_epoch_indices_by_health(health, min_health=3), [0, 1, 3])


# ---------------------------------------------------------------------------
# 60-second epochs
# ---------------------------------------------------------------------------

class TestAssignHealth60s(unittest.TestCase):
    DURATION = 60.0

    def test_health_values(self) -> None:
        health = assign_epoch_health(_health_df(), self.DURATION, _n_epochs(self.DURATION))
        self.assertEqual(health, [5, 1, 5, 2])

    def test_epoch_spanning_boundary_min_taken(self) -> None:
        # epoch [60, 120] overlaps [60,90]=5 AND [90,120]=1 → min=1
        health = assign_epoch_health(_health_df(), self.DURATION, 2)
        self.assertEqual(health[1], 1)

    def test_valid_indices_min3(self) -> None:
        health = assign_epoch_health(_health_df(), self.DURATION, _n_epochs(self.DURATION))
        self.assertEqual(get_valid_epoch_indices_by_health(health, min_health=3), [0, 2])


# ---------------------------------------------------------------------------
# 20-second epochs
# ---------------------------------------------------------------------------

class TestAssignHealth20s(unittest.TestCase):
    DURATION = 20.0

    def test_health_values(self) -> None:
        health = assign_epoch_health(_health_df(), self.DURATION, _n_epochs(self.DURATION))
        self.assertEqual(health, [5, 5, 5, 5, 1, 1, 5, 5, 5, 2, 2, None])

    def test_valid_indices_min3(self) -> None:
        health = assign_epoch_health(_health_df(), self.DURATION, _n_epochs(self.DURATION))
        valid = get_valid_epoch_indices_by_health(health, min_health=3)
        self.assertEqual(valid, [0, 1, 2, 3, 6, 7, 8])


# ---------------------------------------------------------------------------
# get_valid_epoch_indices_by_health: pure-logic tests
# ---------------------------------------------------------------------------

class TestGetValidByHealth(unittest.TestCase):
    def test_all_pass(self) -> None:
        self.assertEqual(get_valid_epoch_indices_by_health([5, 4, 3], min_health=3), [0, 1, 2])

    def test_all_fail(self) -> None:
        self.assertEqual(get_valid_epoch_indices_by_health([1, 2], min_health=3), [])

    def test_all_none_excluded(self) -> None:
        self.assertEqual(get_valid_epoch_indices_by_health([None, None], min_health=1), [])

    def test_empty_list(self) -> None:
        self.assertEqual(get_valid_epoch_indices_by_health([], min_health=3), [])

    def test_threshold_boundary_inclusive(self) -> None:
        result = get_valid_epoch_indices_by_health([3, 2, 3], min_health=3)
        self.assertEqual(result, [0, 2])

    def test_mixed_none_and_values(self) -> None:
        result = get_valid_epoch_indices_by_health([5, None, 3, None, 1], min_health=3)
        self.assertEqual(result, [0, 2])

    def test_default_min_health_is_3(self) -> None:
        result = get_valid_epoch_indices_by_health([5, 2, 4])
        self.assertEqual(result, [0, 2])


# ---------------------------------------------------------------------------
# Edge cases for assign_epoch_health
# ---------------------------------------------------------------------------

class TestAssignHealthEdgeCases(unittest.TestCase):
    def test_empty_health_df(self) -> None:
        df = pd.DataFrame(columns=["epoch_start_s", "epoch_end_s", "health"])
        self.assertEqual(assign_epoch_health(df, 30.0, 3), [None, None, None])

    def test_zero_epochs_requested(self) -> None:
        self.assertEqual(assign_epoch_health(_health_df(), 30.0, 0), [])

    def test_single_epoch_full_coverage_uses_minimum(self) -> None:
        health = assign_epoch_health(_health_df(), 210.0, 1)
        self.assertEqual(health, [1])  # min(5,5,5,1,5,5,2) = 1

    def test_epoch_beyond_baseline_returns_none(self) -> None:
        df = pd.DataFrame([{"epoch_start_s": 0, "epoch_end_s": 30, "health": 4}])
        health = assign_epoch_health(df, 30.0, 3)
        self.assertEqual(health[0], 4)
        self.assertIsNone(health[1])
        self.assertIsNone(health[2])


if __name__ == "__main__":
    unittest.main()
