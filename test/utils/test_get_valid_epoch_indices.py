"""Unit tests for pyblinker.epoch_detection.get_valid_epoch_indices."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from pyblinker.epoch_detection import get_valid_epoch_indices


class _MockEpochs:
    """Minimal stand-in for mne.Epochs — only metadata and __len__ are used."""

    def __init__(self, metadata: pd.DataFrame | None, n: int) -> None:
        self.metadata = metadata
        self._n = n

    def __len__(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# No metadata
# ---------------------------------------------------------------------------

class TestNoMetadata(unittest.TestCase):
    def test_none_metadata_returns_all(self) -> None:
        epochs = _MockEpochs(metadata=None, n=5)
        self.assertEqual(get_valid_epoch_indices(epochs), [0, 1, 2, 3, 4])

    def test_empty_epochs_no_metadata(self) -> None:
        epochs = _MockEpochs(metadata=None, n=0)
        self.assertEqual(get_valid_epoch_indices(epochs), [])


# ---------------------------------------------------------------------------
# is_bad_epoch column
# ---------------------------------------------------------------------------

class TestIsBadEpoch(unittest.TestCase):
    def test_all_good(self) -> None:
        meta = pd.DataFrame({"is_bad_epoch": [False, False, False]})
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 3)), [0, 1, 2])

    def test_first_epoch_bad(self) -> None:
        meta = pd.DataFrame({"is_bad_epoch": [True, False, False]})
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 3)), [1, 2])

    def test_all_bad(self) -> None:
        meta = pd.DataFrame({"is_bad_epoch": [True, True, True]})
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 3)), [])

    def test_alternating_bad(self) -> None:
        meta = pd.DataFrame({"is_bad_epoch": [False, True, False, True, False]})
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 5)), [0, 2, 4])

    def test_string_true_false_flags(self) -> None:
        meta = pd.DataFrame({"is_bad_epoch": ["true", "false", "true"]})
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 3)), [1])

    def test_numeric_one_zero_flags(self) -> None:
        meta = pd.DataFrame({"is_bad_epoch": [1, 0, 1]})
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 3)), [1])

    def test_single_bad_epoch(self) -> None:
        meta = pd.DataFrame({"is_bad_epoch": [True]})
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 1)), [])

    def test_single_good_epoch(self) -> None:
        meta = pd.DataFrame({"is_bad_epoch": [False]})
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 1)), [0])


# ---------------------------------------------------------------------------
# is_good_epoch column
# ---------------------------------------------------------------------------

class TestIsGoodEpoch(unittest.TestCase):
    def test_all_good(self) -> None:
        meta = pd.DataFrame({"is_good_epoch": [True, True, True]})
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 3)), [0, 1, 2])

    def test_middle_not_good(self) -> None:
        meta = pd.DataFrame({"is_good_epoch": [True, False, True]})
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 3)), [0, 2])

    def test_none_good(self) -> None:
        meta = pd.DataFrame({"is_good_epoch": [False, False, False]})
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 3)), [])


# ---------------------------------------------------------------------------
# No recognised flag columns → treat all as valid
# ---------------------------------------------------------------------------

class TestUnrecognisedColumns(unittest.TestCase):
    def test_unrelated_column_returns_all(self) -> None:
        meta = pd.DataFrame({"epoch_id": [10, 20, 30]})
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 3)), [0, 1, 2])

    def test_empty_dataframe_returns_all(self) -> None:
        meta = pd.DataFrame(index=range(4))
        self.assertEqual(get_valid_epoch_indices(_MockEpochs(meta, 4)), [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Combined bad + good flags
# ---------------------------------------------------------------------------

class TestCombinedFlags(unittest.TestCase):
    def test_bad_takes_priority(self) -> None:
        meta = pd.DataFrame({
            "is_bad_epoch":  [False, True,  False],
            "is_good_epoch": [True,  True,  False],
        })
        epochs = _MockEpochs(meta, 3)
        self.assertEqual(get_valid_epoch_indices(epochs), [0])


if __name__ == "__main__":
    unittest.main()
