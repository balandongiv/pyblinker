"""Unit tests for channel modality inference helpers."""

import unittest

from pyblinker.utils.modality import infer_modality


class TestInferModality(unittest.TestCase):
    """Validate the ``infer_modality`` utility function."""

    def test_prefix_based_inference(self) -> None:
        """Channels with hyphen prefixes resolve to the prefix label."""

        self.assertEqual(infer_modality("EEG-E8"), "eeg")
        self.assertEqual(infer_modality("EOG-EEG-eog_vert_left"), "eog")
        self.assertEqual(infer_modality("EAR-Left"), "ear")

    def test_keyword_detection_without_separator(self) -> None:
        """Keyword lookup handles channels without separators."""

        self.assertEqual(infer_modality("EEG001"), "eeg")
        self.assertEqual(infer_modality("earclip"), "ear")

    def test_fallback_lowercase(self) -> None:
        """Unknown channel names are lowercased as a fallback."""

        self.assertEqual(infer_modality("Fp1"), "fp1")

    def test_handles_whitespace_and_empty(self) -> None:
        """Whitespace is ignored and empty names produce an empty string."""

        self.assertEqual(infer_modality("  EEG-E8  "), "eeg")
        self.assertEqual(infer_modality(""), "")


if __name__ == "__main__":  # pragma: no cover - convenience
    unittest.main()
