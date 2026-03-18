from __future__ import annotations

import unittest

import mne
import numpy as np

from pyblinker.blinker.default_setting import DEFAULT_PARAMS, build_blink_params
from pyblinker.blinker.get_blink_positions import get_blink_position
from pyblinker.blinker.pyblinker import BlinkDetector


class TestBlinkDetectorParams(unittest.TestCase):
    def setUp(self):
        info = mne.create_info(
            ["EEG 001", "EEG 002"],
            sfreq=100.0,
            ch_types=["eeg", "eeg"],
        )
        data = np.zeros((2, 200), dtype=float)
        self.raw = mne.io.RawArray(data, info, verbose=False)

    def test_build_blink_params_copies_defaults(self):
        params = build_blink_params()
        self.assertEqual(params["std_threshold"], DEFAULT_PARAMS["std_threshold"])
        self.assertTrue(
            np.array_equal(params["z_thresholds"], DEFAULT_PARAMS["z_thresholds"])
        )
        self.assertIsNot(params["z_thresholds"], DEFAULT_PARAMS["z_thresholds"])

    def test_constructor_accepts_dict_overrides(self):
        custom_thresholds = np.array([[0.5, 0.8], [1.5, 4.0]])
        detector = BlinkDetector(
            self.raw.copy(),
            blink_params={
                "std_threshold": 2.25,
                "min_good_blinks": 4,
                "z_thresholds": custom_thresholds,
            },
        )

        self.assertEqual(detector.params["std_threshold"], 2.25)
        self.assertEqual(detector.params["min_good_blinks"], 4)
        self.assertTrue(
            np.array_equal(detector.params["z_thresholds"], custom_thresholds)
        )
        self.assertIsNot(detector.params["z_thresholds"], custom_thresholds)
        self.assertEqual(DEFAULT_PARAMS["std_threshold"], 1.50)

    def test_direct_keyword_overrides_take_precedence(self):
        detector = BlinkDetector(
            self.raw.copy(),
            blink_params={"std_threshold": 2.25, "min_good_blinks": 4},
            std_threshold=3.0,
            min_good_blinks=7,
        )

        self.assertEqual(detector.params["std_threshold"], 3.0)
        self.assertEqual(detector.params["min_good_blinks"], 7)

    def test_constructor_accepts_all_default_param_keys_as_keywords(self):
        detector = BlinkDetector(self.raw.copy(), **build_blink_params())

        self.assertEqual(set(detector.params), set(DEFAULT_PARAMS))
        for key, expected in DEFAULT_PARAMS.items():
            actual = detector.params[key]
            if isinstance(expected, np.ndarray):
                self.assertTrue(np.array_equal(actual, expected), msg=key)
                self.assertIsNot(actual, expected)
            elif key == "sfreq":
                self.assertEqual(actual, float(self.raw.info["sfreq"]))
            else:
                self.assertEqual(actual, expected, msg=key)

    def test_correlation_threshold_aliases_keep_z_thresholds_in_sync(self):
        params = build_blink_params(
            {
                "correlation_threshold_bottom": 0.82,
                "correlation_threshold": 0.97,
            }
        )

        self.assertTrue(
            np.array_equal(params["z_thresholds"][0], np.array([0.82, 0.97]))
        )
        self.assertEqual(params["correlation_threshold_top"], 0.97)

    def test_min_event_sep_override_is_respected(self):
        signal = np.zeros(40, dtype=float)
        signal[5:9] = 10.0
        signal[12:16] = 10.0

        without_sep_override = get_blink_position(
            {
                "std_threshold": 1.5,
                "min_event_len": 0.02,
                "sfreq": 100,
            },
            blink_component=signal,
            progress_bar=False,
        )
        with_sep_override = get_blink_position(
            {
                "std_threshold": 1.5,
                "min_event_len": 0.02,
                "min_event_sep": 0.05,
                "sfreq": 100,
            },
            blink_component=signal,
            progress_bar=False,
        )

        self.assertEqual(len(without_sep_override), 2)
        self.assertTrue(with_sep_override.empty)

    def test_unknown_blink_parameter_raises_type_error(self):
        with self.assertRaises(TypeError):
            BlinkDetector(self.raw.copy(), not_a_real_blink_param=123)


if __name__ == "__main__":
    unittest.main()
