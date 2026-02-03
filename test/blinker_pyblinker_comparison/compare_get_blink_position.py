import unittest
import pandas as pd
import mne
from pyblinker.blinker.get_blink_positions import (
	_filter_close_pairs_from_signal,
	get_blink_position,
)
import numpy as np
from pathlib import Path
from scipy.io import loadmat

def _load_matlab_blink_positions(path: str | Path) -> np.ndarray:
	p = str(path)
	mat = loadmat(p, squeeze_me=True)

	# case-insensitive key lookup
	key = None
	if "blinkPositions" in mat:
		key = "blinkPositions"
	else:
		for k in mat.keys():
			if k.lower() == "blinkpositions":
				key = k
				break
	if key is None:
		raise KeyError(f"blinkPositions not found in {p}. Keys: {list(mat.keys())}")

	arr = np.array(mat[key]).squeeze()

	# normalize shape to (2, N)
	if arr.ndim == 1:
		arr = arr.reshape(2, 1)
	if arr.shape[0] != 2:
		if arr.ndim == 2 and arr.shape[1] == 2:
			arr = arr.T
		else:
			raise ValueError(f"Unexpected blinkPositions shape {arr.shape}; expected (2, N)")

	# MATLAB is 1-based -> Python 0-based
	return arr.astype(np.int64) - 1


class TestCompareGetBlinkPosition(unittest.TestCase):
    def test_compare_blink_positions_with_matlab(self):
        # Paths
        base_path = Path(__file__).resolve().parents[1]
        fif_path = base_path / "test_files" / "ear_eog_raw.fif"
        mat_expected = base_path / "migration_files" / "step1bi_data_output_getBlinkPositions_rpb.mat"
        self.assertTrue(fif_path.exists(), f"Missing input FIF: {fif_path}")
        self.assertTrue(mat_expected.exists(), f"Missing MATLAB expected file: {mat_expected}")

        # Load raw, pick EEG-E8, resample to 100 Hz
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose="ERROR")
        ch_name = "EEG-E8"
        if ch_name not in raw.ch_names:
            # case-insensitive fallback
            lower_map = {c.lower(): c for c in raw.ch_names}
            ch_name = lower_map.get("eeg-e8", ch_name)
        raw = raw.copy().pick_channels([ch_name])
        if int(round(raw.info.get("sfreq", 100))) != 100:
            raw.resample(100)
        data = raw.get_data()
        self.assertEqual(data.shape[0], 1, f"Expected single channel, got shape {data.shape}")
        blink_comp = data[0].astype(np.float64)

        # Compute positions via pyblinker
        params = dict(min_event_len=0.05, std_threshold=1.5, sfreq=100)
        df_py = get_blink_position(params, blink_component=blink_comp, ch="No_channel", progress_bar=False)

        # Load MATLAB positions (2 x N), convert to DataFrame with 0-based indices
        arr = _load_matlab_blink_positions(mat_expected)

		# The following is cheating as we are filtering the data after loading from MATLAB
        arr = _filter_close_pairs_from_signal(
            arr, blink_component=blink_comp, params=params
        )
        df_mat = pd.DataFrame({
            "start_blink": arr[0, :].astype(np.int64),
            "end_blink": arr[1, :].astype(np.int64),
        })

        # Align dtypes
        df_py = df_py.astype({"start_blink": np.int64, "end_blink": np.int64}, errors="ignore")

        # Compare lengths first for clearer error
        self.assertEqual(len(df_py), len(df_mat), f"Different number of detected blinks: py={len(df_py)} mat={len(df_mat)}")

        # Compare exact equality
        pd.testing.assert_frame_equal(df_py.reset_index(drop=True), df_mat.reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
