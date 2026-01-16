import unittest
import pandas as pd
import mne
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from pyblinker.blinker.get_blink_positions import get_blink_position


class TestCompareFitBlink(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up test by loading FIF, extracting EEG-E8, resampling to 100 Hz,
        and running get_blink_position.
        """
        # Paths
        fif_path = Path("test/test_files/ear_eog_raw.fif")
        cls.mat_path = Path("test/migration_files/step1bii_data_output_process_FitBlinks.mat")
        
        # Verify files exist
        if not fif_path.exists():
            raise FileNotFoundError(f"Missing input FIF: {fif_path}")
        if not cls.mat_path.exists():
            raise FileNotFoundError(f"Missing MATLAB output file: {cls.mat_path}")
        
        # Load raw FIF file, pick EEG-E8 channel, resample to 100 Hz
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose="ERROR")
        ch_name = "EEG-E8"
        if ch_name not in raw.ch_names:
            # Case-insensitive fallback
            lower_map = {c.lower(): c for c in raw.ch_names}
            ch_name = lower_map.get("eeg-e8", ch_name)
        
        raw = raw.copy().pick_channels([ch_name])
        if int(round(raw.info.get("sfreq", 100))) != 100:
            raw.resample(100)
        
        data = raw.get_data()
        if data.shape[0] != 1:
            raise ValueError(f"Expected single channel, got shape {data.shape}")
        
        cls.blink_comp = data[0].astype(np.float64)
        
        # Compute blink positions via pyblinker
        params = dict(min_event_len=0.05, std_threshold=1.5, sfreq=100)
        cls.df_py = get_blink_position(
            params, blink_component=cls.blink_comp, ch="No_channel", progress_bar=False
        )
        
        # Load MATLAB blinkFits using scipy.io.loadmat
        mat_data = loadmat(
            str(cls.mat_path), 
            squeeze_me=True, 
            simplify_cells=True, 
            struct_as_record=False
        )
        
        # Extract blinkFits (may be at top level or nested)
        cls.blink_fits = None
        if 'blinkFits' in mat_data:
            cls.blink_fits = mat_data['blinkFits']
        else:
            # Search in nested structures
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, dict):
                    if 'blinkFits' in value:
                        cls.blink_fits = value['blinkFits']
                        break
        
        # Ensure blinkFits was found
        if cls.blink_fits is None:
            raise KeyError(f"blinkFits not found in {cls.mat_path}")
        
        # Convert the array/list into pandas DataFrame
        try:
            cls.df_blink_fits = pd.DataFrame(cls.blink_fits)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to convert blinkFits to DataFrame. "
                f"Expected list of dicts or array-like structure, "
                f"got {type(cls.blink_fits)}: {e}"
            )
    
    def test_blink_fits_loaded(self):
        """Test that blinkFits was loaded and is non-empty."""
        self.assertIsNotNone(self.blink_fits, "blinkFits should not be None")
        self.assertTrue(len(self.blink_fits) > 0, "blinkFits should be non-empty")
    
    def test_blink_fits_dataframe(self):
        """Test that blinkFits was successfully converted to DataFrame."""
        self.assertIsInstance(self.df_blink_fits, pd.DataFrame, 
                            "blinkFits should be converted to DataFrame")
        self.assertGreater(len(self.df_blink_fits), 0, 
                          "DataFrame should have rows")
        self.assertGreater(len(self.df_blink_fits.columns), 0, 
                          "DataFrame should have columns")


if __name__ == "__main__":
    unittest.main(verbosity=2)
