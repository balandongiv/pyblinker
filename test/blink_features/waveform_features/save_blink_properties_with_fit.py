import logging
from pathlib import Path

import mne
import numpy as np

from pyblinker.utils.epochs import slice_raw_into_epochs
from pyblinker.blink_features.blink_events import generate_blink_dataframe
from pyblinker.segment_blink_properties import compute_segment_blink_properties

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "test" / "test_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load raw test file
    raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)

    # Slice into epochs
    segments, _, _, _ = slice_raw_into_epochs(
        raw,
        epoch_len=30.0,
        blink_label=None,
        progress_bar=False,
    )

    # Blink dataframe
    blink_df = generate_blink_dataframe(
        segments,
        channel="EEG-E8",
        blink_label=None,
        progress_bar=False,
    )

    # Processing parameters
    params = {
        "base_fraction": 0.5,
        "shut_amp_fraction": 0.9,
        "p_avr_threshold": 3,
        "z_thresholds": np.array([[0.9, 0.98], [2.0, 5.0]]),
    }

    # Run blink property extraction with fitting enabled
    df = compute_segment_blink_properties(
        segments,
        params,
        blink_df=blink_df,
        channel="EEG-E8",
        run_fit=True,
        progress_bar=False,
    )

    # Save DataFrame to pickle
    out_path = OUTPUT_DIR / "blink_properties_with_fit.pkl"
    df.to_pickle(out_path)

    logger.info("Saved DataFrame to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
