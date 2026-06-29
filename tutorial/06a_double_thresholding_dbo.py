"""Double-thresholding blink detection (``blink_position_strategy_dbo``) tutorial.

Self-contained demonstration of the migrated double-thresholding detector using
only the standalone ``pyblinker`` package (plus the external ``autoreject`` and
``blink_evaluation`` packages) — no dependency on the parent
``blink_detection_llm`` repository.

Two-stage thresholding:
  Stage A  Autoreject identifies which epochs are likely blink-heavy.
  Stage B  A per-channel sample-level threshold is estimated from those
           flagged epochs using ``center + k * MAD`` robust statistics.
           The center can be the median (default, more robust) or the mean
           (more sensitive to large peaks, more conservative threshold).
  Stage C  Blink regions are located via ``scan_threshold_crossings_kleifges``
           using the Stage B threshold.

This mirrors ``tutorial/10d_strategy_autoreject_drop_threshold.py`` in the parent
repo and reproduces the same results documented in
``tutorial/output_10d_strategy.md``.

Requirements (all pip-installable, no parent repo needed)::

    pip install pyblinker[double-thresholding]
    pip install git+https://github.com/balandongiv/blink_evaluation.git
"""

import logging
from pathlib import Path

import mne

from blink_evaluation import evaluate_channels, load_ground_truth_annotations
from pyblinker.double_thresholding import blink_position_strategy_dbo
from pyblinker.epoch_detection import (
    get_valid_epoch_indices,
    prepare_epoch_detection_input,
)
from pyblinker.io.eeg_channels import (
    load_brain_region_channels,
    load_raw_with_brain_channels,
)

logger = logging.getLogger(__name__)

TUTORIAL_DIR = Path(__file__).resolve().parent

FIF_PATH = Path(
    r"D:\dataset\drowsy_driving_raja_processed\S1\S01_20170519_043933\seg_data_raw\eeg_eog_raw.fif"
)
CSV_PATH = Path(
    r"D:\dataset\drowsy_driving_raja\human_label_annotation_eeg\S1\S01_20170519_043933\ear_eog.csv"
)
BRAIN_REGION_YAML = TUTORIAL_DIR / "brain_region.yaml"
EPOCH_DURATION_S = 30.0
FILTER_LOW = 1.0
FILTER_HIGH = 20.0
RESAMPLE_RATE = 100

# Stage A: autoreject settings
AUTOREJECT_RANDOM_STATE = 42
MIN_FLAGGED_EPOCHS = 1          # fall back when fewer flagged epochs are found

# Stage B: robust threshold settings
STD_THRESHOLD = 3.5             # k in: threshold = center + k * (1.4826 * MAD)
CENTER_METHOD = "median"        # "median" (robust, detects more blinks) or
                                # "mean"   (pulled by peaks, more conservative)

VERBOSE = True                  # print Stage A/B diagnostic lines


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("=== blink_position_strategy_dbo  center_method=%s ===", CENTER_METHOD)
    brain_channels = load_brain_region_channels(BRAIN_REGION_YAML)
    raw = load_raw_with_brain_channels(FIF_PATH, brain_channels)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_DURATION_S, preload=True, verbose="ERROR"
    )
    prepared = prepare_epoch_detection_input(
        epochs,
        pick_types_options={"eeg": True},
        filter_low=FILTER_LOW,
        filter_high=FILTER_HIGH,
        resample_rate=RESAMPLE_RATE,
    )
    # It is possible to drop epochs before we proceed with any of the pipeline
    valid_epoch_indices = get_valid_epoch_indices(epochs)

    setting = {
        "autoreject_random_state": AUTOREJECT_RANDOM_STATE,
        "std_threshold": STD_THRESHOLD,
        "center_method": CENTER_METHOD,
        "min_flagged_epochs": MIN_FLAGGED_EPOCHS,
        "verbose": VERBOSE,
    }
    channel_results = blink_position_strategy_dbo(
        prepared,
        valid_epoch_indices,
        setting=setting,
    )

    gt_annotations = load_ground_truth_annotations(CSV_PATH, EPOCH_DURATION_S)

    scored = evaluate_channels(
        channel_results,
        gt_annotations,
        epoch_duration=EPOCH_DURATION_S,
    )

    best = scored.best_channel_result
    print(f"\nn_flagged_epochs={best['n_flagged']}  used_all_epochs={best['used_all_epochs']}")
    print(
        f"blink_region_threshold={best['blink_region_threshold']:.6f}  "
        f"center={best['threshold_center']:.6f}  "
        f"dispersion={best['threshold_dispersion']:.6f}"
    )

    em = scored.best_eval_result.event_metrics
    print(f"\nbest_channel={scored.best_channel}")
    print(f"tp={em.tp}  fp={em.fp}  fn={em.fn}")
    print(f"precision={em.precision:.4f}  recall={em.recall:.4f}  f1={em.f1:.4f}")
    print(f"\n=== Lane Summary (top 10) ===")
    print(scored.lane_summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
