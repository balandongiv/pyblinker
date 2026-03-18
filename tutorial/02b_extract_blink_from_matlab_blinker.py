"""Extract blink annotations in MNE from MATLAB Blinker (tutorial).

Overview
- End-to-end example that runs the MATLAB Blinker plugin on an EDF file, converts the
  exported blink boundaries into `mne.Annotations`, and visualizes them interactively.

What this script does
1) Creates or locates the MNE sample dataset converted to EDF via
   `test.data_setup.ensure_mne_sample_edf`.
2) Invokes MATLAB EEGLAB + Blinker (via the validation harness helper
   `src.matlab_runner.execute_blinker`) to export
   blink fits for the EDF recording.
3) Reads the exported table (`processed['blinkFits']`), extracts left/right zero-crossing
   indices, and maps them to onset/duration in seconds.
4) Attaches the annotations to the EDF `Raw` object and opens an interactive MNE plot
   with the blink intervals highlighted.

Prerequisites
- MATLAB must be installed and available on your system PATH.
- EEGLAB must be installed. Either:
  - Set environment variable `EEGLAB_ROOT` to the EEGLAB directory, or
  - Update `DEFAULT_EEGLAB_ROOT` below to a valid path on your machine.
- The Blinker plugin should be installed under EEGLAB (default plugin name here: "Blinker1.2.0").
- The validation helper checkout must be available either:
  - As a parent workspace containing `src/matlab_runner/execute_blinker.py`, or
  - Via `BLINKER_VALIDATION_ROOT=<path-to-blinker_pyblinker_validation>`.

Notes and assumptions
- The Blinker export is expected to include a table-like entry at `processed['blinkFits']` with
  columns containing left/right zero-crossing sample indices. Candidate column names tried (in order):
  - Left:  `leftZero`, `left_zero`
  - Right: `rightZero`, `right_zero`
  If none of these exist, the script raises a clear error.
- MATLAB indices are 1-based. We convert sample indices to onsets/durations in seconds using the
  EDF sampling frequency. Any off-by-one at the boundary is negligible for visualization.
- The script filters invalid intervals where right <= left and will warn if any are dropped.
- At the end, an interactive figure opens (`raw.plot(block=True, ...)`); close it to finish.

How to run (tutorial style)
- This file intentionally executes at import/run time; no `if __name__ == "__main__"` guard is used.
- Typical flow on Windows:
  1) Ensure MATLAB and EEGLAB are installed.
  2) Optionally set `EEGLAB_ROOT` in your environment to the EEGLAB folder.
  3) Run this script (e.g., from your IDE or `python tutorial/extract_blink_from_matlab_blinker.py`).

Troubleshooting
- If you see a KeyError about missing columns, check the actual columns in `processed['blinkFits']`
  and adjust `left_candidates`/`right_candidates` as needed.
- If MATLAB/EEGLAB cannot be found, verify `EEGLAB_ROOT` and your MATLAB installation.
- If the validation helper cannot be imported, verify `BLINKER_VALIDATION_ROOT`
  or run the script from the nested validation workspace layout.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, cast

from test.data_setup import ensure_mne_sample_edf
import mne

from pyblinker.utils.evaluation.dataframe_ops import pick_first_match

DEFAULT_EEGLAB_ROOT = Path(r"D:\code development\matlab_plugin\eeglab2025.1.0")


def _resolve_validation_root() -> Path:
    env_root = os.environ.get("BLINKER_VALIDATION_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        helper_path = candidate / "src" / "matlab_runner" / "execute_blinker.py"
        if helper_path.exists():
            return candidate
        raise FileNotFoundError(
            "BLINKER_VALIDATION_ROOT does not contain src/matlab_runner/execute_blinker.py: "
            f"{candidate}"
        )

    for candidate in Path(__file__).resolve().parents:
        helper_path = candidate / "src" / "matlab_runner" / "execute_blinker.py"
        if helper_path.exists():
            return candidate

    raise ModuleNotFoundError(
        "Could not locate the validation helper checkout. "
        "Set BLINKER_VALIDATION_ROOT to the blinker_pyblinker_validation repository root."
    )


VALIDATION_ROOT = _resolve_validation_root()
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from src.matlab_runner import execute_blinker


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

logging.info("Starting FIF -> EDF conversion")
edf_path = ensure_mne_sample_edf()
logging.info("Starting the Blinker")
eeglab_root_env = os.environ.get("EEGLAB_ROOT")
if eeglab_root_env:
    eeglab_root = Path(eeglab_root_env)
    logging.info("Using EEGLAB_ROOT from environment: %s", eeglab_root)
else:
    eeglab_root = DEFAULT_EEGLAB_ROOT
    logging.info(
        "EEGLAB_ROOT environment variable not set. Falling back to default path: %s",
        eeglab_root,
    )

logging.info("Running MATLAB Blinker exports")
processed = cast(
    Mapping[str, Any],
    execute_blinker.run_blinker_batch(
        edf_path=edf_path,
        eeglab_root=eeglab_root,
        project_root=execute_blinker.DEFAULT_PROJECT_ROOT,
        blinker_plugin="Blinker1.2.0",
        overwrite=True,
    ),
)

if "blinkFits" not in processed:
    raise KeyError(
        "Expected key 'blinkFits' in processed results. Present keys: "
        f"{list(processed.keys())}"
    )

df = processed["blinkFits"]

# Load the EDF
raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
sfreq = float(raw.info.get("sfreq", 0.0))
if not sfreq or sfreq <= 0:
    raise ValueError(
        "Invalid sampling frequency in EDF header (sfreq <= 0). Cannot compute annotation times."
    )

left_candidates = ["leftZero", "left_zero"]
right_candidates = ["rightZero", "right_zero"]


lkey = pick_first_match(df.columns, left_candidates)
rkey = pick_first_match(df.columns, right_candidates)

# Validate that required columns were found
if lkey is None or rkey is None:
    raise KeyError(
        "Required blink boundary columns not found in 'blinkFits'. "
        f"Tried left={left_candidates}, right={right_candidates}. Present columns: {list(df.columns)}"
    )

# Clean and validate intervals
sub = df[[lkey, rkey]].dropna().copy()
# Ensure numeric and integer-like
for k in (lkey, rkey):
    sub[k] = (sub[k].astype(float)).round().astype(int)

# Filter invalid intervals (right must be strictly greater than left)
pre_n = len(sub)
sub = sub[sub[rkey] > sub[lkey]]
removed = pre_n - len(sub)
if removed:
    logging.warning("Dropped %d invalid intervals where %s <= %s", removed, rkey, lkey)

if len(sub) == 0:
    raise ValueError(
        "No valid blink intervals remaining after cleaning/filtering. Check the input data and columns."
    )

# MATLAB indices are 1-based; convert sample indices to seconds relative to start
onsets = (sub[lkey].to_numpy() - 1) / sfreq
# Duration in seconds; using difference is fine (off by one sample is negligible)
durations = (sub[rkey].to_numpy() - sub[lkey].to_numpy()) / sfreq
descriptions = ["blink"] * len(sub)

ann = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
raw.set_annotations(ann)
logging.info("Attached %d blink annotations to raw", len(ann))

raw.plot(block=True, title="Blink Annotations from MATLAB Blinker")
