# Blink Segmentation and Refinement

## Purpose
Once a candidate blink region is identified, `pyblinker` refines the start and end points to ensure consistent feature extraction. Accurate segmentation is critical because metrics like "duration", "closing speed", and "amplitude" depend entirely on these boundaries.

## Segmentation Strategies
Different modalities and research questions require different definitions of "start" and "end". `pyblinker` supports several strategies, specified during the refinement step:

*   **`"base"`**: Uses the "outer" limits, capturing the full deviation from baseline before the main rise/fall.
*   **`"zero"`**: Uses **zero-crossing** (or threshold-crossing) points. This is the standard definition for blink duration in many EOG studies.
*   **`"tent"`**: Fits a tent-like shape (peak and linear slopes) to define the event.
*   **`"half_base"` / `"half_zero"`**: Defines boundaries at 50% of the peak amplitude relative to the base or zero-crossing level (Full Width at Half Maximum - FWHM).
*   **`"threshold_interpolation"` (EAR specific)**: Finds the precise time where the signal crosses a specific EAR threshold, using interpolation to achieve sub-sample accuracy.

## Refinement Process
Refinement typically happens when converting continuous data into epochs (`slice_raw_into_mne_epochs_refine_annot`).

1.  **Input**: Raw signal and candidate blink regions (coarse onsets/offsets).
2.  **Search**: For each candidate, the algorithm searches within a small window (expanding if necessary) for the precise landmarks defined by the strategy.
3.  **Result**: New metadata fields (e.g., `blink_onset_ear`, `refined_start_sample`, `interpolated_closing_slope`) are attached to the epochs.

### Tuning Parameters
Users can tune the refinement behavior, especially for EAR:
*   **Threshold**: The EAR value defining "closed" vs. "open".
*   **Extension**: How far to search outside the candidate region if the crossing isn't found immediately.

### Flowchart

```mermaid
graph TD
    A[Candidate Blink Region] --> B{Select Strategy}
    B -- Zero-Crossing --> C[Find Zero/Baseline Crossings]
    B -- Base --> D[Find Deviation from Baseline]
    B -- Threshold --> E[Find Value Crossings]
    E --> F[Interpolate Exact Time]
    C --> G[Refined Start/End]
    D --> G
    F --> G
    G --> H[Update Epoch Metadata]
```

## Single-channel modality configuration

`slice_raw_into_mne_epochs_refine_annot` now requires an explicit, single-channel selection per modality. The helper `_prepare_epochs_and_modalities` in `pyblinker/segmentation/refinement.py` enforces:

* **EAR**: A `"channel"` entry is mandatory. Missing/empty values or multiple matches raise `ValueError`.
* **EEG/EOG**: Optional. Supplying no channel disables refinement for that modality; supplying an invalid or non-unique channel raises `ValueError`.
* Each enabled modality extracts epoch data with `epochs.get_data(picks=[idx])`, so downstream refinement receives a 1D vector per epoch—no implicit averaging across channels.
* Blink annotations are filtered by `blink_label` prior to per-epoch refinement; onsets/durations stay in seconds, while epoch-local bounds remain in samples.

### EAR threshold configuration
* `slice_raw_into_mne_epochs_refine_annot` no longer accepts an `ear_threshold` convenience argument; EAR thresholds must be defined inside the segmentation config under the `"ear"` key (including `seg_type="threshold_interpolation"` and extension/padding settings) before calling the helper. The configuration in `tutorial/5_ear_energy_feature_tutorial.py` shows the full set of recommended EAR parameters alongside optional EEG/EOG entries.
* Code: `pyblinker/segmentation/refinement.py` (`_prepare_segmentation_config`, `slice_raw_into_mne_epochs_refine_annot`).
* Unit tests: `test/blink_feature_ear/energy/test_energy_features.py` builds the explicit EAR segmentation config when refining epochs for energy metrics.

### Verification
* Code: `pyblinker/segmentation/refinement.py` (`_prepare_epochs_and_modalities`, `_refine_epoch_modalities`, `slice_raw_into_mne_epochs_refine_annot`).
* Tutorials: `tutorial/05c_minimal_blink_feature_tutorial.py`, `tutorial/5_ear_energy_feature_tutorial.py`.
* Tests: `test/segmentation/test_ear_refinement_outputs.py`, `test/segmentation/test_refine_annot_by_channel.py`.

## Related Code

*   **`pyblinker/segmentation/refinement.py`**: The core module for refinement. Contains `slice_raw_into_mne_epochs_refine_annot` and shared peak-refinement helpers.
*   **`pyblinker/segmentation/ear.py`**: EAR-specific interpolation helpers used by the segmentation pipeline.
*   **`pyblinker/fitutils/`**: Contains utility functions for fitting shapes and finding crossings (e.g., `ear_crossing.py`).
*   **`pyblinker/blink_features/ear_metrics/refinement.py`**: Implementation of EAR-specific refinement logic used by the segmentation helpers.

## Tutorials

*   **`tutorial/03b_ear_threshold_crossing_tutorial.py`**:
    A detailed, executable exploration of the "threshold interpolation" strategy. It generates synthetic EAR signals and visualizes exactly how the "left crossing", "minimum", and "right crossing" are calculated using linear interpolation between samples.
*   **`tutorial/03a_ear_threshold_blink_refinement.py`**:
    Shows how to apply the refinement logic to real data. It demonstrates the effect of changing the `threshold` parameter on the detected blink duration.
*   **`tutorial/03c_ear_threshold_multi_candidate_refinement.py`**:
    Covers complex scenarios: what happens when a single candidate region actually contains two distinct blinks (double blink)? This tutorial demonstrates the logic that splits or selects the appropriate event.
*   **`tutorial/03d_understand_diff_in_blink_position.py`**:
    A comparative script that runs different segmentation strategies (e.g., "zero-crossing" vs. "50% recovery") on the same blink, printing the start/end times side-by-side to highlight the impact of the chosen definition.

## Unit Tests

*   **`test/segmentation/test_ear_refinement_outputs.py`**:
    Verifies that EAR-based epoch slicing and multi-threshold refinement outputs match the stored FIF and CSV reference artifacts.
*   **`test/test_refined_blink_flow.py`**:
    The primary integration test for the refinement module. It mocks a user workflow: Input coarse annotations -> Run Refinement -> Check output metadata.
*   **`test/test_ear_threshold_refinement.py`**:
    Tests the robustness of the threshold search. It includes test cases for edge conditions: signals that barely cross the threshold, signals that cross multiple times, and signals that start below the threshold.
*   **`test/test_ear_crossing.py`**:
    Validates the low-level math functions (like `find_threshold_crossing_triplet`). It ensures that the sub-sample interpolation is mathematically correct.
*   **`test/segmentation/test_refine_annot_by_channel.py`**:
    Verifies that refinement can be applied independently to different channels (e.g., refining an EEG blink using EEG data while simultaneously refining an EAR blink using video data) without cross-contamination.
