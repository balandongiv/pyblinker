# Kinematics vs Waveform Blink Feature Calculations (EEG)

## Executive Summary
`KinematicBlinkFeatureExtractor` does **not** produce the requested EEG morphology/timing fields (durations, shut times, peak times/amplitudes, per-side amp/velocity ratios, or tent/zero closing/reopening fields) as direct outputs; it emits aggregated per-epoch statistics over a smaller set of kinematic stems. Most requested fields are therefore **Not Available** in its output schema, and only limited approximations are partially inferable from aggregated metrics. The waveform pathway (`BlinkProperties` in `extract_blink_properties.py`) does use the shared morphology and kinematic helper functions listed in this task. By contrast, the kinematics extractor path used by `test_kinematics_eeg_only_config.py` calls `compute_segment_kinematics -> compute_blink_kinematic_metrics` and does **not** call the listed morphology core functions nor `normalize_modality` in that execution path.

## Scope and Trace

### Files inspected
- `pyblinker/blink_features/kinematics/kinematic_features.py`
- `pyblinker/blink_features/kinematics/per_blink.py`
- `pyblinker/blink_features/kinematics/core_metrics.py`
- `pyblinker/blink_features/waveform_features/extract_blink_properties.py`
- `pyblinker/blink_features/morphology/core_metrics.py`
- `pyblinker/blink_features/_blink_metrics_shared.py`
- `test/blink_features/kinematics/test_kinematics_eeg_only_config.py`
- `test/blinker_pyblinker_comparison/test_c_BlinkProperties.py`

### Call flow traced (kinematics test path)
1. `test_kinematics_eeg_only_config.py` constructs `KinematicBlinkFeatureExtractor` and calls `compute(...)`.
2. `KinematicBlinkFeatureExtractor.compute` slices each blink window and calls `compute_segment_kinematics(...)`.
3. `compute_segment_kinematics` delegates to `compute_blink_kinematic_metrics(...)`.
4. Extractor aggregates per-blink values into `mean/std/cv` columns.

### Call flow traced (waveform comparison path)
1. `test_c_BlinkProperties.py` constructs `BlinkProperties(...)`.
2. `BlinkProperties` pipeline calls:
   - morphology: `compute_blink_durations`, `compute_time_zero_shut`, `compute_time_base_shut`, `compute_blink_peak_times`
   - kinematics: `compute_blink_velocity`, `compute_amp_vel_ratio_zero_to_max`, `compute_amp_vel_ratio_base`, `compute_amp_vel_ratio_tent`, `compute_inter_blink_max_vel`
   - modality normalization: `normalize_modality`

## Availability Table for Requested Fields

Legend:
- **Available** = explicit output field from `KinematicBlinkFeatureExtractor.compute` result columns.
- **Derivable** = can be estimated from available outputs/intermediates without re-running full waveform pipeline (often lossy due to aggregation).
- **Not Available** = neither emitted nor reasonably reconstructable from extractor outputs.

| Requested field | Status | Source (if any) | Notes |
|---|---|---|---|
| `duration_base` | Not Available | N/A in kinematics output schema | Duration metrics are produced in waveform `compute_blink_durations`, not in kinematics extractor columns. |
| `duration_zero` | Not Available | N/A | Same as above. |
| `duration_tent` | Not Available | N/A | Same as above. |
| `duration_half_base` | Not Available | N/A | Same as above. |
| `duration_half_zero` | Not Available | N/A | Same as above. |
| `time_shut_base` | Not Available | N/A | Computed only in waveform `compute_time_base_shut`. |
| `time_shut_zero` | Not Available | N/A | Computed only in waveform `compute_time_zero_shut`. |
| `time_shut_tent` | Not Available | N/A | Computed only in waveform `compute_time_base_shut` (tent branch). |
| `closing_time_zero` | Not Available | N/A | Computed only in waveform `compute_time_zero_shut`. |
| `reopening_time_zero` | Not Available | N/A | Computed only in waveform `compute_time_zero_shut`. |
| `closing_time_tent` | Not Available | N/A | Computed only in waveform `compute_time_base_shut` (fitted branch). |
| `reopening_time_tent` | Not Available | N/A | Computed only in waveform `compute_time_base_shut` (fitted branch). |
| `peak_time_blink` | Not Available | N/A | Computed only in waveform `compute_blink_peak_times`. |
| `peak_time_tent` | Not Available | N/A | Computed only in waveform `compute_blink_peak_times`. |
| `inter_blink_max_amp` | Not Available | N/A | Computed only in waveform `compute_blink_peak_times`. |
| `inter_blink_max_vel_base` | Derivable (limited) | `...__kinematic__inter_blink_max_vel_<stat>__<ch>` for style `base` | Only aggregated by epoch (`mean/std/cv`), not per-blink `inter_blink_max_vel_base`. Lossy approximation. |
| `inter_blink_max_vel_zero` | Derivable (limited) | same metric stem under style `zero` (if metadata includes zero style) | Aggregated only; not the waveform per-blink field. |
| `peak_max_blink` | Not Available | N/A | No peak-amplitude output field in kinematic extractor schema. |
| `peak_max_tent` | Not Available | N/A | No tent peak field in kinematic extractor schema. |
| `neg_amp_vel_ratio_base` | Not Available | N/A | Kinematic extractor emits averaged `amp_vel_ratio_base`, not side-specific pos/neg fields. |
| `pos_amp_vel_ratio_base` | Not Available | N/A | Same as above. |
| `neg_amp_vel_ratio_zero` | Not Available | N/A | Kinematic extractor emits averaged `amp_vel_ratio_zero_to_max`. |
| `pos_amp_vel_ratio_zero` | Not Available | N/A | Same as above. |
| `neg_amp_vel_ratio_tent` | Not Available | N/A | Kinematic extractor emits averaged `amp_vel_ratio_tent`. |
| `pos_amp_vel_ratio_tent` | Not Available | N/A | Same as above. |
| `aver_left_velocity` | Not Available | N/A | Internal intermediate in waveform fitting context; not emitted by kinematic extractor. |
| `aver_right_velocity` | Not Available | N/A | Internal intermediate in waveform fitting context; not emitted by kinematic extractor. |

## Missing Fields (Not Available)
- `duration_base`
- `duration_zero`
- `duration_tent`
- `duration_half_base`
- `duration_half_zero`
- `time_shut_base`
- `time_shut_zero`
- `time_shut_tent`
- `closing_time_zero`
- `reopening_time_zero`
- `closing_time_tent`
- `reopening_time_tent`
- `peak_time_blink`
- `peak_time_tent`
- `inter_blink_max_amp`
- `peak_max_blink`
- `peak_max_tent`
- `neg_amp_vel_ratio_base`
- `pos_amp_vel_ratio_base`
- `neg_amp_vel_ratio_zero`
- `pos_amp_vel_ratio_zero`
- `neg_amp_vel_ratio_tent`
- `pos_amp_vel_ratio_tent`
- `aver_left_velocity`
- `aver_right_velocity`

## Logic Comparison: Kinematics vs Waveform

### Short answer
They are **not fully the same execution logic**.

### Shared vs different function usage

#### Used in waveform extraction (`BlinkProperties`)
- `normalize_modality`
- `compute_blink_durations`
- `compute_blink_peak_times`
- `compute_time_base_shut`
- `compute_time_zero_shut`
- `compute_blink_velocity`
- `compute_inter_blink_max_vel`
- `compute_amp_vel_ratio_base`
- `compute_amp_vel_ratio_tent`
- `compute_amp_vel_ratio_zero_to_max`

#### Used in kinematics extractor path (`KinematicBlinkFeatureExtractor`)
- `compute_segment_kinematics`
- `compute_blink_kinematic_metrics`

Notably, this path does **not** call the morphology core functions above and does **not** call `normalize_modality` in the extractor execution chain.

### Naming and semantic mismatches
- Waveform outputs explicit per-blink side-specific ratio fields (`pos_*`, `neg_*`) and timing/duration fields.
- Kinematics extractor returns style-suffixed kinematic stems aggregated to epoch-level stats (`mean/std/cv`).
- `amp_vel_ratio_zero_to_max` in kinematic metrics is a single averaged value, not separated into pos/neg as in waveform output.
- `inter_blink_max_vel` in kinematic extractor is not exposed as waveform-style `inter_blink_max_vel_base` / `inter_blink_max_vel_zero` per blink; only style-specific aggregated stats can approximate those.

### Landmark definitions
- Waveform duration/timing computations explicitly use landmark columns (`left_base/right_base`, `left_zero/right_zero`, `left_x_intercept/right_x_intercept`, etc.).
- Kinematics extractor segments by metadata windows for each style and computes segment-level metrics; it does not compute the waveform timing landmark outputs themselves.

## Confirmation for `test_kinematics_eeg_only_config.py`

### Requested confirmation: shared metric functions exercised?
For the EEG-only kinematics test path, the following applies:

- `pyblinker.blink_features.kinematics.core_metrics`
  - `compute_blink_velocity`: **Not directly exercised** by this test path.
  - `compute_inter_blink_max_vel`: **Not directly exercised** by this test path.
  - `compute_amp_vel_ratio_base`: **Not directly exercised** by this test path.
  - `compute_amp_vel_ratio_tent`: **Not directly exercised** by this test path.
  - `compute_amp_vel_ratio_zero_to_max`: **Not directly exercised** by this test path.

- `pyblinker.blink_features.morphology.core_metrics`
  - `compute_blink_durations`: **Not exercised**.
  - `compute_blink_peak_times`: **Not exercised**.
  - `compute_time_base_shut`: **Not exercised**.
  - `compute_time_zero_shut`: **Not exercised**.

- `pyblinker.blink_features._blink_metrics_shared.normalize_modality`
  - **Not exercised** in the kinematics extractor call chain used by this test.

The test does validate presence of kinematic metric columns (`amp_vel_ratio_base`, `amp_vel_ratio_tent`, `amp_vel_ratio_zero_to_max`, `blink_velocity`, `inter_blink_max_vel`), but those values come through `compute_blink_kinematic_metrics` rather than the listed per-blink helper functions.

## Evidence / Trace Notes
- `KinematicBlinkFeatureExtractor.compute` builds output columns from `KINEMATIC_METRIC_STEMS` and aggregates to `mean/std/cv`, then calls `compute_segment_kinematics` per window.
- `compute_segment_kinematics` delegates to `compute_blink_kinematic_metrics`.
- `compute_blink_kinematic_metrics` computes compact segment-level kinematic values (including averaged amp/vel ratio fields) and returns style-suffixed metric keys.
- `BlinkProperties` imports and calls both morphology core functions and the listed kinematic helper functions, and applies `normalize_modality`.
- The comparison test (`test_c_BlinkProperties.py`) asserts waveform columns including durations, shut times, peak timing/amplitude, and pos/neg amp-velocity ratio fields, confirming those belong to waveform extraction outputs.
