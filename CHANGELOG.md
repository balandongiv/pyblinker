# Changelog

## [Unreleased]

## [0.4.1] - 2026-03-18

### Added
- Add `pyblinker.utils.evaluation.event_tables.prepare_event_tables(...)` so PyBlinker comparison tests can prepare MATLAB-vs-PyBlinker event tables without importing from the external validation harness.

### Changed
- Align `tutorial/01a_basic_usage.py` with the explicit legacy-default blink-parameter profile used in validation experiments, including `min_good_blinks = 10`.
- Resolve `tutorial/02b_extract_blink_from_matlab_blinker.py` against the sibling validation workspace instead of a vendored `src/` copy.

### Removed
- Remove the stale validation-harness duplicate tree under `pyblinker/src/`.

## [0.3.9] - 2026-03-06

### Added
- Add `tutorial/05a_all_channel_feature_aggregation_tutorial.py` documenting combined EAR/EEG/EOG aggregation of blink-count, energy, frequency-domain, morphology, and kinematic features in a single per-epoch DataFrame workflow.
- Extend epoch-level legacy morphology aggregation to compute `mean`, `std`, and `cv` for each legacy metric via the shared `_safe_stats` convention.

### Changed
- Refactor energy feature extraction orchestration to mirror morphology-style epoch/modality/channel delegation (`compute_energy_features -> _compute_epoch_record -> _compute_channel_record`), including per-modality style-window planning reused across channels.
- Refactor kinematics epoch aggregation to mirror morphology’s `compute -> _compute_epoch_record -> _compute_channel_record` call graph, including modality/style planning helpers and single per-channel/per-epoch blink-frame construction reused across style stats.
- Refactor energy, kinematics, and morphology feature schema/header generation into dedicated `column_headers.py` modules backed by a shared `blink_features/utils/column_headers_common.py` helper, while preserving legacy column names/order/alias behavior and existing feature computations.
- Refactor feature-family compute setup to use shared epoch-context utilities (`_epoch_context.py`) across energy, frequency-domain, and kinematics extractors.
- Remove temporary `test/blink_features/test_shared_refactor_helpers.py` helper script and keep validation through the canonical full-suite runner.
- Remove onset/duration-style discovery from shared style helpers and standardize on frame-based start/end metadata windows.
- Extract kinematics low-level numeric helper routines into `pyblinker/blink_features/kinematics/helpers.py` and consume them from the epoch extractor.
- Align EAR-only kinematics expected columns with threshold-interpolation landmarks (`th_interpolation`, `th_point`, and legacy ` interpolated_threshold`) and assert the full generated metric set via `build_expected_metrics`.
- Correct EAR-only kinematics expected-metric schema to use `modality="ear"` in `build_expected_metrics`, matching EAR feature column prefixes.
- Refactor EAR-only kinematics test to define `REQUIRED_KINEMATICS_METRICS` via `build_expected_metrics` and shared/style-suffixed metric generation for parity with EEG/EOG kinematics tests.
- Refactor EOG-only kinematics expected metric assertions to use `build_expected_metrics` with programmatic per-landmark shared/style-suffixed metric generation for `base`, `tent`, and `zero`.
- Refactor EEG-only kinematics test expected metric construction to generate shared and style-suffixed metrics programmatically, reducing duplication while preserving expected columns.
- Refactor epoch-level energy feature aggregation to use modality-aware style segmentation windows (EEG/EOG style landmarks and EAR threshold-interpolation windows) instead of deprecated onset/duration windowing.
- Refactor frequency-domain wavelet epoch aggregation to use channel-aware frame-based segmentation windows (EEG/EOG style landmarks plus `half`/`peak` aliases, and EAR threshold interpolation mapped to `th_point`) instead of deprecated onset/duration windows.
- Rename aggregated legacy morphology outputs to the new fully qualified naming format (`{modality}__{style}__morphology__{metric}_{stat}__{channel}`) for EEG/EOG legacy metric exports.
- Update morphology integration tests to assert the new legacy naming convention with mean/std/cv variants in full-modality and EEG-only coverage.
- Add shared refactor scaffolding for blink features: `BlinkerConfig`/`DEFAULT_CONFIG`, shared constants, style-window helpers, and a reusable compute skeleton module.
- Preserve legacy DataFrame-column compatibility by forcing object-typed column indexes in feature outputs used by baseline regression pickles.
- Harden optional wavelet dependency behavior so missing `PyWavelets` now degrades gracefully with warnings and NaN outputs instead of import-time failure.
- Align morphology epoch feature extraction with shared `_epoch_context` setup utilities and deduplicate energy/frequency-domain window parsing through new `pyblinker/blink_features/_style_windows.py`.

### Fixed
- Drop blink rows with NaN boundary indices before amplitude-velocity ratio computation to prevent `ValueError: cannot convert float NaN to integer` and safely skip malformed blink candidates.
- Restore `blink_count` epoch aggregation implementation for modality-specific metadata columns so blink event feature extraction imports and executes correctly again.
- Remove unused `frame_from_records` import in kinematics feature extraction to satisfy Ruff F401 lint checks.
- Restore baseline-compatible EAR style-window precedence for frequency-domain aggregation (`th_interpolation` preferred over `th_point`) and restore morphology style discovery behavior in epoch aggregation after the shared-context refactor.
- Remove style-priority window fallback from morphology blink-landmark frame building so `_build_blink_landmark_frame` no longer derives `max_blink`/`max_value` from style windows when no explicit peak metadata is present.
- Emit an actionable runtime warning when required legacy morphology columns are missing from `blink_df`, including the missing column names and guidance to review `_REQUIRED_LEGACY_MORPHOLOGY_METRICS`.

### Removed
- Remove deprecated `pyblinker/outside_annotation/` modules from the core package.
- Remove temporary morphology regression test `test_morphology_refine_window_selection.py` and rely on existing morphology suite coverage.


## [0.3.8] - 2026-02-28

### Fixed
- Restore epoch blink-count feature extraction in `blink_count` for channel picks by computing per-modality counts from refined start landmarks when present, then falling back to modality/generic blink onset metadata; also emit the legacy `"<modality>__ncount__<channel>"` output expected by event-feature regression tests while preserving `blink_count_<modality>` columns for aggregation.

## [0.3.7] - 2026-02-28

### Fixed
- Update epoch blink counting to prefer per-epoch `n_blinks` totals when available and to count modality-specific blink totals from refined start landmarks (`start__th_point__ear`, `start__refine__eeg`, `start__refine__eog`) before falling back to onset/duration windows.

## [0.3.6] - 2026-02-26

### Fixed
- Avoid `TypeError` in morphology epoch aggregation when a modality/channel has no detected blinks by returning an empty legacy-metrics mapping instead of `None`.

## [0.3.5] - 2026-02-25

### Fixed
- Correct EEG-only kinematics expected-column generation in `test_kinematics_eeg_only_config.py` by mapping style-specific metrics to the appropriate landmarks (`base`, `tent`, and `zero`) so the test no longer asserts impossible column combinations.

## [0.3.4] - 2026-02-24

### Fixed
- Emit legacy EOG morphology metrics even when EEG channels are also present, restoring expected combined-modality morphology columns in `test_morphology_ear_eeg_eog.py`.

## [0.3.3] - 2026-02-24

### Fixed
- Preserve the original EOG channel label casing in epoch energy and frequency-domain feature column names, while keeping non-EOG channel labels uppercased for backward-compatible EAR/EEG outputs.

## [0.3.2] - 2026-02-23

### Fixed
- Replace deprecated MNE `pick_channels(...)` usage with `pick(...)` in blinker-vs-pyblinker comparison tests to remove legacy warning noise during unittest execution (including `test_b_fitblink`).

## [0.3.1] - 2026-02-22

### Fixed
- Complete the EEG morphology Excel regression test by comparing value-by-value against `test/blink_features/morphology/expected_output_new_naming.xlsx`, including explicit handling for duplicated/mislabeled `peak_time_tent` fixture headers.
- Add `openpyxl` to `requirements-dev.txt` so Excel-based regression tests run reliably in development and CI environments.

## [0.3.0] - 2026-02-22

### Added
- Extend epoch-level legacy morphology aggregation to compute `mean`, `std`, and `cv` for each legacy metric via the shared `_safe_stats` convention.

### Changed
- Refactor energy, kinematics, and morphology feature schema/header generation into dedicated `column_headers.py` modules backed by a shared `blink_features/utils/column_headers_common.py` helper, while preserving legacy column names/order/alias behavior and existing feature computations.
- Refactor feature-family compute setup to use shared epoch-context utilities (`_epoch_context.py`) across energy, frequency-domain, and kinematics extractors.
- Remove temporary `test/blink_features/test_shared_refactor_helpers.py` helper script and keep validation through the canonical full-suite runner.
- Remove onset/duration-style discovery from shared style helpers and standardize on frame-based start/end metadata windows.
- Extract kinematics low-level numeric helper routines into `pyblinker/blink_features/kinematics/helpers.py` and consume them from the epoch extractor.
- Rename aggregated legacy morphology outputs to the new fully qualified naming format (`{modality}__{style}__morphology__{metric}_{stat}__{channel}`) for EEG/EOG legacy metric exports.
- Update morphology integration tests to assert the new legacy naming convention with mean/std/cv variants in full-modality and EEG-only coverage.

### Fixed
- Emit an actionable runtime warning when required legacy morphology columns are missing from `blink_df`, including the missing column names and guidance to review `_REQUIRED_LEGACY_MORPHOLOGY_METRICS`.

## [0.2.2] - 2026-02-20

### Fixed
- Prevent pytest from collecting the comparison-test path helper as a test by renaming `test_file_path(...)` to `get_test_file_path(...)` in `test/blinker_pyblinker_comparison/utils.py`.
- Update all blinker-vs-pyblinker comparison tests to import and call `get_test_file_path(...)`, preserving both per-file pytest execution and aggregate unittest execution via `test/run_all_tests.py`.

## [0.2.1] - 2026-02-20

### Fixed
- Make `test/blinker_pyblinker_comparison` path handling discovery-safe by resolving test fixture files relative to the test package instead of the process working directory.
- Clean up comparison test imports and shared path utilities so the comparison tests execute consistently both via `python -m unittest <module>` and through `python test/run_all_tests.py`.

## [0.2.0] - 2026-02-20

### Added
- Add a dedicated `test/README.md` documenting supported single-module and batch test commands, with `python -m unittest <dotted.module>` as the recommended single-test workflow.

### Changed
- Refactor energy, kinematics, and morphology feature schema/header generation into dedicated `column_headers.py` modules backed by a shared `blink_features/utils/column_headers_common.py` helper, while preserving legacy column names/order/alias behavior and existing feature computations.
- Refactor feature-family compute setup to use shared epoch-context utilities (`_epoch_context.py`) across energy, frequency-domain, and kinematics extractors.
- Remove temporary `test/blink_features/test_shared_refactor_helpers.py` helper script and keep validation through the canonical full-suite runner.
- Remove onset/duration-style discovery from shared style helpers and standardize on frame-based start/end metadata windows.
- Extract kinematics low-level numeric helper routines into `pyblinker/blink_features/kinematics/helpers.py` and consume them from the epoch extractor.
- Rebuild `test/run_all_tests.py` to restore runnable discovery, enforce deterministic ordering by test id, and print clear per-module plus per-test execution progress in CI-friendly output.
- Keep batch discovery rooted at `test/` with repository root as `top_level_dir`, and ensure repository-root import path setup is applied consistently before discovery/runs.

## [0.1.11] - 2026-02-12

### Fixed
- Fix morphology style-window extraction to prioritize frame-based landmark boundaries (`start__...`/`end__...`) over onset/duration metadata so per-window metrics no longer collapse to empty windows when duration fields are zero.
- Update morphology per-window metric extraction to slice `signal[start_frame:end_frame]` directly from frame windows (without round-tripping through onset/duration conversion).
- Add regression coverage ensuring `zero` style windows are built from landmark frame indices when present.

## [0.1.10] - 2026-02-11

### Changed
- Refactor energy, kinematics, and morphology feature schema/header generation into dedicated `column_headers.py` modules backed by a shared `blink_features/utils/column_headers_common.py` helper, while preserving legacy column names/order/alias behavior and existing feature computations.
- Refactor feature-family compute setup to use shared epoch-context utilities (`_epoch_context.py`) across energy, frequency-domain, and kinematics extractors.
- Remove temporary `test/blink_features/test_shared_refactor_helpers.py` helper script and keep validation through the canonical full-suite runner.
- Remove onset/duration-style discovery from shared style helpers and standardize on frame-based start/end metadata windows.
- Extract kinematics low-level numeric helper routines into `pyblinker/blink_features/kinematics/helpers.py` and consume them from the epoch extractor.
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Move inter-blink max velocity epoch assertions into kinematics coverage and keep morphology epoch extraction focused on morphology-only legacy fields.
- Expand EEG-only kinematic and morphology aggregation tests to explicitly assert newly exposed kinematic sign-split ratios/side velocities and legacy morphology timing/peak/inter-blink fields.
- Refactor epoch kinematic extraction to expose per-blink sign-split amplitude/velocity ratios and average side velocities via shared kinematics core-metric utilities.
- Refactor epoch morphology extraction to expose peak time/max and inter-blink max-velocity legacy outputs using shared morphology/kinematics core-metric utilities for parity with BlinkProperties comparisons.

### Fixed
- Tighten MATLAB parity in `get_blink_statistic` by converting zero-crossing blink boundaries with nearest-integer semantics (`rint`) before building the inclusive blink mask.
- Strengthen comparison coverage by using `atol = 1e-4` in `test_a2_stat.py`, confirming stable statistics under a stricter tolerance.

## [0.1.9] - 2026-02-11

### Fixed
- Correct `get_blink_statistic` to report `number_blinks` from the full blink-fit table (matching MATLAB `extractBlinks`), while still excluding invalid rows for downstream quality/cutoff statistics.
- This restores MATLAB-parity for blink statistics and makes the blinker comparison tests pass consistently when run individually or through the aggregate runner.

## [0.1.8] - 2026-02-11

### Fixed
- Align `get_blink_statistic` with MATLAB `extractBlinks` semantics by dropping invalid blink-fit rows before all downstream statistics and mask calculations.
- Match MATLAB blink amplitude ratio masking behavior with inclusive zero-crossing ranges and robust handling of empty inside/outside positive-signal samples.
- Return deterministic NaN-based statistics when fewer than two top-threshold blink fits are available, mirroring MATLAB early-exit behavior for cutoff statistics.

## [0.1.7] - 2026-02-11

### Fixed
- Stabilize FitBlinks correlation outputs (`leftR2`, `rightR2`) for MATLAB comparison by normalizing returned R values to single-precision representation.
- Relax strict floating-point equality in `step_b_fitblink.py` to MATLAB-appropriate absolute tolerance (`atol=1e-6`) to avoid false failures from numeric noise.


## [0.1.6] - 2026-02-11

### Fixed
- Restore MATLAB-consistent left half-height landmark indexing in FitBlinks geometry (base and zero half-height frames).
- Restore MATLAB-consistent inter-blink max-velocity fields used by BlinkProperties comparison outputs.
- Guard amplitude-velocity ratio velocity-index bounds to avoid out-of-range indexing on short boundary blinks.


## [0.1.5] - 2026-02-11

### Changed
- Refactor energy, kinematics, and morphology feature schema/header generation into dedicated `column_headers.py` modules backed by a shared `blink_features/utils/column_headers_common.py` helper, while preserving legacy column names/order/alias behavior and existing feature computations.
- Refactor feature-family compute setup to use shared epoch-context utilities (`_epoch_context.py`) across energy, frequency-domain, and kinematics extractors.
- Remove temporary `test/blink_features/test_shared_refactor_helpers.py` helper script and keep validation through the canonical full-suite runner.
- Remove onset/duration-style discovery from shared style helpers and standardize on frame-based start/end metadata windows.
- Extract kinematics low-level numeric helper routines into `pyblinker/blink_features/kinematics/helpers.py` and consume them from the epoch extractor.
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Refactor blink-position detection and blink-property extraction flows into MATLAB-aligned helper stages for clearer parity-oriented maintenance.

### Fixed
- Align blink start/end candidate scanning and minimum-separation masking with MATLAB `getBlinkPositions` semantics.
- Correct baseline and half-height landmark indexing offsets used during FitBlinks geometry calculations to match MATLAB frame conventions.
- Fix inter-blink maximum-velocity interval computation so BlinkProperties uses MATLAB-style successive-peak differences.


## [0.1.4] - 2026-01-20

### Changed
- Refactor energy, kinematics, and morphology feature schema/header generation into dedicated `column_headers.py` modules backed by a shared `blink_features/utils/column_headers_common.py` helper, while preserving legacy column names/order/alias behavior and existing feature computations.
- Refactor feature-family compute setup to use shared epoch-context utilities (`_epoch_context.py`) across energy, frequency-domain, and kinematics extractors.
- Remove temporary `test/blink_features/test_shared_refactor_helpers.py` helper script and keep validation through the canonical full-suite runner.
- Remove onset/duration-style discovery from shared style helpers and standardize on frame-based start/end metadata windows.
- Extract kinematics low-level numeric helper routines into `pyblinker/blink_features/kinematics/helpers.py` and consume them from the epoch extractor.
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Expand blink-property metric definitions and MATLAB-to-Python column mapping documentation.

### Fixed
- Align blink-property timing, shut-time defaults, and inter-blink velocity intervals with MATLAB to support row-by-row comparisons.

## [0.1.3] - 2026-01-19

### Changed
- Refactor energy, kinematics, and morphology feature schema/header generation into dedicated `column_headers.py` modules backed by a shared `blink_features/utils/column_headers_common.py` helper, while preserving legacy column names/order/alias behavior and existing feature computations.
- Refactor feature-family compute setup to use shared epoch-context utilities (`_epoch_context.py`) across energy, frequency-domain, and kinematics extractors.
- Remove temporary `test/blink_features/test_shared_refactor_helpers.py` helper script and keep validation through the canonical full-suite runner.
- Remove onset/duration-style discovery from shared style helpers and standardize on frame-based start/end metadata windows.
- Extract kinematics low-level numeric helper routines into `pyblinker/blink_features/kinematics/helpers.py` and consume them from the epoch extractor.
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Render FitBlinks tutorial waveforms as scatter points with a faint line trace behind them and pad report windows on the left by 10 samples for clearer landmark visibility.

## [0.1.1] - 2026-01-19

### Fixed
- Resolve FitBlinks tutorial input paths relative to the repository and write reports to the tutorial_outputs folder.

## [0.1.0] - 2026-01-18

### Added
- Add MATLAB FitBlinks HTML reporting utilities to compare MATLAB output with Python-derived landmarks using consistent plotting and captions.

## [0.0.74] - 2026-01-17

### Fixed
- Keep FitBlinks y-intersect comparison values in signal units without indexing offsets.

## [0.0.73] - 2026-01-17

### Fixed
- Adjust FitBlinks comparison indexing so y-intersect values remain in signal units.

## [0.0.72] - 2026-01-17

### Fixed
- Align FitBlinks comparison indexing for intersection outputs without extra y-intersect offset.

## [0.0.71] - 2026-01-17

### Fixed
- Emit NaN half-height landmarks when thresholds are never crossed to match MATLAB fit behavior.

## [0.0.70] - 2026-01-17

### Fixed
- Correct right-base index handling in FitBlinks to match MATLAB downstroke minima behavior.

## [0.0.69] - 2026-01-17

### Fixed
- Align FitBlinks indexing, baseline selection, and line-intersection handling with MATLAB to preserve blink counts and fit outputs.

## [0.0.68] - 2026-01-16

### Fixed
- Match MATLAB blink position separation handling by allowing candidates exactly at the minimum separation threshold.

### Changed
- Refactor energy, kinematics, and morphology feature schema/header generation into dedicated `column_headers.py` modules backed by a shared `blink_features/utils/column_headers_common.py` helper, while preserving legacy column names/order/alias behavior and existing feature computations.
- Refactor feature-family compute setup to use shared epoch-context utilities (`_epoch_context.py`) across energy, frequency-domain, and kinematics extractors.
- Remove temporary `test/blink_features/test_shared_refactor_helpers.py` helper script and keep validation through the canonical full-suite runner.
- Remove onset/duration-style discovery from shared style helpers and standardize on frame-based start/end metadata windows.
- Extract kinematics low-level numeric helper routines into `pyblinker/blink_features/kinematics/helpers.py` and consume them from the epoch extractor.
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Refactor blink position detection into vectorized helpers for improved efficiency without altering behavior.

## [0.0.67] - 2026-01-15

### Changed
- Refactor energy, kinematics, and morphology feature schema/header generation into dedicated `column_headers.py` modules backed by a shared `blink_features/utils/column_headers_common.py` helper, while preserving legacy column names/order/alias behavior and existing feature computations.
- Refactor feature-family compute setup to use shared epoch-context utilities (`_epoch_context.py`) across energy, frequency-domain, and kinematics extractors.
- Remove temporary `test/blink_features/test_shared_refactor_helpers.py` helper script and keep validation through the canonical full-suite runner.
- Remove onset/duration-style discovery from shared style helpers and standardize on frame-based start/end metadata windows.
- Extract kinematics low-level numeric helper routines into `pyblinker/blink_features/kinematics/helpers.py` and consume them from the epoch extractor.
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Compute morphology epoch metrics with shared core duration/shut-time helpers while emitting both legacy flat column names and fully-qualified style-aware names.
- Replace lazy morphology package exports with explicit imports aligned to the kinematics package.
- Add morphology aggregation logging and clarify core metric responsibilities in documentation strings.
- Refactor morphology epoch extraction into helper steps with pipeline-oriented docstrings.

### Added
- EEG-only morphology test coverage for style-aware duration columns and legacy aliases.

## [0.0.66] - 2026-01-14

### Fixed
- Draw EEG epoch landmark markers behind EEG scatter points in the EEG landmark tutorial to keep EEG-E8 dots on top without changing marker sizes.
