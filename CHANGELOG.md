# Changelog

## [0.1.5] - 2026-01-21

### Fixed
- Align blink candidate minimum-separation removal with MATLAB logic by treating candidates at or under the separation threshold as overlapping.

### Changed
- Centralize close-pair filtering in blink position detection so pipeline and comparisons share MATLAB-aligned logic.
- Use repository-relative paths in migration comparison tests for portability.
- Use repository-relative fixture paths in extract_blinks_eeg migration comparison.

## [0.1.4] - 2026-01-20

### Changed
- Expand blink-property metric definitions and MATLAB-to-Python column mapping documentation.

### Fixed
- Align blink-property timing, shut-time defaults, and inter-blink velocity intervals with MATLAB to support row-by-row comparisons.

## [0.1.3] - 2026-01-19

### Changed
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
- Refactor blink position detection into vectorized helpers for improved efficiency without altering behavior.

## [0.0.67] - 2026-01-15

### Changed
- Compute morphology epoch metrics with shared core duration/shut-time helpers while emitting both legacy flat column names and fully-qualified style-aware names.
- Replace lazy morphology package exports with explicit imports aligned to the kinematics package.
- Add morphology aggregation logging and clarify core metric responsibilities in documentation strings.
- Refactor morphology epoch extraction into helper steps with pipeline-oriented docstrings.

### Added
- EEG-only morphology test coverage for style-aware duration columns and legacy aliases.

## [0.0.66] - 2026-01-14

### Fixed
- Draw EEG epoch landmark markers behind EEG scatter points in the EEG landmark tutorial to keep EEG-E8 dots on top without changing marker sizes.
