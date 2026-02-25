# Refactor roadmap

## Phase– safe refactors (no behaviour change)

- [✔] **Centralise shared constants** – Create a module `blink_features/constants.py` containing `_STATS`, the metric registry per family, and modality inference. Add a configuration dataclass (e.g., `BlinkerConfig`) encapsulating thresholds like `shut_amp_fraction`, `base_fraction`, `p_avr_threshold` and `z_thresholds`.  
  Audit: Added `pyblinker/blink_features/constants.py` and `pyblinker/blink_features/default_setting.py`; updated aggregate defaults to consume `DEFAULT_CONFIG`. Migration note: defaults remain backward-compatible and no public API was removed.
- [ ] **To delete, no excuse** – Delete the onset/duration-prefix approach referenced by `utils.style_windows.available_styles(...)`.  
  Pending: onset/duration discovery is still present in morphology/kinematics paths; removal postponed to avoid behavior regressions.
- [✔] **Extract style/window helpers** – Implement `utils.style_windows.available_styles(...)` and `style_windows.extract_windows(...)` with configurable prefixes.  
  Audit: Added shared helper module at `pyblinker/blink_features/utils/style_windows.py` with reusable style discovery and window extraction logic.
- [ ] **Modularise kinematic feature code** – Move low-level kinematic helpers into `kinematics/helpers.py`.  
  Pending: kinematic helper extraction was not completed in this pass.
- [✔] **Mark optional dependencies** – Wrap optional external imports (e.g., `pywt`) with fallbacks/messages.  
  Audit: Updated `pyblinker/blink_features/frequency_domain/features.py` so missing PyWavelets returns NaN outputs with warnings instead of import-time hard failure.

## Phase 2 – consolidation (shared loop skeletons & config migration)

- [✔] **Design a common compute skeleton** – Create `compute_features(...)` orchestration utility.  
  Audit: Added `pyblinker/blink_features/compute_skeleton.py` implementing shared channel/modality/style iteration and DataFrame assembly.
- [ ] **Refactor existing extractors** – Delegate morphology/kinematics/energy/frequency extractors to shared skeleton.  
  Pending: full delegation is incomplete; current extractors remain mostly family-specific to preserve baseline outputs.
- [ ] **Migrate configuration** – Replace scattered constants with `BlinkerConfig` and expose config via high-level APIs.  
  Pending: config is introduced and partially wired (`aggregate.py`), but high-level extractor APIs are not yet fully migrated.

## Phase 3 – API cleanup & legacy deprecation

- [ ] **Deprecate legacy metrics** – Document deprecations and migration paths.  
  Pending: not started.
- [ ] **Remove quarantined code** – Remove dead modules after usage verification.  
  Pending: not started.
- [ ] **Simplify module boundaries** – Split large modules into cohesive subpackages.  
  Pending: not started.
- [ ] **Enhance tests** – Add parameterized/shared-skeleton/config tests and optional-dependency skip coverage.  
  Pending: not started in this refactor pass.

---

# Design proposal: shared compute skeleton

```python
from typing import Callable, Dict, Iterable, List, Sequence, Mapping
import pandas as pd

def compute_features(
    epochs: mne.Epochs,
    picks: Sequence[str] | None,
    metrics_by_style: Dict[str, Sequence[str]],
    compute_func: Callable[[str, np.ndarray, float, str], Mapping[str, float]],
    config: BlinkerConfig,
    *,
    family: str,
) -> pd.DataFrame:
    """Shared orchestration logic for blink feature extraction.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE epochs object with metadata.
    picks : list of str or None
        Channels to process; if None, default channels are resolved.
    metrics_by_style : dict
        Maps style names to the list of metric names for that style.
    compute_func : callable
        Function called as `compute_func(style, segment, sfreq, modality)` to
        compute metric values for a single segment.
    config : BlinkerConfig
        Configuration object providing thresholds and constants.
    family : str
        Name of the feature family (e.g., ``"morphology"``).

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` with namespaced feature columns.
    """
    # Resolve sampling frequency and channels
    sfreq = float(epochs.info['sfreq'])
    ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
        epochs=epochs, picks=picks, sfreq=sfreq
    )
    # Infer modalities and styles using shared helpers
    modality_map = {ch: infer_modality(ch, epochs.info) for ch in ch_names}
    styles_by_modality = {
        mod: available_styles(tuple(epochs.metadata.columns), mod)
        for mod in set(modality_map.values())
    }
    # Build output columns
    columns = []
    for mod, channels in group_channels_by_modality(modality_map).items():
        for style in styles_by_modality.get(mod, {"base"}):
            for metric in metrics_by_style[style]:
                for stat in config.stat_names:
                    for ch in channels:
                        columns.append(f"{mod}__{style}__{family}__{metric}_{stat}__{ch}")
    # Compute per-epoch records
    records: List[Dict[str, float]] = []
    for ei in range(n_epochs):
        metadata_row = epochs.metadata.iloc[ei] if isinstance(
            epochs.metadata, pd.DataFrame
        ) else pd.Series(dtype=float)
        record: Dict[str, float] = {}
        for ch, mod in modality_map.items():
            styles = styles_by_modality.get(mod, {"base"})
            for style in styles:
                windows = extract_windows(metadata_row, mod, style, n_times)
                for start_idx, end_idx in windows:
                    segment = channel_data[ch]['raw'][ei][start_idx:end_idx]
                    stats = compute_func(style, segment, sfreq, mod)
                    for metric, values in stats.items():
                        for stat_name, val in values.items():
                            col_name = f"{mod}__{style}__{family}__{metric}_{stat_name}__{ch}"
                            record[col_name] = val
        records.append(record)
    return pd.DataFrame.from_records(records, index=index, columns=columns)
```

This skeleton abstracts the common loop structure. Feature families implement `metrics_by_style` and `compute_func`. Configuration values (stat names, thresholds, etc.) come from the `BlinkerConfig` object. Adopting this design will make it straightforward to add new feature families or modify the style/metric mapping without touching the orchestration code.

---

# Config strategy

* Define a `BlinkerConfig` dataclass containing:

    * `stat_names`: tuple of statistic names (default `("mean", "std", "cv")`).
    * Thresholds (e.g., `base_fraction`, `shut_amp_fraction`, `p_avr_threshold`).
    * Arrays such as `z_thresholds` using `field(default_factory=lambda: np.array(...))` to avoid mutable defaults.
    * Optionally, sub-configs per feature family (e.g., wavelet levels).
* Replace module-level constants (e.g., `_DEFAULT_WAVEFORM_PARAMS`) with references to the config. Pass the config into compute functions so that thresholds are not hidden global state. Provide a global `DEFAULT_CONFIG` instance in `default_setting.py` so existing callers continue to work.
* Consider loading configuration from a YAML or JSON file for user customisation, but maintain a programmatic API for testability.

---

# Test strategy

* Ensure that all tests import `pyblinker` without optional dependencies installed. Wrap optional imports in try/except and skip the corresponding tests with informative messages when dependencies are missing.
* Parameterise tests on modalities and styles. For example, run the shared compute skeleton with different `BlinkerConfig` settings (e.g., varying `stat_names` or thresholds) to ensure flexibility.
* Add unit tests for the shared helpers (`available_styles`, `extract_windows`, modality inference). These tests should cover corner cases such as missing metadata columns, out-of-order start/end keys and non-standard style names.
* Write tests that compare the output of the refactored feature extractors against the pre-refactor baseline to guard against behavioural changes. Use small synthetic signals where possible to avoid large `.fif` files.

---

# Risks and mitigations

| Risk                                                        | Mitigation                                                                                                                                                                                                     |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Breaking the public API (e.g., column naming conventions)   | Preserve the naming pattern {modality}**{style}**{family}**{metric}_{stat}**{channel}. Provide alias columns or deprecation warnings where names change (e.g., _add_legacy_ear_channel_aliases).               |
| Removing legacy metrics used by downstream consumers        | Deprecate metrics in phases and document replacements. Provide feature flags to include legacy metrics until clients migrate.                                                                                  |
| Configuration migration causing different default behaviour | Ensure that default values in `BlinkerConfig` match the current hard-coded constants (e.g., `shut_amp_fraction = 0.9` and `DEFAULT_PARAMS['shut_amp_fraction']`). Write regression tests for default settings. |
| Optional dependencies not installed                         | Guard optional imports and skip feature families requiring them. Provide installation instructions or vendor the dependency.                                                                                   |
| Performance overhead from abstraction                       | Profile the shared compute skeleton; use vectorised NumPy operations where possible to minimise Python loops. Consider caching style windows per epoch to avoid recomputation.                                 |
