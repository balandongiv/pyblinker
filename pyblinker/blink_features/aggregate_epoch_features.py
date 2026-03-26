"""Aggregate multiple epoch-level blink feature families into one frame."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from .energy.energy_features import compute_energy_features
from .frequency_domain import aggregate_frequency_domain_features
from .kinematics.kinematic_features import KinematicBlinkFeatureExtractor
from .morphology import compute_epoch_morphology_features
from .perclos import aggregate_perclos_features

_FEATURE_FAMILY_ALIASES = {
    "energy": "energy",
    "freq": "frequency",
    "frequency": "frequency",
    "kin": "kinematics",
    "kinematic": "kinematics",
    "kinematics": "kinematics",
    "morph": "morphology",
    "morphology": "morphology",
    "perclos": "perclos",
}


def normalize_epoch_feature_families(
    feature_families: Sequence[str] | Iterable[str] | None,
) -> tuple[str, ...]:
    """Normalize user-facing family aliases into canonical epoch feature groups."""

    if feature_families is None:
        return ("energy", "frequency", "kinematics", "morphology")

    normalized: list[str] = []
    for family in feature_families:
        key = str(family).strip().lower()
        canonical = _FEATURE_FAMILY_ALIASES.get(key)
        if canonical is None:
            raise ValueError(
                f"Unsupported pyblinker feature family '{family}'. "
                "Supported families are: energy, frequency, kinematics, morphology, perclos."
            )
        if canonical not in normalized:
            normalized.append(canonical)
    return tuple(normalized)


def compute_epoch_feature_families(
    epochs,
    *,
    picks: Sequence[str] | None = None,
    feature_families: Sequence[str] | Iterable[str] | None = None,
    progress_bar: bool = False,
) -> pd.DataFrame:
    """Compute one or more epoch-level feature families and merge them by row."""

    selected_families = normalize_epoch_feature_families(feature_families)
    resolved_picks = list(picks or epochs.ch_names)

    frames: list[pd.DataFrame] = []
    if "energy" in selected_families:
        frames.append(compute_energy_features(epochs=epochs, picks=resolved_picks))
    if "frequency" in selected_families:
        frames.append(
            aggregate_frequency_domain_features(
                epochs,
                picks=resolved_picks,
                progress_bar=progress_bar,
            )
        )
    if "kinematics" in selected_families:
        frames.append(
            KinematicBlinkFeatureExtractor(epochs=epochs).compute(picks=resolved_picks)
        )
    if "morphology" in selected_families:
        frames.append(compute_epoch_morphology_features(epochs=epochs, picks=resolved_picks))
    if "perclos" in selected_families:
        perclos_df = aggregate_perclos_features(
            epochs,
            requested_picks=resolved_picks,
        )
        keep_columns = [
            column
            for column in perclos_df.columns
            if column not in {"epoch_start", "epoch_end"}
        ]
        if keep_columns:
            frames.append(perclos_df.loc[:, keep_columns].copy())

    if not frames:
        return pd.DataFrame(index=range(len(epochs)))

    combined = pd.concat(frames, axis=1)
    if combined.columns.duplicated().any():
        combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


__all__ = ["compute_epoch_feature_families", "normalize_epoch_feature_families"]
