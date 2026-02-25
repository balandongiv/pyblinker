"""Aggregate blink features across modalities and feature families."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from .default_setting import DEFAULT_CONFIG
from .constants import cast_columns_to_object

from pyblinker.logging import get_logger
logger = get_logger(__name__)

_DEFAULT_WAVEFORM_PARAMS = {
    "base_fraction": DEFAULT_CONFIG.base_fraction,
    "shut_amp_fraction": DEFAULT_CONFIG.shut_amp_fraction,
    "p_avr_threshold": DEFAULT_CONFIG.p_avr_threshold,
    "z_thresholds": DEFAULT_CONFIG.z_thresholds.copy(),
}


def _get_epoch_index(epochs: mne.Epochs) -> pd.Index:
    """Return the canonical epoch index used for feature alignment."""

    if isinstance(epochs.metadata, pd.DataFrame):
        index = pd.Index(epochs.metadata.index, name="epoch")
    else:
        index = pd.RangeIndex(len(epochs), name="epoch")
    return index


def _get_modality_channels(info: mne.Info, modality: str) -> list[str]:
    """Collect channel names that belong to ``modality``."""

    modality_upper = modality.upper()
    channels: list[str] = []
    for ch_name in info["ch_names"]:
        ch_type = info.get_channel_types(picks=[ch_name])[0]
        ch_lower = ch_name.lower()
        if modality_upper == "EEG" and ch_type == "eeg":
            channels.append(ch_name)
        elif modality_upper == "EOG" and ch_type == "eog":
            channels.append(ch_name)
        elif modality_upper == "EAR":
            if "ear" in ch_lower:
                channels.append(ch_name)
            elif ch_type == "misc" and "ear" in ch_lower:
                channels.append(ch_name)
    return channels


def _align_to_epoch_index(df: pd.DataFrame, epoch_index: pd.Index) -> pd.DataFrame:
    """Align ``df`` to the canonical ``epoch_index``."""

    if df is None or df.empty:
        return pd.DataFrame(index=epoch_index)

    aligned = df.copy()
    if "ep" in aligned.columns:
        aligned = aligned.set_index("ep")

    aligned.index = pd.Index(aligned.index, name="epoch")
    aligned = aligned.reindex(epoch_index)
    return aligned


def _namespace_df(df: pd.DataFrame, modality: str, family: str) -> pd.DataFrame:
    """Namespace columns using ``{modality}__{family}__`` pattern."""

    if df.empty:
        return df

    namespaced = df.copy()
    namespaced.columns = [f"{modality}__{family}__{col}" for col in namespaced.columns]
    return namespaced


def _rename_event_columns(columns: Iterable[str], modality: str) -> dict[str, str]:
    """Remove modality suffixes from blink totals/rates to avoid duplicates."""

    suffix = f"_{modality.lower()}"
    rename: dict[str, str] = {}
    for col in columns:
        if col.startswith("blink_total") and col.endswith(suffix):
            rename[col] = col[: -len(suffix)]
        elif col.startswith("blink_rate") and col.endswith(suffix):
            rename[col] = col[: -len(suffix)]
    return rename


def _compute_waveform_epoch_features(
    epochs: mne.Epochs,
    picks: Sequence[str],
    params: Mapping[str, Any],
    *,
    run_fit: bool,
    progress_bar: bool,
) -> pd.DataFrame:
    """Aggregate waveform features per epoch for the provided channels."""

    from pyblinker.segment_blink_properties import compute_segment_blink_properties

    epoch_index = _get_epoch_index(epochs)
    if not picks:
        return pd.DataFrame(index=epoch_index)

    params = {**_DEFAULT_WAVEFORM_PARAMS, **dict(params)}
    frames: list[pd.DataFrame] = []
    position_to_epoch = {pos: epoch_index[pos] for pos in range(len(epoch_index))}

    for ch in picks:
        epochs_ch = epochs.copy().pick(ch, verbose=False)
        try:
            blink_df = compute_segment_blink_properties(
                epochs_ch,
                params,
                channel=ch,
                run_fit=run_fit,
                progress_bar=progress_bar,
                long_format=True,
            )
        except Exception as exc:  # pragma: no cover - logged for diagnosis
            logger.warning(
                "Waveform feature computation failed for channel %s: %s",
                ch,
                exc,
            )
            continue

        if not isinstance(blink_df, pd.DataFrame) or blink_df.empty:
            continue
        if "seg_id" not in blink_df.columns:
            continue

        numeric = blink_df.select_dtypes(include=[np.number]).drop(
            columns=[c for c in ("seg_id", "blink_id") if c in blink_df.columns],
            errors="ignore",
        )
        if numeric.empty:
            continue

        grouped = numeric.groupby(blink_df["seg_id"]).mean()
        mapped_index = grouped.index.map(position_to_epoch.get)
        grouped.index = pd.Index(mapped_index, name="epoch")
        grouped = grouped.reindex(epoch_index)
        grouped.columns = [f"{col}_{ch}" for col in grouped.columns]
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(index=epoch_index)

    combined = pd.concat(frames, axis=1)
    combined = combined.reindex(epoch_index)
    return combined


def _compute_family_features(
    epochs: mne.Epochs,
    modality: str,
    family: str,
    picks: Sequence[str],
    epoch_index: pd.Index,
    *,
    progress_bar: bool,
    waveform_params: Mapping[str, Any] | None,
    waveform_run_fit: bool,
) -> pd.DataFrame:
    """Dispatch feature computation for a modality/family pair."""

    family_key = family.lower()
    if family_key == "events":
        from .blink_events.event_features import aggregate_blink_event_features

        df = aggregate_blink_event_features(epochs, picks=picks)
        df = df.rename(columns=_rename_event_columns(df.columns, modality))
    elif family_key == "energy":
        from .energy.energy_features import compute_energy_features

        df = compute_energy_features(epochs, picks=picks)
    elif family_key == "freq":
        from .frequency_domain.aggregate import aggregate_frequency_domain_features

        df = aggregate_frequency_domain_features(
            epochs, picks=picks, progress_bar=progress_bar
        )
    elif family_key == "kin":
        from .kinematics.kinematic_features import compute_kinematic_features

        df = compute_kinematic_features(epochs, picks=picks)
    elif family_key == "morph":
        from .morphology.epoch_features import compute_epoch_morphology_features

        df = compute_epoch_morphology_features(epochs, picks=picks)
    elif family_key == "wave":
        params_dict: dict[str, Any] = dict(_DEFAULT_WAVEFORM_PARAMS)
        if waveform_params is not None:
            params_dict.update(dict(waveform_params))
        df = _compute_waveform_epoch_features(
            epochs,
            picks,
            params_dict,
            run_fit=waveform_run_fit,
            progress_bar=progress_bar,
        )
    else:
        raise ValueError(f"Unknown feature family: {family}")

    aligned = _align_to_epoch_index(df, epoch_index)
    family_label = "freq" if family_key == "freq" else family_key
    return _namespace_df(aligned, modality, family_label)


def _load_extra_features(
    epoch_index: pd.Index, metadata_csv_path: str | Path | None
) -> list[pd.DataFrame]:
    """Load auxiliary inputs such as CSV blink counts."""

    pieces: list[pd.DataFrame] = []
    if not metadata_csv_path:
        return pieces

    try:
        csv_df = pd.read_csv(Path(metadata_csv_path))
    except FileNotFoundError:
        logger.warning("Blink count CSV not found at %s", metadata_csv_path)
    else:
        if "epoch_id" in csv_df.columns:
            csv_df = csv_df.set_index("epoch_id")
        else:
            csv_df = csv_df.set_index(csv_df.columns[0])
        csv_df = csv_df.reindex(epoch_index)
        csv_df.columns = [f"META__events__{col}" for col in csv_df.columns]
        for col in csv_df.columns:
            csv_df[col] = pd.to_numeric(csv_df[col], errors="coerce")
        pieces.append(csv_df)
    return pieces


def aggregate_blink_features(
    raw_or_epochs: mne.io.BaseRaw | mne.Epochs,
    *,
    epoch_len: float = 30.0,
    blink_label: str | None = None,
    progress_bar: bool = False,
    include_modalities: tuple[str, ...] = ("EEG", "EOG", "EAR"),
    feature_families: tuple[str, ...] = (
        "events",
        "energy",
        "freq",
        "kin",
        "morph",
        "wave",
    ),
    waveform_params: Mapping[str, Any] | None = None,
    waveform_run_fit: bool = True,
    metadata_csv_path: str | Path | None = None,
    segmentation_type: Mapping[str, Any] | None = None,
    ear_threshold: float | None = None,
) -> pd.DataFrame:
    """Return consolidated blink features across modalities and families.

    Parameters
    ----------
    raw_or_epochs : mne.io.BaseRaw | mne.Epochs
        Continuous recording or pre-computed epochs used for feature
        extraction.
    epoch_len : float, optional
        Epoch length in seconds when ``raw_or_epochs`` is a Raw instance.
    blink_label : str | None, optional
        Annotation label describing blinks. ``None`` uses all annotations.
    progress_bar : bool, optional
        Whether to display progress bars during computations.
    include_modalities : tuple of str, optional
        Modalities to include (e.g., ``("EEG", "EOG", "EAR")``).
    feature_families : tuple of str, optional
        Feature families to compute (``"events"``, ``"energy"``, ``"freq"``,
        ``"kin"``, ``"morph"``, ``"wave"``).
    waveform_params : Mapping[str, Any] | None, optional
        Optional overrides for waveform feature parameters.
    waveform_run_fit : bool, optional
        Whether to run the blink waveform fitting routine.
    metadata_csv_path : str | Path | None, optional
        Optional CSV file containing per-epoch metadata to merge.
    segmentation_type : Mapping[str, Any] | None, optional
        Required when ``raw_or_epochs`` is a Raw instance. Provides explicit
        single-channel selections per modality under ``"ear"``, ``"eeg"``, and
        ``"eog"``.
    ear_threshold : float | None, optional
        Convenience override merged into the EAR segmentation config when
        ``raw_or_epochs`` is raw input.

    Returns
    -------
    pandas.DataFrame
        Consolidated blink features indexed by epoch.
    """

    if isinstance(raw_or_epochs, mne.io.BaseRaw):
        from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot

        if segmentation_type is None:
            raise ValueError(
                "segmentation_type must be provided with explicit single-channel selections for raw inputs."
            )
        segmentation_config = dict(segmentation_type)
        if ear_threshold is not None:
            ear_config = dict(segmentation_config.get("ear", {}))
            ear_config.setdefault("threshold", ear_threshold)
            ear_config.setdefault("seg_type", "threshold_interpolation")
            segmentation_config["ear"] = ear_config
        epochs = slice_raw_into_mne_epochs_refine_annot(
            raw_or_epochs,
            epoch_len=epoch_len,
            blink_label=blink_label,
            progress_bar=progress_bar,
            segmentation_type=segmentation_config,
        )
    elif isinstance(raw_or_epochs, mne.Epochs):
        epochs = raw_or_epochs
    else:  # pragma: no cover - defensive
        raise TypeError("raw_or_epochs must be an mne Raw or Epochs instance")

    epoch_index = _get_epoch_index(epochs)

    from pyblinker.utils import require_channels

    modality_channels: dict[str, list[str]] = {}
    for modality in include_modalities:
        picks = _get_modality_channels(epochs.info, modality)
        if picks:
            modality_channels[modality] = picks
        else:
            logger.warning("No channels found for modality %s", modality)

    if not modality_channels:
        raise ValueError("No matching modalities found in the provided data")

    pieces: list[pd.DataFrame] = []
    for modality, picks in modality_channels.items():
        require_channels(epochs, picks)
        for family in feature_families:
            try:
                features_df = _compute_family_features(
                    epochs,
                    modality,
                    family,
                    picks,
                    epoch_index,
                    progress_bar=progress_bar,
                    waveform_params=waveform_params,
                    waveform_run_fit=waveform_run_fit,
                )
            except Exception as exc:  # pragma: no cover - logged for visibility
                logger.warning(
                    "Failed to compute %s features for %s: %s",
                    family,
                    modality,
                    exc,
                )
                continue
            if features_df.empty:
                continue
            pieces.append(features_df)

    pieces.extend(_load_extra_features(epoch_index, metadata_csv_path))

    if pieces:
        result = pd.concat(pieces, axis=1)
    else:
        result = pd.DataFrame(index=epoch_index)

    if not result.empty:
        all_nan = [col for col in result.columns if result[col].isna().all()]
        if all_nan:
            result = result.drop(columns=all_nan)

    result = result.reindex(epoch_index)
    result.index.name = "epoch"

    if not result.empty:
        try:
            epoch_values = epoch_index.astype(int)
        except (TypeError, ValueError):
            epoch_values = np.arange(len(epoch_index), dtype=int)
        epoch_series = pd.Series(epoch_values, index=epoch_index, name="epoch")
        result = result.assign(epoch=epoch_series)
        feature_cols = [col for col in result.columns if col != "epoch"]
        result = result.loc[:, ["epoch"] + sorted(feature_cols)]

    return cast_columns_to_object(result)


__all__ = ["aggregate_blink_features"]
