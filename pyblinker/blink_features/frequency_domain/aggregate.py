"""Aggregate wavelet blink features across epochs."""

from __future__ import annotations
from typing import Dict, List, Mapping, Sequence, Set

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyblinker.logging import get_logger

from .features import _compute_wavelet_energies
from ..constants import DEFAULT_BLINKER_CONFIG, BlinkerConfig
from ..energy.helpers import _safe_stats
from ..utils.compute_skeleton import build_epoch_metadata_row, prepare_compute_context
from ..utils.style_windows import available_styles, extract_windows

logger = get_logger(__name__)


def _feature_channel_name(channel_name: str, modality: str) -> str:
    """Return output-channel label for feature columns by modality."""

    return channel_name if modality == "eog" else channel_name.upper()
def _compute_epoch_wavelet_record(
    *,
    epoch_index: int,
    metadata_row: pd.Series | Mapping[str, object],
    sfreq: float,
    n_times: int,
    channel_data: Dict[str, np.ndarray],
    modality_by_channel: Dict[str, str],
    available_styles_by_modality: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute style-aware wavelet energies for all channels in a single epoch."""

    record: Dict[str, float] = {}
    for ch, modality in modality_by_channel.items():
        style_windows = _channel_style_windows(
            metadata_row=metadata_row,
            modality=modality,
            available_styles=available_styles_by_modality.get(modality, set()),
            n_times=n_times,
        )
        signal = channel_data[ch]["raw"][epoch_index]
        for style, windows in style_windows.items():
            per_level: Dict[int, List[float]] = {i: [] for i in range(1, 5)}
            for start_idx, end_idx in windows:
                if start_idx >= n_times:
                    continue
                sl = slice(max(0, start_idx), min(end_idx, n_times))
                if sl.stop <= sl.start:
                    continue
                segment = signal[sl]
                if getattr(segment, "size", 0) == 0:
                    continue
                energies = _compute_wavelet_energies(segment, sfreq)
                for lvl, val in enumerate(energies, start=1):
                    per_level[lvl].append(float(val))

            for lvl in range(1, 5):
                stats = _safe_stats(per_level[lvl])
                for stat_name, value in stats.items():
                    key = f"{modality}__{style}__energy__wavelet_energy_d{lvl}_{stat_name}__{_feature_channel_name(ch, modality)}"
                    record[key] = value
    return record


def _available_styles(metadata_columns: Sequence[str] | None, modality: str) -> Set[str]:
    """Return frame-based segmentation styles present in metadata for a modality."""

    return available_styles(metadata_columns, modality, onset_prefix=None, duration_prefix=None)


def _style_windows(metadata_row: Mapping[str, object], modality: str, style: str, n_times: int) -> List[tuple[int, int]]:
    """Extract frame-aligned blink windows as ``(start_sample, end_sample)`` tuples."""

    return extract_windows(metadata_row, modality, style, n_times)


def _channel_style_windows(
    *,
    metadata_row: Mapping[str, object],
    modality: str,
    available_styles: Set[str],
    n_times: int,
) -> Dict[str, List[tuple[int, int]]]:
    """Resolve output wavelet styles to frame windows by modality."""

    style_windows: Dict[str, List[tuple[int, int]]] = {}
    if modality in {"eeg", "eog"}:
        if "zero" in available_styles:
            style_windows["zero"] = _style_windows(metadata_row, modality, "zero", n_times)
        if "base" in available_styles:
            style_windows["base"] = _style_windows(metadata_row, modality, "base", n_times)
        if "tent" in available_styles:
            style_windows["tent"] = _style_windows(metadata_row, modality, "tent", n_times)

        if "half_base" in available_styles:
            style_windows["half"] = _style_windows(metadata_row, modality, "half_base", n_times)
        elif "half_zero" in available_styles:
            style_windows["half"] = _style_windows(metadata_row, modality, "half_zero", n_times)

        if "tent" in style_windows:
            style_windows["peak"] = style_windows["tent"]
        elif "base" in style_windows:
            style_windows["peak"] = style_windows["base"]
    elif modality == "ear":
        if "th_interpolation" in available_styles:
            style_windows["th_point"] = _style_windows(metadata_row, modality, "th_interpolation", n_times)
        elif "th_point" in available_styles:
            style_windows["th_point"] = _style_windows(metadata_row, modality, "th_point", n_times)

    return style_windows


class FrequencyDomainBlinkFeatureExtractor:
    """Compute wavelet-energy blink features from MNE objects."""

    def __init__(
        self,
        epochs: mne.Epochs | None = None,
        raw: mne.io.BaseRaw | None = None,
        config: BlinkerConfig = DEFAULT_BLINKER_CONFIG,
    ):
        self.epochs = epochs
        self.raw = raw
        self.config = config

    def _sampling_frequency(self) -> float:
        """Return sampling frequency from available MNE object."""
        if hasattr(self, "epochs") and self.epochs is not None:
            return float(self.epochs.info["sfreq"])
        if hasattr(self, "raw") and self.raw is not None:
            return float(self.raw.info["sfreq"])
        raise ValueError("Neither self.epochs nor self.raw defined (need MNE object).")

    def compute(
        self,
        picks: str | Sequence[str] | None = None,
        *,
        progress_bar: bool = True,
    ) -> pd.DataFrame:
        """Compute DWT energies for each epoch.

        Parameters
        ----------
        picks : str | list of str | None, optional
            Channel name(s) to include. When multiple channels are supplied
            they are processed per modality/channel. ``None`` uses all channels.
        progress_bar : bool, optional
            Display a progress bar during epoch processing. Defaults to
            ``True``.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed like ``epochs`` with columns ``ep`` and
            ``wavelet_energy_d1_{modality}`` .. ``wavelet_energy_d4_{modality}``
            for each detected modality.

        Notes
        -----
        A warning is logged when the sampling frequency is below 30 Hz
        because wavelet features become unreliable at low rates.
        """

        context = prepare_compute_context(
            epochs=self.epochs,
            picks=picks,
            style_getter=_available_styles,
        )
        sfreq = context.sfreq
        channel_data = context.channel_data
        index = context.index
        n_epochs = context.n_epochs
        n_times = context.n_times

        modality_by_channel = context.modality_map
        eeg_styles = context.styles_by_modality.get("eeg", set())
        available_styles_by_modality: Dict[str, Set[str]] = {}
        for modality in set(modality_by_channel.values()):
            styles = context.styles_by_modality.get(modality, set())
            if modality == "eog" and eeg_styles:
                styles = styles | eeg_styles
            available_styles_by_modality[modality] = styles

        records: List[Dict[str, float]] = []
        for ei in tqdm(
            range(n_epochs),
            desc="Wavelet energies",
            unit="epoch",
            disable=not progress_bar,
        ):
            metadata_row = build_epoch_metadata_row(self.epochs, ei)
            record = _compute_epoch_wavelet_record(
                epoch_index=ei,
                metadata_row=metadata_row,
                sfreq=sfreq,
                n_times=n_times,
                channel_data=channel_data,
                modality_by_channel=modality_by_channel,
                available_styles_by_modality=available_styles_by_modality,
            )
            record["ep"] = index[ei]
            records.append(record)

        df = pd.DataFrame.from_records(records, index=index)
        df.columns = pd.Index([str(col) for col in df.columns], dtype=object)
        logger.debug("Frequency-domain feature DataFrame shape: %s", df.shape)
        return df


def aggregate_frequency_domain_features(
    epochs: mne.Epochs,
    picks: str | Sequence[str] | None = None,
    *,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """Convenience function to compute frequency-domain blink features.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs instance containing the blink data.
    picks : str | list of str | None, optional
        Channel name(s) to include. When multiple channels are provided they
        are aggregated per modality after computing channel-level energies.
        ``None`` uses all channels.
    progress_bar : bool, optional
        Display a progress bar during epoch processing. Defaults to ``True``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an ``ep`` column denoting the epoch index and
        wavelet-energy features.
    """

    extractor = FrequencyDomainBlinkFeatureExtractor(epochs=epochs)
    return extractor.compute(picks=picks, progress_bar=progress_bar)
