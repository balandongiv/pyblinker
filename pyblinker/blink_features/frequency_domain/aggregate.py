"""Aggregate wavelet blink features across epochs."""

from __future__ import annotations
from typing import Dict, List, Mapping, Sequence, Set

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyblinker._logging import get_logger

from .features import _compute_wavelet_energies
from ..energy.helpers import compute_basic_statistics
from ..utils.aggregation import prepare_epoch_channel_data

# from ..constants import cast_columns_to_object
from .._epoch_context import (
    available_styles_by_modality,
    build_epoch_context,
    get_metadata_row,
)
from .._style_windows import style_windows_from_metadata

logger = get_logger(__name__)


def _feature_channel_name(channel_name: str, modality: str) -> str:
    """Return output-channel label for feature columns by modality."""

    return channel_name if modality == "eog" else channel_name.upper()


def _compute_epoch_record(
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
        style_windows = style_windows_from_metadata(
            metadata_row=metadata_row,
            modality=modality,
            available_styles=available_styles_by_modality.get(modality, set()),
            n_times=n_times,
            include_half=True,
            include_peak=True,
            ear_mode="map_to_th_point",
            ear_priority="th_interpolation_first",
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
                stats = compute_basic_statistics(per_level[lvl])
                for stat_name, value in stats.items():
                    key = f"{modality}__{style}__energy__wavelet_energy_d{lvl}_{stat_name}__{_feature_channel_name(ch, modality)}"
                    record[key] = value
    return record


class FrequencyDomainBlinkFeatureExtractor:
    """Compute wavelet-energy blink features from MNE objects."""

    def __init__(
        self, epochs: mne.Epochs | None = None, raw: mne.io.BaseRaw | None = None
    ):
        self.epochs = epochs
        self.raw = raw

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

        ctx = build_epoch_context(self.epochs, picks)
        ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
            epochs=self.epochs,
            picks=ctx.ch_names,
            sfreq=ctx.sfreq,
        )

        modality_by_channel = ctx.modality_by_channel
        available_styles_for_modality = available_styles_by_modality(
            ctx.metadata_cols,
            set(modality_by_channel.values()),
            include_eeg_for_eog=True,
        )

        records: List[Dict[str, float]] = []
        for ei in tqdm(
            range(n_epochs),
            desc="Wavelet energies",
            unit="epoch",
            disable=not progress_bar,
        ):
            metadata_row = get_metadata_row(self.epochs, ei)
            record = _compute_epoch_record(
                epoch_index=ei,
                metadata_row=metadata_row,
                sfreq=ctx.sfreq,
                n_times=n_times,
                channel_data=channel_data,
                modality_by_channel=modality_by_channel,
                available_styles_by_modality=available_styles_for_modality,
            )
            record["ep"] = index[ei]
            records.append(record)
        df = pd.DataFrame.from_records(records, index=index)
        # df = frame_from_records(records, index=index)
        logger.debug("Frequency-domain feature DataFrame shape: %s", df.shape)
        # return cast_columns_to_object(df)
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
