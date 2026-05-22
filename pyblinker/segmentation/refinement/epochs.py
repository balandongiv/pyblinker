"""Epoch-level blink refinement entry points."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mne
import pandas as pd
from tqdm import tqdm

from pyblinker._logging import get_logger

from .prep import _prepare_epochs_and_modalities, _prepare_segmentation_config
from .refine_epoch import _refine_epoch_modalities

logger = get_logger(__name__)


def slice_raw_into_mne_epochs_refine_annot(
    raw: mne.io.BaseRaw,
    *,
    epoch_len: float = 30.0,
    blink_label: Optional[str] = "blink",
    progress_bar: bool = True,
    segmentation_type: Optional[dict] = None,
) -> mne.Epochs:
    """Convert a continuous recording into equally spaced epochs with refinement."""

    segment_config = _prepare_segmentation_config(segmentation_type)
    prep = _prepare_epochs_and_modalities(
        raw,
        epoch_len=epoch_len,
        blink_label=blink_label,
        segment_config=segment_config,
    )
    metadata_rows: List[Dict[str, Any]] = []

    iterator = range(prep.n_epochs)
    if progress_bar:
        iterator = tqdm(iterator, desc="Refining blink metadata", unit="epoch")

    for ei in iterator:
        metadata_rows.append(
            _refine_epoch_modalities(
                epoch_index=ei,
                epoch_len=epoch_len,
                epochs=prep.epochs,
                sfreq=prep.sfreq,
                n_samp_epoch=prep.n_samp_epoch,
                blink_onsets_sec=prep.blink_onsets_sec,
                blink_durs_sec=prep.blink_durs_sec,
                data_ear=prep.data_ear,
                data_eeg=prep.data_eeg,
                data_eog=prep.data_eog,
                have_ear=prep.have_ear,
                have_eeg=prep.have_eeg,
                have_eog=prep.have_eog,
                segment_config=segment_config,
            )
        )

    metadata = pd.DataFrame(metadata_rows)
    prep.epochs.metadata = metadata

    logger.debug("Epoch metadata head: %s", metadata.head())
    logger.debug("Exiting slice_raw_into_mne_epochs_refine_annot")
    return prep.epochs


__all__ = ["slice_raw_into_mne_epochs_refine_annot"]
