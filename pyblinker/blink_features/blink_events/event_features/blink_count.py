"""Blink count feature utilities."""

from typing import Dict, Iterable, List, Optional, Union
import numpy as np
import pandas as pd
import mne
from pyblinker.logging import get_logger
from pyblinker.utils.modality import infer_modality

from .utils import normalize_picks


logger = get_logger(__name__)


def blink_count_epoch(
        blinks: Union[List[Dict[str, int]], 'mne.io.BaseRaw', 'mne.Epochs'],
        label: Optional[str] = None
) -> int:
    """Return the number of blinks for a single epoch or from an MNE object.

    Parameters
    ----------
    blinks : list of dict or mne.io.Raw or mne.Epochs
        Blink annotations for one epoch (as a list of dicts) or an MNE Raw object.
    label : str, optional
        If using an MNE Raw object, count only annotations matching this label.

    Returns
    -------
    int
        Total count of blinks.

    Raises
    ------
    NotImplementedError
        If the input is an MNE Epochs object.
    TypeError
        If the input type is not supported.
    """
    logger.info("Calculating blink count for input of type %s", type(blinks))

    if mne and isinstance(blinks, mne.io.BaseRaw):
        logger.debug("Using MNE Raw logic for annotation counting")
        mask = np.ones(len(blinks.annotations), dtype=bool)
        if label is not None:
            mask &= blinks.annotations.description == label
        count = int(mask.sum())
        logger.debug("Found %d blink annotations matching label '%s'", count, label)
        return count

    elif mne and isinstance(blinks, mne.Epochs):
        logger.warning("Blink count from MNE Epochs not implemented")
        raise NotImplementedError("blink_count_epoch does not support MNE Epochs input.")

    elif isinstance(blinks, list):
        logger.debug("Counting %s blinks from list of dicts", len(blinks))
        return len(blinks)

    else:
        logger.error("Unsupported type passed to blink_count_epoch: %s", type(blinks))
        raise TypeError(f"Unsupported input type: {type(blinks)}")
def blink_count(
    epochs: mne.Epochs, picks: str | Iterable[str] | None = None
) -> pd.DataFrame:
    """Count blinks for each epoch using metadata.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoch object whose metadata contains blink onset and duration
        information.
    picks : str or iterable of str, optional
        Channel name(s) whose modality determines which blink onset and
        duration columns are used. When channel names are provided the
        resulting column is named ``blink_count_<modality>`` for each unique
        modality encountered. If omitted, the generic ``blink_onset`` and
        ``blink_duration`` columns are used and the output column is simply
        ``blink_count``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` with a leading ``ep`` column. If no
        channel selections are supplied, a single ``blink_count`` column holds
        the per-epoch counts derived from the generic blink metadata. When one
        or more channels are supplied, columns named ``blink_count_<modality>``
        are returned, one for each modality present among the selections.

    Raises
    ------
    ValueError
        If required metadata columns are missing.
    """
    logger.info("Counting blinks across %d epochs", len(epochs))
    metadata = epochs.metadata
    if metadata is None:
        raise ValueError("Epochs.metadata must contain blink information")

    index = metadata.index if isinstance(metadata, pd.DataFrame) else pd.RangeIndex(len(epochs))
    df = pd.DataFrame(index=index)
    df.insert(0, "ep", index.to_numpy())

    picks_list = normalize_picks(picks) if picks is not None else []
    modalities: List[str] = []
    for ch in picks_list:
        mod = infer_modality(ch)
        if mod not in modalities:
            modalities.append(mod)

    def _count(entry: object) -> int:
        if entry is None:
            return 0
        if isinstance(entry, (list, np.ndarray, pd.Series)):
            arr = pd.Series(entry)
            return int((~pd.isna(arr)).sum())
        if pd.isna(entry):
            return 0
        return 1

    if not picks_list:
        onset_col = "blink_onset"
        duration_col = "blink_duration"
        if onset_col not in metadata or duration_col not in metadata:
            missing = [col for col in [onset_col, duration_col] if col not in metadata]
            raise ValueError(
                "Epochs.metadata missing required blink columns: " + ", ".join(sorted(missing))
            )
        df["blink_count"] = metadata[onset_col].apply(_count).astype(float)
        logger.debug("Blink counts per epoch: %s", df["blink_count"].tolist())
    else:
        for modality in modalities:
            onset_col = f"blink_onset_{modality}"
            duration_col = f"blink_duration_{modality}"
            if onset_col not in metadata or duration_col not in metadata:
                logger.debug(
                    "Modality '%s' missing specific columns; using generic blink columns",
                    modality,
                )
                onset_col = "blink_onset"
                duration_col = "blink_duration"
                if onset_col not in metadata or duration_col not in metadata:
                    missing = [
                        col
                        for col in [onset_col, duration_col]
                        if col not in metadata
                    ]
                    raise ValueError(
                        "Epochs.metadata missing required blink columns: "
                        + ", ".join(sorted(missing))
                    )

            col_name = f"blink_count_{modality}"
            df[col_name] = metadata[onset_col].apply(_count).astype(float)
            logger.debug(
                "Blink counts for modality '%s': %s",
                modality,
                df[col_name].tolist(),
            )

    logger.info("Finished counting blinks")
    return df