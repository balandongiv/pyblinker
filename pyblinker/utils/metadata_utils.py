"""Helpers for manipulating blink-related metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from pyblinker._logging import get_logger

from .dict_utils import contains_key
from .iter_utils import ensure_list

logger = get_logger(__name__)

if TYPE_CHECKING:
    import mne


def onset_entry_to_blinks(onset: Any) -> List[Dict[str, float]]:
    """Convert a ``blink_onset`` metadata entry into blink dictionaries."""

    logger.debug("Entering onset_entry_to_blinks")
    if isinstance(onset, list):
        blinks = [{"onset": float(o)} for o in onset]
    elif onset is None or pd.isna(onset):
        blinks = []
    else:
        blinks = [{"onset": float(onset)}]
    logger.debug("Converted %s to %d blink entries", onset, len(blinks))
    logger.debug("Exiting onset_entry_to_blinks")
    return blinks


def attach_blink_metadata(epochs: "mne.Epochs", blink_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-blink properties and merge them into epoch metadata."""

    logger.debug("Entering attach_blink_metadata")

    sfreq = float(epochs.info["sfreq"])
    selection_map = {orig: new for new, orig in enumerate(epochs.selection)}

    df = blink_df.copy()
    df["epoch_index"] = df["seg_id"].map(selection_map)
    df = df.dropna(subset=["epoch_index"]).reset_index(drop=True)
    df["epoch_index"] = df["epoch_index"].astype(int)

    df["blink_onset"] = df["start_blink"] / sfreq
    df["blink_duration"] = (df["end_blink"] - df["start_blink"]) / sfreq

    group = df.groupby("epoch_index")
    n_epochs = len(epochs)
    epoch_meta = pd.DataFrame(index=range(n_epochs))
    epoch_meta["n_blinks"] = group.size().reindex(epoch_meta.index, fill_value=0)

    def _list_or_nan(series: pd.Series) -> object:
        values = series.dropna().tolist()
        return values if values else float("nan")

    cols_to_attach = [
        c for c in df.columns if c not in {"seg_id", "blink_id", "epoch_index"}
    ]
    for col in cols_to_attach:
        epoch_meta[col] = group[col].apply(_list_or_nan).reindex(epoch_meta.index)

    epoch_meta.index.name = None

    existing = (
        epochs.metadata.copy()
        if isinstance(epochs.metadata, pd.DataFrame)
        else pd.DataFrame(index=range(n_epochs))
    )
    existing = existing.reset_index(drop=True)
    keep_cols = [
        c
        for c in existing.columns
        if (c not in epoch_meta.columns)
        and not (c.startswith("blink_") or c == "n_blinks")
    ]
    merged = existing[keep_cols].join(epoch_meta)
    epochs.metadata = merged
    epochs.metadata.reset_index(drop=True, inplace=True)

    logger.debug("Exiting attach_blink_metadata")
    return df.drop(columns=["epoch_index"])


def sample_windows_from_metadata(
    metadata: pd.Series | Dict[str, Any],
    channel: str,
    sfreq: float,
    n_times: int,
    epoch_index: int,
) -> List[slice]:
    """Convert blink onset/duration metadata to sample windows."""

    logger.debug("Entering sample_windows_from_metadata")
    windows = extract_blink_windows(metadata, channel, epoch_index)
    sample_windows: List[slice] = []
    for onset_s, duration_s in windows:
        sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
        if sl.stop - sl.start > 1:
            sample_windows.append(sl)
    logger.debug("Found %d sample windows", len(sample_windows))
    logger.debug("Exiting sample_windows_from_metadata")
    return sample_windows


def segment_to_samples(
    onset_s: float,
    duration_s: float,
    sfreq: float,
    n_times: int,
) -> slice:
    """Convert blink onset/duration in seconds to a sample slice."""

    logger.debug("Entering segment_to_samples")
    start = int(round(onset_s * sfreq))
    stop = start + int(round(duration_s * sfreq))
    start = max(start, 0)
    stop = min(stop, n_times)
    logger.debug("Blink window samples: start=%d stop=%d", start, stop)
    logger.debug("Exiting segment_to_samples")
    return slice(start, stop)


def extract_blink_windows(
    metadata_row: pd.Series | Mapping[str, object],
    channel: str | None,
    epoch_index: int,
) -> List[Tuple[float, float]]:
    """Extract blink onset/duration pairs for a single epoch."""

    channel_label = channel if channel is not None else "generic"
    logger.debug(
        "Entering extract_blink_windows for channel %s (epoch %d)",
        channel_label,
        epoch_index,
    )

    if not isinstance(metadata_row, (pd.Series, Mapping)):
        logger.error("Unsupported metadata row type: %s", type(metadata_row))
        raise TypeError("metadata_row must be a pandas.Series or mapping")

    ch_lower = channel_label.lower()
    prefer_generic = ch_lower in {"generic", "all", "any", ""}
    if "ear" in ch_lower:
        mod = "ear"
    elif "eog" in ch_lower:
        mod = "eog"
    else:
        mod = "eeg"

    mod_onset_key = f"blink_onset_{mod}"
    mod_duration_key = f"blink_duration_{mod}"

    def _is_missing(val: object) -> bool:
        return val is None or (isinstance(val, float) and np.isnan(val))

    if (
        (not prefer_generic)
        and contains_key(metadata_row, mod_onset_key)
        and contains_key(metadata_row, mod_duration_key)
    ):
        onsets = metadata_row.get(mod_onset_key)
        durations = metadata_row.get(mod_duration_key)
        if _is_missing(onsets) or _is_missing(durations):
            logger.debug(
                "Modality-specific metadata for channel %s contains missing values",
                channel_label,
            )
            windows: List[Tuple[float, float]] = []
            logger.debug("Exiting extract_blink_windows")
            return windows
    else:
        generic_keys = ("blink_onset", "blink_duration")
        missing = [key for key in generic_keys if not contains_key(metadata_row, key)]
        if missing:
            logger.error(
                "Missing blink metadata columns: %s", ", ".join(sorted(missing))
            )
            raise ValueError(
                "Epochs.metadata missing required blink columns: "
                + ", ".join(sorted(missing))
            )
        onsets = metadata_row.get("blink_onset")
        durations = metadata_row.get("blink_duration")
        if _is_missing(onsets) or _is_missing(durations):
            logger.debug(
                "Generic metadata contains missing values for channel %s",
                channel_label,
            )
            windows = []
            logger.debug("Exiting extract_blink_windows")
            return windows

    onsets_list = ensure_list(onsets)
    durations_list = ensure_list(durations)

    windows = []
    for onset, duration in zip(onsets_list, durations_list):
        if _is_missing(onset) or _is_missing(duration):
            continue
        windows.append((float(onset), float(duration)))

    logger.debug("Extracted %d blink windows", len(windows))
    logger.debug("Exiting extract_blink_windows")
    return windows


__all__ = [
    "onset_entry_to_blinks",
    "attach_blink_metadata",
    "sample_windows_from_metadata",
    "segment_to_samples",
    "extract_blink_windows",
]
