"""Epoch and segmentation helper utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyblinker._logging import get_logger
from .channel_utils import normalize_picks, require_channels

logger = get_logger(__name__)


ChannelSelector = Callable[[mne.Epochs], Sequence[str]]


def resolve_channels(
    epochs: mne.Epochs,
    picks: str | Sequence[str] | None,
    *,
    default: ChannelSelector | None = None,
) -> List[str]:
    """Resolve and validate channel names for feature extraction."""

    logger.debug("Resolving channel picks: %s", picks)
    if picks is None:
        ch_names = list(epochs.ch_names) if default is None else list(default(epochs))
    else:
        ch_names = normalize_picks(picks)
    require_channels(epochs, ch_names)
    return ch_names


def build_metric_stat_columns(
    ch_names: Sequence[str], metrics: Sequence[str], stats: Sequence[str]
) -> List[str]:
    """Construct column names for metric/statistic combinations per channel."""

    return [
        f"{metric}_{stat}_{ch}"
        for ch in ch_names
        for metric in metrics
        for stat in stats
    ]


def slice_raw_to_segments(
    raw: mne.io.BaseRaw,
    epoch_len: float = 30.0,
    *,
    progress_bar: bool = True,
) -> List[mne.io.BaseRaw]:
    """Slice a continuous :class:`mne.io.BaseRaw` into fixed-length segments."""

    n_segments = int(raw.times[-1] // epoch_len)
    segments: List[mne.io.BaseRaw] = []
    for i in tqdm(
        range(n_segments), desc="Segmenting", unit="segment", disable=not progress_bar
    ):
        start = i * epoch_len
        stop = start + epoch_len
        seg = raw.copy().crop(tmin=start, tmax=stop, include_tmax=False)
        segments.append(seg)
    logger.info("Created %d segments", n_segments)
    return segments


def slice_raw_into_mne_epochs(
    raw: mne.io.BaseRaw,
    *,
    epoch_len: float = 30.0,
    blink_label: Optional[str] = "blink",
    progress_bar: bool = True,
) -> mne.Epochs:
    """Convert a continuous recording into equally spaced MNE epochs."""

    logger.debug("Entering slice_raw_into_mne_epochs")
    events = mne.make_fixed_length_events(raw, duration=epoch_len)
    sfreq = raw.info["sfreq"]
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=epoch_len - 1 / sfreq,
        baseline=None,
        preload=True,
        verbose=False,
    )
    metadata = pd.DataFrame(
        {"blink_onset": [None] * len(epochs), "blink_duration": [None] * len(epochs)}
    )
    ann = raw.annotations
    if len(ann):
        mask = np.ones(len(ann), dtype=bool)
        if blink_label is not None:
            mask &= ann.description == blink_label
        onsets = ann.onset[mask]
        durations = ann.duration[mask]
        event_times = events[:, 0] / sfreq
        iterator = range(len(event_times))
        if progress_bar:
            iterator = tqdm(iterator, desc="Assigning annotations", unit="epoch")
        for idx in iterator:
            start = event_times[idx]
            stop = start + epoch_len
            in_epoch = (onsets >= start) & (onsets < stop)
            if np.any(in_epoch):
                rel_onsets = onsets[in_epoch] - start
                rel_durations = durations[in_epoch]
                if len(rel_onsets) == 1:
                    metadata.at[idx, "blink_onset"] = float(rel_onsets[0])
                    metadata.at[idx, "blink_duration"] = float(rel_durations[0])
                else:
                    metadata.at[idx, "blink_onset"] = rel_onsets.tolist()
                    metadata.at[idx, "blink_duration"] = rel_durations.tolist()
    epochs.metadata = metadata
    logger.debug("Epoch metadata head: %s", metadata.head())
    logger.debug("Exiting slice_raw_into_mne_epochs")
    return epochs


def slice_raw_into_epochs(
    raw: mne.io.BaseRaw,
    *,
    epoch_len: float = 30.0,
    blink_label: Optional[str] = "blink",
    progress_bar: bool = True,
) -> Tuple[
    List[mne.io.BaseRaw], pd.DataFrame, List[Tuple[int, int]], List[Tuple[float, float]]
]:
    """Slice a raw recording into epochs and count blink annotations."""

    logger.info("Slicing raw into epochs (%.1fs)", epoch_len)

    ann = raw.annotations
    mask = np.ones(len(ann), dtype=bool)
    if blink_label is not None:
        mask &= ann.description == blink_label
    onsets = ann.onset[mask]
    durations = ann.duration[mask]

    total_time = raw.times[-1]
    n_epochs = int(np.ceil(total_time / epoch_len))
    counts: List[int] = [0] * n_epochs
    boundary_pairs: List[Tuple[int, int]] = []
    segments: List[mne.io.BaseRaw] = []
    times: List[Tuple[float, float]] = []

    for i in tqdm(
        range(n_epochs), desc="Cropping epochs", unit="epoch", disable=not progress_bar
    ):
        start = i * epoch_len
        stop = min(start + epoch_len, total_time)
        times.append((start, stop))

        in_epoch = (onsets >= start) & (onsets < stop)
        counts[i] = int(np.sum(in_epoch))
        spans = in_epoch & ((onsets + durations) > stop)
        for _ in np.where(spans)[0]:
            if i + 1 < n_epochs:
                boundary_pairs.append((i, i + 1))

        mini = raw.copy().crop(tmin=start, tmax=stop, include_tmax=False)
        ann_epoch = mini.annotations
        shifted = mne.Annotations(
            onset=ann_epoch.onset - start,
            duration=ann_epoch.duration,
            description=ann_epoch.description,
        )
        mini.set_annotations(shifted)
        segments.append(mini)

    df = pd.DataFrame({"epoch_id": range(n_epochs), "blink_count": counts})
    logger.debug("Blink counts per epoch: %s", counts)
    logger.debug("Cross-boundary pairs: %s", boundary_pairs)
    return segments, df, boundary_pairs, times


def slice_into_mini_raws(
    raw: mne.io.BaseRaw,
    out_dir: Path,
    *,
    epoch_len: float = 30.0,
    blink_label: Optional[str] = "blink",
    save: bool = True,
    overwrite: bool = False,
    report: bool = False,
    progress_bar: bool = True,
) -> Tuple[
    List[mne.io.BaseRaw], pd.DataFrame, List[Tuple[int, int]], Optional["mne.Report"]
]:
    """Slice a raw recording into epochs with optional saving and reporting."""

    logger.debug("Entering slice_into_mini_raws")
    segments, df, boundary_pairs, times = slice_raw_into_epochs(
        raw, epoch_len=epoch_len, blink_label=blink_label, progress_bar=progress_bar
    )
    rep: Optional["mne.Report"] = None
    if save:
        from .io_utils import save_epoch_raws  # Local import to avoid cycles

        save_epoch_raws(segments, times, out_dir, overwrite=overwrite, verbose=False)
        if report:
            from .report_utils import generate_epoch_report

            rep = generate_epoch_report(segments, times, verbose=False)
            rep.save(
                out_dir / "epoch_report.html", overwrite=overwrite, open_browser=False
            )
    logger.debug("Exiting slice_into_mini_raws")
    return segments, df, boundary_pairs, rep


__all__ = [
    "resolve_channels",
    "build_metric_stat_columns",
    "slice_raw_to_segments",
    "slice_raw_into_mne_epochs",
    "slice_raw_into_epochs",
    "slice_into_mini_raws",
]
