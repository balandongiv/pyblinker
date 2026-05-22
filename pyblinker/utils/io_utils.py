"""Input/output helpers for epoch-based workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import mne
from mne.io import BaseRaw
from tqdm import tqdm

from pyblinker._logging import get_logger

logger = get_logger(__name__)


def save_epoch_raws(
    segments: Sequence[mne.io.BaseRaw],
    times: Sequence[Tuple[float, float]],
    out_dir: Path,
    *,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    """Save cropped raw segments to disk."""

    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (segment, span) in enumerate(zip(segments, times)):
        start, stop = span
        fname = out_dir / f"epoch_{idx:04d}_{start:07.2f}s-{stop:07.2f}s_raw.fif"
        if fname.exists() and not overwrite:
            logger.debug("Skipping existing %s", fname)
            continue
        segment.save(fname, overwrite=overwrite, verbose=verbose)


def _update_segment_annotations(
    segments: Sequence[mne.io.BaseRaw],
    refined: Sequence[Dict[str, int]],
    *,
    progress_bar: bool = True,
) -> None:
    logger.info("Updating annotations for %d segments", len(segments))
    idx = 0
    for seg_idx, seg in enumerate(
        tqdm(segments, desc="Segments", disable=not progress_bar)
    ):
        sfreq = seg.info["sfreq"]
        orig_anns = seg.annotations
        n_anns = len(orig_anns)
        new_onsets: List[float] = []
        new_durations: List[float] = []
        new_descriptions: List[str] = []
        for ann_i in tqdm(
            range(n_anns),
            desc=f"Seg {seg_idx} annotations",
            leave=False,
            disable=not progress_bar,
        ):
            blink_info = refined[idx]
            start_frame = blink_info["refined_start_frame"]
            end_frame = blink_info["refined_end_frame"]
            onset = start_frame / sfreq
            duration = (end_frame - start_frame) / sfreq
            desc = orig_anns.description[ann_i]
            new_onsets.append(onset)
            new_durations.append(duration)
            new_descriptions.append(desc)
            idx += 1
        seg.set_annotations(
            mne.Annotations(
                onset=new_onsets,
                duration=new_durations,
                description=new_descriptions,
            )
        )


def prepare_refined_segments(
    raw: str | Path | mne.io.BaseRaw,
    channel: str,
    *,
    epoch_len: float = 30.0,
    keep_epoch_signal: bool = False,
    progress_bar: bool = True,
) -> tuple[list[BaseRaw], list[dict[str, Any]]]:
    """Load and prepare raw segments with refined blink annotations."""

    logger.info("Preparing raw segments for blink features")
    if isinstance(raw, (str, Path)):
        raw = mne.io.read_raw_fif(raw, preload=False, verbose=False)
    if len(raw.annotations) == 0:
        raise ValueError("Raw recording has no annotations to refine")

    from .epoch_utils import slice_raw_into_epochs
    from pyblinker.segmentation.refinement.eeg import refine_blinks_from_epochs

    segments, _, _, _ = slice_raw_into_epochs(
        raw, epoch_len=epoch_len, progress_bar=progress_bar
    )
    refined = refine_blinks_from_epochs(segments, channel)

    if not keep_epoch_signal:
        for blink in refined:
            blink.pop("epoch_signal", None)

    _update_segment_annotations(segments, refined, progress_bar=progress_bar)
    logger.info("Finished preparing %d segments", len(segments))
    return list(segments), refined


__all__ = ["save_epoch_raws", "prepare_refined_segments"]
