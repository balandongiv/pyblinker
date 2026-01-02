"""Blink refinement utilities shared across pipelines."""

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyblinker.logging import get_logger
from pyblinker.blink_features.ear_metrics.refinement import (
    EARRefinementConfig,
    _progressive_search,
)

from .channel_utils import pick_ear_channels_from_raw
from .dict_utils import append_to_slot
from pyblinker.blink_features.blink_events.blink_dataframe import (
    compute_outer_bounds,
)

logger = get_logger(__name__)


def _init_metadata(
    n_epochs: int,
    have_eeg: bool,
    have_eog: bool,
    have_ear: bool,
) -> Dict[str, List[Any]]:
    """Create metadata dict with required (manual) and conditional fields."""

    md: Dict[str, List[Any]] = {
        "blink_onset": [np.nan] * n_epochs,
        "blink_duration": [np.nan] * n_epochs,
        "n_blinks": [0] * n_epochs,
    }
    # Since we may have different ways to define the blink onset per modality, and per crossing like zero crossing, tent crossing, half base crossing.
    # So here, we just create all the possible fields.
    if have_eeg:
        md["blink_onset_eeg"] = [np.nan] * n_epochs
        md["blink_duration_eeg"] = [np.nan] * n_epochs
        md["blink_onset_extremum_eeg"] = [np.nan] * n_epochs
        md["blink_outer_start_eeg"] = [np.nan] * n_epochs
        md["blink_outer_end_eeg"] = [np.nan] * n_epochs
    if have_eog:
        md["blink_onset_eog"] = [np.nan] * n_epochs
        md["blink_duration_eog"] = [np.nan] * n_epochs
        md["blink_onset_extremum_eog"] = [np.nan] * n_epochs
        md["blink_outer_start_eog"] = [np.nan] * n_epochs
        md["blink_outer_end_eog"] = [np.nan] * n_epochs
    if have_ear:
        md["blink_onset_ear"] = [np.nan] * n_epochs
        md["blink_duration_ear"] = [np.nan] * n_epochs
        md["blink_onset_extremum_ear"] = [np.nan] * n_epochs
        md["blink_outer_start_ear"] = [np.nan] * n_epochs
        md["blink_outer_end_ear"] = [np.nan] * n_epochs
        md["refined_start_sample"] = [np.nan] * n_epochs
        md["refined_end_sample"] = [np.nan] * n_epochs
        md["refined_left_threshold"] = [np.nan] * n_epochs
        md["refined_right_threshold"] = [np.nan] * n_epochs
        md["search_window_start_sample"] = [np.nan] * n_epochs
        md["search_window_end_sample"] = [np.nan] * n_epochs
        md["search_window_start_time"] = [np.nan] * n_epochs
        md["search_window_end_time"] = [np.nan] * n_epochs
        md["refinement_succeeded"] = [None] * n_epochs
        md["search_exhausted"] = [None] * n_epochs
        md["extension_seconds_used"] = [np.nan] * n_epochs
        md["extension_attempts"] = [np.nan] * n_epochs
    return md


def _prepare_segmentation_config(
    segmentation_type: Optional[dict],
    ear_threshold: Optional[float],
) -> Dict[str, Any]:
    """Merge segmentation settings with convenience parameters."""

    config: Dict[str, Any] = dict(segmentation_type or {})
    ear_config = dict(config.get("ear", {}))
    if ear_threshold is not None:
        ear_config.setdefault("threshold", ear_threshold)
        ear_config.setdefault("seg_type", "threshold_interpolation")
        config["ear"] = ear_config
    return config


def _compute_epoch_blink_bounds(
    blink_onsets_sec: np.ndarray,
    blink_durs_sec: np.ndarray,
    epoch_start_sec: float,
    epoch_end_sec: float,
    sfreq: float,
    n_samp_epoch: int,
) -> Tuple[List[int], List[int]]:
    """Return coarse blink start and end samples for a given epoch."""

    blink_starts: List[int] = []
    blink_ends: List[int] = []
    for onset_sec, dur_sec in zip(blink_onsets_sec, blink_durs_sec):
        ann_start = float(onset_sec)
        ann_end = float(onset_sec + max(dur_sec, 0.0))
        if max(ann_start, epoch_start_sec) < min(ann_end, epoch_end_sec):
            start_rel = int(
                np.clip(round((ann_start - epoch_start_sec) * sfreq), 0, n_samp_epoch - 1)
            )
            end_rel = int(
                np.clip(
                    round((ann_end - epoch_start_sec) * sfreq) - 1,
                    0,
                    n_samp_epoch - 1,
                )
            )
            if end_rel < start_rel:
                end_rel = start_rel
            blink_starts.append(start_rel)
            blink_ends.append(end_rel)

    return blink_starts, blink_ends


def _select_seg_type(seg_type_value: Any) -> Optional[str]:
    """Normalize segmentation type configuration."""

    if isinstance(seg_type_value, str):
        return seg_type_value
    if isinstance(seg_type_value, Sequence):
        seg_types = [str(item) for item in seg_type_value]
        if not seg_types:
            return None
        if "threshold_interpolation" in seg_types:
            return "threshold_interpolation"
        return seg_types[0]
    return None


def _locate_trough(signal: np.ndarray, start: int, end: int) -> Optional[int]:
    """Return the index of the minimum value between ``start`` and ``end``."""

    if signal.size == 0:
        return None
    n = len(signal)
    start_idx = int(np.clip(start, 0, n - 1))
    end_idx = int(np.clip(end, 0, n - 1))
    if end_idx < start_idx:
        start_idx = end_idx
    window = signal[start_idx : end_idx + 1]
    if window.size == 0:
        return start_idx
    finite_mask = np.isfinite(window)
    if not finite_mask.any():
        return start_idx
    local_min = int(np.nanargmin(window))
    return start_idx + local_min


def _refine_ear_blinks_for_epoch(
    segment: np.ndarray,
    blink_starts: Sequence[int],
    blink_ends: Sequence[int],
    sfreq: float,
    segmentation_config: Optional[dict],
) -> List[Dict[str, Any]]:
    """Refine EAR blinks for a single epoch based on segmentation settings."""

    if segment.size == 0 or not blink_starts:
        return []

    ear_config = (segmentation_config or {}).get("ear", {}) if segmentation_config is not None else {}
    seg_type = _select_seg_type(ear_config.get("seg_type"))
    use_threshold_interpolation = seg_type == "threshold_interpolation"

    def _fallback_refinement(coarse_start: int, coarse_end: int) -> Dict[str, Any]:
        n = len(segment)
        if n == 0:
            start = end = 0
        else:
            start = int(np.clip(coarse_start, 0, n - 1))
            end = int(np.clip(coarse_end, 0, n - 1))
            if end < start:
                end = start
        return {
            "refined_start_sample": int(start),
            "refined_end_sample": int(end),
            "refined_left_threshold": int(start),
            "refined_right_threshold": int(end),
            "search_window_start_sample": int(start),
            "search_window_end_sample": int(end),
            "search_window_start_time": float(start / sfreq),
            "search_window_end_time": float(end / sfreq),
            "refinement_succeeded": False,
            "search_exhausted": True,
            "extension_seconds_used": 0.0,
            "extension_attempts": 0,
        }

    threshold = ear_config.get("threshold")
    if use_threshold_interpolation and threshold is None:
        logger.warning("EAR threshold missing for threshold_interpolation; falling back to coarse bounds.")
        use_threshold_interpolation = False

    config: EARRefinementConfig | None = None
    if use_threshold_interpolation:
        config = EARRefinementConfig(threshold=threshold)
        override_fields = {
            k: v
            for k, v in ear_config.items()
            if k
            in {
                "annotation_time_unit",
                "max_extension",
                "extension_step",
                "padding",
                "extend_before",
                "extend_after",
            }
        }
        if override_fields:
            config = replace(config, **override_fields)

    refinements: List[Dict[str, Any]] = []
    for coarse_start, coarse_end in zip(blink_starts, blink_ends):
        if use_threshold_interpolation and config is not None:
            refinement = _progressive_search(
                segment,
                threshold,
                coarse_start,
                coarse_end,
                sfreq,
                config,
            )
        else:
            refinement = _fallback_refinement(coarse_start, coarse_end)

        refinement["refined_trough_sample"] = _locate_trough(
            segment,
            refinement["refined_start_sample"],
            refinement["refined_end_sample"],
        )
        refinements.append(refinement)

    return refinements


def _append_outer_bounds_from_peaks(
    md: Dict[str, List[Any]],
    epoch_index: int,
    peaks: Sequence[int],
    key_prefix: str,
    n_samp_epoch: int,
) -> None:
    if not peaks:
        return
    bounds = compute_outer_bounds(peaks, n_samp_epoch)
    for outer_start, outer_end in bounds:
        md[f"blink_outer_start_{key_prefix}"][epoch_index] = append_to_slot(
            md[f"blink_outer_start_{key_prefix}"][epoch_index], outer_start
        )
        md[f"blink_outer_end_{key_prefix}"][epoch_index] = append_to_slot(
            md[f"blink_outer_end_{key_prefix}"][epoch_index], outer_end
        )


def _append_peak_refinements(
    md: Dict[str, List[Any]],
    epoch_index: int,
    segment: np.ndarray,
    blink_starts: Sequence[int],
    blink_ends: Sequence[int],
    sfreq: float,
    key_prefix: str,
    refine_func: Callable[[np.ndarray, int, int, int | None], Tuple[int, int, int]],
    n_samp_epoch: int,
) -> None:
    if segment.size == 0 or not blink_starts:
        return
    peaks: List[int] = []
    for start, end in zip(blink_starts, blink_ends):
        refined_start, peak, refined_end = refine_func(segment, start, end, peak_rel_cvat=None)
        peaks.append(int(peak))
        md[f"blink_onset_{key_prefix}"][epoch_index] = append_to_slot(
            md[f"blink_onset_{key_prefix}"][epoch_index], refined_start / sfreq
        )
        md[f"blink_duration_{key_prefix}"][epoch_index] = append_to_slot(
            md[f"blink_duration_{key_prefix}"][epoch_index],
            max(0.0, (refined_end - refined_start) / sfreq),
        )
        md[f"blink_onset_extremum_{key_prefix}"][epoch_index] = append_to_slot(
            md[f"blink_onset_extremum_{key_prefix}"][epoch_index], peak / sfreq
        )

    _append_outer_bounds_from_peaks(md, epoch_index, peaks, key_prefix, n_samp_epoch)


def _append_ear_refinements(
    md: Dict[str, List[Any]],
    epoch_index: int,
    refinements: Sequence[Dict[str, Any]],
    sfreq: float,
    n_samp_epoch: int,
) -> None:
    if not refinements:
        return

    peaks: List[int] = []
    for refinement in refinements:
        start = refinement["refined_start_sample"]
        end = refinement["refined_end_sample"]
        trough = refinement.get("refined_trough_sample")

        peaks.append(int(trough) if trough is not None else int(start))
        md["blink_onset_ear"][epoch_index] = append_to_slot(
            md["blink_onset_ear"][epoch_index], start / sfreq
        )
        md["blink_duration_ear"][epoch_index] = append_to_slot(
            md["blink_duration_ear"][epoch_index], max(0.0, (end - start) / sfreq)
        )
        md["blink_onset_extremum_ear"][epoch_index] = append_to_slot(
            md["blink_onset_extremum_ear"][epoch_index],
            (trough if trough is not None else start) / sfreq,
        )
        md["refined_start_sample"][epoch_index] = append_to_slot(
            md["refined_start_sample"][epoch_index], int(start)
        )
        md["refined_end_sample"][epoch_index] = append_to_slot(
            md["refined_end_sample"][epoch_index], int(end)
        )
        md["refined_left_threshold"][epoch_index] = append_to_slot(
            md["refined_left_threshold"][epoch_index], int(refinement["refined_left_threshold"])
        )
        md["refined_right_threshold"][epoch_index] = append_to_slot(
            md["refined_right_threshold"][epoch_index], int(refinement["refined_right_threshold"])
        )
        md["search_window_start_sample"][epoch_index] = append_to_slot(
            md["search_window_start_sample"][epoch_index],
            int(refinement["search_window_start_sample"]),
        )
        md["search_window_end_sample"][epoch_index] = append_to_slot(
            md["search_window_end_sample"][epoch_index],
            int(refinement["search_window_end_sample"]),
        )
        md["search_window_start_time"][epoch_index] = append_to_slot(
            md["search_window_start_time"][epoch_index],
            float(refinement["search_window_start_time"]),
        )
        md["search_window_end_time"][epoch_index] = append_to_slot(
            md["search_window_end_time"][epoch_index],
            float(refinement["search_window_end_time"]),
        )
        md["refinement_succeeded"][epoch_index] = append_to_slot(
            md["refinement_succeeded"][epoch_index],
            bool(refinement["refinement_succeeded"]),
        )
        md["search_exhausted"][epoch_index] = append_to_slot(
            md["search_exhausted"][epoch_index],
            bool(refinement["search_exhausted"]),
        )
        md["extension_seconds_used"][epoch_index] = append_to_slot(
            md["extension_seconds_used"][epoch_index],
            float(refinement["extension_seconds_used"]),
        )
        md["extension_attempts"][epoch_index] = append_to_slot(
            md["extension_attempts"][epoch_index],
            int(refinement["extension_attempts"]),
        )

    _append_outer_bounds_from_peaks(md, epoch_index, peaks, "ear", n_samp_epoch)


def slice_raw_into_mne_epochs_refine_annot(
    # TODO : We need to enrich this function to support different blink onset selection strategy.
    raw: mne.io.BaseRaw,
    *,
    epoch_len: float = 30.0,
    blink_label: Optional[str] = "blink",
    progress_bar: bool = True,
    segmentation_type: Optional[dict] = None,
    ear_threshold: Optional[float] = None,
) -> mne.Epochs:
    """When to use this refinement, we use this when user is assume to vizualy do the annotation.
    Convert a continuous recording into equally spaced epochs with refinement."""

    events = mne.make_fixed_length_events(raw, duration=epoch_len)
    sfreq = float(raw.info["sfreq"])
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=epoch_len - 1.0 / sfreq,
        baseline=None,
        preload=True,
        verbose=False,
    )

    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, misc=False)
    picks_eog = mne.pick_types(raw.info, eeg=False, eog=True, misc=False)
    picks_ear = pick_ear_channels_from_raw(raw)

    have_eeg = len(picks_eeg) > 0
    have_eog = len(picks_eog) > 0
    have_ear = len(picks_ear) > 0

    data_eeg = epochs.get_data(picks=picks_eeg) if have_eeg else None
    data_eog = epochs.get_data(picks=picks_eog) if have_eog else None
    data_ear = epochs.get_data(picks=picks_ear) if have_ear else None

    n_epochs = len(epochs)
    n_samp_epoch = (
        epochs.get_data(picks=[0]).shape[-1]
        if epochs.info["nchan"] > 0
        else int(round(epoch_len * sfreq))
    )

    ann = raw.annotations
    if blink_label is None:
        sel = np.ones(len(ann), dtype=bool)
    else:
        sel = np.array(
            [(str(d).lower() == blink_label.lower()) for d in ann.description],
            dtype=bool,
        )

    blink_onsets_sec = np.array(ann.onset)[sel]
    blink_durs_sec = np.array(ann.duration)[sel]

    segment_config = _prepare_segmentation_config(segmentation_type, ear_threshold)
    md = _init_metadata(n_epochs, have_eeg, have_eog, have_ear)

    iterator = range(n_epochs)
    if progress_bar:
        iterator = tqdm(iterator, desc="Refining blink metadata", unit="epoch")

    for ei in iterator:
        epoch_start_samp = int(epochs.events[ei, 0])
        epoch_start_sec = epoch_start_samp / sfreq
        epoch_end_sec = epoch_start_sec + epoch_len

        blink_starts, blink_ends = _compute_epoch_blink_bounds(
            blink_onsets_sec,
            blink_durs_sec,
            epoch_start_sec,
            epoch_end_sec,
            sfreq,
            n_samp_epoch,
        )

        n_blinks = len(blink_starts)
        md["n_blinks"][ei] = n_blinks
        if n_blinks == 0:
            continue

        for sr, er in zip(blink_starts, blink_ends):
            onset_sec_rel = sr / sfreq
            duration_sec_rel = max(0.0, (er - sr) / sfreq)
            md["blink_onset"][ei] = append_to_slot(md["blink_onset"][ei], onset_sec_rel)
            md["blink_duration"][ei] = append_to_slot(
                md["blink_duration"][ei], duration_sec_rel
            )

        if have_ear and data_ear is not None:
            seg = data_ear[ei].mean(axis=0)
            refinements = _refine_ear_blinks_for_epoch(
                seg,
                blink_starts,
                blink_ends,
                sfreq,
                segment_config,
            )
            _append_ear_refinements(md, ei, refinements, sfreq, n_samp_epoch)

        if have_eeg and data_eeg is not None:
            seg = data_eeg[ei].mean(axis=0)
            _append_peak_refinements(
                md,
                ei,
                seg,
                blink_starts,
                blink_ends,
                sfreq,
                "eeg",
                refine_local_maximum_stub,
                n_samp_epoch,
            )

        if have_eog and data_eog is not None:
            seg = data_eog[ei].mean(axis=0)
            _append_peak_refinements(
                md,
                ei,
                seg,
                blink_starts,
                blink_ends,
                sfreq,
                "eog",
                refine_local_maximum_stub,
                n_samp_epoch,
            )

    metadata = pd.DataFrame(md)
    epochs.metadata = metadata

    logger.debug("Epoch metadata head: %s", metadata.head())
    logger.debug("Exiting slice_raw_into_mne_epochs_refine_annot")
    return epochs


def refine_ear_extrema_and_threshold_stub(
    signal_segment: np.ndarray,
    start_rel: int,
    end_rel: int,
    ear_threshold: float,
    seg_type: str = "threshold_interpolation",
) -> Tuple[int, int, int]:
    """Return a crude EAR trough refinement.
    We will return the sample index.
    Also, for ear, we use all calculation on the interpolation between the threshold horizontal line and the signal."""

    n = len(signal_segment)
    if n == 0:
        return 0, 0, 0

    rs_stub = max(0, min(start_rel, n - 1))
    re_stub = max(0, min(end_rel, n - 1))
    if rs_stub > re_stub:
        rs_stub = re_stub

    window = signal_segment[rs_stub : re_stub + 1]
    if window.size == 0 or not np.isfinite(window).any():
        trough = rs_stub
    else:
        trough = rs_stub + int(np.nanargmin(window))

    return rs_stub, int(trough), re_stub


def refine_local_maximum_stub(
    signal_segment: np.ndarray,
    start_rel: int,
    end_rel: int,
    peak_rel_cvat: int | None = None,
) -> Tuple[int, int, int]:
    """Return a crude refinement for local maxima in a signal segment."""

    n = len(signal_segment)
    if n == 0:
        return 0, 0, 0

    rs_stub = max(0, min(start_rel, n - 1))
    re_stub = max(0, min(end_rel, n - 1))
    if rs_stub > re_stub:
        rs_stub = re_stub = min(rs_stub, re_stub)

    if peak_rel_cvat is not None and rs_stub <= peak_rel_cvat <= re_stub:
        valid_peak = peak_rel_cvat
    else:
        segment = signal_segment[rs_stub : re_stub + 1]
        max_idx_local = int(np.argmax(segment))
        valid_peak = rs_stub + max_idx_local

    return rs_stub, valid_peak, re_stub


def plot_refined_blinks(
    refined_blinks: Sequence[Dict[str, Any]],
    sfreq: float,
    epoch_len: float,
    *,
    epoch_indices: Optional[Sequence[int]] = None,
    show: bool = False,
) -> List[plt.Figure]:
    """Plot signal segments with refined blink markers."""

    epochs_to_plot: Dict[int, Dict[str, Any]] = {}
    for blink in refined_blinks:
        idx = blink["epoch_index"]
        if epoch_indices is None or idx in epoch_indices:
            if idx not in epochs_to_plot:
                epochs_to_plot[idx] = {"signal": blink["epoch_signal"], "blinks": []}
            epochs_to_plot[idx]["blinks"].append(blink)

    if not epochs_to_plot:
        logger.warning("No epochs selected for plotting")
        return []

    figs: List[plt.Figure] = []
    time_axis = np.arange(0, epoch_len, 1.0 / sfreq)
    for epoch_index, data in epochs_to_plot.items():
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(time_axis, data["signal"], label="Signal")
        for blink in data["blinks"]:
            start_t = blink["refined_start_frame"] / sfreq
            peak_t = blink["refined_peak_frame"] / sfreq
            end_t = blink["refined_end_frame"] / sfreq
            ax.axvline(start_t, color="g", linestyle="--")
            ax.axvline(peak_t, color="r")
            ax.axvline(end_t, color="b", linestyle="--")
        ax.set_title(f"Epoch {epoch_index}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        figs.append(fig)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return figs


def refine_blinks_from_epochs(
    segments: Sequence[mne.io.BaseRaw],
    channel: str,
    *,
    refine_func: Callable[[np.ndarray, int, int, int | None], Tuple[int, int, int]] = refine_local_maximum_stub,
    local_max_prominence: float = 0.01,
    search_expansion_frames: int | None = None,
    value_threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Refine blink annotations within pre-sliced raw segments."""

    logger.info("Refining blinks across %d segments", len(segments))
    refined: List[Dict[str, Any]] = []
    if not segments:
        return refined

    sfreq = float(segments[0].info["sfreq"])
    if search_expansion_frames is None:
        search_expansion_frames = int(0.1 * sfreq)

    for epoch_index, segment in enumerate(segments):
        data = segment.get_data(picks=[channel])
        if data.size == 0:
            continue
        signal = data[0]
        ann = segment.annotations
        for ann_idx in range(len(ann)):
            onset = ann.onset[ann_idx]
            duration = ann.duration[ann_idx]
            start_rel = int(max(0, round(onset * sfreq) - search_expansion_frames))
            end_rel = int(round((onset + duration) * sfreq) + search_expansion_frames)
            end_rel = min(end_rel, len(signal) - 1)
            if end_rel < start_rel:
                end_rel = start_rel
            rs, peak, re = refine_func(signal, start_rel, end_rel, peak_rel_cvat=None)
            refined.append(
                {
                    "epoch_index": epoch_index,
                    "refined_start_frame": rs,
                    "refined_peak_frame": peak,
                    "refined_end_frame": re,
                    "epoch_signal": signal,
                }
            )

    logger.info("Generated %d refined blink entries", len(refined))
    return refined


__all__ = [
    "slice_raw_into_mne_epochs_refine_annot",
    "refine_ear_extrema_and_threshold_stub",
    "refine_local_maximum_stub",
    "plot_refined_blinks",
    "refine_blinks_from_epochs",
]
