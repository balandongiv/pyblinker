"""Reporting helpers for blink validation workflows."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
from tqdm import tqdm

from pyblinker._logging import get_logger

from .channel_utils import pick_ear_channels_from_info
from .iter_utils import ensure_float_list

logger = get_logger(__name__)


def generate_epoch_report(
    segments: Iterable[mne.io.BaseRaw],
    times: Iterable[tuple[float, float]],
    *,
    verbose: bool = False,
) -> mne.Report:
    """Create a simple report visualizing each segment."""

    report = mne.Report(title="Epoch Overview")
    for idx, (segment, span) in enumerate(zip(segments, times)):
        start, stop = span
        fig = segment.plot(
            n_channels=min(10, len(segment.ch_names)),
            scalings="auto",
            title=f"Epoch {idx} ({start:.2f}-{stop:.2f}s)",
            show=False,
            verbose=verbose,
        )
        report.add_figure(fig, title=f"Epoch {idx}", section="epochs")
        plt.close(fig)
    return report


def add_blink_plots_to_report(
    epochs: mne.Epochs,
    *,
    report: Optional[mne.Report] = None,
    section: str = "Blink validation",
    pad_pre: float = 0.5,
    pad_post: float = 0.5,
    limit_per_epoch: Optional[int] = None,
    decim: Optional[int] = None,
    include_modalities: Iterable[str] = ("eeg", "eog", "ear"),
    progress_bar: bool = True,
) -> mne.Report:
    """Add per-blink plots to an :class:`mne.Report` for manual validation."""

    if report is None:
        report = mne.Report(title="Blink validation")

    md = epochs.metadata
    if md is None or (
        "blink_onset" not in md.columns and "blink_duration" not in md.columns
    ):
        raise RuntimeError("Epochs.metadata is missing manual blink fields.")

    sfreq = float(epochs.info["sfreq"])
    times = epochs.times
    n_times = times.size
    epoch_len = times[-1] - times[0] + (1.0 / sfreq)

    picks_eeg = (
        mne.pick_types(epochs.info, eeg=True, eog=False, misc=False)
        if "eeg" in include_modalities
        else []
    )
    picks_eog = (
        mne.pick_types(epochs.info, eeg=False, eog=True, misc=False)
        if "eog" in include_modalities
        else []
    )
    picks_ear = (
        pick_ear_channels_from_info(epochs.info) if "ear" in include_modalities else []
    )

    have_eeg = len(picks_eeg) > 0
    have_eog = len(picks_eog) > 0
    have_ear = len(picks_ear) > 0

    data_eeg = epochs.get_data(picks=picks_eeg) if have_eeg else None
    data_eog = epochs.get_data(picks=picks_eog) if have_eog else None
    data_ear = epochs.get_data(picks=picks_ear) if have_ear else None

    def _refined_lists(ei: int, modality: str) -> Dict[str, List[float]]:
        out = {"onset": [], "duration": [], "extremum": []}
        for key, tgt in [
            (f"blink_onset_{modality}", "onset"),
            (f"blink_duration_{modality}", "duration"),
            (f"blink_onset_extremum_{modality}", "extremum"),
        ]:
            out[tgt] = ensure_float_list(md.iloc[ei].get(key, np.nan))
        return out

    epoch_iter = range(len(epochs))
    if progress_bar:
        epoch_iter = tqdm(epoch_iter, desc="Building blink report", unit="epoch")

    for ei in epoch_iter:
        manual_onsets = ensure_float_list(md.iloc[ei]["blink_onset"])
        manual_durs = ensure_float_list(md.iloc[ei]["blink_duration"])
        n_blinks = min(len(manual_onsets), len(manual_durs))

        if n_blinks == 0:
            continue

        if limit_per_epoch is not None:
            n_blinks = min(n_blinks, int(limit_per_epoch))

        ref_eeg = _refined_lists(ei, "eeg") if have_eeg else None
        ref_eog = _refined_lists(ei, "eog") if have_eog else None
        ref_ear = _refined_lists(ei, "ear") if have_ear else None

        for bi in range(n_blinks):
            m_on = manual_onsets[bi]
            m_dur = manual_durs[bi]
            m_off = m_on + max(0.0, m_dur)

            win_start = m_on
            win_end = m_off
            for ref in (ref_eeg, ref_eog, ref_ear):
                if ref is None:
                    continue
                rs = ref["onset"]
                rd = ref["duration"]
                if bi < len(rs) and bi < len(rd):
                    r_on = rs[bi]
                    r_off = rs[bi] + max(0.0, rd[bi])
                    if not math.isnan(r_on):
                        win_start = min(win_start, r_on)
                    if not math.isnan(r_off):
                        win_end = max(win_end, r_off)

            win_start = max(0.0, win_start - pad_pre)
            win_end = min(epoch_len, win_end + pad_post)

            s0 = int(np.clip(round(win_start * sfreq), 0, n_times - 1))
            s1 = int(np.clip(round(win_end * sfreq), 0, n_times - 1))
            if s1 < s0:
                s1 = s0

            t_seg = times[s0 : s1 + 1]

            def _plot_mod(
                mod: str,
                picks: List[int],
                data_mod: np.ndarray | None,
                ref: Optional[Dict[str, List[float]]],
            ) -> None:
                if not picks or data_mod is None:
                    return
                for c_rel, ch_idx in enumerate(picks):
                    ch_name = epochs.ch_names[ch_idx]
                    y = data_mod[ei, c_rel, s0 : s1 + 1]
                    if decim and decim > 1:
                        y = y[::decim]
                        t = t_seg[::decim]
                    else:
                        t = t_seg

                    fig, ax = plt.subplots(figsize=(7.5, 3.0))
                    line = ax.plot(t, y, lw=1.0, alpha=0.6)[0]
                    ax.scatter(
                        t,
                        y,
                        s=25.0,
                        color=line.get_color(),
                        zorder=3,
                    )
                    ax.set_title(f"Epoch {ei} • Blink {bi} • {mod.upper()} • {ch_name}")
                    ax.set_xlabel("Time from epoch start (s)")
                    ax.set_ylabel("Amplitude")

                    ax.axvline(m_on, linestyle="--", alpha=0.9, label="manual onset")
                    ax.axvline(m_off, linestyle="--", alpha=0.9, label="manual offset")

                    if ref is not None:
                        rs = ref["onset"]
                        rd = ref["duration"]
                        rx = ref["extremum"]
                        if bi < len(rs) and not math.isnan(rs[bi]):
                            ax.axvline(
                                rs[bi],
                                linestyle="-",
                                alpha=0.9,
                                label=f"{mod} onset (refined)",
                            )
                        if bi < len(rd) and not math.isnan(rd[bi]):
                            ax.axvline(
                                rs[bi] + max(0.0, rd[bi]),
                                linestyle="-",
                                alpha=0.9,
                                label=f"{mod} offset (refined)",
                            )
                        if bi < len(rx) and not math.isnan(rx[bi]):
                            ax.axvline(
                                rx[bi],
                                linestyle=":",
                                alpha=0.9,
                                label=f"{mod} extremum",
                            )

                    ax.legend(loc="upper right", fontsize=8, ncol=3)
                    ax.grid(True, alpha=0.2)

                    caption = (
                        f"Blink window (manual): {m_on:.3f}s–{m_off:.3f}s. "
                        f"Padding: −{pad_pre:.2f}/+{pad_post:.2f}s. "
                        f"Sampling: {sfreq:.2f} Hz{f', decim={decim}' if decim else ''}."
                    )
                    try:
                        report.add_figure(
                            fig=fig,
                            title=f"Epoch {ei} | Blink {bi} | {mod.upper()} | {ch_name}",
                            caption=caption,
                            section=section,
                            tags=("blink", "validation", mod, ch_name),
                        )
                    finally:
                        plt.close(fig)

            if have_eeg:
                _plot_mod("eeg", picks_eeg, data_eeg, ref_eeg)
            if have_eog:
                _plot_mod("eog", picks_eog, data_eog, ref_eog)
            if have_ear:
                _plot_mod("ear", picks_ear, data_ear, ref_ear)

    return report


__all__ = ["generate_epoch_report", "add_blink_plots_to_report"]
