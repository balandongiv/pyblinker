from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.utils.evaluation import mat_data
from pyblinker.utils.refinement_utils import slice_raw_into_mne_epochs_refine_annot

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "tutorial_outputs" / "ear_energy_refinement_report"


def _slot_to_list(slot: object) -> List[object]:
    """Normalize metadata slots to lists."""

    if isinstance(slot, list):
        return slot
    if slot is None or (isinstance(slot, float) and np.isnan(slot)):
        return []
    return [slot]


def _plot_epoch_blink_overlay(
    *,
    epoch_index: int,
    blink_index: int,
    times: np.ndarray,
    ear_signal: np.ndarray,
    eeg_signal: np.ndarray,
    md_row,
    sfreq: float,
) -> plt.Figure:
    """Plot a single blink with EAR + EEG overlays and refinement landmarks."""

    refined_start_samples = _slot_to_list(md_row["refined_start_sample"])
    refined_end_samples = _slot_to_list(md_row["refined_end_sample"])
    lowest_samples = _slot_to_list(md_row.get("refined_lowest_point_sample"))
    left_interp = _slot_to_list(md_row.get("left_interpolated_threshold"))
    right_interp = _slot_to_list(md_row.get("right_interpolated_threshold"))
    left_interp_samples = _slot_to_list(md_row.get("left_interpolated_threshold_sample"))
    right_interp_samples = _slot_to_list(md_row.get("right_interpolated_threshold_sample"))
    search_start_samples = _slot_to_list(md_row.get("search_window_start_sample"))
    search_end_samples = _slot_to_list(md_row.get("search_window_end_sample"))

    try:
        start_sample = int(refined_start_samples[blink_index])
        end_sample = int(refined_end_samples[blink_index])
    except (IndexError, ValueError, TypeError):
        start_sample = 0
        end_sample = 0
    lowest_sample = None
    if lowest_samples and len(lowest_samples) > blink_index:
        try:
            lowest_sample = int(lowest_samples[blink_index])
        except (TypeError, ValueError):
            lowest_sample = None

    fig, (ax_ear, ax_eeg) = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]},
    )

    ax_ear.plot(times, ear_signal, label="EAR", color="C0", lw=1.2)
    ax_ear.axvspan(
        times[start_sample] if start_sample < len(times) else 0.0,
        times[end_sample] if end_sample < len(times) else 0.0,
        color="C0",
        alpha=0.08,
        label="Refined window",
    )
    if lowest_sample is not None and 0 <= lowest_sample < len(times):
        ax_ear.axvline(times[lowest_sample], color="C3", lw=1.2, label="Trough/extremum")
        ax_ear.scatter(
            times[lowest_sample],
            ear_signal[lowest_sample],
            color="C3",
            s=30,
            zorder=5,
        )

    def _maybe_line(ax, sample_list: Sequence, color: str, label: str, linestyle: str = "--"):
        if sample_list and len(sample_list) > blink_index:
            try:
                loc = int(sample_list[blink_index])
            except (TypeError, ValueError):
                return
            if 0 <= loc < len(times):
                ax.axvline(times[loc], color=color, lw=1.0, linestyle=linestyle, label=label)

    _maybe_line(ax_ear, refined_start_samples, "C2", "Refined start")
    _maybe_line(ax_ear, refined_end_samples, "C1", "Refined end")
    _maybe_line(ax_ear, left_interp_samples, "C4", "Interpolated left crossing", linestyle=":")
    _maybe_line(ax_ear, right_interp_samples, "C5", "Interpolated right crossing", linestyle=":")
    _maybe_line(ax_ear, search_start_samples, "0.3", "Search window start", linestyle="-.")
    _maybe_line(ax_ear, search_end_samples, "0.3", "Search window end", linestyle="-.")

    if left_interp and len(left_interp) > blink_index:
        try:
            lt = float(left_interp[blink_index])
            if np.isfinite(lt):
                ax_ear.axvline(lt, color="C4", lw=1.0, linestyle=":", alpha=0.7)
        except (TypeError, ValueError):
            pass
    if right_interp and len(right_interp) > blink_index:
        try:
            rt = float(right_interp[blink_index])
            if np.isfinite(rt):
                ax_ear.axvline(rt, color="C5", lw=1.0, linestyle=":", alpha=0.7)
        except (TypeError, ValueError):
            pass

    ax_ear.set_title(f"Epoch {epoch_index} – Blink {blink_index}")
    ax_ear.set_ylabel("EAR (a.u.)")
    ax_ear.grid(True, alpha=0.3)
    ax_ear.legend(loc="upper right", fontsize=8, ncol=2)

    ax_eeg.plot(times, eeg_signal, color="C7", lw=1.0, label="EEG-E8")
    ax_eeg.set_xlabel("Time (s)")
    ax_eeg.set_ylabel("EEG-E8")
    ax_eeg.grid(True, alpha=0.3)
    ax_eeg.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def _build_epoch_report(
    epochs: mne.Epochs,
    ear_channel: str,
    eeg_channel: str,
    output_dir: Path,
) -> Path:
    """Create per-epoch, per-blink visualization report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ear_energy_refinement_report.html"

    md = epochs.metadata
    sfreq = float(epochs.info["sfreq"])
    times = np.arange(epochs.get_data(picks=[ear_channel]).shape[-1]) / sfreq
    ear_data = epochs.get_data(picks=ear_channel)
    eeg_data = epochs.get_data(picks=eeg_channel)

    report = mne.Report(title="EAR + EEG Blink Refinement")
    for ei, row in md.iterrows():
        refined_start_samples = _slot_to_list(row["refined_start_sample"])
        n_blinks = len(refined_start_samples)
        for bi in range(n_blinks):
            fig = _plot_epoch_blink_overlay(
                epoch_index=ei,
                blink_index=bi,
                times=times,
                ear_signal=ear_data[ei, 0, :],
                eeg_signal=eeg_data[ei, 0, :],
                md_row=row,
                sfreq=sfreq,
            )
            report.add_figure(
                fig,
                title=f"Epoch {ei} – Blink {bi}",
                tags=("ear-refinement", f"epoch-{ei}"),
            )
            plt.close(fig)
    report.save(report_path, overwrite=True, open_browser=False)
    return report_path


def main() -> Tuple[mne.Epochs, str]:
    """Run EAR refinement tutorial and generate visual report."""

    # ------------------------------------------------------------------
    # Load raw data and annotations
    # ------------------------------------------------------------------
    raw_path = PROJECT_ROOT / "manual_annotation_feature_calculation_data" / "ear_eog.fif"
    csv_path = PROJECT_ROOT / "manual_annotation_feature_calculation_data" / "ear_eog.csv"

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    raw.set_annotations(mat_data.read_annotations_as_mne(csv_path))

    # ------------------------------------------------------------------
    # Select channels
    # ------------------------------------------------------------------
    ear_channel = "EAR-avg_ear"
    eeg_channel = "EEG-E8"
    required_channels = [ear_channel, eeg_channel]
    missing = [ch for ch in required_channels if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Required channels missing from raw: {missing}")
    raw.pick(required_channels)

    SEGMENT_CONFIG = {
        "ear": {
            "seg_type": [
                "base",
                "zero",
                "tent",
                "half_base",
                "threshold_interpolation",
            ],
            "threshold": 0.22,
            "annotation_time_unit": "seconds",
            "max_extension": 0.35,
            "extension_step": 0.05,
            "padding": 0.05,
            "extend_before": True,
            "extend_after": True,
        },
        "eeg": {"seg_type": [], "threshold": None},
        "eog": {"seg_type": [], "threshold": None},
    }

    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw,
        epoch_len=30.0,
        blink_label=None,
        segmentation_type=SEGMENT_CONFIG,
    )

    # Compute energy features
    df = compute_energy_features(epochs, picks=ear_channel)
    print(f"Computed energy features with shape {df.shape}")

    # Build visualization report
    report_path = _build_epoch_report(epochs, ear_channel, eeg_channel, OUTPUT_DIR)
    print(f"Saved EAR refinement report to: {report_path}")

    return epochs, str(report_path)


if __name__ == "__main__":
    main()
