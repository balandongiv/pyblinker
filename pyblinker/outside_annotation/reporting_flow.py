"""Reporting utilities for refined blink flow outputs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


def _format_metrics(row: pd.Series, keys: Sequence[str]) -> str:
    """Format selected metrics from a row into human-readable text.

    Parameters
    ----------
    row : pd.Series
        Row of blink metrics (seconds, amplitude units of the plotted channel).
    keys : Sequence[str]
        Column names to include if present in ``row``.

    Returns
    -------
    str
        Multi-line string with ``key: value`` pairs rounded to 4 decimals for floats.
    """

    lines = []
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, float):
            value = round(value, 4)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _compute_overlay_indices(
    start: int,
    end: int,
    base_sfreq: float,
    overlay_len: int,
    overlay_sfreq: float | None,
) -> tuple[int, int]:
    """Compute overlay index range aligned to base sampling.

    Parameters
    ----------
    start : int
        Inclusive start sample in the base signal.
    end : int
        Inclusive end sample in the base signal.
    base_sfreq : float
        Sampling frequency (Hz) of the base signal.
    overlay_len : int
        Total number of samples in the overlay signal.
    overlay_sfreq : float | None
        Sampling frequency (Hz) of the overlay signal; defaults to ``base_sfreq`` when None.

    Returns
    -------
    tuple[int, int]
        Overlay start/end sample indices clamped to ``[0, overlay_len - 1]``.
    """

    derived_sfreq = base_sfreq if overlay_sfreq is None else overlay_sfreq
    derived_sfreq = float(derived_sfreq)

    start_time = start / base_sfreq
    end_time = end / base_sfreq

    overlay_start = int(np.clip(round(start_time * derived_sfreq), 0, overlay_len - 1))
    overlay_end = int(np.clip(round(end_time * derived_sfreq), overlay_start, overlay_len - 1))
    return overlay_start, overlay_end


def _determine_report_threshold(results: pd.DataFrame, threshold_value: float | None) -> tuple[float | None, str | None]:
    """Return a single representative threshold for the entire report."""

    if threshold_value is not None:
        return float(threshold_value), "user"

    status_pattern = re.compile(r"^threshold_(?P<value>[^_]+)_ear_threshold_status$")
    candidates: list[tuple[float, int, int]] = []
    for col in results.columns:
        match = status_pattern.match(col)
        if not match:
            continue
        try:
            theta = float(match.group("value"))
        except ValueError:
            continue
        status_series = results[col].astype(str)
        ok_count = int((status_series == "ok").sum())
        found_by_col = f"threshold_{match.group('value')}_ear_threshold_found_by"
        found_count = 0
        if found_by_col in results.columns:
            found_count = int(results[found_by_col].notna().sum())
        candidates.append((theta, ok_count, found_count))

    if candidates:
        candidates.sort(key=lambda item: (-item[1], -item[2], item[0]))
        top_ok = candidates[0][1]
        best_candidates = [c for c in candidates if c[1] == top_ok]
        best_found = max(c[2] for c in best_candidates)
        best = [c for c in best_candidates if c[2] == best_found][0]
        return float(best[0]), "auto_flat"

    if "selected_threshold_value" in results.columns:
        selected_values = pd.to_numeric(results["selected_threshold_value"], errors="coerce").dropna()
        if not selected_values.empty:
            mode_value = float(selected_values.mode().iat[0])
            return mode_value, "auto"

    if "threshold_value" in results.columns:
        threshold_values = pd.to_numeric(results["threshold_value"], errors="coerce").dropna()
        if not threshold_values.empty:
            return float(threshold_values.mode().iat[0]), "auto"

    return None, None


def build_refined_blink_report(
    *,
    results: pd.DataFrame,
    signal: np.ndarray,
    sfreq: float,
    channel_name: str,
    overlay_signal: np.ndarray | None = None,
    overlay_sfreq: float | None = None,
    overlay_label: str = "EAR-avg_ear",
    plot_overlay: bool = False,
    plot_signal_as_scatter: bool = False,
    mark_threshold_crossings: bool = False,
    threshold_value: float | None = None,
    output_path: Path | None = None,
    pad_seconds: float = 0.1,
    max_plots: int | None = None,
    metrics_keys: Iterable[str] = (
        "peak_max_blink",
        "peak_time_blink",
        "duration_zero",
        "duration_base",
        "closing_time_zero",
        "reopening_time_zero",
    ),
) -> mne.Report:
    """Generate an MNE report visualizing refined blink boundaries and metrics.

    Parameters
    ----------
    results : pd.DataFrame
        Blink metrics including refined start/end samples and threshold metadata.
        Time-related columns are seconds; sample indices are integer sample counts.
    signal : np.ndarray
        Base signal to plot (e.g., EEG or EAR), sampled at ``sfreq``.
    sfreq : float
        Sampling frequency of ``signal`` in Hertz.
    channel_name : str
        Name used in plot labels/legends.
    overlay_signal : np.ndarray | None, optional
        Secondary signal to overlay on a twin axis; should be aligned to the same time base.
    overlay_sfreq : float | None, optional
        Sampling frequency of ``overlay_signal``; defaults to ``sfreq`` when None.
    overlay_label : str, optional
        Legend label for the overlay signal.
    plot_overlay : bool, optional
        Whether to plot ``overlay_signal`` when provided.
    plot_signal_as_scatter : bool, optional
        If True, use scatter + thin line for the base signal.
    mark_threshold_crossings : bool, optional
        If True, mark threshold crossings and minimum with low-opacity markers.
    threshold_value : float | None, optional
        Explicit threshold to draw; if None, uses a single representative threshold derived
        from the results.
    output_path : Path | None, optional
        Destination for the generated HTML report; directories are created as needed.
    pad_seconds : float, optional
        Padding (seconds) around each blink window for plotting.
    max_plots : int | None, optional
        Maximum number of blinks to include; useful for large datasets.
    metrics_keys : Iterable[str], optional
        Column names to include in the inset metrics text on each plot.

    Returns
    -------
    mne.Report
        Generated report object with figures and summary HTML added.
    """

    report = mne.Report(title="Refined Blink Validation")
    n_samples = signal.shape[0]
    pad_samples = int(round(pad_seconds * sfreq))

    annotation_rows = len(results)
    total_candidates = len(results)
    rows = list(results.itertuples(index=False))
    if max_plots is not None:
        rows = rows[: int(max_plots)]

    plotted_count = len(rows)
    skipped_count = max(total_candidates - plotted_count, 0)
    zero_crossing_failures = None
    if "zero_crossing_found" in results.columns:
        zero_crossing_failures = int((~results["zero_crossing_found"].astype(bool)).sum())

    representative_threshold, threshold_origin = _determine_report_threshold(
        results, threshold_value
    )

    for idx, row in enumerate(rows):
        left = int(getattr(row, "refined_left_zero", getattr(row, "left_zero", 0)))
        right = int(getattr(row, "refined_right_zero", getattr(row, "right_zero", 0)))
        start = max(0, left - pad_samples)
        end = min(n_samples - 1, right + pad_samples)

        left_time = float(
            getattr(row, "ear_threshold_left_time", getattr(row, "left_time", left / sfreq))
        )
        right_time = float(
            getattr(row, "ear_threshold_right_time", getattr(row, "right_time", right / sfreq))
        )
        if not np.isfinite(left_time):
            left_time = left / sfreq
        if not np.isfinite(right_time):
            right_time = right / sfreq

        min_time = getattr(row, "ear_threshold_min_time", None)
        if min_time is not None and (pd.isna(min_time) or np.isinf(min_time)):
            min_time = None
        min_sample = getattr(row, "refined_min_zero", getattr(row, "min_zero", None))
        if min_sample is None and min_time is not None:
            min_sample = int(np.clip(round(min_time * sfreq), 0, n_samples - 1))
        elif min_sample is not None:
            min_sample = int(min_sample)
            if min_time is None:
                min_time = min_sample / sfreq

        window_times = np.arange(start, end + 1, dtype=float) / sfreq
        window_signal = signal[start : end + 1]

        fig, ax = plt.subplots(figsize=(9, 3))
        if plot_signal_as_scatter:
            ax.scatter(
                window_times,
                window_signal,
                s=14,
                alpha=0.85,
                color="C0",
                label=channel_name,
            )
            ax.plot(
                window_times,
                window_signal,
                lw=0.8,
                alpha=0.25,
                color="C0",
                label=None,
            )
        else:
            ax.plot(
                window_times,
                window_signal,
                lw=1.0,
                alpha=0.85,
                color="C0",
                label=channel_name,
            )
        ax.axvline(left_time, color="C1", linestyle="--", label="Left threshold crossing")
        ax.axvline(right_time, color="C2", linestyle="--", label="Right threshold crossing")

        chosen_threshold = representative_threshold
        plot_threshold_origin = threshold_origin
        if chosen_threshold is None:
            chosen_threshold = getattr(row, "selected_threshold_value", None)
            plot_threshold_origin = getattr(row, "threshold_selection_mode", None)
            if chosen_threshold is None:
                chosen_threshold = getattr(row, "threshold_value", None)
        if chosen_threshold is not None:
            ax.axhline(
                chosen_threshold,
                color="C5",
                linestyle=":",
                lw=1.0,
                label=f"Threshold = {float(chosen_threshold):.3f}"
                + (f" ({plot_threshold_origin})" if plot_threshold_origin else ""),
            )

        # Mark key landmarks directly on the plot for clarity.
        y_min = float(np.min(window_signal))
        y_max = float(np.max(window_signal))
        y_span = max(y_max - y_min, 1e-6)
        y_offset = 0.1 * y_span

        min_value = None
        if min_sample is not None and 0 <= min_sample < n_samples:
            min_value = float(signal[min_sample])

        crossing_times = [left_time, right_time]
        if chosen_threshold is not None:
            crossing_values = [float(chosen_threshold), float(chosen_threshold)]
        else:
            crossing_values = [
                float(signal[int(np.clip(left, 0, n_samples - 1))]),
                float(signal[int(np.clip(right, 0, n_samples - 1))]),
            ]

        if min_time is not None and min_value is not None:
            crossing_times.insert(1, float(min_time))
            crossing_values.insert(1, min_value)

        if mark_threshold_crossings:
            ax.scatter(
                crossing_times,
                crossing_values,
                color="black",
                zorder=5,
                s=32,
                marker="*",
                alpha=0.45,
                label="Threshold landmarks",
            )

        ax.annotate(
            "Left threshold crossing",
            xy=(left_time, crossing_values[0]),
            xytext=(left_time, crossing_values[0] + y_offset),
            arrowprops=dict(arrowstyle="->", color="C1"),
            fontsize=8,
            ha="center",
        )

        if min_time is not None and min_value is not None:
            ax.annotate(
                "Minimum EAR",
                xy=(min_time, min_value),
                xytext=(min_time, min_value - y_offset),
                arrowprops=dict(arrowstyle="->", color="C7"),
                fontsize=8,
                ha="center",
            )

        ax.annotate(
            "Right threshold crossing",
            xy=(right_time, crossing_values[-1]),
            xytext=(right_time, crossing_values[-1] - y_offset),
            arrowprops=dict(arrowstyle="->", color="C2"),
            fontsize=8,
            ha="center",
        )

        # Maximum absolute amplitude within the window.
        max_idx = int(np.argmax(np.abs(window_signal)))
        max_time = window_times[max_idx]
        max_amp = window_signal[max_idx]
        ax.scatter([max_time], [max_amp], color="C3", zorder=5, label="Max |amplitude|")
        ax.annotate(
            f"Max |amp| = {max_amp:.3f}",
            xy=(max_time, max_amp),
            xytext=(max_time, max_amp * 1.1 if max_amp != 0 else 0.1),
            arrowprops=dict(arrowstyle="->", color="C3"),
            fontsize=8,
            ha="center",
        )

        overlay_ax = None
        if plot_overlay and overlay_signal is not None:
            overlay_start, overlay_end = _compute_overlay_indices(
                start=start,
                end=end,
                base_sfreq=sfreq,
                overlay_len=overlay_signal.shape[0],
                overlay_sfreq=overlay_sfreq,
            )
            overlay_times = np.arange(overlay_start, overlay_end + 1, dtype=float) / (
                sfreq if overlay_sfreq is None else float(overlay_sfreq)
            )
            overlay_window = overlay_signal[overlay_start : overlay_end + 1]

            overlay_ax = ax.twinx()
            overlay_ax.plot(
                overlay_times,
                overlay_window,
                lw=1.0,
                alpha=0.75,
                color="C4",
                label=overlay_label,
            )
            overlay_ax.set_ylabel(overlay_label)

        ax.set_title(f"Candidate {getattr(row, 'candidate_id', idx)} • {channel_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(channel_name)
        ax.grid(alpha=0.25)

        metrics_text = _format_metrics(pd.Series(row._asdict()), list(metrics_keys))
        if metrics_text:
            ax.text(
                0.99,
                0.99,
                metrics_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"),
            )

        handles, labels = ax.get_legend_handles_labels()
        if overlay_ax is not None:
            overlay_handles, overlay_labels = overlay_ax.get_legend_handles_labels()
            handles.extend(overlay_handles)
            labels.extend(overlay_labels)
        if handles:
            ax.legend(handles, labels, loc="upper right")

        caption = (
            f"Threshold crossings at {left / sfreq:.3f}s and {right / sfreq:.3f}s. "
            f"Sampling rate: {sfreq:.2f} Hz. "
            f"Segment {start}–{end} ({(end - start) / sfreq:.3f}s)."
        )
        if chosen_threshold is not None:
            suffix = f" ({plot_threshold_origin})" if plot_threshold_origin else ""
            caption += f" Threshold value: {float(chosen_threshold):.3f}{suffix}."
        report.add_figure(
            fig=fig,
            title=f"Blink {idx}",
            caption=caption,
            section="Refined blinks",
            tags=("blink", "refined", channel_name),
        )
        plt.close(fig)

    summary_rows = [
        ("Source CSV rows (blinks)", annotation_rows),
        ("Total refined blinks", total_candidates),
        ("Blinks plotted", plotted_count),
    ]
    if skipped_count:
        reason = "max_plots limit" if max_plots is not None else "not plotted"
        summary_rows.append((f"Skipped ({reason})", skipped_count))
    if zero_crossing_failures is not None:
        summary_rows.append(("Zero-crossing failures", zero_crossing_failures))
    if representative_threshold is not None:
        label = (
            "Plot threshold (user-provided)" if threshold_origin == "user" else "Plot threshold (auto mode)"
        )
        summary_rows.append((label, f"{float(representative_threshold):.3f}"))
    if {"selected_threshold_value", "threshold_selection_mode"} <= set(results.columns):
        mode_counts = results["threshold_selection_mode"].astype(str).value_counts()
        if not mode_counts.empty:
            summary_rows.append(
                ("Threshold selection modes", "; ".join(f"{k}: {v}" for k, v in mode_counts.items()))
            )
    if "threshold_selection_reason" in results.columns:
        reasons = results["threshold_selection_reason"].dropna().astype(str)
        if not reasons.empty:
            reason_counts = reasons.value_counts()
            top_reason = reason_counts.index[0]
            summary_rows.append(
                ("Threshold selection rationale", f"{top_reason} ({int(reason_counts.iloc[0])}x)")
            )

    summary_html = """<table style='border-collapse: collapse;'>
    <thead><tr><th style='text-align:left;padding:4px;'>Metric</th>
    <th style='text-align:left;padding:4px;'>Value</th></tr></thead><tbody>"""
    for label, value in summary_rows:
        summary_html += (
            f"<tr><td style='padding:4px;border-top:1px solid #ddd;'>{label}</td>"
            f"<td style='padding:4px;border-top:1px solid #ddd;'>{value}</td></tr>"
        )
    summary_html += "</tbody></table>"
    report.add_html(
        title="Refined blink summary",
        html=summary_html,
        section="Summary",
        tags=("summary", "refined", channel_name),
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save(output_path, overwrite=True, open_browser=False)

    return report


__all__ = ["build_refined_blink_report"]
