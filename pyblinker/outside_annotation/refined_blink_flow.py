"""Blink region refinement and metric computation flow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.waveform_features.extract_blink_properties import (
    BlinkProperties,
)
from pyblinker.blinker.default_setting import DEFAULT_PARAMS
from pyblinker.blinker.fit_blink import FitBlinks
from pyblinker.blinker.zero_crossing import left_right_zero_crossing
from pyblinker.utils.statistics_utils import get_max_blink

logger = logging.getLogger(__name__)


@dataclass
class RefinementConfig:
    annotation_csv: Path
    fif_path: Path
    channel: str
    buffer_seconds: float = 0.25
    output_path: Path | None = None
    run_fit: bool = True


@dataclass
class RefinementArtifacts:
    """Artifacts captured during a single refinement run."""

    results: pd.DataFrame
    signal: np.ndarray
    sfreq: float


class BlinkRegionRefinementFlow:
    """Flow for refining annotated blink regions and extracting metrics."""

    def __init__(self, config: RefinementConfig):
        self.config = config
        self.params: Dict[str, float] = dict(DEFAULT_PARAMS)
        self.last_signal: np.ndarray | None = None
        self.last_sfreq: float | None = None

    def load_signal(self) -> Tuple[np.ndarray, float]:
        """Load the target channel from the FIF file."""

        raw = mne.io.read_raw_fif(
            self.config.fif_path, preload=True, verbose="ERROR"
        )
        sfreq = float(raw.info["sfreq"])
        try:
            signal = raw.get_data(picks=self.config.channel)[0]
        except Exception as exc:  # pragma: no cover - defensive channel lookup
            raise ValueError(f"Channel {self.config.channel} not found") from exc

        self.last_signal = signal
        self.last_sfreq = sfreq
        return signal, sfreq

    def load_annotations(self) -> pd.DataFrame:
        """Load blink annotations from CSV and compute derived timing fields."""

        annotations = pd.read_csv(self.config.annotation_csv)
        missing_cols = {"onset", "duration"} - set(annotations.columns)
        if missing_cols:
            raise ValueError(
                f"Annotation file is missing required columns: {sorted(missing_cols)}"
            )

        annotations = annotations.copy()
        annotations["end_time"] = annotations["onset"] + annotations["duration"]
        annotations["candidate_id"] = np.arange(len(annotations))
        return annotations

    def refine_candidates(
        self, signal: np.ndarray, sfreq: float, annotations: pd.DataFrame
    ) -> pd.DataFrame:
        """Refine approximate blink regions using zero-crossing detection."""

        n_samples = signal.shape[0]
        buffer_samples = int(round(self.config.buffer_seconds * sfreq))

        records: List[Dict[str, float | str | bool | int]] = []
        for row in annotations.itertuples(index=False):
            start_sample = int(np.clip(round(row.onset * sfreq), 0, n_samples - 1))
            end_sample = int(
                np.clip(round(row.end_time * sfreq), start_sample, n_samples - 1)
            )

            outer_start = max(0, start_sample - buffer_samples)
            outer_end = min(n_samples - 1, end_sample + buffer_samples)

            max_value, max_blink = get_max_blink(signal, start_sample, end_sample)

            zero_crossing_found = True
            try:
                left_zero, right_zero = left_right_zero_crossing(
                    signal,
                    max_blink,
                    outer_start,
                    outer_end,
                    signal_type="eeg",
                )
            except Exception:
                logger.exception(
                    "Failed to compute zero crossings; falling back to annotation bounds",
                    extra={"candidate_id": row.candidate_id},
                )
                left_zero, right_zero = np.nan, np.nan
                zero_crossing_found = False

            if np.isnan(left_zero):
                left_zero = start_sample
                zero_crossing_found = False
            if np.isnan(right_zero):
                right_zero = end_sample
                zero_crossing_found = False

            records.append(
                {
                    "candidate_id": int(row.candidate_id),
                    "onset": float(row.onset),
                    "end_time": float(row.end_time),
                    "duration": float(row.duration),
                    "description": getattr(row, "description", None),
                    "start_time": float(row.onset),
                    "start_blink": start_sample,
                    "end_blink": end_sample,
                    "outer_start": outer_start,
                    "outer_end": outer_end,
                    "max_value": max_value,
                    "max_blink": max_blink,
                    "refined_left_zero": int(left_zero),
                    "refined_right_zero": int(right_zero),
                    "zero_crossing_found": bool(zero_crossing_found),
                }
            )

        refined = pd.DataFrame.from_records(records)
        logger.info(
            "Loaded %s blink candidates; %s had zero-crossings",
            len(refined),
            int(refined["zero_crossing_found"].sum()),
        )
        return refined

    def _fit_blinks(self, signal: np.ndarray, sfreq: float, rows: pd.DataFrame) -> pd.DataFrame:
        params = dict(self.params)
        params["sfreq"] = sfreq

        fit_rows = rows[
            [
                "candidate_id",
                "start_blink",
                "end_blink",
                "outer_start",
                "outer_end",
                "refined_left_zero",
                "refined_right_zero",
                "max_value",
                "max_blink",
            ]
        ].rename(
            columns={"refined_left_zero": "left_zero", "refined_right_zero": "right_zero"}
        )
        fit_rows = fit_rows.set_index("candidate_id")

        fitter = FitBlinks(candidate_signal=signal, df=fit_rows, params=params)
        fitter.dprocess_segment_raw(run_fit=self.config.run_fit)
        frame_blinks = getattr(fitter, "frame_blinks", None)
        if frame_blinks is None or frame_blinks.empty:
            logger.warning("FitBlinks returned no frames after processing")
            return pd.DataFrame()

        if "candidate_id" not in frame_blinks.columns:
            frame_blinks = frame_blinks.reset_index().rename(
                columns={"index": "candidate_id"}
            )

        return frame_blinks

    def compute_metrics(
        self, signal: np.ndarray, sfreq: float, refined: pd.DataFrame
    ) -> pd.DataFrame:
        """Run FitBlinks and BlinkProperties using the refined regions."""

        try:
            frame_blinks = self._fit_blinks(signal, sfreq, refined)
        except Exception:
            logger.exception("FitBlinks failed; no metrics will be produced")
            return pd.DataFrame()

        if frame_blinks.empty:
            return pd.DataFrame()

        params = dict(self.params)
        params["sfreq"] = sfreq

        try:
            blink_properties = BlinkProperties(
                signal, frame_blinks, sfreq, params, fitted=self.config.run_fit
            )
        except Exception:
            logger.exception("BlinkProperties failed; returning empty result set")
            return pd.DataFrame()

        logger.info("Computed blink properties for %s candidates", len(blink_properties.df))
        return blink_properties.df

    def run(self) -> pd.DataFrame:
        """Execute the full refinement + metric computation flow."""

        signal, sfreq = self.load_signal()
        annotations = self.load_annotations()
        refined = self.refine_candidates(signal, sfreq, annotations)

        metrics = self.compute_metrics(signal, sfreq, refined)
        if metrics.empty or "candidate_id" not in metrics.columns:
            metrics = pd.DataFrame(columns=["candidate_id"])

        merged = refined.merge(
            metrics,
            on="candidate_id",
            how="left",
            suffixes=("_annotation", ""),
        )

        merged["fit_success"] = False
        merged["properties_success"] = False
        if not metrics.empty and "candidate_id" in metrics.columns:
            successful_ids = set(metrics["candidate_id"].astype(int))
            merged["fit_success"] = merged["candidate_id"].isin(successful_ids)
            merged["properties_success"] = merged["candidate_id"].isin(successful_ids)

        fit_count = int(merged["fit_success"].sum())
        properties_count = int(merged["properties_success"].sum())
        logger.info("FitBlinks successful for %s candidates", fit_count)
        logger.info("BlinkProperties computed for %s candidates", properties_count)

        if self.config.output_path:
            self.save_output(merged, self.config.output_path)

        return merged

    def run_with_artifacts(self) -> RefinementArtifacts:
        """Run and capture signal/sfreq alongside results."""

        results = self.run()
        if self.last_signal is None or self.last_sfreq is None:
            raise RuntimeError("Refinement artifacts are unavailable; run() did not set them.")
        return RefinementArtifacts(results=results, signal=self.last_signal, sfreq=self.last_sfreq)

    @staticmethod
    def save_output(results: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info("Wrote refined blink results to %s", output_path)
