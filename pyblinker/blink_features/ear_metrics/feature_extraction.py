"""EAR blink feature extraction built around threshold-refined boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pyblinker.fitutils.ear_crossing import (
    ThresholdCrossingError,
    compute_threshold_slopes,
    find_threshold_crossing_triplet,
)
from pyblinker.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EARFeatureConfig:
    """Configuration for EAR-based blink feature extraction."""

    baseline_window: float = 0.2
    classification_threshold: Optional[float] = None
    context_window: Optional[float] = None
    percentiles: Sequence[int] = field(default_factory=lambda: (5, 10, 90, 95))
    slope_max_expansion_seconds: float = 0.0
    slope_expansion_step_seconds: float = 0.01
    slope_plateau_policy: str = "midpoint"


def _compute_baseline(signal: np.ndarray, start: int, baseline_samples: int) -> float:
    """Return the average EAR before the blink onset for baseline depth calculation.

    Parameters
    ----------
    signal : np.ndarray
        1D EAR signal in raw units.
    start : int
        Sample index of the refined blink onset.
    baseline_samples : int
        Number of samples to average before onset (already converted from seconds).

    Returns
    -------
    float
        Baseline EAR value preceding the blink window. Falls back to the onset value
        when insufficient samples exist.
    """

    if baseline_samples <= 0:
        return float(signal[start])
    baseline_start = max(0, start - baseline_samples)
    if baseline_start == start:
        return float(signal[start])
    return float(np.mean(signal[baseline_start:start]))


def _safe_gradient(values: np.ndarray, dt: float) -> np.ndarray:
    """Compute the gradient with zero-padding for very small arrays.

    Parameters
    ----------
    values : np.ndarray
        Input samples.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    np.ndarray
        Gradient estimate with the same shape as ``values``.
    """

    if values.size < 2:
        return np.zeros_like(values, dtype=float)
    return np.gradient(values, dt)


def _compute_base_features(
    *,
    signal: np.ndarray,
    sfreq: float,
    start_sample: int,
    end_sample: int,
    blink_type: Optional[str],
    feature_config: EARFeatureConfig,
) -> Tuple[Dict[str, float | str], Dict[str, float | np.ndarray]]:
    """Compute threshold-independent features for a blink window.

    Parameters
    ----------
    signal : np.ndarray
        Full EAR signal (raw units).
    sfreq : float
        Sampling frequency in Hertz.
    start_sample : int
        Refined blink onset sample.
    end_sample : int
        Refined blink offset sample (inclusive).
    blink_type : str | None
        Optional blink label from annotations.
    feature_config : EARFeatureConfig
        Configuration controlling context padding, baseline window, and percentiles.

    Returns
    -------
    tuple[dict, dict]
        A tuple of:
        - base_features: scalar metrics independent of thresholding.
        - transient_arrays: reusable arrays (window, velocity, context) and indices.
    """

    dt = 1.0 / sfreq

    context_start = start_sample
    context_end = end_sample
    if feature_config.context_window:
        pad = int(round(feature_config.context_window * sfreq))
        context_start = max(0, start_sample - pad)
        context_end = min(signal.shape[0] - 1, end_sample + pad)

    window = signal[start_sample : end_sample + 1]
    context_window = signal[context_start : context_end + 1]

    baseline_samples = int(round(feature_config.baseline_window * sfreq))
    baseline = _compute_baseline(signal, start_sample, baseline_samples)

    mean_val = float(np.mean(context_window))
    median_val = float(np.median(context_window))
    std_val = float(np.std(context_window))
    var_val = float(np.var(context_window))
    mad_val = float(np.median(np.abs(context_window - median_val)))
    iqr_val = float(np.percentile(context_window, 75) - np.percentile(context_window, 25))
    skewness = float(
        np.mean(((context_window - mean_val) / (std_val + 1e-12)) ** 3)
    )
    kurtosis = float(
        np.mean(((context_window - mean_val) / (std_val + 1e-12)) ** 4)
    )

    percentile_values = np.percentile(context_window, feature_config.percentiles)
    percentile_dict = {
        f"ear_p{p}": float(val) for p, val in zip(feature_config.percentiles, percentile_values)
    }

    min_idx_local = int(np.argmin(window))
    min_value = float(window[min_idx_local])
    min_sample = start_sample + min_idx_local
    time_of_min = min_sample / sfreq

    velocity = _safe_gradient(window, dt)
    acceleration = _safe_gradient(velocity, dt)

    max_closing_speed = float(np.min(velocity))
    max_opening_speed = float(np.max(velocity))

    max_negative_acceleration = float(np.min(acceleration))
    max_positive_acceleration = float(np.max(acceleration))

    max_negative_slope = max_closing_speed
    max_positive_slope = max_opening_speed

    closing_velocity = velocity[: min_idx_local + 1]
    reopening_velocity = velocity[min_idx_local:]
    mean_closing_slope = float(np.mean(closing_velocity)) if closing_velocity.size else float("nan")
    mean_reopening_slope = (
        float(np.mean(reopening_velocity)) if reopening_velocity.size else float("nan")
    )

    closure_time = (min_sample - start_sample) * dt
    reopening_time = (end_sample - min_sample) * dt

    base_features: Dict[str, float | str] = {
        "blink_type_original": blink_type,
        "ear_mean": mean_val,
        "ear_median": median_val,
        "ear_std": std_val,
        "ear_var": var_val,
        "ear_mad": mad_val,
        "ear_iqr": iqr_val,
        "ear_skewness": skewness,
        "ear_kurtosis": kurtosis,
        "ear_min": min_value,
        "ear_max": float(np.max(context_window)),
        "ear_time_of_min": time_of_min,
        "ear_baseline": baseline,
        "ear_blink_depth": float(baseline - min_value),
        "max_closing_speed": max_closing_speed,
        "max_opening_speed": max_opening_speed,
        "max_negative_slope": max_negative_slope,
        "max_positive_slope": max_positive_slope,
        "mean_closing_slope": mean_closing_slope,
        "mean_reopening_slope": mean_reopening_slope,
        "max_negative_acceleration": max_negative_acceleration,
        "max_positive_acceleration": max_positive_acceleration,
        "time_to_close": float(closure_time),
        "time_to_reopen": float(reopening_time),
    }
    base_features.update(percentile_dict)

    transient_arrays = {
        "window": window,
        "velocity": velocity,
        "min_sample": int(min_sample),
        "context_window": context_window,
    }
    return base_features, transient_arrays


def _compute_threshold_features(
    *,
    signal: np.ndarray,
    sfreq: float,
    start_sample: int,
    end_sample: int,
    min_sample: int,
    window: np.ndarray,
    threshold: float,
    feature_config: EARFeatureConfig,
    blink_type: Optional[str],
) -> Dict[str, float | str | bool]:
    """Compute threshold-dependent metrics for a single threshold value.

    Parameters
    ----------
    signal : np.ndarray
        Full EAR signal in raw units.
    sfreq : float
        Sampling frequency in Hertz.
    start_sample : int
        Refined blink onset sample.
    end_sample : int
        Refined blink offset sample (inclusive).
    min_sample : int
        Sample index of the minimum EAR within the blink window.
    window : np.ndarray
        Blink window slice from ``start_sample`` to ``end_sample`` (inclusive).
    threshold : float
        EAR threshold used for crossings and under-threshold metrics.
    feature_config : EARFeatureConfig
        Threshold-dependent configuration (slope expansion, classification overrides).
    blink_type : str | None
        Optional blink label from annotations; preferred over computed classification.

    Returns
    -------
    dict
        Metrics tied to the provided threshold, including slopes, durations, AUC,
        classification outcome (prefers CSV label), computed classification, and
        crossing metadata.
    """

    dt = 1.0 / sfreq
    slope_metrics: Dict[str, float | str | bool] = {
        "ear_threshold_closing_slope": float("nan"),
        "ear_threshold_opening_slope": float("nan"),
        "ear_threshold_left_time": float("nan"),
        "ear_threshold_min_time": float("nan"),
        "ear_threshold_right_time": float("nan"),
        "ear_threshold_found_by": "unattempted",
        "ear_threshold_status": "failed",
    }

    try:
        max_expansion = int(round(feature_config.slope_max_expansion_seconds * sfreq))
        expansion_step = int(max(1, round(feature_config.slope_expansion_step_seconds * sfreq)))
        t = np.arange(signal.shape[0]) / sfreq
        triplet = find_threshold_crossing_triplet(
            signal,
            theta=threshold,
            t=t,
            window=(start_sample, end_sample),
            max_expansion=max_expansion,
            expansion_step=expansion_step,
            plateau_policy=feature_config.slope_plateau_policy,  # type: ignore[arg-type]
        )
        closing_slope, opening_slope = compute_threshold_slopes(triplet, threshold)
        slope_metrics.update(
            {
                "ear_threshold_closing_slope": closing_slope,
                "ear_threshold_opening_slope": opening_slope,
                "ear_threshold_left_time": triplet.left.time,
                "ear_threshold_min_time": triplet.minimum_time,
                "ear_threshold_right_time": triplet.right.time,
                "ear_threshold_found_by": triplet.found_by,
                "ear_threshold_status": triplet.status,
            }
        )
    except ThresholdCrossingError:
        slope_metrics["ear_threshold_status"] = "failed"

    under_threshold_mask = window < threshold
    closed_duration = float(under_threshold_mask.sum() * dt)
    closed_fraction = float(np.mean(under_threshold_mask)) if window.size else float("nan")
    auc_below = float(np.sum((threshold - window[under_threshold_mask]) * dt))

    classification_threshold = (
        feature_config.classification_threshold
        if feature_config.classification_threshold is not None
        else threshold
    )
    min_value = float(window[int(min_sample - start_sample)])
    computed_classification = "full" if min_value < classification_threshold else "partial"
    blink_classification = (
        str(blink_type) if blink_type is not None and str(blink_type) else computed_classification
    )

    threshold_features: Dict[str, float | str | bool] = {
        "closed_duration_seconds": closed_duration,
        "closed_fraction": closed_fraction,
        "time_under_threshold_seconds": closed_duration,
        "time_under_threshold_fraction": closed_fraction,
        "auc_below_threshold": auc_below,
        "classification_threshold": float(classification_threshold),
        "blink_classification": blink_classification,
        "blink_classification_computed": computed_classification,
        "threshold_value": float(threshold),
    }
    threshold_features.update(slope_metrics)
    return threshold_features


THRESHOLD_METRIC_KEYS = {
    "closed_duration_seconds",
    "closed_fraction",
    "time_under_threshold_seconds",
    "time_under_threshold_fraction",
    "auc_below_threshold",
    "classification_threshold",
    "blink_classification",
    "blink_classification_computed",
    "threshold_value",
    "ear_threshold_closing_slope",
    "ear_threshold_opening_slope",
    "ear_threshold_left_time",
    "ear_threshold_min_time",
    "ear_threshold_right_time",
    "ear_threshold_found_by",
    "ear_threshold_status",
}


def compute_blink_features(
    signal: np.ndarray,
    sfreq: float,
    threshold: float,
    start_sample: int,
    end_sample: int,
    blink_type: Optional[str],
    feature_config: EARFeatureConfig,
) -> Dict[str, object]:
    """Compute EAR-derived features for a single blink window.

    Parameters
    ----------
    signal : np.ndarray
        Full EAR signal in raw units.
    sfreq : float
        Sampling frequency in Hertz.
    threshold : float
        Threshold to evaluate for threshold-dependent metrics.
    start_sample : int
        Refined blink onset sample (inclusive).
    end_sample : int
        Refined blink offset sample (inclusive).
    blink_type : str | None
        Optional blink label.
    feature_config : EARFeatureConfig
        Configuration controlling baseline window, slope search, percentiles, and classification.

    Returns
    -------
    dict
        Structured features containing:
        - ``base_features``: threshold-independent metrics.
        - threshold-dependent metrics for the provided threshold.
        - ``blink_type_original``: passthrough of the blink label.
    """

    start_sample = int(max(0, start_sample))
    end_sample = int(min(signal.shape[0] - 1, max(start_sample, end_sample)))

    base_features, transient = _compute_base_features(
        signal=signal,
        sfreq=sfreq,
        start_sample=start_sample,
        end_sample=end_sample,
        blink_type=blink_type,
        feature_config=feature_config,
    )

    min_sample = int(transient["min_sample"])
    window = transient["window"]
    threshold_features = _compute_threshold_features(
        signal=signal,
        sfreq=sfreq,
        start_sample=start_sample,
        end_sample=end_sample,
        min_sample=min_sample,
        window=window,
        threshold=threshold,
        feature_config=feature_config,
        blink_type=blink_type,
    )

    features: Dict[str, object] = {
        **base_features,
        **threshold_features,
        "blink_type_original": blink_type,
    }

    return features


class EARBlinkFeatureExtractor:
    """Compute per-blink EAR metrics given refined boundaries."""

    def __init__(
        self,
        signal: np.ndarray,
        sfreq: float,
        threshold: float | None = None,
        feature_config: Optional[EARFeatureConfig] = None,
        threshold_column: str = "threshold_value",
    ):
        """Create an EAR feature extractor.

        Parameters
        ----------
        signal : np.ndarray
            Full EAR signal (raw units).
        sfreq : float
            Sampling frequency in Hertz.
        threshold : float | None
            Default threshold to apply when the refined table does not provide one.
        feature_config : EARFeatureConfig, optional
            Configuration controlling baseline, percentiles, slopes, and classification.
        threshold_column : str
            Column name containing the threshold used for each refined blink.
        """
        self.signal = np.asarray(signal, dtype=float)
        self.sfreq = float(sfreq)
        self.threshold = threshold
        self.threshold_column = threshold_column
        self.feature_config = feature_config or EARFeatureConfig()
        self._threshold_store: List[Dict[str, float | str | bool]] = []

    @property
    def threshold_store(self) -> List[Dict[str, float | str | bool]]:
        """Return per-row threshold metrics captured during the last table build.

        Provided for compatibility with earlier multi-threshold workflows.
        """

        return list(self._threshold_store)

    def build_feature_table(self, refined: pd.DataFrame) -> pd.DataFrame:
        """Attach EAR-based blink features to refined annotation rows.

        Parameters
        ----------
        refined : pd.DataFrame
            DataFrame containing at least ``refined_start_sample`` and
            ``refined_end_sample`` columns (from refinement).

        Returns
        -------
        pd.DataFrame
            Input rows augmented with base EAR metrics and threshold-dependent columns.
        """

        required_cols = {"refined_start_sample", "refined_end_sample"}
        missing_cols = required_cols - set(refined.columns)
        if missing_cols:
            raise ValueError(
                f"Refined annotations are missing required columns: {sorted(missing_cols)}"
            )

        has_threshold_column = self.threshold_column in refined.columns
        if not has_threshold_column and self.threshold is None:
            raise ValueError(
                f"Refined annotations must include `{self.threshold_column}` or a default "
                "threshold must be provided."
            )

        records: List[Dict[str, float | str | bool]] = []
        self._threshold_store = []
        for row in refined.to_dict(orient="records"):
            theta = (
                float(row[self.threshold_column])
                if has_threshold_column
                else float(self.threshold)  # type: ignore[arg-type]
            )
            features = compute_blink_features(
                signal=self.signal,
                sfreq=self.sfreq,
                threshold=theta,
                start_sample=int(row["refined_start_sample"]),
                end_sample=int(row["refined_end_sample"]),
                blink_type=row.get("blink_type"),
                feature_config=self.feature_config,
            )
            combined = {
                **row,
                **features,
                self.threshold_column: float(theta),
                "refined_duration": float(
                    (row["refined_end_sample"] - row["refined_start_sample"]) / self.sfreq
                ),
            }
            records.append(combined)
            threshold_metrics = {
                key: value
                for key, value in features.items()
                if key in THRESHOLD_METRIC_KEYS
            }
            threshold_metrics[self.threshold_column] = float(theta)
            self._threshold_store.append(threshold_metrics)

        df = pd.DataFrame.from_records(records)
        logger.info("Computed EAR features for %s blinks", len(df))
        return df
