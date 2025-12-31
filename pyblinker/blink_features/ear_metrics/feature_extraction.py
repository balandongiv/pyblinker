"""EAR blink feature extraction built around threshold-refined boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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


def _compute_slopes_from_samples(
    signal: np.ndarray,
    start_sample: int,
    lowest_sample: int,
    end_sample: int,
    sfreq: float,
) -> tuple[float, float]:
    """Return closing/opening slopes using refined boundaries and lowest point.

    Slopes are derived directly from EAR values at the provided sample indices. When
    samples are invalid, out of order, or coincide in time, ``nan`` is returned for
    the corresponding slope.
    """

    n_samples = signal.shape[0]
    if (
        start_sample < 0
        or end_sample >= n_samples
        or lowest_sample < start_sample
        or lowest_sample > end_sample
    ):
        return float("nan"), float("nan")

    dt = 1.0 / sfreq
    closing_duration = (lowest_sample - start_sample) * dt
    opening_duration = (end_sample - lowest_sample) * dt

    closing_slope = float("nan")
    opening_slope = float("nan")
    if closing_duration > 0:
        closing_slope = float((signal[lowest_sample] - signal[start_sample]) / closing_duration)
    if opening_duration > 0:
        opening_slope = float((signal[end_sample] - signal[lowest_sample]) / opening_duration)

    return closing_slope, opening_slope


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
    lowest_point_sample: int | float | None,
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
    lowest_point_sample : int | float | None
        Refined lowest EAR sample within the blink interval, if available.
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
        Metrics tied to the provided threshold, including slopes derived from refined
        boundaries + lowest point, durations, AUC, classification outcome (prefers CSV
        label), and computed classification.
    """

    dt = 1.0 / sfreq
    slope_metrics: Dict[str, float] = {
        "ear_threshold_closing_slope": float("nan"),
        "ear_threshold_opening_slope": float("nan"),
    }

    resolved_lowest_sample: int | None = None
    if lowest_point_sample is not None and np.isfinite(lowest_point_sample):
        candidate = int(lowest_point_sample)
        if start_sample <= candidate <= end_sample and 0 <= candidate < signal.shape[0]:
            resolved_lowest_sample = candidate
    if resolved_lowest_sample is None and start_sample <= min_sample <= end_sample:
        resolved_lowest_sample = int(min_sample)

    if resolved_lowest_sample is not None:
        closing_slope, opening_slope = _compute_slopes_from_samples(
            signal=signal,
            start_sample=start_sample,
            lowest_sample=resolved_lowest_sample,
            end_sample=end_sample,
            sfreq=sfreq,
        )
        slope_metrics.update(
            {
                "ear_threshold_closing_slope": closing_slope,
                "ear_threshold_opening_slope": opening_slope,
            }
        )

    under_threshold_mask = window < threshold
    closed_duration = float(under_threshold_mask.sum() * dt)
    closed_fraction = float(np.mean(under_threshold_mask)) if window.size else float("nan")
    auc_below = float(np.sum((threshold - window[under_threshold_mask]) * dt))

    classification_threshold = (
        feature_config.classification_threshold
        if feature_config.classification_threshold is not None
        else threshold
    )
    min_index_for_metrics = (
        resolved_lowest_sample if resolved_lowest_sample is not None else int(min_sample)
    )
    min_offset = min_index_for_metrics - start_sample
    min_value = float("nan")
    if 0 <= min_offset < window.size:
        min_value = float(window[int(min_offset)])
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


def compute_blink_features(
    signal: np.ndarray,
    sfreq: float,
    threshold: float,
    start_sample: int,
    end_sample: int,
    lowest_point_sample: int | float | None,
    blink_type: Optional[str],
    feature_config: EARFeatureConfig,
) -> Dict[str, object]:
    """Compute EAR-derived features for a single blink window and threshold.

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
    lowest_point_sample : int | float | None
        Lowest EAR sample within the refined interval, if available.
    blink_type : str | None
        Optional blink label.
    feature_config : EARFeatureConfig
        Configuration controlling baseline window, slope search, percentiles, and classification.

    Returns
    -------
    dict
        Structured features containing:
        - threshold-independent metrics (e.g., baseline, min).
        - threshold-dependent metrics tied to the provided ``threshold``.
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
    threshold_metrics = _compute_threshold_features(
        signal=signal,
        sfreq=sfreq,
        start_sample=start_sample,
        end_sample=end_sample,
        min_sample=min_sample,
        lowest_point_sample=lowest_point_sample,
        window=window,
        threshold=threshold,
        feature_config=feature_config,
        blink_type=blink_type,
    )

    features: Dict[str, object] = {
        **base_features,
        **threshold_metrics,
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
    ):
        """Create an EAR feature extractor.

        Parameters
        ----------
        signal : np.ndarray
            Full EAR signal (raw units).
        sfreq : float
            Sampling frequency in Hertz.
        threshold : float | None, optional
            Optional fixed threshold to use when refined rows do not carry one.
        feature_config : EARFeatureConfig, optional
            Configuration controlling baseline, percentiles, slopes, and classification.
        """
        self.signal = np.asarray(signal, dtype=float)
        self.sfreq = float(sfreq)
        self.threshold = threshold
        self.feature_config = feature_config or EARFeatureConfig()

    def build_feature_table(self, refined: pd.DataFrame) -> pd.DataFrame:
        """Attach EAR-based blink features to refined annotation rows.

        Parameters
        ----------
        refined : pd.DataFrame
            DataFrame containing ``refined_start_sample`` and ``refined_end_sample``.
            A ``threshold_value`` column is expected unless the extractor was created
            with a fixed ``threshold``.

        Returns
        -------
        pd.DataFrame
            Input rows augmented with base EAR metrics and threshold-dependent scalars.
        """

        required_cols = {"refined_start_sample", "refined_end_sample", "refined_lowest_point_sample"}
        missing_cols = required_cols - set(refined.columns)
        if missing_cols:
            raise ValueError(
                f"Refined annotations are missing required columns: {sorted(missing_cols)}"
            )

        records: List[Dict[str, float | str | bool]] = []
        for row in refined.to_dict(orient="records"):
            threshold_value = row.get("threshold_value", self.threshold)
            if threshold_value is None:
                raise ValueError(
                    "Refined annotations must include a 'threshold_value' column or the extractor "
                    "must be initialized with a fixed threshold."
                )
            threshold_value = float(threshold_value)

            features = compute_blink_features(
                signal=self.signal,
                sfreq=self.sfreq,
                threshold=threshold_value,
                start_sample=int(row["refined_start_sample"]),
                end_sample=int(row["refined_end_sample"]),
                lowest_point_sample=row.get("refined_lowest_point_sample"),
                blink_type=row.get("blink_type"),
                feature_config=self.feature_config,
            )
            combined = {
                **row,
                **{
                    key: value
                    for key, value in features.items()
                    if not isinstance(value, (dict, list, tuple, np.ndarray))
                },
                "threshold_value": threshold_value,
                "refined_duration": float(
                    (row["refined_end_sample"] - row["refined_start_sample"]) / self.sfreq
                ),
            }
            records.append(combined)

        df = pd.DataFrame.from_records(records)
        logger.info("Computed EAR features for %s blinks", len(df))
        return df
