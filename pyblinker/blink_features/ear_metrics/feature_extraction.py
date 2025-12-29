"""EAR blink feature extraction built around threshold-refined boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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


def _normalize_thresholds(
    thresholds: float | Sequence[float], extra_threshold: float | None = None
) -> List[float]:
    """Return a deterministic, deduplicated threshold list preserving order.

    Parameters
    ----------
    thresholds : float | Sequence[float]
        Single threshold or iterable of thresholds to evaluate.
    extra_threshold : float | None
        Optional additional threshold to append (e.g., explicit plot value).

    Returns
    -------
    list[float]
        Ordered, de-duplicated threshold list.
    """

    if isinstance(thresholds, (str, bytes)):
        raise TypeError("Thresholds must be numeric, not a string.")

    if isinstance(thresholds, Sequence):
        threshold_list = [float(t) for t in thresholds]
    else:
        threshold_list = [float(thresholds)]

    if extra_threshold is not None:
        threshold_list.append(float(extra_threshold))

    # Deduplicate while preserving order to avoid collisions in the per-threshold mapping.
    unique_thresholds = list(dict.fromkeys(threshold_list))
    if not unique_thresholds:
        raise ValueError("At least one threshold value is required for feature extraction.")

    return unique_thresholds


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


def _flatten_threshold_metrics(
    threshold_metrics: Mapping[float, Mapping[str, float | str | bool]]
) -> Dict[str, float | str | bool]:
    """Flatten per-threshold metrics into prefixed keys for tabular inspection.

    Parameters
    ----------
    threshold_metrics : Mapping[float, Mapping[str, float | str | bool]]
        Metrics keyed by threshold value.

    Returns
    -------
    dict
        Flattened metrics with keys of the form ``threshold_<value>_<metric>`` (all
        values are scalar: float, int, bool, or str).
    """

    flat: Dict[str, float | str | bool] = {}
    for theta, metrics in threshold_metrics.items():
        prefix = f"threshold_{theta:.6g}_"
        for key, value in metrics.items():
            flat[f"{prefix}{key}"] = value
    return flat


def _score_threshold_candidate(metrics: Mapping[str, float | str]) -> float:
    """Score a threshold candidate for deterministic selection.

    Higher is better. The policy prioritizes:
    1. Successful threshold crossings (status == \"ok\").
    2. Steeper opening/closing slopes (sum of absolute slopes).
    3. Mid-range closed_fraction (penalizes extreme 0/1 saturation).

    Parameters
    ----------
    metrics : Mapping[str, float | str]
        Per-threshold metrics from ``_compute_threshold_features``.

    Returns
    -------
    float
        Selection score (``-inf`` when crossings fail).
    """

    status = metrics.get("ear_threshold_status")
    if status != "ok":
        return -np.inf

    closing = float(metrics.get("ear_threshold_closing_slope", 0.0) or 0.0)
    opening = float(metrics.get("ear_threshold_opening_slope", 0.0) or 0.0)
    slope_score = abs(closing) + abs(opening)

    closed_fraction = metrics.get("closed_fraction")
    if closed_fraction is None or not np.isfinite(closed_fraction):
        fraction_penalty = 0.5
    else:
        fraction_penalty = abs(float(closed_fraction) - 0.5)

    return slope_score - fraction_penalty


def _select_threshold(
    *,
    threshold_metrics: Mapping[float, Mapping[str, float | str | bool]],
    candidates: Sequence[float],
    user_threshold: float | None,
) -> Dict[str, float | str | None]:
    """Select which threshold to surface for plotting and legacy fields.

    Parameters
    ----------
    threshold_metrics : Mapping[float, Mapping[str, float | str | bool]]
        Per-threshold metric dictionary keyed by the threshold value used.
    candidates : Sequence[float]
        Thresholds that are eligible for automatic selection (typically the list
        the caller asked to evaluate).
    user_threshold : float | None
        Explicit threshold to honor for plotting; bypasses auto selection when set.

    Returns
    -------
    dict
        Selection metadata containing the chosen value, selection mode, reason,
        and the candidate list considered.
    """

    if user_threshold is not None:
        return {
            "value": float(user_threshold),
            "mode": "user",
            "reason": "explicit plot threshold",
            "candidates": list(candidates),
        }

    if len(candidates) == 1:
        only = float(candidates[0])
        return {
            "value": only,
            "mode": "user",
            "reason": "single threshold provided",
            "candidates": list(candidates),
        }

    best_value: float | None = None
    best_score = -np.inf
    for theta in candidates:
        metrics = threshold_metrics.get(theta)
        if metrics is None:
            continue
        score = _score_threshold_candidate(metrics)
        if score > best_score:
            best_score = score
            best_value = theta

    if best_value is not None and np.isfinite(best_score):
        return {
            "value": float(best_value),
            "mode": "auto",
            "reason": "highest slope/closed-fraction score",
            "candidates": list(candidates),
        }

    # Fallback: pick the median candidate deterministically.
    sorted_candidates = list(candidates)
    sorted_candidates.sort()
    fallback = float(sorted_candidates[len(sorted_candidates) // 2])
    return {
        "value": fallback,
        "mode": "fallback",
        "reason": "no successful thresholds; median candidate used",
        "candidates": list(candidates),
    }


def _prepare_table_features(
    *,
    features: Mapping[str, object],
    selected_metrics: Mapping[str, object],
    start_sample: int,
    end_sample: int,
    sfreq: float,
) -> Dict[str, float | str | bool]:
    """Flatten and filter features for DataFrame construction."""

    flattened_thresholds = features.get("threshold_metrics_flat", {})
    selection = features.get("selected_threshold", {})

    scalar_features = {
        key: value
        for key, value in features.items()
        if key
        not in {
            "threshold_metrics_flat",
            "thresholds",
            "base_features",
            "selected_threshold",
            "selected_threshold_metrics",
        }
        and not isinstance(value, (dict, list, tuple, np.ndarray))
    }

    scalar_selected = {
        key: value
        for key, value in selected_metrics.items()
        if not isinstance(value, (dict, list, tuple, np.ndarray))
    }

    combined = {
        **scalar_features,
        **scalar_selected,
        **{f"selected_{key}": value for key, value in scalar_selected.items()},
        **flattened_thresholds,
        "selected_threshold_value": selection.get("value"),
        "threshold_selection_mode": selection.get("mode"),
        "threshold_selection_reason": selection.get("reason"),
    }
    combined["time_under_threshold_fraction"] = combined.get("closed_fraction", float("nan"))
    refined_duration = float((end_sample - start_sample) / sfreq)
    combined["refined_duration"] = max(refined_duration, 0.0)
    return combined


def compute_blink_features(
    signal: np.ndarray,
    sfreq: float,
    threshold: float | Sequence[float],
    start_sample: int,
    end_sample: int,
    blink_type: Optional[str],
    feature_config: EARFeatureConfig,
    plot_threshold: float | None = None,
) -> Dict[str, object]:
    """Compute EAR-derived features for a single blink window.

    Parameters
    ----------
    signal : np.ndarray
        Full EAR signal in raw units.
    sfreq : float
        Sampling frequency in Hertz.
    threshold : float | Sequence[float]
        Single threshold or list of thresholds to evaluate for threshold-dependent metrics.
    start_sample : int
        Refined blink onset sample (inclusive).
    end_sample : int
        Refined blink offset sample (inclusive).
    blink_type : str | None
        Optional blink label.
    feature_config : EARFeatureConfig
        Configuration controlling baseline window, slope search, percentiles, and classification.
    plot_threshold : float | None
        Optional explicit threshold to use for plotting/legacy columns. When ``None``,
        auto-selection chooses among ``threshold`` candidates.

    Returns
    -------
    dict
        Structured features containing:
        - ``base_features``: threshold-independent metrics.
        - ``thresholds``: per-threshold metric dictionaries keyed by value.
        - ``threshold_metrics_flat``: flattened per-threshold metrics for easy tabular use.
        - ``selected_threshold``: selection metadata (value, mode, reason, candidates).
        - ``selected_threshold_metrics``: metrics tied to the chosen threshold.
        - ``table_features``: flattened, scalar-only metrics ready for DataFrame rows.
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

    evaluation_thresholds = _normalize_thresholds(threshold)
    computed_thresholds = _normalize_thresholds(threshold, extra_threshold=plot_threshold)

    threshold_metrics: Dict[float, Dict[str, float | str | bool]] = {}
    min_sample = int(transient["min_sample"])
    window = transient["window"]
    for theta in computed_thresholds:
        threshold_metrics[theta] = _compute_threshold_features(
            signal=signal,
            sfreq=sfreq,
            start_sample=start_sample,
            end_sample=end_sample,
            min_sample=min_sample,
            window=window,
            threshold=theta,
            feature_config=feature_config,
            blink_type=blink_type,
        )

    selection = _select_threshold(
        threshold_metrics=threshold_metrics,
        candidates=evaluation_thresholds,
        user_threshold=plot_threshold,
    )

    selected_metrics = threshold_metrics.get(selection["value"], {})
    threshold_metrics_flat = _flatten_threshold_metrics(threshold_metrics)

    features: Dict[str, object] = {
        "base_features": base_features,
        "thresholds": threshold_metrics,
        "threshold_metrics_flat": threshold_metrics_flat,
        "selected_threshold": selection,
        "selected_threshold_metrics": selected_metrics,
        # Convenience copies for backward compatibility with existing flat columns.
        "blink_type_original": blink_type,
    }
    features.update(base_features)
    features["table_features"] = _prepare_table_features(
        features=features,
        selected_metrics=selected_metrics,
        start_sample=start_sample,
        end_sample=end_sample,
        sfreq=sfreq,
    )

    return features


class EARBlinkFeatureExtractor:
    """Compute per-blink EAR metrics given refined boundaries."""

    def __init__(
        self,
        signal: np.ndarray,
        sfreq: float,
        threshold: float | Sequence[float],
        feature_config: Optional[EARFeatureConfig] = None,
        plot_threshold: float | None = None,
    ):
        """Create an EAR feature extractor.

        Parameters
        ----------
        signal : np.ndarray
            Full EAR signal (raw units).
        sfreq : float
            Sampling frequency in Hertz.
        threshold : float | Sequence[float]
            Threshold(s) to evaluate for threshold-dependent metrics.
        feature_config : EARFeatureConfig, optional
            Configuration controlling baseline, percentiles, slopes, and classification.
        plot_threshold : float | None
            Explicit threshold to surface in legacy columns/plots; defaults to automatic
            selection among ``threshold`` candidates.
        """
        self.signal = np.asarray(signal, dtype=float)
        self.sfreq = float(sfreq)
        self.thresholds = threshold
        self.plot_threshold = plot_threshold
        self.feature_config = feature_config or EARFeatureConfig()

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
            Input rows augmented with base EAR metrics, per-threshold metrics,
            flattened per-threshold columns, and legacy selected-threshold fields
            for plotting/report compatibility.
        """

        required_cols = {"refined_start_sample", "refined_end_sample"}
        missing_cols = required_cols - set(refined.columns)
        if missing_cols:
            raise ValueError(
                f"Refined annotations are missing required columns: {sorted(missing_cols)}"
            )

        records: List[Dict[str, float | str | bool]] = []
        for row in refined.to_dict(orient="records"):
            features = compute_blink_features(
                signal=self.signal,
                sfreq=self.sfreq,
                threshold=self.thresholds,
                start_sample=int(row["refined_start_sample"]),
                end_sample=int(row["refined_end_sample"]),
                blink_type=row.get("blink_type"),
                feature_config=self.feature_config,
                plot_threshold=self.plot_threshold,
            )
            records.append({**row, **features["table_features"]})

        df = pd.DataFrame.from_records(records)
        logger.info("Computed EAR features for %s blinks", len(df))
        return df
