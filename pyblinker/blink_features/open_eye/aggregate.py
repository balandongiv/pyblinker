"""Aggregate open-eye features across epochs."""

from typing import Iterable, Dict, List, Any
import pandas as pd
import numpy as np
from pyblinker._logging import get_logger

from .features import (
    baseline_mean_epoch,
    baseline_drift_epoch,
    baseline_std_epoch,
    baseline_mad_epoch,
    perclos_epoch,
    eye_opening_rms_epoch,
    micropause_count_epoch,
    zero_crossing_rate_epoch,
)

logger = get_logger(__name__)


def aggregate_open_eye_features(
    blinks: Iterable[Dict[str, Any]],
    sfreq: float,
    n_epochs: int,
) -> pd.DataFrame:
    """Aggregate open-eye metrics for multiple epochs.

    Parameters
    ----------
    blinks : Iterable[dict]
        Blink annotations with ``epoch_index`` and ``epoch_signal``.
    sfreq : float
        Sampling frequency in Hertz.
    n_epochs : int
        Number of epochs to aggregate.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by epoch containing open-eye features.
    """
    logger.info("Aggregating open-eye features over %d epochs", n_epochs)

    per_epoch_signals: List[np.ndarray | None] = [None for _ in range(n_epochs)]
    per_epoch_blinks: List[List[Dict[str, Any]]] = [list() for _ in range(n_epochs)]

    for blink in blinks:
        idx = blink["epoch_index"]
        if 0 <= idx < n_epochs:
            per_epoch_blinks[idx].append(blink)
            if per_epoch_signals[idx] is None:
                per_epoch_signals[idx] = np.asarray(blink["epoch_signal"], dtype=float)

    records = []
    for idx in range(n_epochs):
        signal = per_epoch_signals[idx]
        blinks_epoch = per_epoch_blinks[idx]
        record = {"epoch": idx}
        if signal is None:
            record.update(
                {
                    "baseline_mean": float("nan"),
                    "baseline_drift": float("nan"),
                    "baseline_std": float("nan"),
                    "baseline_mad": float("nan"),
                    "perclos": float("nan"),
                    "eye_opening_rms": float("nan"),
                    "micropause_count": 0,
                    "zero_crossing_rate": float("nan"),
                }
            )
        else:
            record["baseline_mean"] = baseline_mean_epoch(signal, blinks_epoch)
            record["baseline_drift"] = baseline_drift_epoch(signal, blinks_epoch, sfreq)
            record["baseline_std"] = baseline_std_epoch(signal, blinks_epoch)
            record["baseline_mad"] = baseline_mad_epoch(signal, blinks_epoch)
            record["perclos"] = perclos_epoch(signal, blinks_epoch)
            record["eye_opening_rms"] = eye_opening_rms_epoch(signal, blinks_epoch)
            record["micropause_count"] = micropause_count_epoch(
                signal, blinks_epoch, sfreq
            )
            record["zero_crossing_rate"] = zero_crossing_rate_epoch(
                signal, blinks_epoch
            )
        records.append(record)

    df = pd.DataFrame.from_records(records).set_index("epoch")
    logger.debug("Aggregated open-eye DataFrame shape: %s", df.shape)
    return df
