"""Aggregate EAR baseline and extrema features."""

from typing import Iterable, Dict, Any, List
import pandas as pd
import numpy as np
from pyblinker._logging import get_logger

from .features import ear_before_blink_avg_epoch, ear_extrema_epoch

logger = get_logger(__name__)


def aggregate_ear_features(
    blinks: Iterable[Dict[str, Any]],
    sfreq: float,
    n_epochs: int,
    lookback: float = 3.0,
) -> pd.DataFrame:
    """Aggregate EAR features for multiple epochs.

    Parameters
    ----------
    blinks : Iterable[dict]
        Blink annotations with ``epoch_index`` and ``epoch_signal``.
    sfreq : float
        Sampling frequency in Hertz.
    n_epochs : int
        Number of epochs to aggregate.
    lookback : float, optional
        Seconds to average before the first blink, by default ``3.0``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by epoch with EAR baseline and extrema features.
    """
    logger.info("Aggregating EAR features over %d epochs", n_epochs)

    per_epoch_signal: List[np.ndarray | None] = [None for _ in range(n_epochs)]
    per_epoch_blinks: List[List[Dict[str, Any]]] = [list() for _ in range(n_epochs)]

    for blink in blinks:
        idx = blink["epoch_index"]
        if 0 <= idx < n_epochs:
            per_epoch_blinks[idx].append(blink)
            if per_epoch_signal[idx] is None:
                per_epoch_signal[idx] = np.asarray(blink["epoch_signal"], dtype=float)

    records = []
    for idx in range(n_epochs):
        signal = per_epoch_signal[idx]
        blink_list = per_epoch_blinks[idx]
        record = {"epoch": idx}
        if signal is None:
            record.update(
                {
                    "EAR_Before_Blink_left_avg": float("nan"),
                    "EAR_Before_Blink_right_avg": float("nan"),
                    "EAR_left_min": float("nan"),
                    "EAR_right_min": float("nan"),
                    "EAR_left_max": float("nan"),
                    "EAR_right_max": float("nan"),
                }
            )
        else:
            pre = ear_before_blink_avg_epoch(signal, blink_list, sfreq, lookback)
            extrema = ear_extrema_epoch(signal)
            record.update(
                {
                    "EAR_Before_Blink_left_avg": pre,
                    "EAR_Before_Blink_right_avg": pre,
                    "EAR_left_min": extrema["ear_min"],
                    "EAR_right_min": extrema["ear_min"],
                    "EAR_left_max": extrema["ear_max"],
                    "EAR_right_max": extrema["ear_max"],
                }
            )
        records.append(record)

    df = pd.DataFrame.from_records(records).set_index("epoch")
    logger.debug("Aggregated EAR DataFrame shape: %s", df.shape)
    return df
