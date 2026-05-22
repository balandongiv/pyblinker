"""Aggregate blink energy features across epochs."""

from __future__ import annotations
from pyblinker._logging import get_logger

from typing import Any, Dict, Iterable, List
import pandas as pd

from .helpers import compute_basic_statistics
from .per_blink import compute_blink_energy

logger = get_logger(__name__)

_METRICS = (
    "blink_signal_energy",
    "teager_kaiser_energy",
    "blink_line_length",
    "blink_velocity_integral",
)
_STATS = ("mean", "std", "cv")


def aggregate_energy_features(
    blinks: Iterable[Dict[str, Any]],
    sfreq: float,
    n_epochs: int,
) -> pd.DataFrame:
    """Aggregate energy metrics for each epoch.

    Parameters
    ----------
    blinks : Iterable[dict]
        Blink annotations with an ``epoch_index`` field.
    sfreq : float
        Sampling frequency in Hertz.
    n_epochs : int
        Number of epochs to aggregate.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by epoch with energy features.
    """
    logger.info("Aggregating energy features over %d epochs", n_epochs)
    per_epoch: List[List[Dict[str, Any]]] = [list() for _ in range(n_epochs)]
    for blink in blinks:
        idx = blink["epoch_index"]
        if 0 <= idx < n_epochs:
            per_epoch[idx].append(blink)

    records: List[Dict[str, float]] = []
    for epoch_blinks in per_epoch:
        energies: List[float] = []
        tkeo_vals: List[float] = []
        lengths: List[float] = []
        vel_ints: List[float] = []
        for blink in epoch_blinks:
            metrics = compute_blink_energy(blink, sfreq)
            energies.append(metrics["blink_signal_energy"])
            tkeo_vals.append(metrics["teager_kaiser_energy"])
            lengths.append(metrics["blink_line_length"])
            vel_ints.append(metrics["blink_velocity_integral"])
        stats_energy = compute_basic_statistics(energies)
        stats_tkeo = compute_basic_statistics(tkeo_vals)
        stats_len = compute_basic_statistics(lengths)
        stats_vel = compute_basic_statistics(vel_ints)
        record: Dict[str, float] = {}
        for metric, stats in zip(
            _METRICS, (stats_energy, stats_tkeo, stats_len, stats_vel)
        ):
            for stat_name, value in stats.items():
                record[f"{metric}_{stat_name}"] = value
        records.append(record)

    df = pd.DataFrame.from_records(records, index=pd.RangeIndex(n_epochs))
    logger.debug("Aggregated energy DataFrame shape: %s", df.shape)
    return df
