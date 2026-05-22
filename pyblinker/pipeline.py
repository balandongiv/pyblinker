"""Main pipeline entry for feature extraction."""

from functools import partial
from typing import Callable, Dict, Iterable, Optional, Sequence

import mne
import pandas as pd

from pyblinker._logging import get_logger

from .blink_features.blink_events.classification import (
    aggregate_classification_features,
)
from .blink_features.blink_events.event_features import (
    aggregate_blink_event_features,
)
from .blink_features.ear_metrics import aggregate_ear_features
from .blink_features.energy import aggregate_energy_features
from .blink_features.frequency_domain import aggregate_frequency_domain_features
from .blink_features.kinematics import aggregate_kinematic_features
from .blink_features.open_eye import aggregate_open_eye_features
from .blink_features.waveform_features import aggregate_waveform_features

logger = get_logger(__name__)

FEATURE_AGGREGATORS: Dict[str, Callable[..., pd.DataFrame]] = {
    "ear": aggregate_ear_features,
    "classification": aggregate_classification_features,
    "kinematics": aggregate_kinematic_features,
    "energy": aggregate_energy_features,
    "open_eye": aggregate_open_eye_features,
    "frequency": aggregate_frequency_domain_features,
    "waveform": aggregate_waveform_features,
}


def extract_features(
    blinks: Iterable[Dict[str, int]],
    sfreq: float,
    epoch_len: float,
    n_epochs: int,
    features: Sequence[str] | None = None,
    raw_segments: Optional[Sequence[mne.io.BaseRaw]] = None,
) -> pd.DataFrame:
    """Extract blink features using provided blink annotations.

    Parameters
    ----------
    blinks : Iterable[Dict[str, int]]
        Blink annotations for each detected blink. Each record must contain
        ``epoch_index`` (``int``), ``epoch_signal`` (1D array),
        ``refined_start_frame`` (``int``), ``refined_peak_frame`` (``int``),
        and ``refined_end_frame`` (``int``). These fields match the format
        used throughout the feature modules and specify the blink location
        relative to its epoch.
    sfreq : float
        Sampling frequency of the recording.
    epoch_len : float
        Length of each epoch in seconds.
    n_epochs : int
        Total number of epochs.
    features : Sequence[str] | None, optional
        Feature groups to compute. Values from
        :func:`aggregate_blink_event_features` (``"blink_count"``, ``"blink_rate"``,
        ``"ibi"``), ``"morphology``, ``"kinematics``, ``"energy``, ``"open_eye``,
        ``"ear"``, ``"waveform`` and ``"classification"`` are recognized. ``None`` computes all
        available features.
    raw_segments : Sequence[mne.io.BaseRaw] | None, optional
        Collection of 30-second raw segments with annotations. Required when
        ``"blink_interval_dist"`` is among ``features``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with aggregated features per epoch.
    """
    logger.info("Starting feature extraction")

    df_features = [
        aggregate_blink_event_features(blinks, sfreq, epoch_len, n_epochs, features)
    ]

    feature_aggregators = FEATURE_AGGREGATORS.copy()
    feature_aggregators["classification"] = partial(
        aggregate_classification_features, epoch_len=epoch_len
    )

    features_to_run = features or feature_aggregators.keys()
    for feature_name in features_to_run:
        agg_func = feature_aggregators.get(feature_name)
        if agg_func is not None:
            df_features.append(agg_func(blinks, sfreq, n_epochs))

    df = pd.concat(df_features, axis=1)

    logger.info("Finished feature extraction")
    return df
