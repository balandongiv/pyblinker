"""I/O helpers for EAR blink refinement tutorials."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import mne
import pandas as pd
import numpy as np

from pyblinker._logging import get_logger

logger = get_logger(__name__)


def load_ear_channel(fif_path: Path, channel: str) -> Tuple[np.ndarray, float]:
    """Load a single EAR channel from a FIF recording."""

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    try:
        signal = raw.get_data(picks=channel)[0]
    except Exception as exc:  # pragma: no cover - defensive channel lookup
        raise ValueError(f"Channel {channel} not found in {fif_path}") from exc

    logger.info("Loaded EAR channel %s at %s Hz", channel, sfreq)
    return signal, sfreq


def load_coarse_blinks(csv_path: Path) -> pd.DataFrame:
    """Load coarse blink annotations."""

    annotations = pd.read_csv(csv_path)
    missing_cols = {"onset", "duration"} - set(annotations.columns)
    if missing_cols:
        raise ValueError(
            f"Annotation file is missing required columns: {sorted(missing_cols)}"
        )
    logger.info("Loaded %s coarse blink annotations", len(annotations))
    return annotations
