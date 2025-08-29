
from typing import Iterable, Dict, Any, List, Sequence
import logging
import pandas as pd
from ..energy.helpers import _extract_blink_windows, _segment_to_samples

logger = logging.getLogger(__name__)




def _sample_windows_from_metadata(
    metadata: pd.Series | Dict[str, Any],
    channel: str,
    sfreq: float,
    n_times: int,
    epoch_index: int,
) -> List[slice]:
    """Convert blink onset/duration metadata to sample windows.

    Parameters
    ----------
    metadata : pandas.Series or dict
        Metadata row containing blink onset and duration information.
    channel : str
        Channel name used to infer which metadata columns to read.
    sfreq : float
        Sampling frequency in Hertz.
    n_times : int
        Number of samples in each epoch.
    epoch_index : int
        Index of the epoch, used for error messages.

    Returns
    -------
    list of slice
        Sample index windows for each blink within the epoch.
    """

    logger.debug("Extracting sample windows from metadata")
    windows = _extract_blink_windows(metadata, channel, epoch_index)
    sample_windows: List[slice] = []
    for onset_s, duration_s in windows:
        sl = _segment_to_samples(onset_s, duration_s, sfreq, n_times)
        if sl.stop - sl.start > 1:
            sample_windows.append(sl)
    logger.debug("Found %d sample windows", len(sample_windows))
    return sample_windows
