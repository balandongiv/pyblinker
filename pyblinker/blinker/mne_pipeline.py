import mne
import numpy as np
import pandas as pd

from pyblinker._logging import get_logger
from pyblinker.blink_features.waveform_features.extract_blink_properties import (
    BlinkProperties,
)
from pyblinker.segmentation.geometry import (
    get_max_blink,
    left_right_zero_crossing,
    create_left_right_base,
)

logger = get_logger(__name__)


def get_mne_blink(detector, channel=None):
    """Run the MNE-based alternative blink detection pipeline.

    This pipeline skips the legacy fitting/statistics/masking stages and
    runs MNE's `find_eog_events` followed directly by `BlinkProperties`.
    """
    logger.info("Starting MNE-based blink detection pipeline.")

    detector.prepare_raw_signal()
    raw = detector.raw_data
    sfreq = detector.sfreq

    # 1. MNE find_eog_events
    # This automatically picks an EOG channel if channel is None,
    # or it uses the specified channel.
    logger.info(f"Finding EOG events using MNE. Channel: {channel}")
    try:
        events = mne.preprocessing.find_eog_events(raw, ch_name=channel)
    except Exception as e:
        logger.error(f"MNE find_eog_events failed: {e}")
        return [], None, 0, pd.DataFrame(), [], None

    n_events = len(events)
    if n_events == 0:
        logger.warning("No blinks detected by MNE.")
        return [], channel, 0, pd.DataFrame(), [], None

    # MNE returns events where the first column is the peak sample relative to raw.first_samp.
    peaks = events[:, 0] - raw.first_samp

    # Find the actual channel name used by MNE if none was provided
    if channel is None:
        try:
            eog_inds = mne.pick_types(raw.info, meg=False, eeg=False, eog=True)
            ch_names = [raw.ch_names[i] for i in eog_inds]
            channel = ch_names[0] if ch_names else raw.ch_names[0]
            logger.info(f"MNE automatically selected channel: {channel}")
        except Exception:
            channel = raw.ch_names[0]

    data, _ = raw.get_data(picks=channel, return_times=True)
    signal = data[0]
    data_size = signal.shape[0]

    # 2. Build blink regions
    # We create a window around each peak to serve as the start/end bounds.
    # A standard blink is roughly 0.4 - 0.5s long. We use +/- 0.3s here for bounds.
    window_samples = int(0.3 * sfreq)

    df_list = []
    for peak in peaks:
        start_blink = max(0, peak - window_samples)
        end_blink = min(data_size - 1, peak + window_samples)
        df_list.append({
            "start_blink": start_blink,
            "end_blink": end_blink,
            "max_blink": peak,
            "max_value": signal[peak],
            "outer_start": max(0, peak - window_samples - int(0.1*sfreq)),
            "outer_end": min(data_size - 1, peak + window_samples + int(0.1*sfreq)),
        })

    df = pd.DataFrame(df_list)

    # 3. Find zero crossings
    df[["left_zero", "right_zero"]] = df.apply(
        lambda row: left_right_zero_crossing(
            signal,
            int(row["max_blink"]),
            int(row["outer_start"]),
            int(row["outer_end"]),
        ),
        axis=1,
        result_type="expand",
    )

    # 4. Find bases using create_left_right_base
    df = create_left_right_base(signal, df)

    if df.empty:
        logger.warning("No valid blink regions found after base crossing.")
        return [], channel, 0, pd.DataFrame(), [], None

    # 5. Go directly to BlinkProperties
    logger.info("Extracting BlinkProperties directly (skipping fitting).")
    df_out = BlinkProperties(
        candidate_signal=signal,
        df=df,
        srate=sfreq,
        params=detector.params,
        fitted=False,  # Skip tent-slope fitting
    ).df

    # 6. Create Annotations
    annot = detector.create_annotations(df_out)

    # 7. Visualization
    fig_data = detector.generate_viz(signal, df_out) if detector.viz_data else []

    n_good_blinks = len(df_out)
    logger.info(f"MNE blink detection completed. {n_good_blinks} blinks detected.")

    # Create a mock ch_selected row to maintain API compatibility
    ch_selected = pd.DataFrame([{"ch": channel, "number_good_blinks": n_good_blinks}])

    return annot, channel, n_good_blinks, df_out, fig_data, ch_selected
