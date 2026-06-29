"""Double-thresholding blink detector core: two-stage threshold detection.

Stage A — Autoreject epoch-level screening identifies which epochs are suspicious.
Stage B — A robust ``center + k * MAD`` threshold is estimated from those flagged
           epochs.
Stage C — Blink regions are found via ``scan_threshold_crossings_kleifges`` using the
           Stage B threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..blinker.default_setting import build_blink_params
from ..blinker.get_blink_positions import scan_threshold_crossings_kleifges
from ..epoch_detection.epoch_channel import map_concatenated_blinks_to_epochs
from ..epoch_detection.epoch_input import PreparedEpochDetectionInput
from ..epoch_detection.pipeline_utils import build_epoch_boundaries, build_signal_by_epoch

from .autoreject_epoch_screener import screen_epochs_with_autoreject
from .blink_threshold import compute_flagged_epoch_threshold


def blink_position_strategy_dbo(
    prepared: PreparedEpochDetectionInput,
    valid_epoch_indices: list[int],
    *,
    setting: dict | None = None,
    **kwargs,
) -> list[dict]:
    """Run double-thresholding blink detection per channel and return results.

    Parameters
    ----------
    prepared:
        Pre-processed epoch data.
    valid_epoch_indices:
        Indices of valid (non-dropped) epochs.
    setting:
        Optional configuration dict.  Supported keys:

        - ``autoreject_random_state`` (int, default 42)
        - ``std_threshold`` (float, default 1.5)
        - ``center_method`` (str, default ``"median"``) — ``"median"`` or ``"mean"``
        - ``min_flagged_epochs`` (int, default 1)
        - ``verbose`` (bool, default False)
    **kwargs:
        Additional overrides merged on top of ``setting``.

    Returns
    -------
    list of dict, one per channel, each with standard keys:
        ``channel``, ``df_positions``, ``mapped_candidates``, ``signal_by_epoch``
    and double-thresholding diagnostic keys:
        ``flagged_valid_epoch_indices``, ``n_flagged``, ``used_all_epochs``,
        ``blink_region_threshold``, ``threshold_center``, ``threshold_dispersion``.
    """
    options = dict(setting or {})
    options.update(kwargs)

    autoreject_random_state = int(options.get("autoreject_random_state", 42))
    # std_threshold: multiplier for the MAD dispersion term in Stage B.
    # 1.5 matches the original Kleifges/BLINKER detection multiplier, giving
    # recall that meets or exceeds the single-threshold baseline while keeping
    # false positives well below it. (The old default of 3.0 was borrowed from
    # BLINKER's blink-quality classification context and was too conservative
    # for detection.)
    std_threshold = float(options.get("std_threshold", 1.5))
    center_method = str(options.get("center_method", "median"))
    min_flagged_epochs = int(options.get("min_flagged_epochs", 1))
    verbose = bool(options.get("verbose", False))

    params = build_blink_params({})
    params["sfreq"] = float(prepared.sfreq)
    sfreq = params["sfreq"]

    min_blink_frames = float(params["min_event_len"] * sfreq)

    # ------------------------------------------------------------------ Stage A
    screen_result = screen_epochs_with_autoreject(
        prepared,
        valid_epoch_indices,
        random_state=autoreject_random_state,
        min_flagged_epochs=min_flagged_epochs,
        verbose=verbose,
    )

    # ------------------------------------------------------------------ Stage B
    threshold_result = compute_flagged_epoch_threshold(
        prepared,
        valid_epoch_indices,
        screen_result.flagged_valid_epoch_indices,
        std_threshold=std_threshold,
        center_method=center_method,
        verbose=verbose,
    )

    epoch_boundaries = build_epoch_boundaries(
        len(valid_epoch_indices), prepared.epoch_length_samples
    )
    valid_indices = np.asarray(valid_epoch_indices, dtype=int)

    # ------------------------------------------------------------------ Stage C
    results: list[dict] = []
    for channel_idx, channel_name in enumerate(prepared.channel_names):
        concatenated_signal = prepared.data[valid_indices, channel_idx, :].reshape(-1)

        blink_threshold = float(threshold_result.thresholds[channel_name])
        start_blinks, end_blinks = scan_threshold_crossings_kleifges(
            concatenated_signal,
            blink_threshold,
            min_blink_frames,
            progress_bar=False,
            channel_name=channel_name,
        )

        df_positions = pd.DataFrame({"start_blink": start_blinks, "end_blink": end_blinks})
        mapped_candidates = map_concatenated_blinks_to_epochs(
            df_positions,
            channel=channel_name,
            valid_epoch_indices=valid_epoch_indices,
            epoch_boundaries=epoch_boundaries,
            sfreq=prepared.sfreq,
        )

        results.append(
            {
                "channel": channel_name,
                "df_positions": df_positions,
                "mapped_candidates": mapped_candidates,
                "signal_by_epoch": build_signal_by_epoch(prepared, channel_idx),
                "flagged_valid_epoch_indices": screen_result.flagged_valid_epoch_indices,
                "n_flagged": screen_result.n_flagged,
                "used_all_epochs": threshold_result.used_all_epochs,
                "blink_region_threshold": blink_threshold,
                "threshold_center": float(threshold_result.centers[channel_name]),
                "threshold_dispersion": float(threshold_result.dispersions[channel_name]),
            }
        )
    return results


__all__ = ["blink_position_strategy_dbo"]
