"""Kleifges 2017 per-channel blink detection strategy."""

from __future__ import annotations

import pandas as pd

from ..blinker.default_setting import build_blink_params
from ..blinker.get_blink_positions import compute_basic_statistics, scan_threshold_crossings_kleifges
from ..epoch_detection.epoch_channel import map_concatenated_blinks_to_epochs
from ..epoch_detection.epoch_input import PreparedEpochDetectionInput
from ..epoch_detection.pipeline_utils import build_epoch_boundaries, build_signal_by_epoch


def kleifges_strategy(
    prepared: PreparedEpochDetectionInput,
    valid_epoch_indices: list[int],
) -> list[dict]:
    """Run Kleifges 2017 blink detection on each channel and map results to epochs.

    Uses the threshold-crossing scan without minimum-separation filtering,
    matching the bare Kleifges approach.

    Returns a list of dicts, one per channel, each containing:
    - ``channel``: channel name
    - ``df_positions``: DataFrame of raw blink-position candidates
    - ``mapped_candidates``: DataFrame of epoch-relative blink candidates
    - ``signal_by_epoch``: dict mapping epoch_index -> 1-D filtered signal array
    """
    params = build_blink_params({})
    params["sfreq"] = float(prepared.sfreq)

    epoch_boundaries = build_epoch_boundaries(len(valid_epoch_indices), prepared.epoch_length_samples)

    results = []
    for channel_index, channel_name in enumerate(prepared.channel_names):
        concatenated_signal = prepared.data[valid_epoch_indices, channel_index, :].reshape(-1)

        min_blink_frames, threshold = compute_basic_statistics(params, concatenated_signal)
        start_blinks, end_blinks = scan_threshold_crossings_kleifges(
            concatenated_signal,
            float(threshold),
            min_blink_frames,
            progress_bar=False,
            channel_name=channel_name,
        )
        df_positions = pd.DataFrame({"start_blink": start_blinks, "end_blink": end_blinks})
        mapped_positions = map_concatenated_blinks_to_epochs(
            df_positions,
            channel=channel_name,
            valid_epoch_indices=valid_epoch_indices,
            epoch_boundaries=epoch_boundaries,
            sfreq=prepared.sfreq,
        )
        signal_by_epoch = build_signal_by_epoch(prepared, channel_index)
        results.append(
            {
                "channel": channel_name,
                "df_positions": df_positions,
                "mapped_candidates": mapped_positions,
                "signal_by_epoch": signal_by_epoch,
            }
        )
    return results


__all__ = ["kleifges_strategy"]
