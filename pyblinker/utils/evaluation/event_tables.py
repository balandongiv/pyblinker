"""Helpers for turning PyBlinker and MATLAB payloads into comparable event tables."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def prepare_event_tables(
    py_payload: Mapping[str, object],
    blinker_payload: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return comparable 1-based blink event tables from PyBlinker and MATLAB payloads."""

    py_events = (
        pd.DataFrame(py_payload["events"])[["left_zero", "right_zero", "max_value"]]
        .rename(
            columns={
                "left_zero": "start_blink",
                "right_zero": "end_blink",
                "max_value": "maxValue",
            }
        )
        .copy()
    )
    py_events[["start_blink", "end_blink"]] = (
        py_events[["start_blink", "end_blink"]].astype(int) + 1
    )
    py_events = py_events.sort_values("start_blink", kind="mergesort").reset_index(drop=True)

    blinker_events = (
        pd.DataFrame(blinker_payload["frames"]["blinkFits"])[["leftZero", "rightZero", "maxValue"]]
        .rename(columns={"leftZero": "start_blink", "rightZero": "end_blink"})
        .sort_values("start_blink", kind="mergesort")
        .reset_index(drop=True)
    )
    return py_events, blinker_events
