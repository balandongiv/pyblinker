from __future__ import annotations

import pandas as pd

from pyblinker.utils.evaluation.event_tables import prepare_event_tables


def test_prepare_event_tables_converts_py_zero_based_boundaries():
    py_payload = {
        "events": pd.DataFrame(
            {
                "left_zero": [9],
                "right_zero": [19],
                "max_value": [1.5],
            }
        )
    }
    blinker_payload = {
        "frames": {
            "blinkFits": pd.DataFrame(
                {
                    "leftZero": [10],
                    "rightZero": [20],
                    "maxValue": [1.5],
                }
            )
        }
    }

    py_events, blinker_events = prepare_event_tables(py_payload, blinker_payload)

    expected = pd.DataFrame(
        {
            "start_blink": [10],
            "end_blink": [20],
            "maxValue": [1.5],
        }
    )

    pd.testing.assert_frame_equal(py_events, expected)
    pd.testing.assert_frame_equal(blinker_events, expected)
