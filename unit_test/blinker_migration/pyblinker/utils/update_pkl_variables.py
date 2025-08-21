import pickle
from pathlib import Path
from typing import Any, Dict

RENAME_MAP = {
    "baseFraction": "base_fraction",
    "shutAmpFraction": "shut_amp_fraction",
    "pAVRThreshold": "p_avr_threshold",
    "stdThreshold": "std_threshold",
    "minEventLen": "min_event_len",
    "minEventSep": "min_event_sep",
    "correlationThresholdTop": "correlation_threshold_top",
    "correlationThresholdBottom": "correlation_threshold_bottom",
    "correlationThresholdMiddle": "correlation_threshold_middle",
    "blinkAmpRange_1": "blink_amp_range_1",
    "blinkAmpRange_2": "blink_amp_range_2",
    "goodRatioThreshold": "good_ratio_threshold",
    "minGoodBlinks": "min_good_blinks",
    "keepSignals": "keep_signals",
    "correlationThreshold": "correlation_threshold",
    "startBlinks": "start_blink",
    "endBlinks": "end_blink",
    "maxValue": "max_value",
    "maxFrame": "max_blink",
    "max_frame": "max_blink",
    "outerStarts": "outer_start",
    "outerEnds": "outer_end",
    "leftOuter": "outer_start",
    "rightOuter": "outer_end",
    "outer_starts": "outer_start",
    "outer_ends": "outer_end",
    "leftZero": "left_zero",
    "rightZero": "right_zero",
    "leftBase": "left_base",
    "rightBase": "right_base",
    "maxPosVelFrame": "max_pos_vel_frame",
    "maxNegVelFrame": "max_neg_vel_frame",
    "leftBaseHalfHeight": "left_base_half_height",
    "rightBaseHalfHeight": "right_base_half_height",
    "leftZeroHalfHeight": "left_zero_half_height",
    "rightZeroHalfHeight": "right_zero_half_height",
    "numberBlinks": "number_blinks",
    "numberGoodBlinks": "number_good_blinks",
    "blinkAmpRatio": "blink_amp_ratio",
    "bestMedian": "best_median",
    "bestRobustStd": "best_robust_std",
    "goodRatio": "good_ratio",
    "averLeftVelocity": "aver_left_velocity",
    "averRightVelocity": "aver_right_velocity",
    "leftXIntercept": "left_x_intercept",
    "rightXIntercept": "right_x_intercept",
    "leftRange": "left_range",
    "rightRange": "right_range",
    "leftSlope": "left_slope",
    "rightSlope": "right_slope",
    "leftXIntercept_int": "left_x_intercept_int",
    "rightXIntercept_int": "right_x_intercept_int",
    "xIntersect": "x_intersect",
    "yIntersect": "y_intersect",
    "peaksPosVelBase": "peaks_pos_vel_base",
    "peaksPosVelZero": "peaks_pos_vel_zero",
    "durationBase": "duration_base",
    "durationZero": "duration_zero",
    "durationTent": "duration_tent",
    "durationHalfBase": "duration_half_base",
    "durationHalfZero": "duration_half_zero",
    "posAmpVelRatioZero": "pos_amp_vel_ratio_zero",
    "negAmpVelRatioZero": "neg_amp_vel_ratio_zero",
    "posAmpVelRatioBase": "pos_amp_vel_ratio_base",
    "negAmpVelRatioBase": "neg_amp_vel_ratio_base",
    "posAmpVelRatioTent": "pos_amp_vel_ratio_tent",
    "negAmpVelRatioTent": "neg_amp_vel_ratio_tent",
    "timeShutBase": "time_shut_base",
    "timeShutZero": "time_shut_zero",
    "timeShutTent": "time_shut_tent",
    "closingTimeZero": "closing_time_zero",
    "reopeningTimeZero": "reopening_time_zero",
    "closingTimeTent": "closing_time_tent",
    "reopeningTimeTent": "reopening_time_tent",
    "peakMaxBlink": "peak_max_blink",
    "peakMaxTent": "peak_max_tent",
    "peakTimeBlink": "peak_time_blink",
    "peakTimeTent": "peak_time_tent",
    "interBlinkMaxAmp": "inter_blink_max_amp",
    "interBlinkMaxVelBase": "inter_blink_max_vel_base",
    "interBlinkMaxVelZero": "inter_blink_max_vel_zero",
}


def rename_keys(data: Any, rename_map: Dict[str, str]) -> Any:
    if isinstance(data, dict):
        return {
            rename_map.get(k, k): rename_keys(v, rename_map) for k, v in data.items()
        }
    if isinstance(data, list):
        return [rename_keys(item, rename_map) for item in data]
    return data


def update_pkl_file(path: str) -> None:
    file_path = Path(path)
    with file_path.open("rb") as f:
        data = pickle.load(f)
    data = rename_keys(data, RENAME_MAP)
    with file_path.open("wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    update_pkl_file("../../../test_files/blink_properties_fits.pkl")
    update_pkl_file("../../../test_files/file_test_blink_position.pkl")
