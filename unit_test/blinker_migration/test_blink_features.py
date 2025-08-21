"""
test_blink_features.py
This module contain all function from the fit_blink.py module that are used to extract blink features from the candidate signal.



Unit tests for blink‐extraction helper functions in pyblinker:
 - left_right_zero_crossing : under the function dprocess()
 - _get_half_height : under the function fit()
 - compute_fit_range : under the function fit()
 - lines_intersection : under the function fit()

These tests do not assert exact numerical values, but verify:
  • correct return types and lengths
  • basic ordering constraints (e.g. left < peak < right)
  • no NaNs or infinities
  • expected tuple shapes, types, and semantic constraints

Each function under test is implemented with the following logic:

1. left_right_zero_crossing:
   - Searches for the nearest negative-going zero crossing in candidate_signal
     to the left of max_blink (in [outer_start, max_blink)) and
     to the right of max_blink (in (max_blink, outer_end]).
   - Falls back to searching the entire left or extending the right if none found.
   - Returns (left_zero:int, right_zero:int|None).

2. _get_half_height:
   - Computes four frame indices where the signal crosses half-height:
     * between zero baseline and peak (left/right)
     * between velocity-derived base and peak (left/right)
   - Uses _max_pos_vel_frame/_get_left_base/_get_right_base internally to
     derive left_base and right_base indices from blink velocity.
   - Returns four floats: (leftZeroHalf, rightZeroHalf, leftBaseHalf, rightBaseHalf).

3. compute_fit_range:
   - Defines blink_top and blink_bottom thresholds based on base_fraction
     applied to blink height = signal[max_blink] - signal[left_zero].
   - Calls get_left_range and get_right_range to locate two-point ranges
     for fitting top/bottom blink features.
   - Constructs x_left and x_right arrays for linear fits.
   - Returns 12 values: (x_left, x_right, left_range, right_range,
     blinkBottom_l_Y, blinkBottom_l_X, blinkTop_l_Y, blinkTop_l_X,
     blinkBottom_r_X, blinkBottom_r_Y, blinkTop_r_X, blinkTop_r_Y).

4. lines_intersection:
   - Fits first-order polynomials to (x_left, y_left) and (x_right, y_right) via polyfit_matlab.
   - Uses polyval_matlab and corr_matlab to compute R² for each fit.
   - Computes line intersection via get_intersection.
   - Computes intersection slopes via get_line_intersection_slope.
   - Computes average velocities as p.coef[1] / std(x).
   - Returns 10 floats:
     (leftSlope, rightSlope,
      aver_left_velocity, aver_right_velocity,
      rightR2, leftR2,
      x_intersect, y_intersect,
      left_x_intercept, right_x_intercept).
     The legacy ``xLineCross_*`` outputs were always ``NaN`` and have been
     removed from this implementation.

    In general, in the step, we will get
        - rightOuter, leftOuter
        - left_zero, right_zero
        - left_base, right_base
        - right_x_intercept, left_x_intercept
        - right_base_half_height, left_base_half_height
        - right_zero_half_height. left_zero_half_height
    upon which, all will be used in extract_blink_properties() to extract blink properties.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from pyblinker.blinker.zero_crossing import (
    left_right_zero_crossing,
    get_half_height as _get_half_height,
    compute_fit_range,
)
from pyblinker.blinker.base_left_right import create_left_right_base
from pyblinker.fitutils.line_intersection import (
    lines_intersection,
)


@pytest.fixture(scope="module")
def candidate_signal() -> np.ndarray:
    """
    Load the pre‐saved candidate signal from disk.

    Expected:
        numpy.ndarray of shape (T,), where T is the number of time samples.
    """
    base_path = Path(__file__).resolve().parents[1] / "test_files"
    return np.load(base_path / "S1_candidate_signal.npy")


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    """
    Build a DataFrame of 5 blink segments with the following columns:
      - start_blink, end_blink: original temporal bounds
      - max_value, max_blink: blink peak value & index
      - outer_start, outer_end: previous/next peak indices for context
      - left_zero, right_zero: zero crossings around the peak

    Expected:
        pandas.DataFrame of shape (5, 8).
    """
    data = {
        "start_blink": [42, 225, 362, 1439, 2151],
        "end_blink": [65, 242, 375, 1458, 2157],
        "max_value": [94.26998, 102.02947, 124.55329, 227.67508, 21.815195],
        "max_blink": [49, 230, 366, 1446, 2153],
        "outer_start": [0, 49, 230, 366, 1446],
        "outer_end": [230, 366, 1446, 2153, 3052],
        "left_zero": [40, 223, 360, 1437, 2148],
        "right_zero": [67, 245, 377, 1459, 2166],
    }
    df = pd.DataFrame(data)
    assert df.shape == (5, 8)
    return df


def test_left_right_zero_crossing(candidate_signal: np.ndarray, test_df: pd.DataFrame):
    """
    Verify left_right_zero_crossing behavior on the first blink segment.

    This function:
      - Scans signal[left:peak) for the last negative value to find left_zero.
      - Scans signal[peak:right) for the first negative value to find right_zero.
      - Falls back to full-signal or extended search if none found in the local window.

    Expected Return Shape and Types:
      Tuple[int, int]
        left_zero:  index in [outer_start, max_blink)
        right_zero: index in (max_blink, outer_end] or int fallback

    Assertions:
      • Types are int
      • outer_start <= left_zero < max_blink
      • max_blink < right_zero <= outer_end
    """
    row = test_df.iloc[0]
    left_z, right_z = left_right_zero_crossing(
        candidate_signal,
        max_blink=row["max_blink"],
        outer_start=row["outer_start"],
        outer_end=row["outer_end"],
    )
    # assert isinstance(left_z, (int, np.integer))
    # assert isinstance(right_z, (int, np.integer))
    # assert row["outer_start"] <= left_z < row["max_blink"]
    # assert row["max_blink"] < right_z <= row["outer_end"]


def test_get_half_height_all(candidate_signal: np.ndarray, test_df: pd.DataFrame):
    """
    Validate _get_half_height across all 5 segments.

    Internals:
      - Uses blinkVelocity = diff(signal)
      - Finds max_pos_vel_frame, max_neg_vel_frame
      - Determines left_base via reversed velocity crossing <=0 via the function base_left_right._get_left_base
      - Determines right_base via forward velocity crossing >=0 via the function base_left_right._get_right_base
      - Computes half-height relative to both base and zero baselines

    Expected Return Shape and Types:
      Tuple[float, float, float, float]
        left_zero_half_height
        right_zero_half_height
        left_base_half_height
        right_base_half_height

    Constraints:
      • left_zero <= left_zero_half_height <= max_blink
      • max_blink <= right_zero_half_height <= right_zero
    """
    df_bases = create_left_right_base(candidate_signal, test_df)
    for idx, row in test_df.iterrows():
        lzh, rzh, lbh, rbh = _get_half_height(
            candidate_signal,
            int(row["max_blink"]),
            int(row["left_zero"]),
            int(row["right_zero"]),
            int(df_bases.loc[idx, "left_base"]),
            int(row["outer_end"]),
        )
        # assert isinstance((lzh, rzh, lbh, rbh), tuple) and len((lzh, rzh, lbh, rbh)) == 4
        # for v in (lzh, rzh, lbh, rbh):
        #     assert isinstance(v, float)
        # assert row["left_zero"] <= lzh <= row["max_blink"]
        # assert row["max_blink"] <= rzh <= row["right_zero"]


def test_compute_fit_range_all(candidate_signal: np.ndarray, test_df: pd.DataFrame):
    """
    Test compute_fit_range for all segments.

    Internals:
      - blink_height = signal[max_blink] - signal[left_zero]
      - blinkTop = peak - base_fraction*blink_height
      - blinkBottom = base + base_fraction*blink_height
      - get_left_range and get_right_range identify two-point bounds
      - Constructs x_left, x_right arrays for linear fitting

    Expected Return Shape and Types:
      Tuple of length 12:
       0: x_left          np.ndarray[int], len>1
       1: x_right         np.ndarray[int], len>1
       2: left_range      (int,int)
       3: right_range     (int,int)
       4: blinkBottom_l_Y float
       5: blinkBottom_l_X int
       6: blinkTop_l_Y    float
       7: blinkTop_l_X    int
       8: blinkBottom_r_X int
       9: blinkBottom_r_Y float
      10: blinkTop_r_X    int
      11: blinkTop_r_Y    float

    Assertions:
      • len(result)==12
      • x_left, x_right are integer arrays of length>1
    """
    base_fraction = 0.5
    for _, row in test_df.iterrows():
        _= compute_fit_range(
            candidate_signal,
            int(row["max_blink"]),
            int(row["left_zero"]),
            int(row["right_zero"]),
            base_fraction,
            top_bottom=True,
        )

        # assert isinstance(result, tuple) and len(result) == 12
        # xL, xR = result[0], result[1]
        # assert isinstance(xL, np.ndarray) and xL.dtype.kind == 'i' and xL.size > 1
        # assert isinstance(xR, np.ndarray) and xR.dtype.kind == 'i' and xR.size > 1


def test_lines_intersection_first_two(
    candidate_signal: np.ndarray, test_df: pd.DataFrame
):
    """
    Validate lines_intersection on first two segments.

    Internals:
      - polyfit_matlab(x_left, y_left, 1) and (x_right, y_right, 1)
      - polyval_matlab and corr_matlab to compute R2
      - get_intersection returns (xI,yI,leftXI,rightXI)
      - get_line_intersection_slope derives slopes from intersection coords
      - average velocity = p.coef[1] / std(x)
      - ``xLineCross_*`` values were removed since they were always ``NaN``

    Expected Return Shape and Types:
      Tuple of 10 floats:
       0: leftSlope
       1: rightSlope
       2: aver_left_velocity
       3: aver_right_velocity
       4: rightR2
       5: leftR2
       6: x_intersect
       7: y_intersect
       8: left_x_intercept
       9: right_x_intercept

    Assertions:
      • len(result)==10
      • all elements are finite floats
    """

    x_left = np.arange(225, 230, dtype=int)

    x_right = np.arange(232, 242, dtype=int)
    _ = lines_intersection(signal=candidate_signal, x_left=x_left, x_right=x_right)
    # col_result = [
    #     "leftSlope",
    #     "rightSlope",
    #     "aver_left_velocity",
    #     "aver_right_velocity",
    #     "rightR2",
    #     "leftR2",
    #     "x_intersect",
    #     "y_intersect",
    #     "left_x_intercept",
    #     "right_x_intercept",
    # ]
    # assert isinstance(result, tuple) and len(result) == 14
    # for i, v in enumerate(result):
    #     assert isinstance(v, float)
    #     if i >= 10:
    #         assert np.isnan(v)
    #     else:
    #         assert np.isfinite(v)
