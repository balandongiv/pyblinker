import logging
import warnings

import numpy as np
import pandas as pd

from .zero_crossing import (
    get_half_height,
    compute_fit_range,
    left_right_zero_crossing,
)
from .base_left_right import create_left_right_base
from ..fitutils.line_intersection import lines_intersection


logger = logging.getLogger(__name__)


class FitBlinks:

    def __init__(self, candidate_signal=None, df=None, params=None):
        # candidateSignal    IC or channel time course of blinks to be fitted
        self.candidate_signal = candidate_signal
        self.df = df
        self.frame_blinks = []
        self.base_fraction = params["base_fraction"]

        # Column lists produced by helper functions
        self.cols_half_height = [
            "left_zero_half_height",
            "right_zero_half_height",
            "left_base_half_height",
            "right_base_half_height",
        ]
        self.cols_fit_range = [
            "x_left",
            "x_right",
            "left_range",
            "right_range",
            "blink_bottom_point_l_y",
            "blink_bottom_point_l_x",
            "blink_top_point_l_y",
            "blink_top_point_l_x",
            "blink_bottom_point_r_x",
            "blink_bottom_point_r_y",
            "blink_top_point_r_x",
            "blink_top_point_r_y",
        ]
        self.cols_lines_intesection = [
            "left_slope",
            "right_slope",
            "aver_left_velocity",
            "aver_right_velocity",
            "rightR2",
            "leftR2",
            "x_intersect",
            "y_intersect",
            "left_x_intercept",
            "right_x_intercept",
        ]

    def get_max_blink(self, start_idx, end_idx):
        """Return the maximum value and its index within ``start_idx`` and ``end_idx``.

        Parameters
        ----------
        start_idx : int or float
            Starting sample index of the blink window.
        end_idx : int or float
            Ending sample index of the blink window.

        Returns
        -------
        tuple[float, int]
            ``(max_value, frame_index_at_max)`` within the specified range.
        """

        start_idx = int(start_idx)
        end_idx = int(end_idx)

        blink_range = np.arange(start_idx, end_idx + 1, dtype=int)
        blink_frame = self.candidate_signal[start_idx : end_idx + 1]
        # One-pass for max value and index
        max_idx = np.argmax(blink_frame)
        max_val = blink_frame[max_idx]
        max_fr = blink_range[max_idx]
        return max_val, max_fr

    def dprocess_segment_raw(self, *, run_fit: bool = False) -> None:
        """Process blink metadata for a raw segment.

        If the DataFrame lacks essential columns (``outer_start``, ``outer_end``,
        ``left_zero`` or ``right_zero``) the method falls back to
        :meth:`dprocess`, mirroring the original Matlab implementation where
        only blink start and end samples were provided.

        Parameters
        ----------
        run_fit : bool, optional
            If ``True`` the :meth:`fit` routine is executed. This step may
            introduce ``NaN`` values in the resulting frame due to aggressive
            range estimation. The default is ``False`` which skips the fitting
            stage.
        """

        required_cols = {"outer_start", "outer_end", "left_zero", "right_zero"}
        if not required_cols.issubset(self.df.columns):
            self.dprocess(run_fit=run_fit)
            return

        # Compute the maximum value within each blink interval
        self.df[["max_value", "max_blink_alternative"]] = self.df.apply(
            lambda row: self.get_max_blink(row["start_blink"], row["end_blink"]),
            axis=1,
            result_type="expand",
        )

        # Compute baseline information required by downstream features
        self.frame_blinks = create_left_right_base(self.candidate_signal, self.df)

        if run_fit:
            msg = "Running fit() may drop blinks due to NaNs in fit range"
            logger.warning(msg)
            if logger.isEnabledFor(logging.DEBUG):
                warnings.warn(msg, RuntimeWarning)
            self.fit()

    def dprocess(self, *, run_fit: bool = True) -> None:
        """Compute blink boundaries and optional fits.

        This routine reproduces the legacy Matlab approach where blink fits were
        always computed using only ``start_blink`` and ``end_blink`` indices.

        Parameters
        ----------
        run_fit : bool, optional
            If ``True`` also execute :meth:`fit`. Defaults to ``True``.
        """

        data_size = (
            self.candidate_signal.size
        )  # store locally to avoid repeated lookups

        # Find the max_frame index and max_value at that max_frame index
        self.df[["max_value", "max_blink"]] = self.df.apply(
            lambda row: self.get_max_blink(row["start_blink"], row["end_blink"]),
            axis=1,
            result_type="expand",
        )
        # Ensure the max_blink is integer
        self.df["max_blink"] = self.df["max_blink"].astype(int)

        # Shifts for outer start/end
        self.df["outer_start"] = self.df["max_blink"].shift(1, fill_value=0)
        self.df["outer_end"] = self.df["max_blink"].shift(-1, fill_value=data_size)

        # Add columns for leftZero/rightZero
        self.df[["left_zero", "right_zero"]] = self.df.apply(
            lambda row: left_right_zero_crossing(
                self.candidate_signal,
                row["max_blink"],
                row["outer_start"],
                row["outer_end"],
            ),
            axis=1,
            result_type="expand",
        )

        # Perform fitting calculations
        if run_fit:
            self.fit()

    def fit(self):
        """Run baseline fitting and associated calculations for each blink.

        Main method to create base line fits, compute half-height, fit ranges,
        and line intersections.

        If no valid blink segments remain after baseline creation or filtering,
        the method returns an empty ``DataFrame`` with all expected columns. This
        prevents downstream ``Columns must be same length as key`` errors.
        """
        # candidate_signal = self.candidate_signal  # Local reference for efficiency

        # Create left and right base lines
        self.frame_blinks = create_left_right_base(self.candidate_signal, self.df)

        # Baseline generation may drop every potential blink. Provide an empty
        # DataFrame with the correct schema so later operations don't fail.
        if self.frame_blinks.empty:
            # No valid blink regions after baseline calculation
            all_cols = (
                list(self.df.columns)
                + ["left_base", "right_base"]
                + self.cols_half_height
                + self.cols_fit_range
                + ["nsize_x_left", "nsize_x_right"]
                + self.cols_lines_intesection
            )
            self.frame_blinks = pd.DataFrame(columns=all_cols)
            return

        # Get half height
        self.frame_blinks[self.cols_half_height] = self.frame_blinks.apply(
            lambda row: get_half_height(
                self.candidate_signal,
                row["max_blink"],
                row["left_zero"],
                row["right_zero"],
                row["left_base"],
                row["outer_end"],
            ),
            axis=1,
            result_type="expand",
        )

        # Compute fit ranges
        self.frame_blinks[self.cols_fit_range] = self.frame_blinks.apply(
            lambda row: compute_fit_range(
                self.candidate_signal,
                row["max_blink"],
                row["left_zero"],
                row["right_zero"],
                self.base_fraction,
                top_bottom=True,
            ),
            axis=1,
            result_type="expand",
        )

        # Drop rows with NaN values
        self.frame_blinks.dropna(inplace=True)
        self.frame_blinks["nsize_x_left"] = self.frame_blinks["x_left"].apply(len)
        self.frame_blinks["nsize_x_right"] = self.frame_blinks["x_right"].apply(len)

        # Keep only rows with nsize_x_left > 1 and nsize_x_right > 1
        self.frame_blinks = self.frame_blinks[
            (self.frame_blinks["nsize_x_left"] > 1)
            & (self.frame_blinks["nsize_x_right"] > 1)
        ].reset_index(drop=True)

        # Filtering for valid fit ranges may remove all rows. Return an empty
        # DataFrame with the expected schema so subsequent code does not raise
        # a length mismatch error.
        if self.frame_blinks.empty:
            all_cols = (
                list(self.df.columns)
                + ["left_base", "right_base"]
                + self.cols_half_height
                + self.cols_fit_range
                + ["nsize_x_left", "nsize_x_right"]
                + self.cols_lines_intesection
            )
            self.frame_blinks = pd.DataFrame(columns=all_cols)
            return

        # Calculate line intersections

        self.frame_blinks[self.cols_lines_intesection] = self.frame_blinks.apply(
            lambda row: lines_intersection(
                signal=self.candidate_signal,
                x_right=row["x_right"],
                x_left=row["x_left"],
            ),
            axis=1,
            result_type="expand",
        )
