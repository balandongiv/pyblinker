from pyblinker._logging import get_logger

import numpy as np
import pandas as pd

from pyblinker.segmentation.geometry import (
    compute_fit_range,
    create_left_right_base,
    get_half_height,
    get_max_blink,
    left_right_zero_crossing,
    lines_intersection,
)


logger = get_logger(__name__)


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

    def _empty_frame_blinks(self) -> pd.DataFrame:
        """Return an empty blink-fit frame with the downstream-required schema."""

        base_cols = list(self.df.columns) if self.df is not None else []
        derived_cols = [
            "max_value",
            "max_blink",
            "outer_start",
            "outer_end",
            "left_zero",
            "right_zero",
            "max_pos_vel_frame",
            "max_neg_vel_frame",
            "left_base",
            "right_base",
            *self.cols_half_height,
            *self.cols_fit_range,
            "nsize_x_left",
            "nsize_x_right",
            *self.cols_lines_intesection,
        ]
        all_cols = list(dict.fromkeys(base_cols + derived_cols))
        return pd.DataFrame(columns=all_cols)

    def dprocess_segment_raw(self, *, run_fit: bool = False) -> None:
        """Process blink metadata for a raw segment.

        If the DataFrame lacks essential columns (``outer_start``, ``outer_end``,
        ``left_zero`` or ``right_zero``) the method falls back to
        :meth:`dprocess`, mirroring the original Matlab implementation where
        only blink start and end samples were provided.

        But if those columns are present, which is the case in upgrade versions, where
        those columns are already computed, then this can skip self.dprocess().
        Parameters
        ----------
        run_fit : bool, optional
            If ``True`` the :meth:`fit` routine is executed. This step may
            introduce ``NaN`` values in the resulting frame due to aggressive
            range estimation. The default is ``False`` which skips the fitting
            stage.
        """
        if self.df is None or self.df.empty:
            self.frame_blinks = self._empty_frame_blinks()
            return

        required_cols = {"outer_start", "outer_end", "left_zero", "right_zero"}
        if not required_cols.issubset(self.df.columns):
            self.dprocess(run_fit=run_fit)
            # return

        # Compute the maximum value within each blink interval
        # self.df[["max_value", "max_blink"]] = self.df.apply(
        #     lambda row: get_max_blink(
        #         self.candidate_signal, row["start_blink"], row["end_blink"]
        #     ),
        #     axis=1,
        #     result_type="expand",
        # )
        if not {"max_value", "max_blink"}.issubset(self.df.columns):
            # Compute the maximum value within each blink interval
            self.df[["max_value", "max_blink"]] = self.df.apply(
                lambda row: get_max_blink(
                    self.candidate_signal, row["start_blink"], row["end_blink"]
                ),
                axis=1,
                result_type="expand",
            )

        if run_fit:
            logger.warning(
                "Running fit() may drop blinks due to NaNs in fit range",
                extra={"run_fit": run_fit},
            )
            self.fit()
        else:
            # Compute baseline information required by downstream features
            self.frame_blinks = create_left_right_base(self.candidate_signal, self.df)

    def dprocess(self, *, run_fit: bool = True) -> None:
        """Compute blink boundaries and optional fits.

        This routine reproduces the legacy Matlab approach where blink fits were
        always computed using only ``start_blink`` and ``end_blink`` indices.

        Parameters
        ----------
        run_fit : bool, optional
            If ``True`` also execute :meth:`fit`. Defaults to ``True``.
        """
        if self.df is None or self.df.empty:
            self.frame_blinks = self._empty_frame_blinks()
            return

        data_size = (
            self.candidate_signal.size
        )  # store locally to avoid repeated lookups

        # Find the max_frame index and max_value at that max_frame index
        self.df[["max_value", "max_blink"]] = self.df.apply(
            lambda row: get_max_blink(
                self.candidate_signal, row["start_blink"], row["end_blink"]
            ),
            axis=1,
            result_type="expand",
        )
        # Ensure the max_blink is integer
        self.df["max_blink"] = self.df["max_blink"].astype(int)

        # Shifts for outer start/end
        self.df["outer_start"] = self.df["max_blink"].shift(1, fill_value=0)
        self.df["outer_end"] = self.df["max_blink"].shift(-1, fill_value=data_size - 1)

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

        # MATLAB leaves the downstream fit fields unset when a blink cannot
        # produce both velocity extrema and base landmarks (for example when
        # the trailing zero-crossing lands on the final sample). Mirror that
        # behavior by skipping the remaining fit stages for those rows.
        fit_ready_mask = self.frame_blinks[
            ["max_pos_vel_frame", "max_neg_vel_frame"]
        ].notna().all(axis=1)

        def _safe_half_height(row):
            required = [
                row.get("max_blink"),
                row.get("left_zero"),
                row.get("right_zero"),
                row.get("left_base"),
                row.get("outer_end"),
            ]
            if any(pd.isna(value) for value in required):
                return (np.nan, np.nan, np.nan, np.nan)
            try:
                return get_half_height(
                    self.candidate_signal,
                    row["max_blink"],
                    row["left_zero"],
                    row["right_zero"],
                    row["left_base"],
                    row["outer_end"],
                )
            except (ValueError, IndexError, TypeError):
                return (np.nan, np.nan, np.nan, np.nan)

        def _safe_fit_range(row):
            required = [
                row.get("max_blink"),
                row.get("left_zero"),
                row.get("right_zero"),
            ]
            if any(pd.isna(value) for value in required):
                return tuple(np.nan for _ in self.cols_fit_range)
            try:
                return compute_fit_range(
                    self.candidate_signal,
                    row["max_blink"],
                    row["left_zero"],
                    row["right_zero"],
                    self.base_fraction,
                    top_bottom=True,
                )
            except (ValueError, IndexError, TypeError):
                return tuple(np.nan for _ in self.cols_fit_range)

        # Get half height only for rows that remain valid after baseline setup.
        half_height_values = pd.DataFrame(
            np.nan,
            index=self.frame_blinks.index,
            columns=self.cols_half_height,
        )
        if fit_ready_mask.any():
            computed_half_height = self.frame_blinks.loc[
                fit_ready_mask
            ].apply(
                _safe_half_height,
                axis=1,
                result_type="expand",
            )
            computed_half_height.columns = self.cols_half_height
            half_height_values.update(computed_half_height)
        self.frame_blinks[self.cols_half_height] = half_height_values

        # Compute fit ranges only for rows that remain valid after baseline setup.
        fit_range_values = pd.DataFrame(
            index=self.frame_blinks.index,
            columns=self.cols_fit_range,
            dtype=object,
        )
        fit_range_values.loc[:, :] = np.nan
        if fit_ready_mask.any():
            computed_fit_ranges = self.frame_blinks.loc[
                fit_ready_mask
            ].apply(
                _safe_fit_range,
                axis=1,
                result_type="expand",
            )
            computed_fit_ranges.columns = self.cols_fit_range
            for col in self.cols_fit_range:
                fit_range_values.loc[fit_ready_mask, col] = computed_fit_ranges[col]
        self.frame_blinks[self.cols_fit_range] = fit_range_values

        def _range_size(value):
            if isinstance(value, (list, np.ndarray)):
                return len(value)
            return 0

        self.frame_blinks["nsize_x_left"] = self.frame_blinks["x_left"].apply(
            _range_size
        )
        self.frame_blinks["nsize_x_right"] = self.frame_blinks["x_right"].apply(
            _range_size
        )

        # Calculate line intersections only for valid ranges
        line_cols = self.cols_lines_intesection
        line_values = pd.DataFrame(
            np.nan, index=self.frame_blinks.index, columns=line_cols
        )
        valid_mask = (self.frame_blinks["nsize_x_left"] > 1) & (
            self.frame_blinks["nsize_x_right"] > 1
        )
        if valid_mask.any():
            line_values.loc[valid_mask] = self.frame_blinks.loc[valid_mask].apply(
                lambda row: lines_intersection(
                    signal=self.candidate_signal,
                    x_right=row["x_right"],
                    x_left=row["x_left"],
                ),
                axis=1,
                result_type="expand",
            )

        self.frame_blinks[line_cols] = line_values
