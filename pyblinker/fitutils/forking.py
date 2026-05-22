"""Utility functions replicating select MATLAB helpers in Python."""

from __future__ import annotations
from pyblinker._logging import get_logger


import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.stats import pearsonr, spearmanr, kendalltau

logger = get_logger(__name__)


def mad(arr: np.ndarray, axis: int | None = None, keepdims: bool = True) -> np.ndarray:
    """Compute median absolute deviation using the MATLAB approach."""
    median = np.median(arr, axis=axis, keepdims=True)
    mad = np.median(np.abs(arr - median), axis=axis, keepdims=keepdims)[0]
    return mad


def corr(x, y=None, type="Pearson", rows="all", tail="both", weights=None):
    """Compute correlation coefficients mimicking MATLAB's ``corr``.

    Parameters:
        x: array-like, shape (n_samples, n_features1)
        y: array-like, shape (n_samples, n_features2), optional
        type: 'Pearson' (default), 'Kendall', or 'Spearman'
        rows: 'all' (default), 'complete', or 'pairwise'
        tail: 'both' (default), 'right', or 'left'
        weights: array-like of shape (n_samples,), optional

    Returns:
        coef: correlation coefficient(s)
        pval: p-value(s)
    """
    x = np.asarray(x)
    if y is not None:
        y = np.asarray(y)
    else:
        y = x

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_samples = x.shape[0]

    # Handle weights
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape[0] != n_samples:
            raise ValueError(
                "Weights must be of the same length as the number of samples."
            )
    else:
        weights = np.ones(n_samples)

    # Handle 'rows' parameter
    if rows == "all":
        # Use all candidate_signal (including NaNs)
        pass
    elif rows == "complete":
        # Use only rows with no missing values
        valid_mask = ~np.isnan(x).any(axis=1)
        valid_mask = valid_mask & ~np.isnan(y).any(axis=1)
        x = x[valid_mask]
        y = y[valid_mask]
        weights = weights[valid_mask]
    elif rows == "pairwise":
        # For each pair, use rows with no missing values in either column
        pass  # Not implemented in this version
    else:
        raise ValueError("Unknown 'rows' parameter.")

    n_features1 = x.shape[1]
    n_features2 = y.shape[1]
    coef = np.zeros((n_features1, n_features2))
    pval = np.zeros((n_features1, n_features2))

    # Handle 'type' parameter
    if type.lower() == "pearson":
        # Compute Pearson correlation
        for i in range(n_features1):
            for j in range(n_features2):
                xi = x[:, i]
                yj = y[:, j]
                if len(xi) < 2:
                    coef[i, j] = np.nan
                    pval[i, j] = np.nan
                    continue
                if np.all(weights == 1):
                    r, p = pearsonr(xi, yj)
                else:
                    r = weighted_corr(xi, yj, weights)
                    p = np.nan  # p-value not computed with weights
                coef[i, j] = r
                # Adjust p-value according to 'tail' parameter
                if tail == "both":
                    pval[i, j] = p
                elif tail == "right":
                    if r >= 0:
                        pval[i, j] = p / 2
                    else:
                        pval[i, j] = 1 - p / 2
                elif tail == "left":
                    if r <= 0:
                        pval[i, j] = p / 2
                    else:
                        pval[i, j] = 1 - p / 2
                else:
                    raise ValueError("Unknown 'tail' parameter.")
    elif type.lower() == "spearman":
        # Compute Spearman correlation
        for i in range(n_features1):
            for j in range(n_features2):
                xi = x[:, i]
                yj = y[:, j]
                if len(xi) < 2:
                    coef[i, j] = np.nan
                    pval[i, j] = np.nan
                    continue
                r, p = spearmanr(xi, yj)
                coef[i, j] = r
                # Adjust p-value according to 'tail' parameter
                if tail == "both":
                    pval[i, j] = p
                elif tail == "right":
                    if r >= 0:
                        pval[i, j] = p / 2
                    else:
                        pval[i, j] = 1 - p / 2
                elif tail == "left":
                    if r <= 0:
                        pval[i, j] = p / 2
                    else:
                        pval[i, j] = 1 - p / 2
                else:
                    raise ValueError("Unknown 'tail' parameter.")
    elif type.lower() == "kendall":
        # Compute Kendall correlation
        for i in range(n_features1):
            for j in range(n_features2):
                xi = x[:, i]
                yj = y[:, j]
                if len(xi) < 2:
                    coef[i, j] = np.nan
                    pval[i, j] = np.nan
                    continue
                r, p = kendalltau(xi, yj)
                coef[i, j] = r
                # Adjust p-value according to 'tail' parameter
                if tail == "both":
                    pval[i, j] = p
                elif tail == "right":
                    if r >= 0:
                        pval[i, j] = p / 2
                    else:
                        pval[i, j] = 1 - p / 2
                elif tail == "left":
                    if r <= 0:
                        pval[i, j] = p / 2
                    else:
                        pval[i, j] = 1 - p / 2
                else:
                    raise ValueError("Unknown 'tail' parameter.")
    else:
        raise ValueError("Unknown 'type' parameter.")

    return coef, pval


def weighted_corr(x, y, w):
    """Compute weighted Pearson correlation coefficient used by ``corr``."""
    w_sum = np.sum(w)
    w_mean_x = np.sum(w * x) / w_sum
    w_mean_y = np.sum(w * y) / w_sum
    cov_xy = np.sum(w * (x - w_mean_x) * (y - w_mean_y)) / w_sum
    var_x = np.sum(w * (x - w_mean_x) ** 2) / w_sum
    var_y = np.sum(w * (y - w_mean_y) ** 2) / w_sum
    return cov_xy / np.sqrt(var_x * var_y)


def polyfit(x, y, n):
    """Fit a polynomial of degree ``n`` using MATLAB's ``polyfit`` logic.

    Parameters:
    x : array_like, shape (M,)
        x-coordinates of the M sample points (independent variable).
    y : array_like, shape (M,)
        y-coordinates of the sample points (dependent variable).
    n : int
        Degree of the fitting polynomial.

    Returns:
    p : ndarray, shape (n+1,)
        Polynomial coefficients, highest power first.
    S : dict
        A dictionary containing diagnostic information:
        - 'R': Triangular factor from QR decomposition of the Vandermonde matrix.
        - 'df': Degrees of freedom.
        - 'normr': Norm of the residuals.
        - 'rsquared': Coefficient of determination (R-squared).
    mu : ndarray, shape (2,)
        Mean and standard deviation of x for centering and scaling.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    # Center and scale x
    mx = x.mean()
    sx = x.std(ddof=1)
    mu = np.array([mx, sx])
    x_scaled = (x - mu[0]) / mu[1]

    # Construct the Vandermonde matrix
    V = np.vander(x_scaled, n + 1)

    # Solve least squares problem
    p, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
    p = p.flatten()  # Coefficients are already in descending order

    # Compute the QR decomposition
    Q, R = qr(V, mode="economic")

    # Degrees of freedom
    df = max(0, len(y) - (n + 1))

    # Norm of residuals
    if residuals.size > 0:
        normr = np.sqrt(residuals[0])
    else:
        r = y - V @ p
        normr = np.linalg.norm(r)

    # R-squared
    y_mean = y.mean()
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = normr**2
    rsquared = 1 - ss_res / ss_tot

    S = {"R": R, "df": df, "normr": normr, "rsquared": rsquared}

    return p, S, mu


def polyval(p, x, S=None, mu=None):
    """Evaluate a polynomial using MATLAB's ``polyval`` logic.

    Parameters:
    p : array_like
        Polynomial coefficients in descending powers.
    x : array_like
        Points at which to evaluate the polynomial.
    S : dict, optional
        Structure containing optional outputs from polyfit:
        S = {'R': array_like, 'df': int, 'normr': float}
    mu : array_like, optional
        Centering and scaling parameters.

    Returns:
    y : ndarray
        Evaluated polynomial at x.
    delta : ndarray or None
        Prediction error estimates at x.
    """
    logger.debug("Entering polyval")

    # Ensure p is a vector or empty
    if not (np.ndim(p) == 1 or len(p) == 0):
        raise ValueError("Invalid P")

    nc = len(p)

    # Adjust x if mu is provided
    if mu is not None:
        x = (x - mu[0]) / mu[1]

    # Evaluate polynomial using Horner's method
    y = np.zeros_like(x, dtype=np.result_type(x, p))
    if nc > 0:
        y[:] = p[0]
    for i in range(1, nc):
        y = x * y + p[i]

    delta = None
    if S is not None:
        # Extract parameters from S
        if isinstance(S, dict):
            R = np.array(S["R"])
            df = S["df"]
            normr = S["normr"]
        else:
            raise ValueError("S must be a dictionary with keys R, df, normr")

        # Check if R is singular
        if np.linalg.det(R) == 0:
            logger.warning("Singular matrix R. Skipping delta calculation.")
            logger.debug("Exiting polyval")
            return y, None

        # Construct Vandermonde matrix for x
        x_flat = x.flatten()
        V = np.zeros((len(x_flat), nc), dtype=x.dtype)
        V[:, -1] = 1
        for j in range(nc - 2, -1, -1):
            V[:, j] = x_flat * V[:, j + 1]

        # Solve R.T * E.T = V.T
        try:
            E_T = solve_triangular(R.T, V.T, lower=True, check_finite=False)
            E = E_T.T
        except np.linalg.LinAlgError as e:
            logger.exception("Error solving triangular system: %s", e)
            logger.debug("Exiting polyval")
            return y, None

        e = np.sqrt(1 + np.sum(E**2, axis=1))

        if df == 0:
            delta = np.full_like(e, np.inf)
        else:
            delta = normr / np.sqrt(df) * e

        delta = delta.reshape(x.shape)

    logger.debug("Exiting polyval")
    return y, delta


def get_intersection(p, q, u, v):
    """Return the intersection of two lines fitted to segments.

    Parameters
    ----------
    p : array_like
        Coefficients ``[slope, intercept]`` of the first line.
    q : array_like
        Coefficients ``[slope, intercept]`` of the second line.
    u : array_like
        ``[mean, std]`` of the first line's x values used during fitting.
    v : array_like
        ``[mean, std]`` of the second line's x values used during fitting.

    Returns
    -------
    tuple
        ``(x_intersect, y_intersect, x_intercept1, x_intercept2)`` where
        ``x_intercept1`` and ``x_intercept2`` are the x-intercepts of the
        fitted lines.
    """
    if p[0] == 0:
        x_intercept1 = np.nan
    else:
        x_intercept1 = (p[0] * u[0] - p[1] * u[1]) / p[0]

    if q[0] == 0:
        x_intercept2 = np.nan
    else:
        x_intercept2 = (q[0] * v[0] - q[1] * v[1]) / q[0]

    denom = p[0] * v[1] - q[0] * u[1]
    if denom == 0:
        x_intersect = np.nan
        y_intersect = np.nan
    else:
        numer = (
            u[0] * p[0] * v[1]
            - v[0] * q[0] * u[1]
            + q[1] * v[1] * u[1]
            - p[1] * u[1] * v[1]
        )
        x_intersect = numer / denom
        y_intersect = p[0] * (x_intersect - u[0]) / u[1] + p[1]

    return x_intersect, y_intersect, x_intercept1, x_intercept2
