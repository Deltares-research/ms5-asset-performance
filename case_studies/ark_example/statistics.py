"""
Statistical utilities for reliability analysis.

Provides:
- Pf estimation from samples with confidence intervals
- Beta ↔ Pf conversions
- PDF operations (normalization, moments, quantiles)
- MCS quality metrics
"""

from typing import Tuple, Sequence, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats


# -----------------------------------------------------------------------------
# Pf estimation
# -----------------------------------------------------------------------------

def pf_from_samples(
    g: NDArray,
    weights: Optional[NDArray] = None,
) -> float:
    """
    Estimate failure probability from performance function samples.

    Args:
        g: Performance function values. g < 0 indicates failure.
        weights: Sample weights for importance sampling.

    Returns:
        Estimated failure probability.
    """
    g = np.asarray(g)

    if weights is None:
        weights = np.ones(len(g))
    weights = np.asarray(weights)

    pf = np.sum((g < 0) * weights) / np.sum(weights)
    return float(pf)


def pf_confidence_interval(
    g: NDArray,
    weights: Optional[NDArray] = None,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute confidence interval for Pf estimate.

    Uses normal approximation for binomial proportion.

    Args:
        g: Performance function values.
        weights: Sample weights.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    g = np.asarray(g)
    n = len(g)

    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights)

    pf = pf_from_samples(g, weights)

    # Effective sample size for weighted samples
    n_eff = np.sum(weights)**2 / np.sum(weights**2)

    # Standard error (normal approximation)
    se = np.sqrt(pf * (1 - pf) / n_eff)

    # Z-score for confidence level
    z = stats.norm.ppf(1 - alpha / 2)

    lower = max(0, pf - z * se)
    upper = min(1, pf + z * se)

    return lower, upper


def pf_cov(pf: float, n_samples: int) -> float:
    """
    Coefficient of variation of Pf estimate.

    Measures relative precision of MCS estimate.
    Rule of thumb: CoV < 0.1 for reliable estimate.

    Args:
        pf: Estimated failure probability.
        n_samples: Number of samples.

    Returns:
        Coefficient of variation.
    """
    if pf <= 0 or pf >= 1:
        return float('inf')

    # CoV = std(pf) / pf = sqrt((1-pf)/(n*pf))
    cov = np.sqrt((1 - pf) / (n_samples * pf))
    return float(cov)


def required_samples(pf_target: float, cov_target: float = 0.1) -> int:
    """
    Estimate required samples for target Pf and precision.

    Args:
        pf_target: Target failure probability.
        cov_target: Target coefficient of variation.

    Returns:
        Required number of samples.
    """
    if pf_target <= 0 or pf_target >= 1:
        return int(1e9)

    n = (1 - pf_target) / (pf_target * cov_target**2)
    return int(np.ceil(n))


# -----------------------------------------------------------------------------
# Beta ↔ Pf conversions
# -----------------------------------------------------------------------------

def pf_to_beta(pf: float | NDArray) -> float | NDArray:
    """
    Convert failure probability to reliability index.

    β = -Φ⁻¹(Pf)

    Args:
        pf: Failure probability (scalar or array).

    Returns:
        Reliability index.
    """
    pf = np.asarray(pf)
    pf_clipped = np.clip(pf, 1e-15, 1 - 1e-15)
    beta = -stats.norm.ppf(pf_clipped)

    if pf.ndim == 0:
        return float(beta)
    return beta


def beta_to_pf(beta: float | NDArray) -> float | NDArray:
    """
    Convert reliability index to failure probability.

    Pf = Φ(-β)

    Args:
        beta: Reliability index (scalar or array).

    Returns:
        Failure probability.
    """
    beta = np.asarray(beta)
    pf = stats.norm.cdf(-beta)

    if beta.ndim == 0:
        return float(pf)
    return pf


# -----------------------------------------------------------------------------
# PDF operations
# -----------------------------------------------------------------------------

def pdf_normalize(pdf: NDArray, grid: NDArray) -> NDArray:
    """
    Normalize PDF to integrate to 1.

    Args:
        pdf: PDF values.
        grid: Grid points.

    Returns:
        Normalized PDF.
    """
    pdf = np.asarray(pdf)
    grid = np.asarray(grid)

    integral = np.trapezoid(pdf, grid)
    if integral <= 0:
        return pdf

    return pdf / integral


def pdf_mean(pdf: NDArray, grid: NDArray) -> float:
    """
    Compute mean of distribution from PDF.

    Args:
        pdf: PDF values (should be normalized).
        grid: Grid points.

    Returns:
        Mean value.
    """
    pdf = np.asarray(pdf)
    grid = np.asarray(grid)

    return float(np.trapezoid(grid * pdf, grid))


def pdf_variance(pdf: NDArray, grid: NDArray, mean: Optional[float] = None) -> float:
    """
    Compute variance of distribution from PDF.

    Args:
        pdf: PDF values (should be normalized).
        grid: Grid points.
        mean: Pre-computed mean (computed if None).

    Returns:
        Variance.
    """
    pdf = np.asarray(pdf)
    grid = np.asarray(grid)

    if mean is None:
        mean = pdf_mean(pdf, grid)

    return float(np.trapezoid((grid - mean)**2 * pdf, grid))


def pdf_std(pdf: NDArray, grid: NDArray, mean: Optional[float] = None) -> float:
    """
    Compute standard deviation of distribution from PDF.

    Args:
        pdf: PDF values (should be normalized).
        grid: Grid points.
        mean: Pre-computed mean (computed if None).

    Returns:
        Standard deviation.
    """
    return float(np.sqrt(pdf_variance(pdf, grid, mean)))


def pdf_moments(pdf: NDArray, grid: NDArray) -> dict:
    """
    Compute moments of distribution from PDF.

    Args:
        pdf: PDF values (should be normalized).
        grid: Grid points.

    Returns:
        Dict with mean, std, variance, skewness, kurtosis.
    """
    pdf = np.asarray(pdf)
    grid = np.asarray(grid)

    mean = pdf_mean(pdf, grid)
    var = pdf_variance(pdf, grid, mean)
    std = np.sqrt(var)

    # Skewness: E[(X-μ)³] / σ³
    if std > 0:
        skewness = float(np.trapezoid((grid - mean)**3 * pdf, grid) / std**3)
        kurtosis = float(np.trapezoid((grid - mean)**4 * pdf, grid) / std**4)
    else:
        skewness = 0.0
        kurtosis = 0.0

    return {
        "mean": mean,
        "std": std,
        "variance": var,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def pdf_cdf(pdf: NDArray, grid: NDArray) -> NDArray:
    """
    Compute CDF from PDF via cumulative integration.

    Args:
        pdf: PDF values.
        grid: Grid points.

    Returns:
        CDF values at grid points.
    """
    pdf = np.asarray(pdf)
    grid = np.asarray(grid)

    # Trapezoidal cumulative integration
    dx = np.diff(grid)
    pdf_mid = (pdf[:-1] + pdf[1:]) / 2
    cdf = np.zeros(len(grid))
    cdf[1:] = np.cumsum(pdf_mid * dx)

    # Normalize
    if cdf[-1] > 0:
        cdf /= cdf[-1]

    return cdf


def pdf_quantile(
    pdf: NDArray,
    grid: NDArray,
    q: float | Sequence[float],
) -> float | NDArray:
    """
    Compute quantile(s) from PDF.

    Args:
        pdf: PDF values.
        grid: Grid points.
        q: Quantile(s) to compute (0-1).

    Returns:
        Quantile value(s).
    """
    pdf = np.asarray(pdf)
    grid = np.asarray(grid)
    q = np.asarray(q)

    cdf = pdf_cdf(pdf, grid)

    quantiles = np.interp(q, cdf, grid)

    if q.ndim == 0:
        return float(quantiles)
    return quantiles


def pdf_credible_interval(
    pdf: NDArray,
    grid: NDArray,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute credible interval from PDF.

    Args:
        pdf: PDF values.
        grid: Grid points.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    lower = pdf_quantile(pdf, grid, alpha / 2)
    upper = pdf_quantile(pdf, grid, 1 - alpha / 2)
    return lower, upper


# -----------------------------------------------------------------------------
# Integration utilities
# -----------------------------------------------------------------------------

def integrate_product(
    f1: NDArray,
    f2: NDArray,
    grid: NDArray,
) -> float:
    """
    Integrate product of two functions over grid.

    Args:
        f1: First function values.
        f2: Second function values.
        grid: Grid points.

    Returns:
        Integral of f1 * f2.
    """
    return float(np.trapezoid(f1 * f2, grid))


def interpolate_to_grid(
    values: NDArray,
    old_grid: NDArray,
    new_grid: NDArray,
) -> NDArray:
    """
    Interpolate values to new grid.

    Args:
        values: Values on old grid.
        old_grid: Original grid points.
        new_grid: Target grid points.

    Returns:
        Interpolated values.
    """
    return np.interp(new_grid, old_grid, values, left=0, right=0)
