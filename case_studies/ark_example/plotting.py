"""
Plotting utilities for the D-Sheet piling case study.
"""

from pathlib import Path
from typing import Sequence, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray


# -----------------------------------------------------------------------------
# Surrogate model diagnostics
# -----------------------------------------------------------------------------

def plot_predictions(
    y_true: NDArray,
    y_pred: NDArray,
    title: str = "Surrogate predictions",
    xlabel: str = "Observed",
    ylabel: str = "Predicted",
) -> plt.Figure:
    """
    Scatter plot of predicted vs observed values.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, marker="x", alpha=0.5)

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "k-", label="1:1 line")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


def plot_loss_history(
    losses: NDArray,
    title: str = "Training loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot training loss over epochs.

    Args:
        losses: Array of loss values per epoch.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        log_scale: Use log scale for y-axis.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(losses) + 1), losses)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    return fig


# -----------------------------------------------------------------------------
# Reliability timeline
# -----------------------------------------------------------------------------

def plot_reliability_index(
    times: Sequence[float],
    beta: Sequence[float],
    beta_target: float | None = None,
    title: str = "Reliability index over time",
    xlabel: str = "Time [years]",
    ylabel: str = r"$\beta$ [-]",
) -> plt.Figure:
    """
    Plot reliability index over time.

    Args:
        times: Time points.
        beta: Reliability index values.
        beta_target: Target reliability (horizontal line).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots()
    ax.plot(times, beta, "b-o", markersize=4, label=r"$\beta$")

    if beta_target is not None:
        ax.axhline(beta_target, color="r", linestyle="--", label=f"Target = {beta_target}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


def plot_prior_posterior_comparison(
    times: Sequence[float],
    beta_prior: Sequence[float],
    beta_posterior: Sequence[float],
    beta_target: float | None = None,
    title: str = "Prior vs posterior reliability",
    xlabel: str = "Time [years]",
    ylabel: str = r"$\beta$ [-]",
) -> plt.Figure:
    """
    Compare prior and posterior reliability index.

    Args:
        times: Time points.
        beta_prior: Prior reliability index.
        beta_posterior: Posterior reliability index (after updating).
        beta_target: Target reliability (horizontal line).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots()
    ax.plot(times, beta_prior, "b--", label="Prior")
    ax.plot(times, beta_posterior, "g-", label="Posterior")

    if beta_target is not None:
        ax.axhline(beta_target, color="r", linestyle=":", label=f"Target = {beta_target}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


# -----------------------------------------------------------------------------
# Distributions
# -----------------------------------------------------------------------------

def plot_pdf(
    x: NDArray,
    pdf: NDArray,
    title: str = "Probability density",
    xlabel: str = "x",
    ylabel: str = "Density",
    fill: bool = True,
) -> plt.Figure:
    """
    Plot probability density function.

    Args:
        x: Grid values.
        pdf: PDF values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        fill: Fill under curve.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots()

    if fill:
        ax.fill_between(x, pdf, alpha=0.3)
    ax.plot(x, pdf)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


def plot_corrosion_over_time(
    times: Sequence[float],
    corrosion: Sequence[float],
    corrosion_obs: Sequence[float] | None = None,
    obs_times: Sequence[float] | None = None,
    title: str = "Corrosion over time",
    xlabel: str = "Time [years]",
    ylabel: str = "Corrosion [mm]",
) -> plt.Figure:
    """
    Plot corrosion progression with optional observations.

    Args:
        times: Time points for model.
        corrosion: Modeled corrosion values.
        corrosion_obs: Observed corrosion values.
        obs_times: Times of observations.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots()
    ax.plot(times, corrosion, "b-", label="Model")

    if corrosion_obs is not None and obs_times is not None:
        ax.scatter(obs_times, corrosion_obs, c="r", marker="o", s=50,
                   zorder=5, label="Observations")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


# -----------------------------------------------------------------------------
# Saving utilities
# -----------------------------------------------------------------------------

def save_figure(fig: plt.Figure, filepath: Path | str, dpi: int = 150) -> None:
    """
    Save figure to file.

    Args:
        fig: Matplotlib figure.
        filepath: Output file path.
        dpi: Resolution for raster formats.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_figures_to_pdf(figs: Sequence[plt.Figure], filepath: Path | str) -> None:
    """
    Save multiple figures to a single PDF.

    Args:
        figs: List of matplotlib figures.
        filepath: Output PDF path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filepath) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)


def collect_pngs_to_pdf(png_dir: Path | str, pdf_path: Path | str, dpi: int = 150) -> None:
    """
    Collect all PNGs in a directory into a single PDF.

    Reads existing PNG files (sorted alphabetically), renders each as a
    full-page image in a PDF.  This avoids regenerating figures just to
    create a PDF companion.

    Args:
        png_dir: Directory containing PNG files.
        pdf_path: Output PDF path.
        dpi: Resolution for the PDF pages.
    """
    png_dir = Path(png_dir)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    png_files = sorted(png_dir.glob("*.png"))
    if not png_files:
        return

    with PdfPages(pdf_path) as pdf:
        for png_file in png_files:
            img = plt.imread(str(png_file))
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)


# -----------------------------------------------------------------------------
# Prior vs Posterior comparison plots (replicating timos branch style)
# -----------------------------------------------------------------------------

def plot_beta_prior_posterior(
    times: Sequence[float],
    beta_prior: Sequence[float],
    beta_posterior: Sequence[float],
    beta_req: float = 2.3,
    max_beta: float = 6.0,
    title: str = "",
    xlim: tuple = (50, 80),
    ylim: tuple = (0.5, 6),
) -> plt.Figure:
    """
    Plot reliability index (beta) prior vs posterior over time.

    Replicates the style from visualize_timeline_prior_posterior.py.

    Args:
        times: Time points [years].
        beta_prior: Prior reliability index values.
        beta_posterior: Posterior reliability index values.
        beta_req: Required reliability index (horizontal line).
        max_beta: Cap for infinite beta values.
        title: Plot title (e.g., "Time=50").
        xlim: X-axis limits.
        ylim: Y-axis limits.

    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=(8, 4))

    # Cap infinite values
    beta_prior_capped = [min(b, max_beta) for b in beta_prior]
    beta_posterior_capped = [min(b, max_beta) for b in beta_posterior]

    plt.plot(times, beta_prior_capped, c="b", label="Prior")
    plt.plot(times, beta_posterior_capped, c="r", label="Posterior")
    plt.axhline(beta_req, c="k", linestyle="--", label=f"Requirement ({beta_req})")

    plt.xlabel("Forecast time [yr]", fontsize=12)
    plt.ylabel(r"$\beta$ [-]", fontsize=12)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    if title:
        plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.close()

    return fig


def plot_corrosion_prior_posterior(
    times_forecast: Sequence[float],
    corrosion_mean_prior: Sequence[float],
    corrosion_q05_prior: Sequence[float],
    corrosion_q95_prior: Sequence[float],
    corrosion_mean_posterior: Sequence[float],
    corrosion_q05_posterior: Sequence[float],
    corrosion_q95_posterior: Sequence[float],
    obs_times: Sequence[float] = None,
    obs_values: Sequence[float] = None,
    obs_error_std: float = 0.4,
    alpha: float = 0.05,
    title: str = "",
    xlim: tuple = (50, 80),
    ylim: tuple = (0, 9.5),
) -> plt.Figure:
    """
    Plot corrosion depth progression (prior vs posterior) with observations.

    Args:
        times_forecast: Forecast time points.
        corrosion_mean_prior: Prior mean corrosion.
        corrosion_q05_prior: Prior 5th percentile.
        corrosion_q95_prior: Prior 95th percentile.
        corrosion_mean_posterior: Posterior mean corrosion.
        corrosion_q05_posterior: Posterior 5th percentile.
        corrosion_q95_posterior: Posterior 95th percentile.
        obs_times: Times of observations.
        obs_values: Observed corrosion values.
        obs_error_std: Observation error standard deviation.
        alpha: Confidence level for error bars.
        title: Plot title.
        xlim: X-axis limits.
        ylim: Y-axis limits.

    Returns:
        Matplotlib figure.
    """
    from scipy.stats import norm

    fig = plt.figure(figsize=(8, 4))

    # Prior (blue)
    plt.fill_between(times_forecast, corrosion_q05_prior, corrosion_q95_prior,
                     color="b", alpha=0.3)
    plt.plot(times_forecast, corrosion_q05_prior, color="b", linewidth=0.5)
    plt.plot(times_forecast, corrosion_q95_prior, color="b", linewidth=0.5)
    plt.plot(times_forecast, corrosion_mean_prior, color="b", label="Prior")

    # Posterior (red)
    plt.fill_between(times_forecast, corrosion_q05_posterior, corrosion_q95_posterior,
                     color="r", alpha=0.3)
    plt.plot(times_forecast, corrosion_q05_posterior, color="r", linewidth=0.5)
    plt.plot(times_forecast, corrosion_q95_posterior, color="r", linewidth=0.5)
    plt.plot(times_forecast, corrosion_mean_posterior, color="r", label="Posterior")

    # Observations
    if obs_times is not None and obs_values is not None:
        yerr = obs_error_std * norm.ppf(1 - alpha)
        plt.errorbar(x=obs_times, y=obs_values, yerr=yerr, fmt='o', c="k",
                     capsize=3, label="Observations")

    plt.xlabel("Forecast time [yr]", fontsize=12)
    plt.ylabel("Corrosion [mm]", fontsize=12)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    if title:
        plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.close()

    return fig


def plot_corrosion_forecast_at_time(
    cr_grid: Sequence[float],
    cr_forecast_prior: Dict[float, Sequence[float]],
    cr_forecast_posterior: Dict[float, Sequence[float]],
    obs_times: Sequence[float] = None,
    obs_values: Sequence[float] = None,
    obs_error_std: float = 0.4,
    alpha: float = 0.05,
    start_thickness: float = 9.5,
    xlim: tuple = (50, 80),
    ylim: tuple = (0, 9.5),
) -> plt.Figure:
    """
    Plot corrosion forecast at a specific observation time.

    Shows:
    - Prior corrosion band (full time range)
    - Observations up to current_time
    - Posterior forecast from current_time onwards

    Args:
        current_time: Current observation time.
        obs_times: All observation times.
        obs_values: All observed corrosion values.
        obs_error_std: Observation error standard deviation.
        alpha: Confidence level for error bars.
        xlim: X-axis limits.
        ylim: Y-axis limits.

    Returns:
        Matplotlib figure.
    """
    from scipy.stats import norm
    from scipy.integrate import cumulative_trapezoid

    fig, ax = plt.subplots(figsize=(10, 5))

    cr_forecasts_prior = np.vstack(list(cr_forecast_prior.values()))
    cr_prior_mean = np.trapezoid(cr_forecasts_prior*cr_grid[None, :], cr_grid, axis=1)
    cr_prior_cdf = cumulative_trapezoid(cr_forecasts_prior, cr_grid)
    cr_prior_q05 = cr_grid[np.argmin(np.abs(cr_prior_cdf-alpha), axis=1)]
    cr_prior_q95 = cr_grid[np.argmin(np.abs(cr_prior_cdf-(1-alpha)), axis=1)]
    
    corrosion_prior_mean = cr_prior_mean * start_thickness
    corrosion_prior_q05 = cr_prior_q05 * start_thickness
    corrosion_prior_q95 = cr_prior_q95 * start_thickness
    
    cr_forecasts_posterior = np.vstack(list(cr_forecast_posterior.values()))
    cr_posterior_mean = np.trapezoid(cr_forecasts_posterior * cr_grid[None, :], cr_grid, axis=1)
    cr_posterior_cdf = cumulative_trapezoid(cr_forecasts_posterior, cr_grid)
    cr_posterior_q05 = cr_grid[np.argmin(np.abs(cr_posterior_cdf - alpha), axis=1)]
    cr_posterior_q95 = cr_grid[np.argmin(np.abs(cr_posterior_cdf - (1 - alpha)), axis=1)]

    corrosion_posterior_mean = cr_posterior_mean * start_thickness
    corrosion_posterior_q05 = cr_posterior_q05 * start_thickness
    corrosion_posterior_q95 = cr_posterior_q95 * start_thickness
    
    # Prior band
    times_forecast = list(list(cr_forecast_prior.keys()))
    times_forecast = np.asarray(times_forecast)
    ax.fill_between(times_forecast, corrosion_prior_q05, corrosion_prior_q95,color="b", alpha=0.15)
    ax.plot(times_forecast, corrosion_prior_q05, color="b", linewidth=0.5)
    ax.plot(times_forecast, corrosion_prior_q95, color="b", linewidth=0.5)
    ax.plot(times_forecast, corrosion_prior_mean, color="b", linewidth=1.5, label="Prior")

    # Posterior band
    times_forecast = list(list(cr_forecast_posterior.keys()))
    times_forecast = np.asarray(times_forecast)
    ax.fill_between(times_forecast, corrosion_posterior_q05, corrosion_posterior_q95,color="r", alpha=0.15)
    ax.plot(times_forecast, corrosion_posterior_q05, color="r", linewidth=0.5)
    ax.plot(times_forecast, corrosion_posterior_q95, color="r", linewidth=0.5)
    ax.plot(times_forecast, corrosion_posterior_mean, color="r", linewidth=1.5, label="Posterior")

    # Observations up to current_time
    if obs_times is not None and obs_values is not None:
        yerr = obs_error_std * norm.ppf(1 - alpha)
        ax.errorbar(x=obs_times, y=obs_values, yerr=yerr, fmt="o", c="k", capsize=3, zorder=5, label="Observations")

    ax.set_xlabel("Forecast time [yr]", fontsize=12)
    ax.set_ylabel("Corrosion [mm]", fontsize=12)
    ax.set_title(f"Corrosion Forecast at t_obs = {times_forecast.min():.0f}", fontsize=13, fontweight="bold")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()

    return fig


def plot_moment_capacity_prior_posterior(
    times_forecast: Sequence[float],
    moment_mean_prior: Sequence[float],
    moment_q05_prior: Sequence[float],
    moment_q95_prior: Sequence[float],
    moment_mean_posterior: Sequence[float],
    moment_q05_posterior: Sequence[float],
    moment_q95_posterior: Sequence[float],
    moment_survived: float = None,
    obs_times: Sequence[float] = None,
    obs_moment_cap: Sequence[float] = None,
    title: str = "",
    xlim: tuple = (50, 80),
    ylim: tuple = (100, 800),
) -> plt.Figure:
    """
    Plot moment capacity degradation (prior vs posterior).

    Args:
        times_forecast: Forecast time points.
        moment_mean_prior: Prior mean moment capacity.
        moment_q05_prior: Prior 5th percentile.
        moment_q95_prior: Prior 95th percentile.
        moment_mean_posterior: Posterior mean moment capacity.
        moment_q05_posterior: Posterior 5th percentile.
        moment_q95_posterior: Posterior 95th percentile.
        moment_survived: Survived moment threshold (horizontal line).
        obs_times: Times of observations.
        obs_moment_cap: Observed moment capacity values.
        title: Plot title.
        xlim: X-axis limits.
        ylim: Y-axis limits.

    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=(8, 4))

    # Prior (blue)
    plt.fill_between(times_forecast, moment_q05_prior, moment_q95_prior,
                     color="b", alpha=0.3)
    plt.plot(times_forecast, moment_q05_prior, color="b", linewidth=0.5)
    plt.plot(times_forecast, moment_q95_prior, color="b", linewidth=0.5)
    plt.plot(times_forecast, moment_mean_prior, color="b", label="Prior")

    # Posterior (red)
    plt.fill_between(times_forecast, moment_q05_posterior, moment_q95_posterior,
                     color="r", alpha=0.3)
    plt.plot(times_forecast, moment_q05_posterior, color="r", linewidth=0.5)
    plt.plot(times_forecast, moment_q95_posterior, color="r", linewidth=0.5)
    plt.plot(times_forecast, moment_mean_posterior, color="r", label="Posterior")

    # Observations
    if obs_times is not None and obs_moment_cap is not None:
        plt.scatter(x=obs_times, y=obs_moment_cap, color="k", s=50, zorder=5,
                    label="Observations")

    # Survived moment
    if moment_survived is not None:
        plt.axhline(moment_survived, c="g", linestyle="-", label="Survived moment")

    plt.xlabel("Forecast time [yr]", fontsize=12)
    plt.ylabel("Moment capacity [kNm]", fontsize=12)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    if title:
        plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.close()

    return fig


def plot_moment_forecast_at_time(
    cr_grid: Sequence[float],
    cr_forecast_prior: Dict[float, Sequence[float]],
    cr_forecast_posterior: Dict[float, Sequence[float]],
    moment_cap: float = 750.0,
    survived_times: Sequence[float] = None,
    survived_moments: Sequence[float] = None,
    obs_times: Sequence[float] = None,
    obs_moment_cap: Sequence[float] = None,
    alpha: float = 0.05,
    xlim: tuple = (50, 80),
    ylim: tuple = (0, 800),
) -> plt.Figure:
    """
    Plot moment capacity forecast at a specific observation time.

    Derives moment capacity from corrosion ratio PDFs:
    moment = moment_cap * (1 - cr).

    Shows:
    - Prior moment capacity band (full time range)
    - Observed moment capacity up to current_time
    - Posterior forecast from current_time onwards
    - Survived moments as scatter markers (if provided)

    Args:
        cr_grid: Corrosion ratio grid values.
        cr_forecast_prior: Dict of {time: cr_pdf} for prior forecast.
        cr_forecast_posterior: Dict of {time: cr_pdf} for posterior forecast.
        moment_cap: Initial moment capacity [kNm].
        survived_times: Times at which moment was survived.
        survived_moments: Survived moment values at each time.
        obs_times: All observation times.
        obs_moment_cap: All observed moment capacity values.
        alpha: Confidence level for quantiles.
        xlim: X-axis limits.
        ylim: Y-axis limits.

    Returns:
        Matplotlib figure.
    """
    from scipy.integrate import cumulative_trapezoid

    cr_grid = np.asarray(cr_grid)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Prior: cr PDF → cr mean/quantiles → moment
    cr_forecasts_prior = np.vstack(list(cr_forecast_prior.values()))
    cr_prior_mean = np.trapezoid(cr_forecasts_prior * cr_grid[None, :], cr_grid, axis=1)
    cr_prior_cdf = cumulative_trapezoid(cr_forecasts_prior, cr_grid)
    cr_prior_q05 = cr_grid[np.argmin(np.abs(cr_prior_cdf - alpha), axis=1)]
    cr_prior_q95 = cr_grid[np.argmin(np.abs(cr_prior_cdf - (1 - alpha)), axis=1)]

    moment_prior_mean = moment_cap * (1 - cr_prior_mean)
    moment_prior_q95 = moment_cap * (1 - cr_prior_q05)   # flip: low cr → high moment
    moment_prior_q05 = moment_cap * (1 - cr_prior_q95)

    # Posterior: same derivation
    cr_forecasts_posterior = np.vstack(list(cr_forecast_posterior.values()))
    cr_posterior_mean = np.trapezoid(cr_forecasts_posterior * cr_grid[None, :], cr_grid, axis=1)
    cr_posterior_cdf = cumulative_trapezoid(cr_forecasts_posterior, cr_grid)
    cr_posterior_q05 = cr_grid[np.argmin(np.abs(cr_posterior_cdf - alpha), axis=1)]
    cr_posterior_q95 = cr_grid[np.argmin(np.abs(cr_posterior_cdf - (1 - alpha)), axis=1)]

    moment_posterior_mean = moment_cap * (1 - cr_posterior_mean)
    moment_posterior_q95 = moment_cap * (1 - cr_posterior_q05)
    moment_posterior_q05 = moment_cap * (1 - cr_posterior_q95)

    # Truncate posterior at survived moments
    times_posterior = np.asarray(list(cr_forecast_posterior.keys()))
    if survived_times is not None and survived_moments is not None:
        survived_interp = np.interp(times_posterior, survived_times, survived_moments)
        moment_posterior_mean = np.maximum(moment_posterior_mean, survived_interp)
        moment_posterior_q05 = np.maximum(moment_posterior_q05, survived_interp)
        moment_posterior_q95 = np.maximum(moment_posterior_q95, survived_interp)

    # Prior band
    times_prior = np.asarray(list(cr_forecast_prior.keys()))
    ax.fill_between(times_prior, moment_prior_q05, moment_prior_q95, color="b", alpha=0.15)
    ax.plot(times_prior, moment_prior_q05, color="b", linewidth=0.5)
    ax.plot(times_prior, moment_prior_q95, color="b", linewidth=0.5)
    ax.plot(times_prior, moment_prior_mean, color="b", linewidth=1.5, label="Prior")

    # Posterior band
    ax.fill_between(times_posterior, moment_posterior_q05, moment_posterior_q95, color="r", alpha=0.15)
    ax.plot(times_posterior, moment_posterior_q05, color="r", linewidth=0.5)
    ax.plot(times_posterior, moment_posterior_q95, color="r", linewidth=0.5)
    ax.plot(times_posterior, moment_posterior_mean, color="r", linewidth=1.5, label="Posterior")

    # Observations
    if obs_times is not None and obs_moment_cap is not None:
        ax.scatter(obs_times, obs_moment_cap, color="k", s=50, zorder=5, label="Observations")

    # Survived moments
    if survived_times is not None and survived_moments is not None:
        ax.plot(survived_times, survived_moments, color="g", linewidth=1.5, label="Survived moment")

    ax.set_xlabel("Forecast time [yr]", fontsize=12)
    ax.set_ylabel("Moment capacity [kNm]", fontsize=12)
    ax.set_title(f"Moment Capacity Forecast at t_obs = {times_posterior.min():.0f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()

    return fig


def plot_jpdf_snapshot(
    time: float,
    C50_grid: NDArray,
    C50_prior: NDArray,
    C50_posterior: NDArray,
    corrosion_ratio_grid: NDArray,
    cr_pdf_prior: NDArray,
    cr_pdf_posterior: NDArray,
    fragility_cr: NDArray,
    fragility_pf: NDArray,
    beta_prior: float,
    beta_posterior: float,
    beta_forecast_prior: dict = None,
    beta_forecast_posterior: dict = None,
    obs_times: Sequence[float] = None,
    obs_corrosion: Sequence[float] = None,
    current_cr: float = None,
    moment_cap: float = 750.0,
    start_thickness: float = 9.5,
    beta_req: float = 2.3,
) -> plt.Figure:
    """
    Generate a snapshot image of the JPDF state at a given timestep.

    Creates a 2x2 dashboard showing:
    - Top-left: C50 prior vs posterior PDF
    - Top-right: Beta forecast from current time (prior vs posterior)
    - Bottom-left: Corrosion ratio distribution at current time
    - Bottom-right: Fragility curve with current CR marked

    Args:
        time: Current time [years].
        C50_grid: Grid for C50 values.
        C50_prior: Prior PDF of C50.
        C50_posterior: Posterior PDF of C50.
        corrosion_ratio_grid: Grid for corrosion ratio.
        cr_pdf_prior: Prior PDF of corrosion ratio.
        cr_pdf_posterior: Posterior PDF of corrosion ratio.
        fragility_cr: Corrosion ratios from fragility curve.
        fragility_pf: Pf values from fragility curve.
        beta_prior: Prior reliability index at current time.
        beta_posterior: Posterior reliability index at current time.
        beta_forecast_prior: Dict of {future_time: beta} for prior forecast.
        beta_forecast_posterior: Dict of {future_time: beta} for posterior forecast.
        obs_times: Times of corrosion observations.
        obs_corrosion: Observed corrosion values.
        current_cr: Current corrosion ratio (to mark on plots).
        moment_cap: Moment capacity [kNm].
        start_thickness: Initial wall thickness [mm].
        beta_req: Required reliability index.

    Returns:
        Matplotlib figure with JPDF snapshot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"JPDF Snapshot at t = {time:.0f} years", fontsize=14, fontweight="bold")

    max_beta = 6.0

    # Top-left: C50 distribution
    ax = axes[0, 0]
    ax.fill_between(C50_grid, C50_prior, alpha=0.3, color="b", label="Prior")
    ax.plot(C50_grid, C50_prior, "b-", linewidth=1.5)
    ax.fill_between(C50_grid, C50_posterior, alpha=0.3, color="r", label="Posterior")
    ax.plot(C50_grid, C50_posterior, "r-", linewidth=1.5)
    ax.set_xlabel("C50 [mm]", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("C50 Distribution", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top-right: Beta forecast
    ax = axes[0, 1]
    if beta_forecast_prior is not None and beta_forecast_posterior is not None:
        times_forecast = sorted(beta_forecast_prior.keys())
        beta_prior_vals = [min(beta_forecast_prior[t], max_beta) for t in times_forecast]
        beta_post_vals = [min(beta_forecast_posterior[t], max_beta) for t in times_forecast]

        ax.plot(times_forecast, beta_prior_vals, "b-", linewidth=2, label="Prior forecast")
        ax.plot(times_forecast, beta_post_vals, "r-", linewidth=2, label="Posterior forecast")
        ax.axhline(beta_req, color="k", linestyle="--", linewidth=1.5, label=f"Requirement (β={beta_req})")
        ax.axvline(time, color="gray", linestyle=":", linewidth=1, alpha=0.7, label=f"Current t={time:.0f}")

        ax.set_xlabel("Forecast time [yr]", fontsize=10)
        ax.set_ylabel("β [-]", fontsize=10)
        ax.set_title("Reliability Index Forecast", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max_beta)
    else:
        ax.text(0.5, 0.5, "No forecast data", ha="center", va="center", fontsize=12)
        ax.set_title("Reliability Index Forecast", fontsize=11)

    # Bottom-left: Corrosion ratio distribution
    ax = axes[1, 0]
    ax.fill_between(corrosion_ratio_grid, cr_pdf_prior, alpha=0.3, color="b", label="Prior")
    ax.plot(corrosion_ratio_grid, cr_pdf_prior, "b-", linewidth=1.5)
    ax.fill_between(corrosion_ratio_grid, cr_pdf_posterior, alpha=0.3, color="r", label="Posterior")
    ax.plot(corrosion_ratio_grid, cr_pdf_posterior, "r-", linewidth=1.5)
    if current_cr is not None:
        ax.axvline(current_cr, color="k", linestyle="--", linewidth=1.5, label=f"Observed CR={current_cr:.3f}")
    ax.set_xlabel("Corrosion Ratio [-]", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Corrosion Ratio Distribution at t={time:.0f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(1.0, max(0.5, corrosion_ratio_grid.max() * 1.2)))

    # Bottom-right: Fragility curve
    ax = axes[1, 1]
    ax.plot(fragility_cr, fragility_pf, "k-", linewidth=2, label="Fragility curve")
    ax.fill_between(fragility_cr, fragility_pf, alpha=0.2, color="gray")
    if current_cr is not None:
        pf_at_cr = np.interp(current_cr, fragility_cr, fragility_pf)
        ax.axvline(current_cr, color="r", linestyle="--", linewidth=1.5)
        ax.scatter([current_cr], [pf_at_cr], color="r", s=100, zorder=5,
                   label=f"Current: CR={current_cr:.3f}")
    ax.set_xlabel("Corrosion Ratio [-]", fontsize=10)
    ax.set_ylabel("Pf [-]", fontsize=10)
    ax.set_title("Fragility Curve", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add text annotation with key stats
    stats_text = (
        f"Current (t={time:.0f}):\n"
        f"  Prior:     β={beta_prior:.2f}\n"
        f"  Posterior: β={beta_posterior:.2f}\n"
        f"  C50 prior mean:  {np.trapezoid(C50_grid * C50_prior, C50_grid):.2f} mm\n"
        f"  C50 post mean:   {np.trapezoid(C50_grid * C50_posterior, C50_grid):.2f} mm"
    )
    if obs_times is not None and obs_corrosion is not None:
        n_obs = sum(1 for t in obs_times if t <= time)
        stats_text += f"\n  Observations: {n_obs}"

    fig.text(0.02, 0.02, stats_text, fontsize=9, fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.close()

    return fig


def plot_beta_forecast_at_time(
    current_time: float,
    results: dict,
    beta_req: float = 2.3,
) -> plt.Figure:
    """
    Plot beta forecasts at a specific observation time.

    Shows:
    - Blue solid line: Prior forecast (no updating)
    - Red dots: Posterior beta at each observation time
    - Red dotted lines: Posterior forecasts from past observation times
    - Red dashed line: Posterior forecast from the current (last) observation time
    - Red solid line: Hindcast connecting posterior betas with forecast segments
    - Black dashed: Requirement

    Args:
        current_time: Current observation time.
        results: Pipeline results dict (should contain only results up to current_time).
        beta_req: Required reliability index.

    Returns:
        Matplotlib figure.
    """
    from itertools import chain

    min_beta = 1.0
    max_beta = 3.5

    obs_times = sorted(results.keys())
    first_result = results[obs_times[0]]
    all_forecast_times = sorted(first_result["prior"]["beta_forecast"].keys())
    t_min, t_max = min(all_forecast_times), max(all_forecast_times)

    fig = plt.figure(figsize=(8, 4))

    # Prior forecast (blue solid)
    bf_prior = first_result["prior"]["beta_forecast"]
    times_prior = sorted(bf_prior.keys())
    beta_prior = [min(bf_prior[t], max_beta) for t in times_prior]
    plt.plot(times_prior, beta_prior, c="b", label="Prior")

    # Hindcast: solid red line connecting posterior betas with forecast segments
    hindcast_segments = []
    for prev_t, curr_t in zip(obs_times[:-1], obs_times[1:]):
        prev_beta = min(results[prev_t]["posterior"]["beta"], max_beta)
        curr_beta = min(results[curr_t]["posterior"]["beta"], max_beta)
        prev_forecast_at_curr = results[prev_t]["posterior"]["beta_forecast"].get(curr_t)
        if prev_forecast_at_curr is not None:
            prev_forecast_at_curr = min(prev_forecast_at_curr, max_beta)
            hindcast_segments.append([
                (prev_t, prev_beta),
                (curr_t, prev_forecast_at_curr),
                (curr_t, curr_beta),
            ])

    if hindcast_segments:
        hindcast_points = list(chain.from_iterable(hindcast_segments))
        plt.plot([p[0] for p in hindcast_points], [p[1] for p in hindcast_points], c="r")

    # Posterior dots and forecast lines
    for i, obs_t in enumerate(obs_times):
        result = results[obs_t]
        beta_current = min(result["posterior"]["beta"], max_beta)

        # Red dot
        label = "Posterior" if i == 0 else None
        plt.scatter(obs_t, beta_current, color="r", zorder=5, label=label)

        # Forecast line from this observation time
        if "beta_forecast" in result["posterior"]:
            bf = result["posterior"]["beta_forecast"]
            times_f = sorted(bf.keys())
            betas_f = [min(bf[t], max_beta) for t in times_f]

            is_current = (obs_t == current_time)
            if is_current:
                plt.plot(times_f, betas_f, c="r", linestyle="--")
            else:
                plt.plot(times_f, betas_f, c="r", linestyle="dotted", alpha=0.4)

    # Requirement
    plt.axhline(beta_req, c="k", linestyle="--", label="Requirement")

    plt.xlabel("Forecast time [yr]", fontsize=12)
    plt.ylabel(r"$\beta$ [-]", fontsize=12)
    plt.xlim(t_min, t_max)
    plt.ylim(min_beta, max_beta)
    plt.legend(fontsize=12)
    plt.grid()
    plt.subplots_adjust(bottom=0.15)
    plt.close()

    return fig


def plot_beta_forecasts(
    results: dict,
    beta_req: float = 2.3,
    title: str = "Reliability Index Forecasts",
) -> plt.Figure:
    """
    Plot beta forecasts from each observation time.

    Shows prior forecast (single line) and posterior forecasts (one per obs time).

    Args:
        results: Pipeline results dict with beta_forecast per timestep.
        beta_req: Required reliability index.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    obs_times = sorted(results.keys())
    max_beta = 6.0

    # Colors for different observation times
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(obs_times)))

    # Left plot: Prior forecast (same for all obs times - just show once)
    first_result = results[obs_times[0]]
    if "beta_forecast" in first_result["prior"]:
        bf = first_result["prior"]["beta_forecast"]
        times_forecast = sorted(bf.keys())
        beta_vals = [min(bf[t], max_beta) for t in times_forecast]

        ax1.plot(times_forecast, beta_vals, "b-", linewidth=2, label="Prior")
        ax1.axhline(beta_req, color="k", linestyle="--", linewidth=1.5, label=f"Requirement (β={beta_req})")
        ax1.set_xlabel("Forecast time [yr]", fontsize=11)
        ax1.set_ylabel("β [-]", fontsize=11)
        ax1.set_title("Prior Forecast (no updating)", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max_beta)

    # Right plot: Posterior forecasts from each observation time
    for i, obs_t in enumerate(obs_times):
        result = results[obs_t]
        if "beta_forecast" in result["posterior"]:
            bf = result["posterior"]["beta_forecast"]
            times_forecast = sorted(bf.keys())
            beta_vals = [min(bf[t], max_beta) for t in times_forecast]

            ax2.plot(times_forecast, beta_vals, color=colors[i], linewidth=1.5,
                     label=f"t={obs_t:.0f}", marker="o", markersize=3)

    ax2.axhline(beta_req, color="k", linestyle="--", linewidth=1.5, label=f"Requirement")
    ax2.set_xlabel("Forecast time [yr]", fontsize=11)
    ax2.set_ylabel("β [-]", fontsize=11)
    ax2.set_title("Posterior Forecasts (with Bayesian updating)", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max_beta)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.close()

    return fig


def plot_posterior_forecast_evolution(
    results: dict,
    beta_req: float = 2.3,
    title: str = "Posterior Forecast Evolution",
) -> plt.Figure:
    """
    Single plot showing how CUMULATIVE posterior forecasts evolve with observations.

    Each line is cumulative:
    - Historical segment: posterior betas from observation times <= t_obs
    - Forecast segment: forecasted betas for times > t_obs

    Args:
        results: Pipeline results dict.
        beta_req: Required reliability index.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    obs_times = sorted(results.keys())
    max_beta = 6.0
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(obs_times)))

    # Get the full time range
    first_result = results[obs_times[0]]
    all_forecast_times = sorted(first_result["prior"]["beta_forecast"].keys())
    t_min, t_max = min(all_forecast_times), max(all_forecast_times)

    # Plot prior as reference
    if "beta_forecast" in first_result["prior"]:
        bf = first_result["prior"]["beta_forecast"]
        times_forecast = sorted(bf.keys())
        beta_vals = [min(bf[t], max_beta) for t in times_forecast]

        ax.plot(times_forecast, beta_vals, color="gray", linestyle="--",
                linewidth=2, alpha=0.6, label="Prior (no updating)")

    # Build cumulative lines for each observation time
    for i, obs_t in enumerate(obs_times):
        # Build cumulative line: historical betas + forecast betas
        cumulative_times = []
        cumulative_betas = []

        # Historical segment: posterior betas from past observation times up to obs_t
        for past_t in obs_times:
            if past_t <= obs_t:
                cumulative_times.append(past_t)
                cumulative_betas.append(min(results[past_t]["posterior"]["beta"], max_beta))

        # Forecast segment: forecasted betas for times > obs_t
        result = results[obs_t]
        if "beta_forecast" in result["posterior"]:
            bf = result["posterior"]["beta_forecast"]
            for t in sorted(bf.keys()):
                if t > obs_t:
                    cumulative_times.append(t)
                    cumulative_betas.append(min(bf[t], max_beta))

        ax.plot(cumulative_times, cumulative_betas, color=colors[i], linewidth=2,
                label=f"t_obs={obs_t:.0f}")

        # Mark the observation point
        obs_idx = cumulative_times.index(obs_t)
        ax.scatter([obs_t], [cumulative_betas[obs_idx]], color=colors[i], s=80,
                   zorder=5, edgecolor="k", linewidth=1)

    ax.axhline(beta_req, color="r", linestyle="-", linewidth=2,
               label=f"Requirement (β={beta_req})")
    ax.set_xlabel("Forecast time [yr]", fontsize=12)
    ax.set_ylabel("β [-]", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max_beta)
    ax.set_xlim(t_min, t_max)

    plt.tight_layout()
    plt.close()

    return fig


def plot_beta_forecast_grid(
    results: dict,
    beta_req: float = 2.3,
    ncols: int = 3,
    title: str = "Reliability Index Forecasts per Observation Time",
) -> plt.Figure:
    """
    Create a grid of subplots showing CUMULATIVE beta forecasts at each observation time.

    Each subplot shows cumulative lines where:
    - Historical segment: posterior betas from observation times <= t_obs
    - Forecast segment: forecasted betas for times > t_obs

    Args:
        results: Pipeline results dict with beta_forecast per timestep.
        beta_req: Required reliability index.
        ncols: Number of columns in the grid.
        title: Overall figure title.

    Returns:
        Matplotlib figure with grid of subplots.
    """
    obs_times = sorted(results.keys())
    n_obs = len(obs_times)
    nrows = int(np.ceil(n_obs / ncols))
    max_beta = 6.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    # Get the full time range
    first_result = results[obs_times[0]]
    all_forecast_times = sorted(first_result["prior"]["beta_forecast"].keys())
    t_min, t_max = min(all_forecast_times), max(all_forecast_times)

    # Precompute prior forecast
    bf_prior = first_result["prior"]["beta_forecast"]
    times_prior = sorted(bf_prior.keys())
    beta_prior = [min(bf_prior[t], max_beta) for t in times_prior]

    for idx, current_time in enumerate(obs_times):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # Get results up to current time
        results_up_to_t = {k: v for k, v in results.items() if k <= current_time}
        past_obs_times = sorted(results_up_to_t.keys())

        # Colors for past observations (lighter) to current (darker)
        n_past = len(past_obs_times)
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_past))

        # Plot prior (gray dashed)
        ax.plot(times_prior, beta_prior, color="gray", linestyle="--",
                linewidth=1.5, alpha=0.6, label="Prior")

        # Build cumulative lines for each observation time up to current_time
        for i, obs_t in enumerate(past_obs_times):
            # Build cumulative line: historical betas + forecast betas
            cumulative_times = []
            cumulative_betas = []

            # Historical segment: posterior betas from past observation times up to obs_t
            for past_t in past_obs_times:
                if past_t <= obs_t:
                    cumulative_times.append(past_t)
                    cumulative_betas.append(min(results_up_to_t[past_t]["posterior"]["beta"], max_beta))

            # Forecast segment: forecasted betas for times > obs_t
            result = results_up_to_t[obs_t]
            if "beta_forecast" in result["posterior"]:
                bf = result["posterior"]["beta_forecast"]
                for t in sorted(bf.keys()):
                    if t > obs_t:
                        cumulative_times.append(t)
                        cumulative_betas.append(min(bf[t], max_beta))

            is_current = (obs_t == current_time)
            linewidth = 2 if is_current else 1
            alpha = 1.0 if is_current else 0.6

            ax.plot(cumulative_times, cumulative_betas, color=colors[i],
                    linewidth=linewidth, alpha=alpha)

            # Mark observation point
            obs_idx = cumulative_times.index(obs_t)
            marker_size = 60 if is_current else 30
            ax.scatter([obs_t], [cumulative_betas[obs_idx]], color=colors[i],
                       s=marker_size, zorder=5,
                       edgecolor="k" if is_current else "none",
                       linewidth=1 if is_current else 0)

        # Requirement line
        ax.axhline(beta_req, color="r", linestyle="-", linewidth=1.5, alpha=0.8)

        # Vertical line at current observation time
        ax.axvline(current_time, color="k", linestyle=":", linewidth=0.8, alpha=0.5)

        ax.set_title(f"t_obs = {current_time:.0f}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max_beta)
        ax.set_xlim(t_min, t_max)

        # Only add labels on edge subplots
        if row == nrows - 1 or (row == nrows - 2 and col >= n_obs - (nrows - 1) * ncols):
            ax.set_xlabel("Forecast time [yr]", fontsize=10)
        if col == 0:
            ax.set_ylabel("β [-]", fontsize=10)

    # Hide unused subplots
    for idx in range(n_obs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.close()

    return fig
