"""
Plotting utilities for the D-Sheet piling case study.
"""

from pathlib import Path
from typing import Sequence

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


def plot_failure_probability(
    times: Sequence[float],
    pf: Sequence[float],
    pf_target: float | None = None,
    title: str = "Failure probability over time",
    xlabel: str = "Time [years]",
    ylabel: str = r"$P_f$ [-]",
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot failure probability over time.

    Args:
        times: Time points.
        pf: Failure probability values.
        pf_target: Target failure probability (horizontal line).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        log_scale: Use log scale for y-axis.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots()
    ax.plot(times, pf, "b-o", markersize=4, label=r"$P_f$")

    if pf_target is not None:
        ax.axhline(pf_target, color="r", linestyle="--", label=f"Target = {pf_target:.1e}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
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


# -----------------------------------------------------------------------------
# Prior vs Posterior comparison plots (replicating timos branch style)
# -----------------------------------------------------------------------------

def plot_beta_prior_posterior(
    times: Sequence[float],
    beta_prior: Sequence[float],
    beta_posterior: Sequence[float],
    beta_req: float = 3.8,
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
    pf_prior: float,
    pf_posterior: float,
    beta_prior: float,
    beta_posterior: float,
    pf_forecast_prior: dict = None,
    pf_forecast_posterior: dict = None,
    obs_times: Sequence[float] = None,
    obs_corrosion: Sequence[float] = None,
    current_cr: float = None,
    moment_cap: float = 750.0,
    start_thickness: float = 9.5,
    beta_req: float = 3.8,
) -> plt.Figure:
    """
    Generate a snapshot image of the JPDF state at a given timestep.

    Creates a 2x2 dashboard showing:
    - Top-left: C50 prior vs posterior PDF
    - Top-right: Pf FORECAST from current time (prior vs posterior)
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
        pf_prior: Prior failure probability at current time.
        pf_posterior: Posterior failure probability at current time.
        beta_prior: Prior reliability index at current time.
        beta_posterior: Posterior reliability index at current time.
        pf_forecast_prior: Dict of {future_time: pf} for prior forecast.
        pf_forecast_posterior: Dict of {future_time: pf} for posterior forecast.
        obs_times: Times of corrosion observations.
        obs_corrosion: Observed corrosion values.
        current_cr: Current corrosion ratio (to mark on plots).
        moment_cap: Moment capacity [kNm].
        start_thickness: Initial wall thickness [mm].
        beta_req: Required reliability index.

    Returns:
        Matplotlib figure with JPDF snapshot.
    """
    from scipy.stats import norm as sp_norm

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"JPDF Snapshot at t = {time:.0f} years", fontsize=14, fontweight="bold")

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

    # Top-right: Pf FORECAST (the key plot!)
    ax = axes[0, 1]
    if pf_forecast_prior is not None and pf_forecast_posterior is not None:
        times_forecast = sorted(pf_forecast_prior.keys())
        pf_prior_vals = [pf_forecast_prior[t] for t in times_forecast]
        pf_post_vals = [pf_forecast_posterior[t] for t in times_forecast]

        # Convert to beta
        max_beta = 6.0
        beta_prior_vals = []
        beta_post_vals = []
        for pf in pf_prior_vals:
            pf_clipped = max(pf, 1e-10)
            b = -sp_norm.ppf(pf_clipped)
            beta_prior_vals.append(min(b, max_beta))
        for pf in pf_post_vals:
            pf_clipped = max(pf, 1e-10)
            b = -sp_norm.ppf(pf_clipped)
            beta_post_vals.append(min(b, max_beta))

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
        f"  Prior:     Pf={pf_prior:.2e}, β={beta_prior:.2f}\n"
        f"  Posterior: Pf={pf_posterior:.2e}, β={beta_posterior:.2f}\n"
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


def plot_pf_prior_posterior(
    times: Sequence[float],
    pf_prior: Sequence[float],
    pf_posterior: Sequence[float],
    title: str = "",
    xlim: tuple = (50, 80),
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot failure probability (Pf) prior vs posterior over time.

    Args:
        times: Time points [years].
        pf_prior: Prior failure probability values.
        pf_posterior: Posterior failure probability values.
        title: Plot title.
        xlim: X-axis limits.
        log_scale: Use log scale for y-axis.

    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=(8, 4))

    plt.plot(times, pf_prior, c="b", marker="o", markersize=4, label="Prior")
    plt.plot(times, pf_posterior, c="r", marker="o", markersize=4, label="Posterior")

    plt.xlabel("Forecast time [yr]", fontsize=12)
    plt.ylabel(r"$P_f$ [-]", fontsize=12)
    plt.xlim(xlim)
    if log_scale:
        plt.yscale("log")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    if title:
        plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.close()

    return fig


def plot_beta_forecast_at_time(
    current_time: float,
    results: dict,
    beta_req: float = 3.8,
) -> plt.Figure:
    """
    Plot beta forecasts at a specific observation time (CUMULATIVE).

    For each past observation time t_obs, builds a cumulative line:
    - Historical segment: posterior betas from all observation times <= t_obs
    - Forecast segment: forecasted betas for times > t_obs

    This creates continuous lines from t_min to t_max, where each line shows:
    - What actually happened up to t_obs (based on Bayesian updates)
    - What was forecasted from t_obs onwards

    Args:
        current_time: Current observation time.
        results: Pipeline results dict (should contain only results up to current_time).
        beta_req: Required reliability index.

    Returns:
        Matplotlib figure.
    """
    from scipy.stats import norm as sp_norm

    fig, ax = plt.subplots(figsize=(10, 6))

    obs_times = sorted(results.keys())
    max_beta = 6.0

    # Colormap: older observations are lighter, current is darkest
    n_obs = len(obs_times)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_obs))

    # Get the full time range from the first observation's forecast
    first_result = results[obs_times[0]]
    all_forecast_times = sorted(first_result["prior"]["pf_forecast"].keys())
    t_min, t_max = min(all_forecast_times), max(all_forecast_times)

    # Plot prior forecast (reference, dashed gray)
    if "pf_forecast" in first_result["prior"]:
        pf_forecast = first_result["prior"]["pf_forecast"]
        times_forecast = sorted(pf_forecast.keys())
        beta_forecast = []
        for t in times_forecast:
            pf = max(pf_forecast[t], 1e-10)
            b = -sp_norm.ppf(pf)
            beta_forecast.append(min(b, max_beta))

        ax.plot(times_forecast, beta_forecast, color="gray", linestyle="--",
                linewidth=2, alpha=0.7, label="Prior (no updating)")

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
        if "pf_forecast" in result["posterior"]:
            pf_forecast = result["posterior"]["pf_forecast"]
            for t in sorted(pf_forecast.keys()):
                if t > obs_t:
                    pf = max(pf_forecast[t], 1e-10)
                    b = -sp_norm.ppf(pf)
                    cumulative_times.append(t)
                    cumulative_betas.append(min(b, max_beta))

        # Current observation time gets emphasized
        is_current = (obs_t == current_time)
        linewidth = 2.5 if is_current else 1.5
        alpha = 1.0 if is_current else 0.7

        ax.plot(cumulative_times, cumulative_betas, color=colors[i],
                linewidth=linewidth, alpha=alpha,
                label=f"t_obs={obs_t:.0f}" + (" (current)" if is_current else ""))

        # Mark the observation point with a marker
        obs_idx = cumulative_times.index(obs_t)
        ax.scatter([obs_t], [cumulative_betas[obs_idx]], color=colors[i],
                   s=80 if is_current else 50, zorder=5,
                   edgecolor="k" if is_current else "none",
                   linewidth=1.5 if is_current else 0)

    # Requirement line
    ax.axhline(beta_req, color="r", linestyle="-", linewidth=2,
               label=f"Requirement (β={beta_req})")

    # Vertical line at current observation time
    ax.axvline(current_time, color="k", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("Forecast time [yr]", fontsize=12)
    ax.set_ylabel("β [-]", fontsize=12)
    ax.set_title(f"Reliability Index Forecast at Observation Time t = {current_time:.0f}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max_beta)
    ax.set_xlim(t_min, t_max)

    plt.tight_layout()
    plt.close()

    return fig


def plot_beta_forecasts(
    results: dict,
    beta_req: float = 3.8,
    title: str = "Reliability Index Forecasts",
) -> plt.Figure:
    """
    Plot beta forecasts from each observation time.

    Shows prior forecast (single line) and posterior forecasts (one per obs time).

    Args:
        results: Pipeline results dict with pf_forecast per timestep.
        beta_req: Required reliability index.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    from scipy.stats import norm as sp_norm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    obs_times = sorted(results.keys())
    max_beta = 6.0

    # Colors for different observation times
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(obs_times)))

    # Left plot: Prior forecast (same for all obs times - just show once)
    first_result = results[obs_times[0]]
    if "pf_forecast" in first_result["prior"]:
        pf_forecast = first_result["prior"]["pf_forecast"]
        times_forecast = sorted(pf_forecast.keys())
        beta_forecast = []
        for t in times_forecast:
            pf = max(pf_forecast[t], 1e-10)
            b = -sp_norm.ppf(pf)
            beta_forecast.append(min(b, max_beta))

        ax1.plot(times_forecast, beta_forecast, "b-", linewidth=2, label="Prior")
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
        if "pf_forecast" in result["posterior"]:
            pf_forecast = result["posterior"]["pf_forecast"]
            times_forecast = sorted(pf_forecast.keys())
            beta_forecast = []
            for t in times_forecast:
                pf = max(pf_forecast[t], 1e-10)
                b = -sp_norm.ppf(pf)
                beta_forecast.append(min(b, max_beta))

            ax2.plot(times_forecast, beta_forecast, color=colors[i], linewidth=1.5,
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


def plot_pf_forecasts(
    results: dict,
    title: str = "Failure Probability Forecasts",
) -> plt.Figure:
    """
    Plot Pf forecasts from each observation time.

    Args:
        results: Pipeline results dict with pf_forecast per timestep.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    obs_times = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(obs_times)))

    # Left plot: Prior forecast
    first_result = results[obs_times[0]]
    if "pf_forecast" in first_result["prior"]:
        pf_forecast = first_result["prior"]["pf_forecast"]
        times_forecast = sorted(pf_forecast.keys())
        pf_vals = [pf_forecast[t] for t in times_forecast]

        ax1.plot(times_forecast, pf_vals, "b-", linewidth=2, marker="o", markersize=4, label="Prior")
        ax1.set_xlabel("Forecast time [yr]", fontsize=11)
        ax1.set_ylabel("Pf [-]", fontsize=11)
        ax1.set_title("Prior Forecast (no updating)", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

    # Right plot: Posterior forecasts
    for i, obs_t in enumerate(obs_times):
        result = results[obs_t]
        if "pf_forecast" in result["posterior"]:
            pf_forecast = result["posterior"]["pf_forecast"]
            times_forecast = sorted(pf_forecast.keys())
            pf_vals = [pf_forecast[t] for t in times_forecast]

            ax2.plot(times_forecast, pf_vals, color=colors[i], linewidth=1.5,
                     label=f"t={obs_t:.0f}", marker="o", markersize=3)

    ax2.set_xlabel("Forecast time [yr]", fontsize=11)
    ax2.set_ylabel("Pf [-]", fontsize=11)
    ax2.set_title("Posterior Forecasts (with Bayesian updating)", fontsize=12)
    ax2.legend(fontsize=9, loc="lower right", ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.close()

    return fig


def plot_posterior_forecast_evolution(
    results: dict,
    beta_req: float = 3.8,
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
    from scipy.stats import norm as sp_norm

    fig, ax = plt.subplots(figsize=(10, 6))

    obs_times = sorted(results.keys())
    max_beta = 6.0
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(obs_times)))

    # Get the full time range
    first_result = results[obs_times[0]]
    all_forecast_times = sorted(first_result["prior"]["pf_forecast"].keys())
    t_min, t_max = min(all_forecast_times), max(all_forecast_times)

    # Plot prior as reference
    if "pf_forecast" in first_result["prior"]:
        pf_forecast = first_result["prior"]["pf_forecast"]
        times_forecast = sorted(pf_forecast.keys())
        beta_forecast = []
        for t in times_forecast:
            pf = max(pf_forecast[t], 1e-10)
            b = -sp_norm.ppf(pf)
            beta_forecast.append(min(b, max_beta))

        ax.plot(times_forecast, beta_forecast, color="gray", linestyle="--",
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
        if "pf_forecast" in result["posterior"]:
            pf_forecast = result["posterior"]["pf_forecast"]
            for t in sorted(pf_forecast.keys()):
                if t > obs_t:
                    pf = max(pf_forecast[t], 1e-10)
                    b = -sp_norm.ppf(pf)
                    cumulative_times.append(t)
                    cumulative_betas.append(min(b, max_beta))

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
    beta_req: float = 3.8,
    ncols: int = 3,
    title: str = "Reliability Index Forecasts per Observation Time",
) -> plt.Figure:
    """
    Create a grid of subplots showing CUMULATIVE beta forecasts at each observation time.

    Each subplot shows cumulative lines where:
    - Historical segment: posterior betas from observation times <= t_obs
    - Forecast segment: forecasted betas for times > t_obs

    Args:
        results: Pipeline results dict with pf_forecast per timestep.
        beta_req: Required reliability index.
        ncols: Number of columns in the grid.
        title: Overall figure title.

    Returns:
        Matplotlib figure with grid of subplots.
    """
    from scipy.stats import norm as sp_norm

    obs_times = sorted(results.keys())
    n_obs = len(obs_times)
    nrows = int(np.ceil(n_obs / ncols))
    max_beta = 6.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    # Get the full time range
    first_result = results[obs_times[0]]
    all_forecast_times = sorted(first_result["prior"]["pf_forecast"].keys())
    t_min, t_max = min(all_forecast_times), max(all_forecast_times)

    # Precompute prior forecast
    pf_forecast_prior = first_result["prior"]["pf_forecast"]
    times_prior = sorted(pf_forecast_prior.keys())
    beta_prior = []
    for t in times_prior:
        pf = max(pf_forecast_prior[t], 1e-10)
        b = -sp_norm.ppf(pf)
        beta_prior.append(min(b, max_beta))

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
            if "pf_forecast" in result["posterior"]:
                pf_forecast = result["posterior"]["pf_forecast"]
                for t in sorted(pf_forecast.keys()):
                    if t > obs_t:
                        pf = max(pf_forecast[t], 1e-10)
                        b = -sp_norm.ppf(pf)
                        cumulative_times.append(t)
                        cumulative_betas.append(min(b, max_beta))

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
