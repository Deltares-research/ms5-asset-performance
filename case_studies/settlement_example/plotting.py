"""
Plotting functions for the settlement reliability analysis.

Generates corner plots of the joint (CR, k) PDF, settlement forecast plots
with prior/posterior comparison and hindcast/forecast distinction, residual
settlement PDF plots, reliability index evolution, and animated GIFs.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


def _hdr_level(pdf_2d: NDArray, x_grid: NDArray, y_grid: NDArray, alpha: float = 0.95) -> float:
    """Compute the PDF level that encloses `alpha` fraction of probability mass.

    Sorts all PDF values from highest to lowest, accumulates probability mass
    (using cell areas), and returns the threshold where the cumulative mass
    first reaches `alpha`.

    Args:
        pdf_2d: 2D array of joint PDF values on the (x, y) grid.
        x_grid: 1D array of x grid values.
        y_grid: 1D array of y grid values.
        alpha: Probability mass to enclose (default 0.95).

    Returns:
        PDF threshold level for the HDR contour.
    """
    dx = np.diff(x_grid)
    dy = np.diff(y_grid)
    # Cell areas on the interior grid
    cell_area = dx[:, np.newaxis] * dy[np.newaxis, :]
    # Average PDF to cell centers
    pdf_centers = (pdf_2d[:-1, :-1] + pdf_2d[:-1, 1:] + pdf_2d[1:, :-1] + pdf_2d[1:, 1:]) / 4
    mass = pdf_centers * cell_area

    # Sort descending by PDF value, accumulate mass
    flat_pdf = pdf_centers.flatten()
    flat_mass = mass.flatten()
    order = np.argsort(-flat_pdf)
    cumulative = np.cumsum(flat_mass[order])
    cumulative /= cumulative[-1]

    idx = np.searchsorted(cumulative, alpha)
    idx = min(idx, len(order) - 1)
    return flat_pdf[order[idx]]


def plot_jpdf_snapshot(
    time: float,
    var_names: List[str],
    state: Dict[str, Any],
    true_values: Dict[str, float] = None,
) -> plt.Figure:
    """Corner plot of the grid-based JPDF at a given observation time.

    Center panel: filled contour of the posterior with prior and
    log-likelihood contour overlays and 95% HDR contour. Top/right panels:
    marginal PDFs (prior in blue, posterior in red) with 95% CI.

    Args:
        time: Observation time [days].
        var_names: List of two variable names [axis_0, axis_1].
        state: JPDF state dict with keys "{name}_grid", "{name}_prior",
            "{name}_posterior", "prior", "posterior", "loglikes".
        true_values: Dict mapping variable name to true value.

    Returns:
        Matplotlib Figure.
    """
    true_values = true_values or {}
    v0, v1 = var_names[0], var_names[1]
    g0 = np.array(state[f"{v0}_grid"])
    g1 = np.array(state[f"{v1}_grid"])
    p0_prior = np.array(state[f"{v0}_prior"])
    p0_post = np.array(state[f"{v0}_posterior"])
    p1_prior = np.array(state[f"{v1}_prior"])
    p1_post = np.array(state[f"{v1}_posterior"])
    prior = np.array(state["prior"])
    posterior = np.array(state["posterior"])
    loglikes = np.array(state["loglikes"])
    t0 = true_values.get(v0)
    t1 = true_values.get(v1)

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f"JPDF at t = {time:.0f} days", fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3, 1],
        height_ratios=[1, 3],
        hspace=0.05,
        wspace=0.05,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Bivariate contour (center)
    mesh0, mesh1 = np.meshgrid(g0, g1, indexing="ij")
    ax_main.contourf(mesh0, mesh1, posterior, levels=20, cmap="viridis")
    ax_main.contour(mesh0, mesh1, prior, levels=5, colors="white", linewidths=0.8, linestyles="--", alpha=0.6)
    ax_main.contour(mesh0, mesh1, loglikes, levels=5, colors="magenta", linewidths=0.8, linestyles="--", alpha=0.6)
    level_95 = _hdr_level(posterior, g0, g1, alpha=0.95)
    ax_main.contour(mesh0, mesh1, posterior, levels=[level_95], colors="yellow", linewidths=2, linestyles="-")
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], color="yellow", linewidth=2, linestyle="-", label="95% HDR"),
        Line2D([], [], color="white", linewidth=0.8, linestyle="--", label="Prior"),
        Line2D([], [], color="magenta", linewidth=0.8, linestyle="--", label="Log-likelihood"),
    ]
    if t0 is not None:
        ax_main.axvline(t0, color="red", linestyle="--", linewidth=1.5)
    if t1 is not None:
        ax_main.axhline(t1, color="red", linestyle="--", linewidth=1.5)
    if t0 is not None and t1 is not None:
        ax_main.plot(t0, t1, "rx", markersize=15, markeredgewidth=2.5, zorder=5)
        legend_handles.append(Line2D([], [], color="red", marker="x", linestyle="--", linewidth=1.5, markersize=10, label="True"))
    ax_main.legend(handles=legend_handles, loc="upper right", fontsize=8, facecolor="black", framealpha=0.4, labelcolor="white")
    ax_main.set_xlabel(v0)
    ax_main.set_ylabel(v1)
    ax_main.grid(True, alpha=0.3)

    # Variable 0 marginal (top)
    ax_top.fill_between(g0, p0_prior, alpha=0.3, color="b", label="Prior")
    ax_top.plot(g0, p0_prior, "b-", linewidth=1.5)
    ax_top.fill_between(g0, p0_post, alpha=0.3, color="r", label="Posterior")
    ax_top.plot(g0, p0_post, "r-", linewidth=1.5)
    m0, q025_0, q975_0 = _pdf_stats(g0, p0_post)
    pdf_at_m0 = np.interp(m0, g0, p0_post)
    ax_top.errorbar(m0, pdf_at_m0 * 0.5, xerr=[[max(0.0, m0 - q025_0)], [max(0.0, q975_0 - m0)]],
                    fmt="none", ecolor="r", elinewidth=1.5, capsize=4, capthick=1.5, label="95% CI")
    if t0 is not None:
        ax_top.axvline(t0, color="red", linestyle="--", linewidth=1.5, label="True")
    ax_top.set_ylabel("Density")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)
    ax_top.tick_params(labelbottom=False)

    # Variable 1 marginal (right)
    ax_right.fill_betweenx(g1, p1_prior, alpha=0.3, color="b")
    ax_right.plot(p1_prior, g1, "b-", linewidth=1.5)
    ax_right.fill_betweenx(g1, p1_post, alpha=0.3, color="r")
    ax_right.plot(p1_post, g1, "r-", linewidth=1.5)
    m1, q025_1, q975_1 = _pdf_stats(g1, p1_post)
    pdf_at_m1 = np.interp(m1, g1, p1_post)
    ax_right.errorbar(pdf_at_m1 * 0.5, m1, yerr=[[max(0.0, m1 - q025_1)], [max(0.0, q975_1 - m1)]],
                      fmt="none", ecolor="r", elinewidth=1.5, capsize=4, capthick=1.5)
    if t1 is not None:
        ax_right.axhline(t1, color="red", linestyle="--", linewidth=1.5)
    ax_right.set_xlabel("Density")
    ax_right.grid(True, alpha=0.3)
    ax_right.tick_params(labelleft=False)

    fig.add_subplot(gs[0, 1]).set_visible(False)

    plt.close()

    return fig


def plot_jpdf_prior(
    var_names: List[str],
    state: Dict[str, Any],
    true_values: Dict[str, float] = None,
) -> plt.Figure:
    """Corner plot of the grid-based prior JPDF (before any observations).

    Center panel: filled contour of the prior with log-likelihood overlay
    and 95% HDR contour. Top/right panels: marginal prior PDFs.

    Args:
        var_names: List of two variable names [axis_0, axis_1].
        state: JPDF state dict with keys "{name}_grid", "{name}_prior",
            "prior", "loglikes".
        true_values: Dict mapping variable name to true value.

    Returns:
        Matplotlib Figure.
    """
    true_values = true_values or {}
    v0, v1 = var_names[0], var_names[1]
    g0 = np.array(state[f"{v0}_grid"])
    g1 = np.array(state[f"{v1}_grid"])
    p0 = np.array(state[f"{v0}_prior"])
    p1 = np.array(state[f"{v1}_prior"])
    prior = np.array(state["prior"])
    loglikes = np.array(state["loglikes"])
    t0 = true_values.get(v0)
    t1 = true_values.get(v1)

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("JPDF Prior", fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3, 1],
        height_ratios=[1, 3],
        hspace=0.05,
        wspace=0.05,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    mesh0, mesh1 = np.meshgrid(g0, g1, indexing="ij")
    ax_main.contourf(mesh0, mesh1, prior, levels=20, cmap="viridis")
    ax_main.contour(mesh0, mesh1, loglikes, levels=5, colors="magenta", linewidths=0.8, linestyles="--", alpha=0.6)
    level_95 = _hdr_level(prior, g0, g1, alpha=0.95)
    ax_main.contour(mesh0, mesh1, prior, levels=[level_95], colors="yellow", linewidths=2, linestyles="-")
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], color="yellow", linewidth=2, linestyle="-", label="95% HDR"),
        Line2D([], [], color="white", linewidth=0.8, linestyle="--", label="Prior"),
        Line2D([], [], color="magenta", linewidth=0.8, linestyle="--", label="Log-likelihood"),
    ]
    if t0 is not None:
        ax_main.axvline(t0, color="red", linestyle="--", linewidth=1.5)
    if t1 is not None:
        ax_main.axhline(t1, color="red", linestyle="--", linewidth=1.5)
    if t0 is not None and t1 is not None:
        ax_main.plot(t0, t1, "rx", markersize=15, markeredgewidth=2.5, zorder=5)
        legend_handles.append(Line2D([], [], color="red", marker="x", linestyle="--", linewidth=1.5, markersize=10, label="True"))
    ax_main.legend(handles=legend_handles, loc="upper right", fontsize=8, facecolor="black", framealpha=0.4, labelcolor="white")
    ax_main.set_xlabel(v0)
    ax_main.set_ylabel(v1)
    ax_main.grid(True, alpha=0.3)

    ax_top.fill_between(g0, p0, alpha=0.3, color="b", label="Prior")
    ax_top.plot(g0, p0, "b-", linewidth=1.5)
    if t0 is not None:
        ax_top.axvline(t0, color="red", linestyle="--", linewidth=1.5, label="True")
    ax_top.set_ylabel("Density")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)
    ax_top.tick_params(labelbottom=False)

    ax_right.fill_betweenx(g1, p1, alpha=0.3, color="b")
    ax_right.plot(p1, g1, "b-", linewidth=1.5)
    if t1 is not None:
        ax_right.axhline(t1, color="red", linestyle="--", linewidth=1.5)
    ax_right.set_xlabel("Density")
    ax_right.grid(True, alpha=0.3)
    ax_right.tick_params(labelleft=False)

    fig.add_subplot(gs[0, 1]).set_visible(False)

    plt.close()

    return fig


def save_jpdf_plots(
    results: Dict[float, Dict[str, Any]],
    output_dir: Path,
    var_names: List[str] = None,
    true_values: Dict[str, float] = None,
) -> None:
    """Save JPDF corner plots: one prior-only plot + one per observation time.

    Outputs PNGs to a subdirectory and a collected multi-page PDF.

    Args:
        results: Pipeline results dict (keyed by observation time).
        output_dir: Directory for output files.
        var_names: List of variable names.
        true_values: Dict mapping variable name to true value.
    """

    png_dir = output_dir / "pdfs"
    png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "jpdf.pdf"

    with PdfPages(pdf_path) as pdf:
        first_state = next(iter(results.values()))["jpdf_state"]
        fig_prior = plot_jpdf_prior(
            var_names=var_names,
            state=first_state,
            true_values=true_values,
        )
        fig_prior.savefig(png_dir / "jpdf_prior.png", dpi=150, bbox_inches="tight")
        pdf.savefig(fig_prior)

        for t, data in results.items():
            fig = plot_jpdf_snapshot(
                time=t,
                var_names=var_names,
                state=data["jpdf_state"],
                true_values=true_values,
            )
            fig.savefig(png_dir / f"jpdf_t{t:.0f}.png", dpi=150, bbox_inches="tight")
            pdf.savefig(fig)

    print(f"JPDF PNGs saved to {png_dir}")
    print(f"JPDF PDF  saved to {pdf_path}")


def _pdf_stats(grid: NDArray, pdf: NDArray):
    """Compute mean, 2.5th and 97.5th percentiles from a discrete PDF.

    Args:
        grid: 1D array of grid values.
        pdf: 1D array of PDF values at grid points.

    Returns:
        Tuple of (mean, q025, q975).
    """
    dx = np.diff(grid)
    cdf = np.concatenate([[0], np.cumsum((pdf[:-1] + pdf[1:]) / 2 * dx)])
    cdf /= cdf[-1]
    mean = np.trapezoid(grid * pdf, grid)
    q025 = np.interp(0.025, cdf, grid)
    q975 = np.interp(0.975, cdf, grid)
    return mean, q025, q975


def plot_settlement_forecast(
    time: float,
    forecast_times: NDArray,
    settlement_grids: Dict[float, list],
    settlement_pdfs: Dict[float, list],
    prior_grids: Dict[float, list] = None,
    prior_pdfs: Dict[float, list] = None,
    obs_times: NDArray = None,
    obs_values: NDArray = None,
    obs_error: float = None,
    t_max: float = None,
    y_max: float = None,
) -> plt.Figure:
    """Plot settlement forecast with prior (blue) and posterior (red).

    Distinguishes hindcast (solid lines, up to current time) from forecast
    (dashed lines, after current time). Shows 95% confidence intervals as
    shaded bands and observations as black dots with error bars.

    Args:
        time: Current observation time [days].
        forecast_times: List of all forecast evaluation times.
        settlement_grids: Dict mapping forecast time to settlement grid.
        settlement_pdfs: Dict mapping forecast time to settlement PDF.
        prior_grids, prior_pdfs: Same as above but for the prior.
        obs_times, obs_values: Observation data.
        obs_error: Standard deviation of measurement error [m].
        t_max: Maximum time to display on x-axis.
        y_max: Maximum settlement for y-axis scaling.

    Returns:
        Matplotlib Figure.
    """

    ft_sorted = sorted(forecast_times)
    if t_max is not None:
        ft_sorted = [ft for ft in ft_sorted if ft <= t_max]

    # Prior
    if prior_grids is not None and prior_pdfs is not None:
        means_pr, lo_pr, hi_pr = [], [], []
        for ft in ft_sorted:
            grid = np.array(prior_grids[ft])
            pdf = np.array(prior_pdfs[ft])
            m, q025, q975 = _pdf_stats(grid, pdf)
            means_pr.append(m)
            lo_pr.append(q025)
            hi_pr.append(q975)
        means_pr, lo_pr, hi_pr = np.array(means_pr), np.array(lo_pr), np.array(hi_pr)

    # Posterior
    means, lo, hi = [], [], []
    for ft in ft_sorted:
        grid = np.array(settlement_grids[ft])
        pdf = np.array(settlement_pdfs[ft])
        m, q025, q975 = _pdf_stats(grid, pdf)
        means.append(m)
        lo.append(q025)
        hi.append(q975)
    means, lo, hi = np.array(means), np.array(lo), np.array(hi)

    fig, ax = plt.subplots(figsize=(8, 5))

    ft_arr = np.array(ft_sorted)
    t_split_idx = np.argmin(np.abs(ft_arr - time))
    hind = np.arange(len(ft_arr)) <= t_split_idx
    fore = np.arange(len(ft_arr)) >= t_split_idx

    if prior_grids is not None and prior_pdfs is not None:
        ax.plot(ft_arr[hind], means_pr[hind], "b", linewidth=2, label="Prior hindcast")
        ax.plot(ft_arr[fore], means_pr[fore], "b--", linewidth=2, label="Prior forecast")
        ax.fill_between(ft_arr[hind], lo_pr[hind], hi_pr[hind], alpha=0.15, color="b")
        ax.fill_between(ft_arr[fore], lo_pr[fore], hi_pr[fore], alpha=0.08, color="b")
        ax.plot(ft_arr[fore], lo_pr[fore], "b--", linewidth=0.8, alpha=0.5)
        ax.plot(ft_arr[fore], hi_pr[fore], "b--", linewidth=0.8, alpha=0.5)

    ax.plot(ft_arr[hind], means[hind], "r", linewidth=2, label="Posterior hindcast")
    ax.plot(ft_arr[fore], means[fore], "r--", linewidth=2, label="Posterior forecast")
    ax.fill_between(ft_arr[hind], lo[hind], hi[hind], alpha=0.2, color="r")
    ax.fill_between(ft_arr[fore], lo[fore], hi[fore], alpha=0.1, color="r")
    ax.plot(ft_arr[fore], lo[fore], "r--", linewidth=0.8, alpha=0.5)
    ax.plot(ft_arr[fore], hi[fore], "r--", linewidth=0.8, alpha=0.5)

    ax.axvline(ft_arr[t_split_idx], color="gray", linestyle=":", linewidth=1, alpha=0.7)

    if obs_times is not None and obs_values is not None:
        ci = 1.96 * obs_error if obs_error is not None else None
        ax.errorbar(obs_times, obs_values, yerr=ci, fmt="ko", capsize=4,
                    zorder=5, label="Observations")

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Settlement [m]")
    ax.set_title(f"Settlement Forecast (posterior at t = {time:.0f} days)")
    if t_max is not None:
        ax.set_xlim(0, t_max)
    if y_max is not None:
        ax.set_ylim(0, y_max * 1.05)
    else:
        ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.close()

    return fig


def save_settlement_forecast_plots(
    results: Dict[float, Dict[str, Any]],
    output_dir: Path,
    t_max: float = None,
    obs_error: float = None,
    y_max: float = None,
) -> None:
    """Save settlement forecast plots: one per observation time.

    Args:
        results: Pipeline results dict.
        output_dir: Directory for output files.
        t_max: Maximum time for x-axis.
        obs_error: Observation measurement error for error bars.
        y_max: Maximum settlement for consistent y-axis scaling.
    """

    png_dir = output_dir / "settlement_forecast"
    png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "settlement_forecast.pdf"

    with PdfPages(pdf_path) as pdf:
        for t, data in results.items():
            prior = data["prior"]
            post = data["posterior"]
            fig = plot_settlement_forecast(
                time=t,
                forecast_times=list(post["output_grid"].keys()),
                settlement_grids=post["output_grid"],
                settlement_pdfs=post["output_pdf"],
                prior_grids=prior["output_grid"],
                prior_pdfs=prior["output_pdf"],
                obs_times=np.array(data["obs_times"]),
                obs_values=np.array(data["obs_values"]),
                obs_error=obs_error,
                t_max=t_max,
                y_max=y_max,
            )
            fig.savefig(png_dir / f"settlement_forecast_t{t:.0f}.png", dpi=150, bbox_inches="tight")
            pdf.savefig(fig)

    print(f"Settlement forecast PNGs saved to {png_dir}")
    print(f"Settlement forecast PDF  saved to {pdf_path}")


def plot_settlement_residual(
    time: float,
    prior_grid: NDArray,
    prior_pdf: NDArray,
    posterior_grid: NDArray,
    posterior_pdf: NDArray,
    end_settlement_req: float = None,
    pf_prior: float = None,
    pf_posterior: float = None,
    beta_prior: float = None,
    beta_posterior: float = None,
    y_max: float = None,
) -> plt.Figure:
    """Plot prior and posterior PDFs of residual settlement.

    Shows Gaussian-smoothed PDFs, the allowable settlement requirement line,
    and prior/posterior failure probabilities and reliability indices in the title.

    Args:
        time: Current observation time [days].
        prior_grid, prior_pdf: Prior residual settlement PDF.
        posterior_grid, posterior_pdf: Posterior residual settlement PDF.
        end_settlement_req: Allowable residual settlement [m].
        pf_prior, pf_posterior: Failure probabilities.
        beta_prior, beta_posterior: Reliability indices.
        y_max: Maximum density for consistent y-axis across time steps.

    Returns:
        Matplotlib Figure.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    prior_pdf_smooth = gaussian_filter1d(prior_pdf, sigma=3)
    posterior_pdf_smooth = gaussian_filter1d(posterior_pdf, sigma=3)

    ax.fill_between(prior_grid, prior_pdf_smooth, alpha=0.3, color="b", label="Prior")
    ax.plot(prior_grid, prior_pdf_smooth, "b-", linewidth=1.5)
    ax.fill_between(posterior_grid, posterior_pdf_smooth, alpha=0.3, color="r", label="Posterior")
    ax.plot(posterior_grid, posterior_pdf_smooth, "r-", linewidth=1.5)

    if end_settlement_req is not None:
        ax.axvline(end_settlement_req, color="k", linestyle="--", linewidth=1.5, label=f"Requirement")

    ax.set_xlabel("Residual settlement [m]")
    ax.set_ylabel("Density")

    title = f"Residual Settlement PDF (t = {time:.0f} days)"
    if pf_prior is not None and pf_posterior is not None:
        title += f"\nPrior: Pf={pf_prior:.2e}, \u03b2={beta_prior:.2f}\nPosterior: Pf={pf_posterior:.2e}, \u03b2={beta_posterior:.2f}"
    ax.set_title(title)
    ax.set_xlim(0, 0.65)
    if y_max is not None:
        ax.set_ylim(0, y_max * 1.05)
    else:
        ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.close()

    return fig


def save_settlement_residual_plots(
    results: Dict[float, Dict[str, Any]],
    output_dir: Path,
    end_settlement_req: float = None,
) -> None:
    """Save residual settlement PDF plots with consistent y-axis scaling.

    Computes a global y-max across all time steps (after smoothing) so
    that all frames share the same y-axis range for GIF animation.

    Args:
        results: Pipeline results dict.
        output_dir: Directory for output files.
        end_settlement_req: Allowable residual settlement [m].
    """

    png_dir = output_dir / "settlement_residual"
    png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "settlement_residual.pdf"

    # Compute global y-max across all times (after smoothing)
    y_max = 0
    for data in results.values():
        diff = data["residual"]
        for key in ["prior_pdf", "posterior_pdf"]:
            smoothed = gaussian_filter1d(np.array(diff[key]), sigma=3)
            y_max = max(y_max, smoothed.max())

    with PdfPages(pdf_path) as pdf:
        for t, data in results.items():
            diff = data["residual"]
            fig = plot_settlement_residual(
                time=t,
                prior_grid=np.array(diff["prior_grid"]),
                prior_pdf=np.array(diff["prior_pdf"]),
                posterior_grid=np.array(diff["posterior_grid"]),
                posterior_pdf=np.array(diff["posterior_pdf"]),
                end_settlement_req=end_settlement_req,
                pf_prior=data["prior"]["pf"],
                pf_posterior=data["posterior"]["pf"],
                beta_prior=data["prior"]["beta"],
                beta_posterior=data["posterior"]["beta"],
                y_max=y_max,
            )
            fig.savefig(png_dir / f"settlement_residual_t{t:.0f}.png", dpi=150, bbox_inches="tight")
            pdf.savefig(fig)

    print(f"Residual settlement PNGs saved to {png_dir}")
    print(f"Residual settlement PDF  saved to {pdf_path}")


def plot_beta_over_time(
    results: Dict[float, Dict[str, Any]],
    beta_req: float = None,
) -> plt.Figure:
    """Plot the reliability index (beta) over observation time.

    Shows the prior beta as a horizontal blue line and the posterior beta
    as a red line with dots. Optionally shows the beta requirement.

    Args:
        results: Pipeline results dict.
        beta_req: Required reliability index (shown as dashed black line).

    Returns:
        Matplotlib Figure.
    """

    times = sorted(results.keys())
    beta_prior = results[times[0]]["prior"]["beta"]
    beta_posterior = [results[t]["posterior"]["beta"] for t in times]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(beta_prior, color="b", linewidth=2, label=f"Prior ({beta_prior:.2f})")
    ax.plot(times, beta_posterior, "r-o", linewidth=2, markersize=4, label="Posterior")

    if beta_req is not None:
        ax.axhline(beta_req, color="k", linestyle="--", linewidth=1.5, label=f"Requirement ({beta_req:.1f})")

    ax.set_xlabel("Observation time [days]")
    ax.set_ylabel("\u03b2 [-]")
    ax.set_title("Reliability Index over Time")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.close()

    return fig


def save_beta_over_time_plot(
    results: Dict[float, Dict[str, Any]],
    output_dir: Path,
    beta_req: float = None,
) -> None:
    """Save the beta-over-time plot as a single PNG.

    Args:
        results: Pipeline results dict.
        output_dir: Directory for output file.
        beta_req: Required reliability index.
    """

    fig = plot_beta_over_time(results=results, beta_req=beta_req)
    fig.savefig(output_dir / "beta_over_time.png", dpi=150, bbox_inches="tight")

    print(f"Beta over time saved to {output_dir / 'beta_over_time.png'}")


def plot_jpdf_snapshot_samples(
    time: float,
    var_names: List[str],
    samples: Dict[str, NDArray],
    W_prior: NDArray,
    W_posterior: NDArray,
    true_values: Dict[str, float] = None,
    n_bins: int = 60,
) -> plt.Figure:
    """Corner plot for sample-based JPDF using weighted histograms.

    Center: 2D weighted histogram (posterior). Marginals: 1D weighted
    histograms showing prior (blue) and posterior (red).

    Args:
        time: Current observation time [days].
        var_names: List of two variable names [axis_0, axis_1].
        samples: Dict mapping variable name to 1D sample array.
        W_prior: Normalized prior importance weights.
        W_posterior: Normalized posterior importance weights.
        true_values: Dict mapping variable name to true value.
        n_bins: Number of histogram bins per axis.

    Returns:
        Matplotlib Figure.
    """
    from matplotlib.lines import Line2D
    true_values = true_values or {}
    v0, v1 = var_names[0], var_names[1]
    s0, s1 = samples[v0], samples[v1]
    t0, t1 = true_values.get(v0), true_values.get(v1)

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f"JPDF samples at t = {time:.0f} days", fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3, 1],
        height_ratios=[1, 3],
        hspace=0.05,
        wspace=0.05,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.hist2d(s0, s1, bins=n_bins, weights=W_posterior,
                   cmap="viridis", cmin=1e-30)

    legend_handles = []
    if t0 is not None:
        ax_main.axvline(t0, color="red", linestyle="--", linewidth=1.5)
    if t1 is not None:
        ax_main.axhline(t1, color="red", linestyle="--", linewidth=1.5)
    if t0 is not None and t1 is not None:
        ax_main.plot(t0, t1, "rx", markersize=15, markeredgewidth=2.5, zorder=5)
        legend_handles.append(Line2D([], [], color="red", marker="x", linestyle="--",
                                     linewidth=1.5, markersize=10, label="True"))
    if legend_handles:
        ax_main.legend(handles=legend_handles, loc="upper right", fontsize=8,
                       facecolor="black", framealpha=0.4, labelcolor="white")
    ax_main.set_xlabel(v0)
    ax_main.set_ylabel(v1)
    ax_main.grid(True, alpha=0.3)

    ax_top.hist(s0, bins=n_bins, weights=W_prior, density=True,
                alpha=0.3, color="b", label="Prior")
    ax_top.hist(s0, bins=n_bins, weights=W_posterior, density=True,
                alpha=0.3, color="r", label="Posterior")
    if t0 is not None:
        ax_top.axvline(t0, color="red", linestyle="--", linewidth=1.5)
    ax_top.set_ylabel("Density")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)
    ax_top.tick_params(labelbottom=False)

    ax_right.hist(s1, bins=n_bins, weights=W_prior, density=True,
                  alpha=0.3, color="b", orientation="horizontal")
    ax_right.hist(s1, bins=n_bins, weights=W_posterior, density=True,
                  alpha=0.3, color="r", orientation="horizontal")
    if t1 is not None:
        ax_right.axhline(t1, color="red", linestyle="--", linewidth=1.5)
    ax_right.set_xlabel("Density")
    ax_right.grid(True, alpha=0.3)
    ax_right.tick_params(labelleft=False)

    fig.add_subplot(gs[0, 1]).set_visible(False)

    plt.close()
    return fig


def plot_jpdf_prior_samples(
    var_names: List[str],
    samples: Dict[str, NDArray],
    W_prior: NDArray,
    true_values: Dict[str, float] = None,
    n_bins: int = 60,
) -> plt.Figure:
    """Corner plot for the sample-based prior JPDF.

    Args:
        var_names: List of two variable names [axis_0, axis_1].
        samples: Dict mapping variable name to 1D sample array.
        W_prior: Normalized prior importance weights.
        true_values: Dict mapping variable name to true value.
        n_bins: Number of histogram bins per axis.

    Returns:
        Matplotlib Figure.
    """
    from matplotlib.lines import Line2D
    true_values = true_values or {}
    v0, v1 = var_names[0], var_names[1]
    s0, s1 = samples[v0], samples[v1]
    t0, t1 = true_values.get(v0), true_values.get(v1)

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("JPDF Prior (samples)", fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3, 1],
        height_ratios=[1, 3],
        hspace=0.05,
        wspace=0.05,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.hist2d(s0, s1, bins=n_bins, weights=W_prior,
                   cmap="viridis", cmin=1e-30)

    legend_handles = []
    if t0 is not None:
        ax_main.axvline(t0, color="red", linestyle="--", linewidth=1.5)
    if t1 is not None:
        ax_main.axhline(t1, color="red", linestyle="--", linewidth=1.5)
    if t0 is not None and t1 is not None:
        ax_main.plot(t0, t1, "rx", markersize=15, markeredgewidth=2.5, zorder=5)
        legend_handles.append(Line2D([], [], color="red", marker="x", linestyle="--",
                                     linewidth=1.5, markersize=10, label="True"))
    if legend_handles:
        ax_main.legend(handles=legend_handles, loc="upper right", fontsize=8,
                       facecolor="black", framealpha=0.4, labelcolor="white")
    ax_main.set_xlabel(v0)
    ax_main.set_ylabel(v1)
    ax_main.grid(True, alpha=0.3)

    ax_top.hist(s0, bins=n_bins, weights=W_prior, density=True,
                alpha=0.3, color="b", label="Prior")
    if t0 is not None:
        ax_top.axvline(t0, color="red", linestyle="--", linewidth=1.5)
    ax_top.set_ylabel("Density")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)
    ax_top.tick_params(labelbottom=False)

    ax_right.hist(s1, bins=n_bins, weights=W_prior, density=True,
                  alpha=0.3, color="b", orientation="horizontal")
    if t1 is not None:
        ax_right.axhline(t1, color="red", linestyle="--", linewidth=1.5)
    ax_right.set_xlabel("Density")
    ax_right.grid(True, alpha=0.3)
    ax_right.tick_params(labelleft=False)

    fig.add_subplot(gs[0, 1]).set_visible(False)

    plt.close()
    return fig


def save_jpdf_plots_samples(
    results: Dict[float, Dict[str, Any]],
    output_dir: Path,
    var_names: List[str],
    samples: Dict[str, NDArray],
    W_prior: NDArray,
    W_posterior_per_t: Dict[float, NDArray],
    true_values: Dict[str, float] = None,
) -> None:
    """Save sample-based JPDF corner plots: prior + one per observation time.

    Args:
        results: Pipeline results dict.
        output_dir: Directory for output files.
        var_names: List of variable names.
        samples: Dict mapping variable name to 1D sample array.
        W_prior: Normalized prior weights.
        W_posterior_per_t: Dict mapping obs time to posterior weights.
        true_values: Dict mapping variable name to true value.
    """
    png_dir = output_dir / "pdfs"
    png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "jpdf.pdf"

    with PdfPages(pdf_path) as pdf:
        fig_prior = plot_jpdf_prior_samples(
            var_names=var_names, samples=samples,
            W_prior=W_prior, true_values=true_values,
        )
        fig_prior.savefig(png_dir / "jpdf_prior.png", dpi=150, bbox_inches="tight")
        pdf.savefig(fig_prior)

        for t in results:
            W_post = W_posterior_per_t[t]
            fig = plot_jpdf_snapshot_samples(
                time=t, var_names=var_names, samples=samples,
                W_prior=W_prior, W_posterior=W_post,
                true_values=true_values,
            )
            fig.savefig(png_dir / f"jpdf_t{t:.0f}.png", dpi=150, bbox_inches="tight")
            pdf.savefig(fig)

    print(f"JPDF PNGs saved to {png_dir}")
    print(f"JPDF PDF  saved to {pdf_path}")


def make_gifs(output_dir: Path, duration: int = 500) -> None:
    """Create animated GIFs from PNG sequences in subdirectories.

    Scans each subdirectory of output_dir for PNG files, sorts them by the
    numeric value in the filename (e.g. "t30", "t39"), and assembles them
    into a looping GIF.

    Args:
        output_dir: Parent directory containing PNG subdirectories.
        duration: Frame duration in milliseconds.
    """
    from PIL import Image

    for png_dir in output_dir.iterdir():
        if not png_dir.is_dir():
            continue
        import re
        def _sort_key(p):
            m = re.search(r'(\d+\.?\d*)', p.stem)
            return float(m.group(1)) if m else -1
        pngs = sorted(png_dir.glob("*.png"), key=_sort_key)
        if len(pngs) < 2:
            continue

        frames = [Image.open(p) for p in pngs]
        gif_path = output_dir / f"{png_dir.name}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        print(f"GIF saved to {gif_path}")
