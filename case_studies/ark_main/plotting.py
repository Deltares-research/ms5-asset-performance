"""
Plotting functions for the D-SheetPiling reliability analysis.

Same style as ark_example but without survived moment logic.
"""

from itertools import chain
from pathlib import Path
from typing import Dict, Any, Sequence, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid

from src.plotting import save_figure, collect_pngs_to_pdf, make_gifs


# ---------------------------------------------------------------------------
# Beta forecast
# ---------------------------------------------------------------------------

def plot_beta_forecast_at_time(
    current_time: float,
    results: dict,
    beta_req: float = 2.3,
) -> plt.Figure:
    """Plot beta forecasts at a specific observation time.

    Shows:
    - Blue solid: Prior forecast
    - Red dots: Posterior beta at each observation time
    - Red dotted: Past posterior forecasts
    - Red dashed: Current posterior forecast
    - Red solid: Hindcast connecting posterior betas
    - Black dashed: Requirement
    """
    min_beta = 1.0
    max_beta = 3.5

    obs_times = sorted(results.keys())
    first_result = results[obs_times[0]]
    all_ft = sorted(first_result["prior"]["beta_forecast"].keys())
    t_min, t_max = min(all_ft), max(all_ft)

    fig = plt.figure(figsize=(8, 4))

    # Prior forecast
    bf_prior = first_result["prior"]["beta_forecast"]
    ft_pr = sorted(bf_prior.keys())
    plt.plot(ft_pr, [min(bf_prior[t], max_beta) for t in ft_pr], c="b", label="Prior")

    # Hindcast: solid red traces the prev-obs posterior forecast curve from
    # prev_t to curr_t (every forecast point in between), then jumps to the
    # posterior beta at curr_t. Tracing the curve (instead of a chord) makes
    # the solid line coincide with the dotted forecast for prev_t between
    # obs points.
    hindcast_segments = []
    for prev_t, curr_t in zip(obs_times[:-1], obs_times[1:]):
        bf = results[prev_t]["posterior"]["beta_forecast"]
        curr_beta = min(results[curr_t]["posterior"]["beta"], max_beta)
        seg_times = [t for t in sorted(bf.keys()) if prev_t <= t <= curr_t]
        if not seg_times:
            continue
        seg = [(t, min(bf[t], max_beta)) for t in seg_times]
        seg.append((curr_t, curr_beta))  # vertical jump at curr_t
        hindcast_segments.append(seg)

    if hindcast_segments:
        pts = list(chain.from_iterable(hindcast_segments))
        plt.plot([p[0] for p in pts], [p[1] for p in pts], c="r")

    # Posterior dots and forecast lines
    for i, obs_t in enumerate(obs_times):
        result = results[obs_t]
        beta_cur = min(result["posterior"]["beta"], max_beta)
        plt.scatter(obs_t, beta_cur, color="r", zorder=5,
                    label="Posterior" if i == 0 else None)

        bf = result["posterior"]["beta_forecast"]
        ft = sorted(bf.keys())
        betas = [min(bf[t], max_beta) for t in ft]
        is_current = (obs_t == current_time)
        plt.plot(ft, betas, c="r",
                 linestyle="--" if is_current else "dotted",
                 alpha=1.0 if is_current else 0.4)

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


# ---------------------------------------------------------------------------
# Corrosion forecast
# ---------------------------------------------------------------------------

def plot_corrosion_forecast_at_time(
    cr_grid: Sequence[float],
    cr_forecast_prior: Dict[float, Sequence[float]],
    cr_forecast_posterior: Dict[float, Sequence[float]],
    obs_times: Sequence[float] = None,
    obs_values: Sequence[float] = None,
    obs_error_std: float = 0.4,
    alpha: float = 0.05,
    wall_thickness: float = 9.5,
    xlim: tuple = (50, 80),
    ylim: tuple = (0, 9.5),
) -> plt.Figure:
    """Plot corrosion forecast with prior/posterior bands and observations."""
    cr_grid = np.asarray(cr_grid)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Prior
    cr_pr = np.vstack(list(cr_forecast_prior.values()))
    cr_pr_mean = np.trapezoid(cr_pr * cr_grid[None, :], cr_grid, axis=1)
    cr_pr_cdf = cumulative_trapezoid(cr_pr, cr_grid)
    cr_pr_q05 = cr_grid[np.argmin(np.abs(cr_pr_cdf - alpha), axis=1)]
    cr_pr_q95 = cr_grid[np.argmin(np.abs(cr_pr_cdf - (1 - alpha)), axis=1)]

    times_pr = np.asarray(list(cr_forecast_prior.keys()))
    ax.fill_between(times_pr, cr_pr_q05 * wall_thickness, cr_pr_q95 * wall_thickness,
                    color="b", alpha=0.15)
    ax.plot(times_pr, cr_pr_q05 * wall_thickness, color="b", linewidth=0.5)
    ax.plot(times_pr, cr_pr_q95 * wall_thickness, color="b", linewidth=0.5)
    ax.plot(times_pr, cr_pr_mean * wall_thickness, color="b", linewidth=1.5, label="Prior")

    # Posterior
    cr_po = np.vstack(list(cr_forecast_posterior.values()))
    cr_po_mean = np.trapezoid(cr_po * cr_grid[None, :], cr_grid, axis=1)
    cr_po_cdf = cumulative_trapezoid(cr_po, cr_grid)
    cr_po_q05 = cr_grid[np.argmin(np.abs(cr_po_cdf - alpha), axis=1)]
    cr_po_q95 = cr_grid[np.argmin(np.abs(cr_po_cdf - (1 - alpha)), axis=1)]

    times_po = np.asarray(list(cr_forecast_posterior.keys()))
    ax.fill_between(times_po, cr_po_q05 * wall_thickness, cr_po_q95 * wall_thickness,
                    color="r", alpha=0.15)
    ax.plot(times_po, cr_po_q05 * wall_thickness, color="r", linewidth=0.5)
    ax.plot(times_po, cr_po_q95 * wall_thickness, color="r", linewidth=0.5)
    ax.plot(times_po, cr_po_mean * wall_thickness, color="r", linewidth=1.5, label="Posterior")

    # Observations
    if obs_times is not None and obs_values is not None:
        yerr = obs_error_std * norm.ppf(1 - alpha)
        ax.errorbar(x=obs_times, y=obs_values, yerr=yerr, fmt="o", c="k",
                    capsize=3, zorder=5, label="Observations")

    ax.set_xlabel("Forecast time [yr]", fontsize=12)
    ax.set_ylabel("Corrosion [mm]", fontsize=12)
    ax.set_title(f"Corrosion Forecast at t_obs = {times_po.min():.0f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()
    return fig


# ---------------------------------------------------------------------------
# Moment capacity forecast
# ---------------------------------------------------------------------------

def plot_moment_forecast_at_time(
    cr_grid: Sequence[float],
    cr_forecast_prior: Dict[float, Sequence[float]],
    cr_forecast_posterior: Dict[float, Sequence[float]],
    wall_moment_capacity: float = 750.0,
    obs_times: Sequence[float] = None,
    obs_moment_cap: Sequence[float] = None,
    alpha: float = 0.05,
    xlim: tuple = (50, 80),
    ylim: tuple = (0, 800),
) -> plt.Figure:
    """Plot moment capacity forecast derived from corrosion ratio PDFs.

    moment = wall_moment_capacity * (1 - cr). No survived moment logic.
    """
    cr_grid = np.asarray(cr_grid)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Prior
    cr_pr = np.vstack(list(cr_forecast_prior.values()))
    cr_pr_mean = np.trapezoid(cr_pr * cr_grid[None, :], cr_grid, axis=1)
    cr_pr_cdf = cumulative_trapezoid(cr_pr, cr_grid)
    cr_pr_q05 = cr_grid[np.argmin(np.abs(cr_pr_cdf - alpha), axis=1)]
    cr_pr_q95 = cr_grid[np.argmin(np.abs(cr_pr_cdf - (1 - alpha)), axis=1)]

    m_pr_mean = wall_moment_capacity * (1 - cr_pr_mean)
    m_pr_q95 = wall_moment_capacity * (1 - cr_pr_q05)  # low cr → high moment
    m_pr_q05 = wall_moment_capacity * (1 - cr_pr_q95)

    times_pr = np.asarray(list(cr_forecast_prior.keys()))
    ax.fill_between(times_pr, m_pr_q05, m_pr_q95, color="b", alpha=0.15)
    ax.plot(times_pr, m_pr_q05, color="b", linewidth=0.5)
    ax.plot(times_pr, m_pr_q95, color="b", linewidth=0.5)
    ax.plot(times_pr, m_pr_mean, color="b", linewidth=1.5, label="Prior")

    # Posterior
    cr_po = np.vstack(list(cr_forecast_posterior.values()))
    cr_po_mean = np.trapezoid(cr_po * cr_grid[None, :], cr_grid, axis=1)
    cr_po_cdf = cumulative_trapezoid(cr_po, cr_grid)
    cr_po_q05 = cr_grid[np.argmin(np.abs(cr_po_cdf - alpha), axis=1)]
    cr_po_q95 = cr_grid[np.argmin(np.abs(cr_po_cdf - (1 - alpha)), axis=1)]

    m_po_mean = wall_moment_capacity * (1 - cr_po_mean)
    m_po_q95 = wall_moment_capacity * (1 - cr_po_q05)
    m_po_q05 = wall_moment_capacity * (1 - cr_po_q95)

    times_po = np.asarray(list(cr_forecast_posterior.keys()))
    ax.fill_between(times_po, m_po_q05, m_po_q95, color="r", alpha=0.15)
    ax.plot(times_po, m_po_q05, color="r", linewidth=0.5)
    ax.plot(times_po, m_po_q95, color="r", linewidth=0.5)
    ax.plot(times_po, m_po_mean, color="r", linewidth=1.5, label="Posterior")

    # Observations
    if obs_times is not None and obs_moment_cap is not None:
        ax.scatter(obs_times, obs_moment_cap, color="k", s=50, zorder=5, label="Observations")

    ax.set_xlabel("Forecast time [yr]", fontsize=12)
    ax.set_ylabel("Moment capacity [kNm]", fontsize=12)
    ax.set_title(f"Moment Capacity Forecast at t_obs = {times_po.min():.0f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()
    return fig


# ---------------------------------------------------------------------------
# End of life
# ---------------------------------------------------------------------------

def plot_end_of_life(
    results: dict,
    beta_req: float = 2.3,
    t_start: float = 50.0,
    forecast_key: str = "posterior",
) -> plt.Figure:
    """Bar chart of end-of-life with delta-EOL on secondary axis.

    EOL = first forecast time where beta < beta_req (linear interpolation).
    """
    obs_times = sorted(results.keys())

    eol_values = []
    for t in obs_times:
        bf = results[t][forecast_key]["beta_forecast"]
        ft = sorted(bf.keys())
        betas = [bf[f] for f in ft]

        eol = ft[-1]
        for k in range(len(betas) - 1):
            if betas[k] >= beta_req and betas[k + 1] < beta_req:
                t0, t1 = ft[k], ft[k + 1]
                b0, b1 = betas[k], betas[k + 1]
                eol = t0 + (beta_req - b0) / (b1 - b0) * (t1 - t0)
                break
            elif betas[k] < beta_req:
                eol = ft[k]
                break
        eol_values.append(eol)

    eol_values = np.array(eol_values)
    delta_eol = np.diff(eol_values)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    bar_labels = [f"{t:.0f}" for t in obs_times]
    bars = ax1.bar(bar_labels, eol_values - t_start, bottom=t_start,
                   color="steelblue", edgecolor="k", linewidth=0.5, label="End of life")
    ax1.set_ylabel("End of life [yr]", fontsize=12, color="steelblue")
    ax1.set_ylim(t_start, max(eol_values) * 1.05)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_xlabel("Observation time [yr]", fontsize=12)

    for bar, eol in zip(bars, eol_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, eol + 0.3,
                 f"{eol:.1f}", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    x_delta = np.arange(1, len(obs_times))
    ax2.plot(x_delta, delta_eol, "o-", color="darkorange", linewidth=2,
             markersize=6, label=r"$\Delta$ EOL")
    ax2.axhline(0, color="darkorange", linewidth=0.5, alpha=0.5)
    ax2.set_ylabel(r"$\Delta$ EOL [yr]", fontsize=12, color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")

    fig.suptitle("End-of-Life Estimate per Observation Time",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.close()
    return fig


# ---------------------------------------------------------------------------
# Orchestration: save all plots for a pipeline run
# ---------------------------------------------------------------------------

def save_all_plots(
    results: Dict[float, Dict[str, Any]],
    data: Dict[str, Any],
    config: Dict[str, Any],
    corrosion_model,
    jpdf,
    forecast_times: Sequence[float],
    n_cr_grid: int,
    output_dir: Path,
) -> None:
    """Generate and save all plots for a pipeline run.

    Args:
        results: Pipeline results dict keyed by observation time.
        data: Observation data dict keyed by time string.
        config: Parameters dict from settings.
        corrosion_model: CorrosionModel instance (for CR PDF forecasts).
        jpdf: JPDF instance (for C50 prior/posterior).
        forecast_times: Array of forecast times.
        n_cr_grid: Number of corrosion ratio grid points.
        output_dir: Root output directory (plots go into output_dir/plots/).
    """
    plot_dir = Path(output_dir) / "plots"
    times = sorted(results.keys())

    # Extract observations
    obs_times_list, obs_corrosion_list = [], []
    for t in times:
        key = str(t) if str(t) in data else f"{t:.1f}"
        if key in data and "corrosion" in data[key]:
            obs_times_list.append(t)
            obs_corrosion_list.append(data[key]["corrosion"])

    t_first, t_last = times[0], times[-1]
    wall_moment_capacity = config["wall_moment_capacity"]
    wall_thickness = config["wall_thickness"]
    obs_moment_cap = [wall_moment_capacity * (1 - c / wall_thickness) for c in obs_corrosion_list]
    cr_grid_plot = np.linspace(0.0, 1.0, n_cr_grid)

    def build_cr_forecasts(t_obs):
        key = str(t_obs) if str(t_obs) in data else f"{t_obs:.1f}"
        corr_obs = data[key]["corrosion"] if key in data else None
        future = [ft for ft in forecast_times if ft >= t_obs]
        pr, po = {}, {}
        for ft in future:
            g, p = corrosion_model.corrosion_ratio_pdf(t=ft, param_pdf=jpdf.param_prior)
            pr[ft] = np.interp(cr_grid_plot, g, p, left=0, right=0).tolist()
            g, p = corrosion_model.corrosion_ratio_pdf(
                t=ft, param_pdf=jpdf.param_pdf, last_obs_time=t_obs, last_obs=corr_obs,
            )
            po[ft] = np.interp(cr_grid_plot, g, p, left=0, right=0).tolist()
        return pr, po

    # Beta forecast
    png_dir = plot_dir / "beta_forecast"
    png_dir.mkdir(parents=True, exist_ok=True)
    for t in times:
        results_up_to_t = {k: v for k, v in results.items() if k <= t}
        fig = plot_beta_forecast_at_time(
            current_time=t, results=results_up_to_t, beta_req=config.get("beta_req"),
        )
        save_figure(fig, png_dir / f"beta_forecast_t{int(t):03d}.png")
    collect_pngs_to_pdf(png_dir, plot_dir / "beta_forecast.pdf")
    print(f"Beta forecast plots saved to {png_dir}")

    # Corrosion forecast
    png_dir = plot_dir / "corrosion"
    png_dir.mkdir(parents=True, exist_ok=True)
    for current_t in obs_times_list:
        cr_pr, cr_po = build_cr_forecasts(current_t)
        fig = plot_corrosion_forecast_at_time(
            cr_grid=cr_grid_plot, cr_forecast_prior=cr_pr, cr_forecast_posterior=cr_po,
            obs_times=[t for t in obs_times_list if t <= current_t],
            obs_values=[c for t, c in zip(obs_times_list, obs_corrosion_list) if t <= current_t],
            obs_error_std=config["obs_error_std"], wall_thickness=wall_thickness,
            xlim=(t_first, t_last), ylim=(0, wall_thickness),
        )
        save_figure(fig, png_dir / f"corrosion_t{int(current_t):03d}.png")
    collect_pngs_to_pdf(png_dir, plot_dir / "corrosion.pdf")
    print(f"Corrosion forecast plots saved to {png_dir}")

    # Moment capacity forecast
    png_dir = plot_dir / "moment"
    png_dir.mkdir(parents=True, exist_ok=True)
    for current_t in obs_times_list:
        cr_pr, cr_po = build_cr_forecasts(current_t)
        fig = plot_moment_forecast_at_time(
            cr_grid=cr_grid_plot, cr_forecast_prior=cr_pr, cr_forecast_posterior=cr_po,
            wall_moment_capacity=wall_moment_capacity,
            obs_times=[t for t in obs_times_list if t <= current_t],
            obs_moment_cap=[m for t, m in zip(obs_times_list, obs_moment_cap) if t <= current_t],
            xlim=(t_first, t_last), ylim=(0, wall_moment_capacity * 1.1),
        )
        save_figure(fig, png_dir / f"moment_t{int(current_t):03d}.png")
    collect_pngs_to_pdf(png_dir, plot_dir / "moment.pdf")
    print(f"Moment forecast plots saved to {png_dir}")

    # End of life
    png_dir = plot_dir / "end_of_life"
    png_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_end_of_life(
        results=results, beta_req=config.get("beta_req", 2.3), t_start=config.get("t_start", times[0]),
    )
    save_figure(fig, png_dir / "end_of_life.png")
    collect_pngs_to_pdf(png_dir, plot_dir / "end_of_life.pdf")
    print(f"End-of-life plot saved to {png_dir}")

    # GIFs
    make_gifs(plot_dir)
