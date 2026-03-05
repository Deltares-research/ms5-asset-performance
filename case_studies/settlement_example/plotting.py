from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray


def plot_jpdf_snapshot(
    time: float,
    CR_grid: NDArray,
    CR_prior: NDArray,
    CR_posterior: NDArray,
    k_grid: NDArray,
    k_prior: NDArray,
    k_posterior: NDArray,
    CR_true: float = None,
    k_true: float = None,
) -> plt.Figure:

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
    CR_mesh, k_mesh = np.meshgrid(CR_grid, k_grid, indexing="ij")
    posterior_2d = CR_posterior[:, np.newaxis] * k_posterior[np.newaxis, :]
    ax_main.contourf(CR_mesh, k_mesh, posterior_2d, levels=20, cmap="viridis")
    if CR_true is not None:
        ax_main.axvline(CR_true, color="red", linestyle="--", linewidth=1.5)
    if k_true is not None:
        ax_main.axhline(k_true, color="red", linestyle="--", linewidth=1.5)
    if CR_true is not None and k_true is not None:
        ax_main.plot(CR_true, k_true, "r+", markersize=15, markeredgewidth=2.5, zorder=5)
    ax_main.set_xlabel("CR [-]")
    ax_main.set_ylabel("k [-]")
    ax_main.grid(True, alpha=0.3)

    # CR marginal (top, aligned to x-axis)
    ax_top.fill_between(CR_grid, CR_prior, alpha=0.3, color="b", label="Prior")
    ax_top.plot(CR_grid, CR_prior, "b-", linewidth=1.5)
    ax_top.fill_between(CR_grid, CR_posterior, alpha=0.3, color="r", label="Posterior")
    ax_top.plot(CR_grid, CR_posterior, "r-", linewidth=1.5)
    if CR_true is not None:
        ax_top.axvline(CR_true, color="red", linestyle="--", linewidth=1.5, label=f"True ({CR_true})")
    ax_top.set_ylabel("Density")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)
    ax_top.tick_params(labelbottom=False)

    # k marginal (right, aligned to y-axis)
    ax_right.fill_betweenx(k_grid, k_prior, alpha=0.3, color="b")
    ax_right.plot(k_prior, k_grid, "b-", linewidth=1.5)
    ax_right.fill_betweenx(k_grid, k_posterior, alpha=0.3, color="r")
    ax_right.plot(k_posterior, k_grid, "r-", linewidth=1.5)
    if k_true is not None:
        ax_right.axhline(k_true, color="red", linestyle="--", linewidth=1.5)
    ax_right.set_xlabel("Density")
    ax_right.grid(True, alpha=0.3)
    ax_right.tick_params(labelleft=False)

    # Hide unused corner
    fig.add_subplot(gs[0, 1]).set_visible(False)

    plt.close()

    return fig


def save_jpdf_plots(
    results: Dict[float, Dict[str, Any]],
    output_dir: Path,
    CR_true: float = None,
    k_true: float = None,
) -> None:

    png_dir = output_dir / "pdfs"
    png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "jpdf.pdf"

    with PdfPages(pdf_path) as pdf:
        for t, data in results.items():
            state = data["jpdf_state"]
            fig = plot_jpdf_snapshot(
                time=t,
                CR_grid=np.array(state["CR_grid"]),
                CR_prior=np.array(state["CR_prior"]),
                CR_posterior=np.array(state["CR_posterior"]),
                k_grid=np.array(state["k_grid"]),
                k_prior=np.array(state["k_prior"]),
                k_posterior=np.array(state["k_posterior"]),
                CR_true=CR_true,
                k_true=k_true,
            )
            fig.savefig(png_dir / f"jpdf_t{t:.0f}.png", dpi=150, bbox_inches="tight")
            pdf.savefig(fig)

    print(f"JPDF PNGs saved to {png_dir}")
    print(f"JPDF PDF  saved to {pdf_path}")


def _pdf_stats(grid: NDArray, pdf: NDArray):
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
    obs_times: NDArray = None,
    obs_values: NDArray = None,
    obs_error: float = None,
    t_max: float = None,
) -> plt.Figure:

    ft_sorted = sorted(forecast_times)
    if t_max is not None:
        ft_sorted = [ft for ft in ft_sorted if ft <= t_max]
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
    ax.plot(ft_sorted, means, "r-", linewidth=2, label="Mean")
    ax.fill_between(ft_sorted, lo, hi, alpha=0.2, color="r", label="95% CI")

    if obs_times is not None and obs_values is not None:
        ci = 1.96 * obs_error if obs_error is not None else None
        ax.errorbar(obs_times, obs_values, yerr=ci, fmt="ko", capsize=4,
                    zorder=5, label="Observations")

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Settlement [mm]")
    ax.set_title(f"Settlement Forecast (posterior at t = {time:.0f} days)")
    if t_max is not None:
        ax.set_xlim(0, t_max)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.close()

    return fig


def save_settlement_forecast_plots(
    results: Dict[float, Dict[str, Any]],
    output_dir: Path,
    t_max: float = None,
    obs_error: float = None,
) -> None:

    png_dir = output_dir / "settlement_forecast"
    png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "settlement_forecast.pdf"

    with PdfPages(pdf_path) as pdf:
        for t, data in results.items():
            post = data["posterior"]
            fig = plot_settlement_forecast(
                time=t,
                forecast_times=list(post["settlement_posterior_grid"].keys()),
                settlement_grids=post["settlement_posterior_grid"],
                settlement_pdfs=post["settlement_forecast"],
                obs_times=np.array(data["obs_times"]),
                obs_values=np.array(data["settlement_obs"]),
                obs_error=obs_error,
                t_max=t_max,
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
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(prior_grid, prior_pdf, alpha=0.3, color="b", label="Prior")
    ax.plot(prior_grid, prior_pdf, "b-", linewidth=1.5)
    ax.fill_between(posterior_grid, posterior_pdf, alpha=0.3, color="r", label="Posterior")
    ax.plot(posterior_grid, posterior_pdf, "r-", linewidth=1.5)

    if end_settlement_req is not None:
        ax.axvline(end_settlement_req, color="k", linestyle="--", linewidth=1.5,
                   label=f"Requirement ({end_settlement_req})")

    ax.set_xlabel("End differential settlement [mm]")
    ax.set_ylabel("Density")

    title = f"End Differential Settlement PDF (t = {time:.0f} days)"
    if pf_prior is not None and pf_posterior is not None:
        title += f"\nPrior: Pf={pf_prior:.2e}, \u03b2={beta_prior:.2f}  |  Posterior: Pf={pf_posterior:.2e}, \u03b2={beta_posterior:.2f}"
    ax.set_title(title)
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

    png_dir = output_dir / "settlement_residual"
    png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "settlement_residual.pdf"

    with PdfPages(pdf_path) as pdf:
        for t, data in results.items():
            diff = data["settlement_residual"]
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
            )
            fig.savefig(png_dir / f"settlement_residual_t{t:.0f}.png", dpi=150, bbox_inches="tight")
            pdf.savefig(fig)

    print(f"End diff settlement PNGs saved to {png_dir}")
    print(f"End diff settlement PDF  saved to {pdf_path}")
