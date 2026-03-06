from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


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
    prior_2d = CR_prior[:, np.newaxis] * k_prior[np.newaxis, :]
    posterior_2d = CR_posterior[:, np.newaxis] * k_posterior[np.newaxis, :]
    ax_main.contourf(CR_mesh, k_mesh, posterior_2d, levels=20, cmap="viridis")
    ax_main.contour(CR_mesh, k_mesh, prior_2d, levels=5, colors="white", linewidths=0.8, linestyles="--", alpha=0.6)
    if CR_true is not None:
        ax_main.axvline(CR_true, color="red", linestyle="--", linewidth=1.5)
    if k_true is not None:
        ax_main.axhline(k_true, color="red", linestyle="--", linewidth=1.5)
    if CR_true is not None and k_true is not None:
        ax_main.plot(CR_true, k_true, "rx", markersize=15, markeredgewidth=2.5, zorder=5)
    ax_main.set_xlabel("CR [-]")
    ax_main.set_ylabel("k [m/d]")
    ax_main.grid(True, alpha=0.3)

    # CR marginal (top, aligned to x-axis)
    ax_top.fill_between(CR_grid, CR_prior, alpha=0.3, color="b", label="Prior")
    ax_top.plot(CR_grid, CR_prior, "b-", linewidth=1.5)
    ax_top.fill_between(CR_grid, CR_posterior, alpha=0.3, color="r", label="Posterior")
    ax_top.plot(CR_grid, CR_posterior, "r-", linewidth=1.5)
    if CR_true is not None:
        ax_top.axvline(CR_true, color="red", linestyle="--", linewidth=1.5, label=f"True")
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


def plot_jpdf_prior(
    CR_grid: NDArray,
    CR_prior: NDArray,
    k_grid: NDArray,
    k_prior: NDArray,
    CR_true: float = None,
    k_true: float = None,
) -> plt.Figure:

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

    # Bivariate contour (prior)
    CR_mesh, k_mesh = np.meshgrid(CR_grid, k_grid, indexing="ij")
    prior_2d = CR_prior[:, np.newaxis] * k_prior[np.newaxis, :]
    ax_main.contourf(CR_mesh, k_mesh, prior_2d, levels=20, cmap="viridis")
    if CR_true is not None:
        ax_main.axvline(CR_true, color="red", linestyle="--", linewidth=1.5)
    if k_true is not None:
        ax_main.axhline(k_true, color="red", linestyle="--", linewidth=1.5)
    if CR_true is not None and k_true is not None:
        ax_main.plot(CR_true, k_true, "rx", markersize=15, markeredgewidth=2.5, zorder=5)
    ax_main.set_xlabel("CR [-]")
    ax_main.set_ylabel("k [m/d]")
    ax_main.grid(True, alpha=0.3)

    # CR marginal (top)
    ax_top.fill_between(CR_grid, CR_prior, alpha=0.3, color="b", label="Prior")
    ax_top.plot(CR_grid, CR_prior, "b-", linewidth=1.5)
    if CR_true is not None:
        ax_top.axvline(CR_true, color="red", linestyle="--", linewidth=1.5, label="True")
    ax_top.set_ylabel("Density")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)
    ax_top.tick_params(labelbottom=False)

    # k marginal (right)
    ax_right.fill_betweenx(k_grid, k_prior, alpha=0.3, color="b")
    ax_right.plot(k_prior, k_grid, "b-", linewidth=1.5)
    if k_true is not None:
        ax_right.axhline(k_true, color="red", linestyle="--", linewidth=1.5)
    ax_right.set_xlabel("Density")
    ax_right.grid(True, alpha=0.3)
    ax_right.tick_params(labelleft=False)

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
        # Prior-only plot (use first time step's state for grid/prior)
        first_state = next(iter(results.values()))["jpdf_state"]
        fig_prior = plot_jpdf_prior(
            CR_grid=np.array(first_state["CR_grid"]),
            CR_prior=np.array(first_state["CR_prior"]),
            k_grid=np.array(first_state["k_grid"]),
            k_prior=np.array(first_state["k_prior"]),
            CR_true=CR_true,
            k_true=k_true,
        )
        fig_prior.savefig(png_dir / "jpdf_prior.png", dpi=150, bbox_inches="tight")
        pdf.savefig(fig_prior)

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
    prior_grids: Dict[float, list] = None,
    prior_pdfs: Dict[float, list] = None,
    obs_times: NDArray = None,
    obs_values: NDArray = None,
    obs_error: float = None,
    t_max: float = None,
    y_max: float = None,
) -> plt.Figure:

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

    png_dir = output_dir / "settlement_forecast"
    png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "settlement_forecast.pdf"

    with PdfPages(pdf_path) as pdf:
        for t, data in results.items():
            prior = data["prior"]
            post = data["posterior"]
            fig = plot_settlement_forecast(
                time=t,
                forecast_times=list(post["settlement_posterior_grid"].keys()),
                settlement_grids=post["settlement_posterior_grid"],
                settlement_pdfs=post["settlement_forecast"],
                prior_grids=prior["settlement_prior_grid"],
                prior_pdfs=prior["settlement_forecast"],
                obs_times=np.array(data["obs_times"]),
                obs_values=np.array(data["settlement_obs"]),
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

    fig, ax = plt.subplots(figsize=(8, 5))

    prior_pdf_smooth = gaussian_filter1d(prior_pdf, sigma=3)
    posterior_pdf_smooth = gaussian_filter1d(posterior_pdf, sigma=3)

    ax.fill_between(prior_grid, prior_pdf_smooth, alpha=0.3, color="b", label="Prior")
    ax.plot(prior_grid, prior_pdf_smooth, "b-", linewidth=1.5)
    ax.fill_between(posterior_grid, posterior_pdf_smooth, alpha=0.3, color="r", label="Posterior")
    ax.plot(posterior_grid, posterior_pdf_smooth, "r-", linewidth=1.5)

    if end_settlement_req is not None:
        ax.axvline(end_settlement_req, color="k", linestyle="--", linewidth=1.5, label=f"Requirement")

    ax.set_xlabel("End differential settlement [m]")
    ax.set_ylabel("Density")

    title = f"End Differential Settlement PDF (t = {time:.0f} days)"
    if pf_prior is not None and pf_posterior is not None:
        title += f"""
        \nPrior: Pf={pf_prior:.2e}, \u03b2={beta_prior:.2f}  |  
        Posterior: Pf={pf_posterior:.2e}, \u03b2={beta_posterior:.2f}
        """
    ax.set_title(title)
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

    png_dir = output_dir / "settlement_residual"
    png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "settlement_residual.pdf"

    # Compute global y-max across all times (after smoothing)
    y_max = 0
    for data in results.values():
        diff = data["settlement_residual"]
        for key in ["prior_pdf", "posterior_pdf"]:
            smoothed = gaussian_filter1d(np.array(diff[key]), sigma=3)
            y_max = max(y_max, smoothed.max())

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

    fig = plot_beta_over_time(results=results, beta_req=beta_req)
    fig.savefig(output_dir / "beta_over_time.png", dpi=150, bbox_inches="tight")

    print(f"Beta over time saved to {output_dir / 'beta_over_time.png'}")


def make_gifs(output_dir: Path, duration: int = 500) -> None:
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
