"""
Universal JPDF corner plots.

Provides a single ``plot_jpdf`` function that renders either:
- Grid-based: contour plot of the 2D joint PDF with prior/posterior/loglike overlays
- Sample-based: 2D weighted histogram

Both modes show marginal PDFs on the top and right panels. The two
variables to plot are selected via ``var_names`` (any pair from the JPDF).
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray

from .utils import pdf_stats, hdr_level


def plot_jpdf(
    var_names: List[str],
    state: Dict[str, Any],
    mode: str = "grid",
    samples: Optional[Dict[str, NDArray]] = None,
    W_prior: Optional[NDArray] = None,
    W_posterior: Optional[NDArray] = None,
    true_values: Optional[Dict[str, float]] = None,
    title: Optional[str] = None,
    n_bins: int = 60,
) -> plt.Figure:
    """Universal JPDF corner plot for two selected variables.

    Args:
        var_names: Two variable names to plot [x_axis, y_axis].
        state: JPDF state dict with keys "{name}_grid", "{name}_prior",
            "{name}_posterior". Grid mode also needs "prior", "posterior",
            "loglikes" (2D arrays).
        mode: "grid" for contour plot, "samples" for weighted histogram.
        samples: Dict mapping variable name to 1D sample array
            (required for sample mode).
        W_prior: Normalized prior weights (sample mode).
        W_posterior: Normalized posterior weights (sample mode).
        true_values: Dict mapping variable name to true value.
        title: Figure title. Auto-generated if None.
        n_bins: Number of histogram bins (sample mode only).

    Returns:
        Matplotlib Figure.
    """
    true_values = true_values or {}
    v0, v1 = var_names[0], var_names[1]
    t0 = true_values.get(v0)
    t1 = true_values.get(v1)

    fig = plt.figure(figsize=(8, 8))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

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

    if mode == "grid":
        _plot_grid_center(ax_main, v0, v1, state, t0, t1)
        _plot_grid_marginals(ax_top, ax_right, v0, v1, state, t0, t1)
    elif mode == "samples":
        _plot_sample_center(ax_main, v0, v1, samples, W_posterior, t0, t1, n_bins)
        _plot_sample_marginals(ax_top, ax_right, v0, v1, samples, W_prior, W_posterior, t0, t1, n_bins)

    ax_main.set_xlabel(v0)
    ax_main.set_ylabel(v1)
    ax_main.grid(True, alpha=0.3)
    ax_top.set_ylabel("Density")
    ax_top.grid(True, alpha=0.3)
    ax_top.tick_params(labelbottom=False)
    ax_right.set_xlabel("Density")
    ax_right.grid(True, alpha=0.3)
    ax_right.tick_params(labelleft=False)

    fig.add_subplot(gs[0, 1]).set_visible(False)
    plt.close()
    return fig


# ---------------------------------------------------------------------------
# Grid-based internals
# ---------------------------------------------------------------------------

def _plot_grid_center(ax, v0, v1, state, t0, t1):
    g0 = np.array(state[f"{v0}_grid"])
    g1 = np.array(state[f"{v1}_grid"])
    posterior = np.array(state["posterior"])
    prior = np.array(state["prior"])
    loglikes = np.array(state["loglikes"])

    mesh0, mesh1 = np.meshgrid(g0, g1, indexing="ij")
    ax.contourf(mesh0, mesh1, posterior, levels=20, cmap="viridis")
    ax.contour(mesh0, mesh1, prior, levels=5, colors="white",
               linewidths=0.8, linestyles="--", alpha=0.6)
    ax.contour(mesh0, mesh1, loglikes, levels=5, colors="magenta",
               linewidths=0.8, linestyles="--", alpha=0.6)

    level_95 = hdr_level(posterior, g0, g1, alpha=0.95)
    ax.contour(mesh0, mesh1, posterior, levels=[level_95],
               colors="yellow", linewidths=2, linestyles="-")

    handles = [
        Line2D([], [], color="yellow", lw=2, ls="-", label="95% HDR"),
        Line2D([], [], color="white", lw=0.8, ls="--", label="Prior"),
        Line2D([], [], color="magenta", lw=0.8, ls="--", label="Log-likelihood"),
    ]
    _add_true_markers(ax, handles, t0, t1)
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              facecolor="black", framealpha=0.4, labelcolor="white")


def _plot_grid_marginals(ax_top, ax_right, v0, v1, state, t0, t1):
    g0 = np.array(state[f"{v0}_grid"])
    g1 = np.array(state[f"{v1}_grid"])
    p0_prior = np.array(state[f"{v0}_prior"])
    p0_post = np.array(state[f"{v0}_posterior"])
    p1_prior = np.array(state[f"{v1}_prior"])
    p1_post = np.array(state[f"{v1}_posterior"])

    # Top: variable 0
    ax_top.fill_between(g0, p0_prior, alpha=0.3, color="b", label="Prior")
    ax_top.plot(g0, p0_prior, "b-", linewidth=1.5)
    ax_top.fill_between(g0, p0_post, alpha=0.3, color="r", label="Posterior")
    ax_top.plot(g0, p0_post, "r-", linewidth=1.5)

    m0, q025, q975 = pdf_stats(g0, p0_post)
    pdf_at_m = np.interp(m0, g0, p0_post)
    ax_top.errorbar(m0, pdf_at_m * 0.5,
                    xerr=[[max(0.0, m0 - q025)], [max(0.0, q975 - m0)]],
                    fmt="none", ecolor="r", elinewidth=1.5, capsize=4,
                    capthick=1.5, label="95% CI")
    if t0 is not None:
        ax_top.axvline(t0, color="red", ls="--", lw=1.5, label="True")
    ax_top.legend(fontsize=8)

    # Right: variable 1
    ax_right.fill_betweenx(g1, p1_prior, alpha=0.3, color="b")
    ax_right.plot(p1_prior, g1, "b-", linewidth=1.5)
    ax_right.fill_betweenx(g1, p1_post, alpha=0.3, color="r")
    ax_right.plot(p1_post, g1, "r-", linewidth=1.5)

    m1, q025, q975 = pdf_stats(g1, p1_post)
    pdf_at_m = np.interp(m1, g1, p1_post)
    ax_right.errorbar(pdf_at_m * 0.5, m1,
                      yerr=[[max(0.0, m1 - q025)], [max(0.0, q975 - m1)]],
                      fmt="none", ecolor="r", elinewidth=1.5, capsize=4,
                      capthick=1.5)
    if t1 is not None:
        ax_right.axhline(t1, color="red", ls="--", lw=1.5)


# ---------------------------------------------------------------------------
# Sample-based internals
# ---------------------------------------------------------------------------

def _plot_sample_center(ax, v0, v1, samples, W_posterior, t0, t1, n_bins):
    ax.hist2d(samples[v0], samples[v1], bins=n_bins, weights=W_posterior,
              cmap="viridis", cmin=1e-30)
    handles = []
    _add_true_markers(ax, handles, t0, t1)
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8,
                  facecolor="black", framealpha=0.4, labelcolor="white")


def _plot_sample_marginals(ax_top, ax_right, v0, v1, samples, W_prior, W_posterior, t0, t1, n_bins):
    s0, s1 = samples[v0], samples[v1]

    ax_top.hist(s0, bins=n_bins, weights=W_prior, density=True,
                alpha=0.3, color="b", label="Prior")
    ax_top.hist(s0, bins=n_bins, weights=W_posterior, density=True,
                alpha=0.3, color="r", label="Posterior")
    if t0 is not None:
        ax_top.axvline(t0, color="red", ls="--", lw=1.5)
    ax_top.legend(fontsize=8)

    ax_right.hist(s1, bins=n_bins, weights=W_prior, density=True,
                  alpha=0.3, color="b", orientation="horizontal")
    ax_right.hist(s1, bins=n_bins, weights=W_posterior, density=True,
                  alpha=0.3, color="r", orientation="horizontal")
    if t1 is not None:
        ax_right.axhline(t1, color="red", ls="--", lw=1.5)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_true_markers(ax, handles, t0, t1):
    if t0 is not None:
        ax.axvline(t0, color="red", ls="--", lw=1.5)
    if t1 is not None:
        ax.axhline(t1, color="red", ls="--", lw=1.5)
    if t0 is not None and t1 is not None:
        ax.plot(t0, t1, "rx", markersize=15, markeredgewidth=2.5, zorder=5)
        handles.append(Line2D([], [], color="red", marker="x", ls="--",
                              lw=1.5, markersize=10, label="True"))


def save_jpdf_plots(
    results: Dict[float, Dict[str, Any]],
    output_dir: Path,
    var_names: List[str],
    mode: str = "grid",
    samples: Optional[Dict[str, NDArray]] = None,
    W_prior: Optional[NDArray] = None,
    W_posterior_per_t: Optional[Dict[float, NDArray]] = None,
    true_values: Optional[Dict[str, float]] = None,
) -> None:
    """Save JPDF corner plots: one prior + one per observation time.

    Works for both grid and sample modes. Outputs PNGs to a subdirectory
    and a collected multi-page PDF.

    Args:
        results: Pipeline results dict (keyed by observation time).
        output_dir: Directory for output files.
        var_names: Two variable names to plot.
        mode: "grid" or "samples".
        samples: Sample arrays (sample mode only).
        W_prior: Prior weights (sample mode only).
        W_posterior_per_t: Posterior weights per obs time (sample mode only).
        true_values: Dict mapping variable name to true value.
    """
    png_dir = Path(output_dir) / "pdfs"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(output_dir) / "jpdf.pdf"

    with PdfPages(pdf_path) as pdf:
        # Prior plot
        first_state = next(iter(results.values()))["jpdf_state"]
        W_post_prior = W_prior if mode == "samples" else None
        fig_prior = plot_jpdf(
            var_names=var_names,
            state=first_state,
            mode=mode,
            samples=samples,
            W_prior=W_prior,
            W_posterior=W_post_prior,
            true_values=true_values,
            title="JPDF Prior",
        )
        fig_prior.savefig(png_dir / "jpdf_prior.png", dpi=150, bbox_inches="tight")
        pdf.savefig(fig_prior)

        # One per observation time
        for t, data in results.items():
            W_post = W_posterior_per_t[t] if W_posterior_per_t else None
            fig = plot_jpdf(
                var_names=var_names,
                state=data["jpdf_state"],
                mode=mode,
                samples=samples,
                W_prior=W_prior,
                W_posterior=W_post,
                true_values=true_values,
                title=f"JPDF at t = {t:.0f}",
            )
            fig.savefig(png_dir / f"jpdf_t{t:.0f}.png", dpi=150, bbox_inches="tight")
            pdf.savefig(fig)

    print(f"JPDF PNGs saved to {png_dir}")
    print(f"JPDF PDF  saved to {pdf_path}")
