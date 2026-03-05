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
    ax_main.set_xlabel("CR [-]")
    ax_main.set_ylabel("k [-]")
    ax_main.grid(True, alpha=0.3)

    # CR marginal (top, aligned to x-axis)
    ax_top.fill_between(CR_grid, CR_prior, alpha=0.3, color="b", label="Prior")
    ax_top.plot(CR_grid, CR_prior, "b-", linewidth=1.5)
    ax_top.fill_between(CR_grid, CR_posterior, alpha=0.3, color="r", label="Posterior")
    ax_top.plot(CR_grid, CR_posterior, "r-", linewidth=1.5)
    ax_top.set_ylabel("Density")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)
    ax_top.tick_params(labelbottom=False)

    # k marginal (right, aligned to y-axis)
    ax_right.fill_betweenx(k_grid, k_prior, alpha=0.3, color="b")
    ax_right.plot(k_prior, k_grid, "b-", linewidth=1.5)
    ax_right.fill_betweenx(k_grid, k_posterior, alpha=0.3, color="r")
    ax_right.plot(k_posterior, k_grid, "r-", linewidth=1.5)
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
            )
            fig.savefig(png_dir / f"jpdf_t{t:.0f}.png", dpi=150, bbox_inches="tight")
            pdf.savefig(fig)

    print(f"JPDF PNGs saved to {png_dir}")
    print(f"JPDF PDF  saved to {pdf_path}")
