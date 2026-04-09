"""
Generic plotting utilities.

Provides helper functions for PDF statistics, HDR contours, figure saving,
PNG-to-PDF collection, and animated GIF creation.
"""

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray


def pdf_stats(grid: NDArray, pdf: NDArray) -> Tuple[float, float, float]:
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


def hdr_level(
    pdf_2d: NDArray,
    x_grid: NDArray,
    y_grid: NDArray,
    alpha: float = 0.95,
) -> float:
    """Compute the PDF level that encloses `alpha` fraction of probability mass.

    Args:
        pdf_2d: 2D array of joint PDF values.
        x_grid: 1D array of x grid values.
        y_grid: 1D array of y grid values.
        alpha: Probability mass to enclose (default 0.95).

    Returns:
        PDF threshold level for the HDR contour.
    """
    dx = np.diff(x_grid)
    dy = np.diff(y_grid)
    cell_area = dx[:, np.newaxis] * dy[np.newaxis, :]
    pdf_centers = (
        pdf_2d[:-1, :-1] + pdf_2d[:-1, 1:] + pdf_2d[1:, :-1] + pdf_2d[1:, 1:]
    ) / 4
    mass = pdf_centers * cell_area

    flat_pdf = pdf_centers.flatten()
    flat_mass = mass.flatten()
    order = np.argsort(-flat_pdf)
    cumulative = np.cumsum(flat_mass[order])
    cumulative /= cumulative[-1]

    idx = np.searchsorted(cumulative, alpha)
    idx = min(idx, len(order) - 1)
    return flat_pdf[order[idx]]


def save_figure(fig: plt.Figure, filepath, dpi: int = 150) -> None:
    """Save a figure to disk and close it.

    Args:
        fig: Matplotlib figure.
        filepath: Output file path.
        dpi: Resolution.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def collect_pngs_to_pdf(png_dir: Path, pdf_path: Path, dpi: int = 150) -> None:
    """Collect all PNGs in a directory into a single PDF.

    Args:
        png_dir: Directory containing PNG files.
        pdf_path: Output PDF file path.
        dpi: Resolution for the PDF pages.
    """
    pngs = sorted(Path(png_dir).glob("*.png"), key=_sort_key)
    if not pngs:
        return

    with PdfPages(pdf_path) as pdf:
        for p in pngs:
            from PIL import Image
            img = Image.open(p)
            fig, ax = plt.subplots(figsize=(img.width / dpi, img.height / dpi))
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)


def make_gifs(output_dir: Path, duration: int = 500) -> None:
    """Create animated GIFs from PNG sequences in subdirectories.

    Scans each subdirectory for PNGs, sorts by numeric value in filename,
    and assembles a looping GIF.

    Args:
        output_dir: Parent directory containing PNG subdirectories.
        duration: Frame duration in milliseconds.
    """
    from PIL import Image

    for png_dir in Path(output_dir).iterdir():
        if not png_dir.is_dir():
            continue
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


def _sort_key(p: Path) -> float:
    """Sort key for PNG filenames by embedded numeric value."""
    m = re.search(r"(\d+\.?\d*)", p.stem)
    return float(m.group(1)) if m else -1
