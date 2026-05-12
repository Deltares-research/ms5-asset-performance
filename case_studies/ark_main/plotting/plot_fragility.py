"""
Visualise a cached fragility curve for ark_main.

Two figures:

1. ``fragility_curve.png`` — beta and Pf vs corrosion_rate (twin axes, log Pf).
   Marks FORM-converged points and any points produced by extrapolation /
   importance sampling differently.

2. ``form_alphas_pies.png`` — one pie per fragility point showing the
   FORM importance factors (alpha^2) by variable. Variables that are
   deterministic at every point (alpha == 0 throughout) are dropped from
   the legend; remaining variables get a stable colour across all pies.

Usage:
    python plot_fragility.py
    python plot_fragility.py --lsf-name lsf_wall_anchor
    python plot_fragility.py --min-share 0.02     # group <2% into "other"
"""

import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

from src.io import get_remote_path


_ENV = Path(__file__).resolve().parents[1] / ".env"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_points(cache_dir: Path) -> list[dict]:
    """Load all point_NNNN.json files in order, ignoring manifest."""
    pts = []
    for f in sorted(cache_dir.glob("point_*.json")):
        with open(f) as fh:
            pts.append(json.load(fh))
    pts.sort(key=lambda r: r["index"])
    return pts


# ---------------------------------------------------------------------------
# Fragility curve
# ---------------------------------------------------------------------------

def plot_fragility_curve(points: list[dict], out_path: Path, lsf_name: str) -> None:
    cr = np.array([p["point"]["corrosion_rate"] for p in points])
    beta = np.array([p["beta"] for p in points])
    pf = np.array([p["pf"] for p in points])
    methods = [p["method"] for p in points]
    converged = np.array([p["convergence"] for p in points])

    method_color = {
        "form": "#2E86AB",
        "importance_sampling": "#1F4E79",
        "development": "#7FB3D5",
        "extrapolation": "#7FB3D5",
    }

    fig, (ax_b, ax_p) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # connecting lines (no markers)
    ax_b.plot(cr, beta, color="#2E86AB", linewidth=1.5, alpha=0.6, zorder=1)
    ax_p.plot(cr, pf,   color="#2E86AB", linewidth=1.5, alpha=0.6, zorder=1)

    # markers per method, on top
    for m in set(methods):
        mask = np.array([mt == m for mt in methods]) & converged
        if mask.any():
            color = method_color.get(m, "gray")
            ax_b.scatter(cr[mask], beta[mask], s=42, color=color,
                         edgecolors="k", linewidth=0.5, zorder=3, label=m)
            ax_p.scatter(cr[mask], pf[mask], s=42, color=color,
                         edgecolors="k", linewidth=0.5, zorder=3, label=m)
    bad = ~converged
    if bad.any():
        ax_b.scatter(cr[bad], beta[bad], s=42, marker="x", color="k",
                     zorder=3, label="non-converged")
        ax_p.scatter(cr[bad], pf[bad], s=42, marker="x", color="k",
                     zorder=3, label="non-converged")

    ax_b.set_ylabel(r"Reliability index  $\beta$ [-]")
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=9)
    ax_b.set_title(f"Fragility curve — {lsf_name}")

    ax_p.set_yscale("log")
    ax_p.set_ylabel("Failure probability  $P_f$ [-]")
    ax_p.set_xlabel("Corrosion ratio  $cr = \\Delta t / d_0$  [-]")
    ax_p.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Per-point alpha pies
# ---------------------------------------------------------------------------

def plot_alpha_pies(points: list[dict], out_path: Path, lsf_name: str,
                    min_share: float = 0.02) -> None:
    # Variable order from the first point that has alphas; keep stochastic only.
    all_vars = list(points[0]["alphas"].keys())

    # Drop variables with alpha == 0 across all points (deterministic in FORM).
    alpha2 = {v: np.array([p["alphas"].get(v, 0.0) ** 2 for p in points]) for v in all_vars}
    stoch_vars = [v for v in all_vars if alpha2[v].max() > 1e-10]

    if not stoch_vars:
        print("  no stochastic variables in alphas — skipping pies")
        return

    # Stable colour map
    cmap = plt.get_cmap("tab20")
    var_colors = {v: cmap(i % 20) for i, v in enumerate(stoch_vars)}
    var_colors["other"] = "#bbbbbb"

    n = len(points)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows + 1.0))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    # Track variables that ever appear in a pie (for legend filtering)
    legend_vars: list[str] = []

    for i, p in enumerate(points):
        ax = axes[i // ncols, i % ncols]
        cr = p["point"]["corrosion_rate"]

        a2 = np.array([p["alphas"].get(v, 0.0) ** 2 for v in stoch_vars])
        s = a2.sum()
        if s == 0:
            ax.text(0.5, 0.5, "all alpha = 0", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"cr = {cr:.2f}\nbeta = {p['beta']:.2f} ({p['method']})", fontsize=10)
            ax.axis("off")
            continue
        a2 /= s

        # Group small slices into "other"
        big = a2 >= min_share
        labels = [v for v, k in zip(stoch_vars, big) if k]
        sizes = list(a2[big])
        if (~big).any():
            other = float(a2[~big].sum())
            if other > 0:
                labels.append("other")
                sizes.append(other)

        for v in labels:
            if v not in legend_vars:
                legend_vars.append(v)
        colors = [var_colors[v] for v in labels]

        ax.pie(sizes, colors=colors, startangle=90, wedgeprops=dict(linewidth=0.5, edgecolor="white"),
               radius=1.0)
        ax.set_title(f"cr = {cr:.2f}\n"
                     f"beta = {p['beta']:.2f} ({p['method']})",
                     fontsize=10)

    # blank unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    # Single shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=var_colors[v]) for v in legend_vars]
    fig.legend(handles, legend_vars, loc="lower center", ncol=min(5, len(legend_vars)),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"FORM importance factors $\\alpha^2$ per fragility point — {lsf_name}\n"
        f"(slices < {min_share*100:.0f}% grouped as 'other')",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(lsf_name: str = "lsf_wall", output_dir: str | None = None,
         min_share: float = 0.02) -> None:
    remote = get_remote_path(_ENV)
    cache_dir = remote / "output" / f"fragility_curve_{lsf_name}"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"No cached fragility curve at {cache_dir} — run build_fragility.py first."
        )

    points = load_points(cache_dir)
    if not points:
        raise RuntimeError(f"{cache_dir} is empty — nothing to plot.")
    print(f"Loaded {len(points)} fragility points from {cache_dir}")

    out = Path(output_dir) if output_dir else (remote / "output" / f"fragility_plots_{lsf_name}")
    out.mkdir(parents=True, exist_ok=True)

    plot_fragility_curve(points, out / "fragility_curve.png", lsf_name)
    plot_alpha_pies(points, out / "form_alphas_pies.png", lsf_name, min_share=min_share)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lsf-name", type=str, default="lsf_wall_anchor",
                        help="LSF name (selects fragility_curve_{lsf}/). Default: lsf_wall.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Default: <remote>/output/fragility_plots_{lsf}/.")
    parser.add_argument("--min-share", type=float, default=0.02,
                        help="Group alpha^2 slices smaller than this share into 'other'. Default: 0.02.")
    args = parser.parse_args()
    main(lsf_name=args.lsf_name, output_dir=args.output_dir, min_share=args.min_share)
