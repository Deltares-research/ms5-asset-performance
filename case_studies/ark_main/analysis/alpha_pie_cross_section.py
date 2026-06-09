"""Fragility-vs-corrosion alpha (importance-factor) pie charts per time.

For the single cross-section analysis (``run.py``) the reliability at a
forecast time ``t`` combines two sources of uncertainty:

  * **fragility** — the FORM variables baked into the cached fragility curve
    (soil strengths, loads, model factors, wall stiffness), summarised by the
    conditional reliability ``beta(cr)``;
  * **corrosion** — the spread of the corrosion-rate ``cr`` at time ``t``
    (the power/linear corrosion model), i.e. how uncertain the loss of
    section is.

``run.py`` itself only produces the marginal ``beta(t) = -Phi^{-1}(int Pf(cr)
f(cr; t) dcr)`` — it has no notion of an alpha split because the integration
collapses both sources into one number. To attribute that reliability to the
two sources we run the **1-D nested-FORM search** over the standardised
corrosion variable ``z = Phi^{-1}(F_prior(cr; t))`` (the same machinery the
spatial pipeline uses in ``mcs_method='alphas'`` mode, prior leg):

    xi* = argmin  xi^2 + beta(cr(xi))^2
    beta_T   = sqrt(xi*^2 + beta(cr*)^2)
    alpha_cr = -xi* / beta_T                      (corrosion direction cosine)
    |alpha_fragility| = beta(cr*) / beta_T        (fragility variables, combined)

with the identity ``alpha_cr^2 + |alpha_fragility|^2 = 1``. The two squared
importance factors are the natural pie slices (they sum to 100%):

    corrosion share = alpha_cr^2
    fragility share = |alpha_fragility|^2 = (beta(cr*) / beta_T)^2

This is the **prior** decomposition (no observations), matching ``run.py``'s
``b_prior`` column. ``beta_T`` is the nested-FORM approximation of the same
integral ``run.py`` evaluates, so the two betas track closely.

Reads:
  <remote>/output/cr_pdfs_<lsf>.json   (prior cr-PDFs per t; from export_cr_pdfs)
  <remote>/output/fragility_curve_<lsf>/   (beta + alphas per cr point)

Writes (under the latest run.py results dir for the lsf, or a fallback dir):
  plots/alpha_decomposition/alpha_pie/<frame>.png   one pie per time
  plots/alpha_decomposition/alpha_pie.gif           animation over time
  plots/alpha_decomposition/alpha_pie.pdf           all frames bundled
  plots/alpha_decomposition/alpha_pie_grid.png      pies at obs times
  plots/alpha_decomposition/alpha_share_over_time.png   stacked-area trend

Usage:
    python alpha_pie_cross_section.py
    python alpha_pie_cross_section.py --lsf-name lsf_wall
"""

import json
import sys
from pathlib import Path
from argparse import ArgumentParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from src.io import get_remote_path
from src.plotting import save_figure, make_gifs, collect_pngs_to_pdf
from spatial.fragility import load_points
from spatial.nested_mcs import compute_nested_form_prior


_ENV = Path(__file__).resolve().parents[1] / ".env"

# Slice colours / labels (fragility first, corrosion second).
_COL_FRAG = "#4C72B0"   # steel blue
_COL_CORR = "#DD8452"   # warm orange
_LABELS = ["fragility", "corrosion"]
_COLORS = [_COL_FRAG, _COL_CORR]


def _resolve_out_dir(remote: Path, lsf_name: str) -> Path:
    """Latest run.py results dir for this lsf, else a fallback under output/."""
    results_root = remote / "output" / "results" / lsf_name
    if results_root.exists():
        runs = [p for p in results_root.iterdir() if p.is_dir()]
        if runs:
            latest = max(runs, key=lambda p: p.stat().st_mtime)
            return latest / "plots" / "alpha_decomposition"
    return remote / "output" / f"alpha_decomposition_{lsf_name}"


def compute_shares(remote: Path, lsf_name: str):
    """Return (times, frac_fragility, frac_corrosion, beta_T, cr_star)."""
    cr_file = remote / "output" / f"cr_pdfs_{lsf_name}.json"
    if not cr_file.exists():
        raise FileNotFoundError(
            f"{cr_file} not found. Run io/export_cr_pdfs.py --lsf-name {lsf_name} first."
        )
    data = json.load(open(cr_file))
    cr_grid = np.asarray(data["cr_grid"], dtype=float)
    prior_pdf_per_t = data["prior_pdf_per_t"]
    # Use the times actually present as prior keys, sorted ascending.
    times = np.array(sorted(float(k) for k in prior_pdf_per_t.keys()), dtype=float)

    # Fragility cache: converged points carrying alphas, sorted by cr.
    points = load_points(remote, lsf_name)
    if not points:
        raise SystemExit(f"No converged fragility points with alphas for {lsf_name}.")
    cr_values = np.array([float(p["point"]["corrosion_rate"]) for p in points])
    betas = np.array([float(p["beta"]) for p in points])
    var_names = list(points[0]["alphas"].keys())
    alpha_full = np.array([[float(p["alphas"][v]) for v in var_names] for p in points])
    active_mask = np.any(np.abs(alpha_full) > 1e-6, axis=0)
    active_vars = [v for v, m in zip(var_names, active_mask) if m]
    alpha_table = alpha_full[:, active_mask]

    pre = compute_nested_form_prior(
        cr_grid_export=cr_grid,
        prior_pdf_per_t=prior_pdf_per_t,
        forecast_times=times,
        cr_values_frag=cr_values,
        betas_frag=betas,
        alpha_table=alpha_table,
        active_vars=active_vars,
    )
    alpha_cr = pre["alpha_cr"]                       # (n_t,)
    alpha_basic = pre["alpha_basic"]                 # (n_t, n_active)
    frac_corr = alpha_cr ** 2
    frac_frag = np.sum(alpha_basic ** 2, axis=1)
    # Renormalise to guard against tiny numerical drift off 1.
    total = frac_corr + frac_frag
    total[total == 0] = 1.0
    frac_corr = frac_corr / total
    frac_frag = frac_frag / total
    return times, frac_frag, frac_corr, pre["beta_T"], pre["cr_star"]


def _pie(ax, frag, corr, title):
    ax.pie(
        [frag, corr], labels=_LABELS, colors=_COLORS,
        autopct="%1.0f%%", startangle=90, counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
        textprops={"fontsize": 9},
    )
    ax.set_title(title, fontsize=10)


def make_plots(remote: Path, lsf_name: str, results_dir: Path | None = None):
    times, frag, corr, beta_T, cr_star = compute_shares(remote, lsf_name)
    out_dir = ((Path(results_dir) / "plots" / "alpha_decomposition")
               if results_dir else _resolve_out_dir(remote, lsf_name))
    frames_dir = out_dir / "alpha_pie"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 1) One pie per forecast time (animation frames).
    for t, fr, co, bT in zip(times, frag, corr, beta_T):
        fig, ax = plt.subplots(figsize=(3.6, 3.8))
        _pie(ax, fr, co, f"t = {t:.1f} yr   (beta_T = {bT:.2f})")
        save_figure(fig, frames_dir / f"alpha_pie_{t:07.2f}.png")
        plt.close(fig)

    # GIF (one per subdir of out_dir) + PDF bundle of the frames.
    make_gifs(out_dir)
    collect_pngs_to_pdf(frames_dir, out_dir / "alpha_pie.pdf")

    # 2) Representative grid at observation times (+ t=0).
    obs = json.load(open(remote / "output" / f"cr_pdfs_{lsf_name}.json"))["obs_times"]
    grid_times = [0.0] + list(obs)
    sel = [int(np.argmin(np.abs(times - gt))) for gt in grid_times]
    n = len(sel)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, k in zip(axes, sel):
        _pie(ax, frag[k], corr[k], f"t = {times[k]:.1f} yr  (beta_T={beta_T[k]:.2f})")
    fig.suptitle(
        f"Importance split: fragility vs corrosion  -  {lsf_name} (prior)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out_dir / "alpha_pie_grid.png")
    plt.close(fig)

    # 3) Stacked-area trend over time (complementary, easier to read at a glance).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stackplot(
        times, frag * 100, corr * 100,
        labels=[r"fragility  $\alpha^2$", r"corrosion  $\alpha^2$"],
        colors=_COLORS, alpha=0.9,
    )
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("importance share (%)")
    ax.set_title(f"Fragility vs corrosion importance over time  -  {lsf_name} (prior)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir / "alpha_share_over_time.png")
    plt.close(fig)

    # Console summary at the grid times.
    print(f"\nImportance split (prior) for '{lsf_name}':")
    print(f"{'t':>6} {'beta_T':>8} {'cr*':>7} {'frag%':>7} {'corr%':>7}")
    for k in sel:
        print(f"{times[k]:6.1f} {beta_T[k]:8.3f} {cr_star[k]:7.3f} "
              f"{frag[k]*100:7.1f} {corr[k]*100:7.1f}")
    print(f"\nOutputs -> {out_dir}")


def main(lsf_name: str = "lsf_wall"):
    remote = get_remote_path(_ENV)
    make_plots(remote, lsf_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf-name", type=str, default="lsf_wall")
    args = parser.parse_args()
    main(lsf_name=args.lsf_name)
