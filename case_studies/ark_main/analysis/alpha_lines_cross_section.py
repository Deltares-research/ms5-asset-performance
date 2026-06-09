"""Per-variable FORM alphas over time (incl. corrosion) — cross-section run.

Companion to ``alpha_pie_cross_section.py`` (which collapses the split to two
slices: fragility vs corrosion). Here we expose the **full** design-point
direction, one alpha per random variable, as a function of forecast time, for
both the prior and the (accumulating) posterior leg of ``run.py``.

Method (1-D nested-FORM over the standardised corrosion variable)
-----------------------------------------------------------------
At each forecast time ``t`` the corrosion rate enters via
``z = Phi^{-1}(F_prior(cr; t))``. Its marginal under the leg of interest is
Gaussian ``(mu_z, sigma_z)``:

  * prior      -> ``(0, 1)`` (no observations);
  * posterior  -> ``(m_post, sqrt(v_post))`` from
    ``cr_field.z_moments_under_posterior`` on the posterior cr-PDF, where the
    posterior at time ``t`` conditions on **all observations up to ``t``**
    (exactly how ``run.py``'s ``b_post`` column accumulates obs).

The nested-FORM 1-D search then gives, treating cr as one more variable::

    xi*       = argmin xi^2 + beta(cr(xi))^2          cr(xi)=F_prior^{-1}(Phi(mu_z+sigma_z*xi))
    beta_T    = sqrt(xi*^2 + beta(cr*)^2)
    alpha_cr  = -xi* / beta_T
    alpha_v   = (beta(cr*) / beta_T) * alpha_v(cr*)   for each fragility variable v

with the unit-norm identity ``alpha_cr^2 + sum_v alpha_v^2 = 1``. The
``alpha_v(cr*)`` are the cached fragility direction cosines interpolated at the
design-point ``cr*``; rescaling by ``beta(cr*)/beta_T`` folds in the corrosion
coupling so the full vector (cr + all fragility vars) is unit-norm.

Reads:
  <remote>/output/cr_pdfs_<lsf>.json        (prior + posterior cr-PDFs per t)
  <remote>/output/fragility_curve_<lsf>/    (beta + alphas per cr point)

Writes (under the latest run.py results dir for the lsf, or a fallback dir):
  plots/alpha_lines/alpha_lines_prior.png
  plots/alpha_lines/alpha_lines_posterior.png
  plots/alpha_lines/alpha_lines_prior_vs_posterior.png

Usage:
    python alpha_lines_cross_section.py
    python alpha_lines_cross_section.py --lsf-name lsf_wall
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
from spatial import cr_field
from spatial.nested_mcs import _nested_form_1d


_ENV = Path(__file__).resolve().parents[1] / ".env"
_CR_LABEL = "corrosion"


def _load_fragility(remote: Path, lsf_name: str):
    points = load_points(remote, lsf_name)
    if not points:
        raise SystemExit(f"No converged fragility points with alphas for {lsf_name}.")
    cr_values = np.array([float(p["point"]["corrosion_rate"]) for p in points])
    betas = np.array([float(p["beta"]) for p in points])
    var_names = list(points[0]["alphas"].keys())
    alpha_full = np.array([[float(p["alphas"][v]) for v in var_names] for p in points])
    active_mask = np.any(np.abs(alpha_full) > 1e-6, axis=0)
    active_vars = [v for v, m in zip(var_names, active_mask) if m]
    alpha_table = alpha_full[:, active_mask]              # (n_cr, n_active)
    return cr_values, betas, active_vars, alpha_table


def _decompose(mu_z, sigma_z, F_prior, cr_grid, cr_values, betas, alpha_table):
    """Nested-FORM at one time -> (alpha_cr, alpha_basic[n_active], beta_T, cr*)."""
    xi, crs, b = _nested_form_1d(
        mu_z=float(mu_z), sigma_z=float(sigma_z),
        cr_grid_export=cr_grid, F_prior=F_prior,
        cr_values_frag=cr_values, betas_frag=betas,
    )
    bT = float(np.sqrt(xi * xi + b * b))
    if bT <= 0:
        return 0.0, np.zeros(alpha_table.shape[1]), bT, crs
    alpha_cr = -xi / bT
    alpha_at_crs = np.array([
        float(np.interp(crs, cr_values, alpha_table[:, j]))
        for j in range(alpha_table.shape[1])
    ])
    alpha_basic = (b / bT) * alpha_at_crs
    return alpha_cr, alpha_basic, bT, crs


def compute_alpha_tracks(remote: Path, lsf_name: str):
    """Return times, active_vars, and prior/posterior alpha arrays.

    alpha arrays are dicts with keys 'cr' (n_t,) and 'basic' (n_t, n_active),
    one entry for 'prior' and one for 'posterior'.
    """
    data = json.load(open(remote / "output" / f"cr_pdfs_{lsf_name}.json"))
    cr_grid = np.asarray(data["cr_grid"], dtype=float)
    prior_per_t = data["prior_pdf_per_t"]
    post_per_obs = data["posterior_pdf_per_obs"]
    obs_times = sorted(float(t) for t in data["obs_times"])
    times = np.array(sorted(float(k) for k in prior_per_t.keys()), dtype=float)

    cr_values, betas, active_vars, alpha_table = _load_fragility(remote, lsf_name)
    n_t, n_active = len(times), len(active_vars)

    def post_pdf_at(t):
        """Posterior cr-PDF at forecast time t, conditioned on obs <= t."""
        elig = [to for to in obs_times if to <= t + 1e-9]
        if not elig:
            return None                                   # no obs yet -> prior
        to = max(elig)
        blk = post_per_obs[f"{to:.4f}"]
        return np.asarray(blk[f"{t:.4f}"], dtype=float)

    out = {leg: {"cr": np.zeros(n_t), "basic": np.zeros((n_t, n_active)),
                 "beta_T": np.zeros(n_t), "cr_star": np.zeros(n_t)}
           for leg in ("prior", "posterior")}

    for ti, t in enumerate(times):
        prior_pdf = np.asarray(prior_per_t[f"{t:.4f}"], dtype=float)
        F_prior = cr_field.cdf_on_grid(prior_pdf, cr_grid)

        # Prior leg: mu_z = 0, sigma_z = 1.
        a_cr, a_b, bT, crs = _decompose(0.0, 1.0, F_prior, cr_grid,
                                        cr_values, betas, alpha_table)
        out["prior"]["cr"][ti] = a_cr
        out["prior"]["basic"][ti] = a_b
        out["prior"]["beta_T"][ti] = bT
        out["prior"]["cr_star"][ti] = crs

        # Posterior leg: condition on obs <= t (accumulating, as in run.py).
        post_pdf = post_pdf_at(t)
        if post_pdf is None:
            mu_z, sigma_z = 0.0, 1.0                       # before first obs
        else:
            m_post, v_post = cr_field.z_moments_under_posterior(
                prior_pdf, post_pdf, cr_grid,
            )
            mu_z, sigma_z = m_post, float(np.sqrt(max(v_post, 1e-12)))
        a_cr, a_b, bT, crs = _decompose(mu_z, sigma_z, F_prior, cr_grid,
                                        cr_values, betas, alpha_table)
        out["posterior"]["cr"][ti] = a_cr
        out["posterior"]["basic"][ti] = a_b
        out["posterior"]["beta_T"][ti] = bT
        out["posterior"]["cr_star"][ti] = crs

    return times, active_vars, obs_times, out


def _resolve_out_dir(remote: Path, lsf_name: str) -> Path:
    results_root = remote / "output" / "results" / lsf_name
    if results_root.exists():
        runs = [p for p in results_root.iterdir() if p.is_dir()]
        if runs:
            latest = max(runs, key=lambda p: p.stat().st_mtime)
            return latest / "plots" / "alpha_lines"
    return remote / "output" / f"alpha_lines_{lsf_name}"


def _var_colors(active_vars):
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(active_vars), 1)))
    return {v: cmap[i] for i, v in enumerate(active_vars)}


def _plot_leg(ax, times, active_vars, colors, track, obs_times, title):
    # corrosion: thick dashed black.
    ax.plot(times, track["cr"], "--", color="black", lw=2.6, label=_CR_LABEL, zorder=5)
    for j, v in enumerate(active_vars):
        ax.plot(times, track["basic"][:, j], "-", color=colors[v], lw=1.6, label=v)
    for to in obs_times:
        ax.axvline(to, color="0.8", ls=":", lw=0.8, zorder=0)
    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("alpha")
    ax.set_title(title)
    ax.grid(alpha=0.25)


def _pie_colors(active_vars, colors):
    """Slice colours: corrosion black, then each fragility var (matches lines)."""
    return ["black"] + [colors[v] for v in active_vars]


def _alpha_sq(track, ti, n_active):
    """Squared importance factors [corrosion, *fragility vars] at time index ti."""
    vals = np.concatenate(([track["cr"][ti] ** 2], track["basic"][ti] ** 2))
    s = vals.sum()
    return vals / s if s > 0 else vals


def _pie_vars(ax, vals, slice_colors, title):
    def _fmt(p):
        return f"{p:.0f}%" if p >= 4 else ""
    ax.pie(
        vals, colors=slice_colors, autopct=_fmt, startangle=90,
        counterclock=False, pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 0.6},
        textprops={"fontsize": 7},
    )
    ax.set_title(title, fontsize=9)


def make_plots(remote: Path, lsf_name: str, results_dir: Path | None = None):
    times, active_vars, obs_times, out = compute_alpha_tracks(remote, lsf_name)
    out_dir = ((Path(results_dir) / "plots" / "alpha_lines")
               if results_dir else _resolve_out_dir(remote, lsf_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = _var_colors(active_vars)

    # 1) Prior — single panel.
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_leg(ax, times, active_vars, colors, out["prior"], obs_times,
              f"FORM alphas vs time (prior)  -  {lsf_name}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir / "alpha_lines_prior.png")
    plt.close(fig)

    # 2) Posterior — single panel.
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_leg(ax, times, active_vars, colors, out["posterior"], obs_times,
              f"FORM alphas vs time (posterior, obs<=t)  -  {lsf_name}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir / "alpha_lines_posterior.png")
    plt.close(fig)

    # 3) Side-by-side comparison.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    _plot_leg(axes[0], times, active_vars, colors, out["prior"], obs_times, "prior")
    _plot_leg(axes[1], times, active_vars, colors, out["posterior"], obs_times,
              "posterior (obs <= t)")
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=8)
    fig.suptitle(f"FORM alphas vs time (incl. corrosion)  -  {lsf_name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    save_figure(fig, out_dir / "alpha_lines_prior_vs_posterior.png")
    plt.close(fig)

    # 4) Per-variable alpha^2 pie charts (sum to 100%), prior + posterior.
    n_active = len(active_vars)
    slice_colors = _pie_colors(active_vars, colors)
    slice_labels = [_CR_LABEL] + active_vars
    pies_root = out_dir / "alpha_pies"
    for leg in ("prior", "posterior"):
        frames_dir = pies_root / f"alpha_pie_vars_{leg}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for ti, t in enumerate(times):
            vals = _alpha_sq(out[leg], ti, n_active)
            fig, ax = plt.subplots(figsize=(4.2, 4.4))
            _pie_vars(ax, vals, slice_colors,
                      f"{leg}: t = {t:.1f} yr  (beta_T={out[leg]['beta_T'][ti]:.2f})")
            ax.legend(slice_labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=False, fontsize=7)
            fig.tight_layout()
            save_figure(fig, frames_dir / f"alpha_pie_{t:07.2f}.png")
            plt.close(fig)
        collect_pngs_to_pdf(frames_dir, pies_root / f"alpha_pie_vars_{leg}.pdf")

    make_gifs(pies_root)   # one GIF per leg subdir

    # Representative grid at obs times (+ t=0), one figure per leg.
    grid_times = [0.0] + list(obs_times)
    sel = [int(np.argmin(np.abs(times - gt))) for gt in grid_times]
    ncol = 3
    nrow = int(np.ceil(len(sel) / ncol))
    for leg in ("prior", "posterior"):
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.8 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[len(sel):]:
            ax.axis("off")
        for ax, k in zip(axes, sel):
            _pie_vars(ax, _alpha_sq(out[leg], k, n_active), slice_colors,
                      f"t = {times[k]:.1f} yr  (beta_T={out[leg]['beta_T'][k]:.2f})")
        fig.legend(slice_labels, loc="center left", bbox_to_anchor=(0.99, 0.5),
                   frameon=False, fontsize=8)
        fig.suptitle(
            f"alpha^2 importance per variable ({leg})  -  {lsf_name}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 0.86, 0.97))
        save_figure(fig, out_dir / f"alpha_pie_vars_{leg}_grid.png")
        plt.close(fig)

    # Console summary at obs times.
    print(f"\nFORM alphas (incl. corrosion) for '{lsf_name}'. "
          f"{len(active_vars)} fragility vars + corrosion.")
    print("Dominant alpha per leg at obs times:")
    print(f"{'t':>6} {'leg':>10} {'beta_T':>8} {'cr*':>7}  top contributors (alpha)")
    for leg in ("prior", "posterior"):
        for to in obs_times:
            ti = int(np.argmin(np.abs(times - to)))
            allv = [(_CR_LABEL, out[leg]["cr"][ti])] + list(
                zip(active_vars, out[leg]["basic"][ti]))
            top = sorted(allv, key=lambda kv: -abs(kv[1]))[:3]
            tops = ", ".join(f"{k}={a:+.2f}" for k, a in top)
            print(f"{times[ti]:6.1f} {leg:>10} {out[leg]['beta_T'][ti]:8.3f} "
                  f"{out[leg]['cr_star'][ti]:7.3f}  {tops}")
    print(f"\nOutputs -> {out_dir}")


def main(lsf_name: str = "lsf_wall"):
    make_plots(get_remote_path(_ENV), lsf_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf-name", type=str, default="lsf_wall")
    args = parser.parse_args()
    main(lsf_name=args.lsf_name)
