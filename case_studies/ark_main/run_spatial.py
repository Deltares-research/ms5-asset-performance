"""
Entry point for the spatial-variability MCS.

Sequence:

1-5. ``compute_or_load_pf_grid`` — sample the spatial model, check per-sample
     per-location failure at every cr in the cached fragility curve,
     aggregate to Pf(cr, location), and cache the table. On a re-run with
     the same configuration the cached file is loaded instead.
6.   Integrate the Pf grid over the cr distribution to get Pf at one time
     ``t`` (currently t = 0, where the cr distribution is a delta at
     cr = 0).

Then writes a ``summary.json`` for t = 0 and three plots.

Usage::

    python -m case_studies.ark_main.run_spatial --n-samples 100000
    python -m case_studies.ark_main.run_spatial --n-samples 100000 --theta 400 --rho-0 0.5
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

_ARK = Path(__file__).resolve().parent
sys.path.insert(0, str(_ARK))

# Importing from ``spatial.engine`` triggers ``load_dotenv`` for
# geolib.env and the ``load_settings`` import chain. Do this before any
# other ``src.*`` / ``reliability.*`` imports.
from spatial.engine import compute_or_load_pf_grid

import numpy as np
from scipy import stats as st
from tqdm import tqdm

from src.io import get_remote_path
from reliability.build_fragility import load_settings

from spatial import covariance, fragility, integration, plots


_ENV = _ARK / ".env"
_settings = load_settings()
_remote = get_remote_path(_ENV)


# ----------------------------------------------------------------------
# Prior cr-PDF source — read from the canonical exporter file
#
# ``case_studies/ark_main/io/export_cr_pdfs.py`` produces this file. It
# bundles the prior cr-PDF at every forecast time on a common ``cr_grid``,
# alongside the obs-conditioned posteriors. Reading it here keeps the
# spatial pipeline byte-identical with ``run.py``'s per-section pipeline
# instead of re-deriving the prior from the corrosion model in two places.
# ----------------------------------------------------------------------

def _load_cr_pdfs(lsf_name: str) -> dict:
    """Load ``output/cr_pdfs_<lsf>.json`` or raise with a regen hint."""
    path = _remote / "output" / f"cr_pdfs_{lsf_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Regenerate with:\n"
            f"  python -m case_studies.ark_main.io.export_cr_pdfs "
            f"--lsf-name {lsf_name}"
        )
    return json.load(open(path))


def analyze(
    lsf_name: str = "lsf_wall",
    n_samples: int = 10000,
    seed: int = 42,
    theta: float | None = None,
    rho_0: float | None = None,
) -> None:
    """Run (or load) the spatial MCS, integrate at t = 0, and produce
    ``summary.json`` + plots under ``<remote>/output/spatial_mc_<lsf>/``.
    """
    # Light setup so we can print a meaningful header and prepare what the
    # plots need (``points`` for the cr=0 alphas, ``spec`` for the Cholesky
    # of the realisations panel).
    points = fragility.load_points(_remote, lsf_name)
    var_names = list(points[0]["alphas"].keys())
    L, n_sections, default_theta, default_rho_0, spec = covariance.resolve_config(
        var_names, _settings.get("spatial"), theta, rho_0,
    )
    x = np.linspace(0.0, L, n_sections)
    out_dir = _remote / "output" / f"spatial_mc_{lsf_name}"

    print("=" * 60)
    print("Spatial MCS — sheet-pile wall")
    print("=" * 60)
    print(f"LSF:              {lsf_name}")
    print(f"L:                {L:.1f} m,  n_sections={n_sections}  (spacing {L/(n_sections-1):.1f} m)")
    print(f"default theta:    {default_theta:.1f} m")
    print(f"default rho_0:    {default_rho_0:.3f}")
    print(f"n_cr points:      {len(points)}")
    print(f"n_samples:        {n_samples}")
    print(f"seed:             {seed}")

    # Steps 1-5.
    data = compute_or_load_pf_grid(
        lsf_name=lsf_name, n_samples=n_samples, seed=seed,
        theta=theta, rho_0=rho_0,
    )

    cr_values = np.array(data["cr_values"])
    betas_form = np.array(data["betas"])
    n_fail_section = np.array(data["n_fail_section"])      # (n_cr, n_sections)
    n_fail_system = np.array(data["n_fail_system"])        # (n_cr,)
    pf_section_grid = n_fail_section / n_samples
    pf_system_grid = n_fail_system / n_samples

    # Step 6: integrate Pf(cr) against the **prior** cr_pdf at every t in
    # the forecast grid. We read the prior PDFs from the canonical exporter
    # file rather than reconstructing them from the corrosion model — this
    # keeps the spatial pipeline strictly downstream of ``run.py``'s
    # per-section pipeline (one source of truth for cr distributions).
    crp = _load_cr_pdfs(lsf_name)
    cr_grid_export = np.array(crp["cr_grid"])
    forecast_times = np.array(crp["forecast_times"], dtype=float)

    print(f"\nIntegrating spatial Pf grid against prior cr_pdf(t)...")
    print(f"  Corrosion model: {crp.get('model_type', 'unknown')}")
    print(f"  Forecast grid:   t = {forecast_times[0]:.1f} -> {forecast_times[-1]:.1f}, "
          f"n_t = {len(forecast_times)} (from {Path(crp.get('lsf_name', lsf_name)).name})")

    pf_section_t = np.zeros((len(forecast_times), n_sections))
    pf_system_t = np.zeros(len(forecast_times))
    prior_pdf_per_t = crp["prior_pdf_per_t"]
    for i, t in enumerate(tqdm(forecast_times, desc="t-integration",
                               unit="t", dynamic_ncols=True)):
        cr_pdf = np.asarray(prior_pdf_per_t[f"{t:.4f}"])
        pf_section_t[i], pf_system_t[i] = integration.over_cr(
            pf_section_grid, pf_system_grid, cr_values,
            cr_pdf_grid=cr_grid_export, cr_pdf_values=cr_pdf,
        )

    def _to_beta(pf):
        out = np.full_like(pf, np.nan, dtype=float)
        m = (pf > 0) & (pf < 1)
        out[m] = st.norm.ppf(1.0 - np.clip(pf[m], 1e-300, 1.0 - 1e-15))
        out[pf == 0] = np.inf
        out[pf == 1] = -np.inf
        return out

    beta_system_t = _to_beta(pf_system_t)
    beta_section_t = _to_beta(pf_section_t)
    pf_section_mean_t = pf_section_t.mean(axis=1)

    print(f"\n{'='*60}")
    print(f"Integrated result vs forecast time (prior cr_pdf)")
    print(f"{'='*60}")
    print(f"  t = {forecast_times[0]:>5.1f}:  "
          f"Pf_sys = {pf_system_t[0]:.4e}  (beta = {beta_system_t[0]:.3f})")
    print(f"  t = {forecast_times[-1]:>5.1f}:  "
          f"Pf_sys = {pf_system_t[-1]:.4e}  (beta = {beta_system_t[-1]:.3f})")
    print(f"  Output: {out_dir}")

    summary = {
        "lsf_name": lsf_name,
        "n_samples": n_samples,
        "seed": seed,
        "L": L,
        "n_sections": n_sections,
        "default_theta": default_theta,
        "default_rho_0": default_rho_0,
        "corrosion_model_type": crp.get("model_type", "unknown"),
        "forecast_times": forecast_times.tolist(),
        "pf_system_t": pf_system_t.tolist(),
        "beta_system_t": [float(b) if np.isfinite(b) else None for b in beta_system_t],
        "pf_section_t": pf_section_t.tolist(),   # (n_t, n_sections)
        "x_sections": x.tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots.pf_vs_cr(
        cr_values=cr_values,
        n_fail_section=n_fail_section,
        n_fail_system=n_fail_system,
        betas_form=betas_form,
        n_samples=n_samples,
        out_path=plots_dir / "pf_vs_cr.png",
    )
    plots.pf_vs_time(
        forecast_times=forecast_times,
        pf_section_t=pf_section_t,
        pf_system_t=pf_system_t,
        out_path=plots_dir / "pf_vs_time.png",
    )
    # Per-section beta snapshot at t_end (the most degraded state in the grid).
    i_last = len(forecast_times) - 1
    beta_single_at_tend_section = (
        float(st.norm.ppf(1 - pf_section_mean_t[i_last]))
        if 0 < pf_section_mean_t[i_last] < 1 else np.nan
    )
    plots.section_beta(
        pf_section_t=pf_section_t[i_last],
        x=x,
        beta_single=beta_single_at_tend_section,
        pf_system_t=pf_system_t[i_last],
        out_path=plots_dir / "beta_along_wall.png",
        t_label=f"t = {forecast_times[i_last]:.1f}",
    )
    # Realisations at cr = 0 (the cr point where the spatial spread is
    # measured before integration; useful as a visual sanity check for the
    # spatial sampler regardless of t).
    idx0 = int(np.argmin(np.abs(cr_values - 0.0)))
    L_eff_at_cr0 = covariance.effective_cholesky(
        x, {k: float(v) for k, v in points[idx0]["alphas"].items()}, spec,
    )
    plots.realizations(
        x=x, L_eff=L_eff_at_cr0, beta_single=float(betas_form[idx0]),
        n_show=10, seed=seed,
        out_path=plots_dir / "realizations.png",
    )
    print(f"  Plots: {plots_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf", type=str, default="lsf_wall",
                        help="LSF whose fragility cache to read.")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--theta", type=float, default=None,
                        help="Default spatial correlation length [m]. Falls "
                             "back to settings.json `spatial.default.theta` "
                             "(200 m if absent).")
    parser.add_argument("--rho-0", type=float, default=None,
                        help="Default long-range correlation floor in [0, 1]. "
                             "Falls back to settings.json `spatial.default."
                             "rho_0` (0.3 if absent).")
    args = parser.parse_args()

    analyze(
        lsf_name=args.lsf,
        n_samples=args.n_samples,
        seed=args.seed,
        theta=args.theta,
        rho_0=args.rho_0,
    )
