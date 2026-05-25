"""
Entry point for the spatial-variability MCS.

All configuration lives in ``<remote>/input/spatial_settings.json``. No CLI
flags — edit the file then run::

    python -m case_studies.ark_main.run_spatial

Pipeline:

1-5. ``compute_or_load_pf_grid`` — sample the spatial model, check per-sample
     per-location failure at every cr in the cached fragility curve,
     aggregate to Pf(cr, location), cache the table.
6.   Prior leg — integrate ``pf_grid`` against ``prior_pdf_per_t`` from
     ``cr_pdfs_<lsf>.json`` for every forecast time.
7.   Posterior leg — per obs time, sample cr as a Kriged spatial field
     anchored on the obs at x = 0, propagate to other sections through the
     cr spatial kernel, re-evaluate the per-section LSF; aggregate to
     per-section and system Pf.
8.   Write results (summary, forecasts, plots) into a fresh
     ``<remote>/output/results_spatial/<user>_<datetime>/`` folder.
"""
from __future__ import annotations

import json
import sys
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
from src.plotting import collect_pngs_to_pdf, make_gifs, save_figure
from reliability.build_fragility import load_settings
from plotting.timeline import plot_beta_forecast_at_time

from spatial import (
    cache_dir as _cache_dir,
    checkpoint as spatial_checkpoint,
    covariance,
    cr_field,
    fragility,
    integration,
    nested_mcs,
    plots,
    posterior_mcs,
    results_dir as _results_dir,
)


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


def _load_spatial_settings() -> dict:
    """Load ``<remote>/input/spatial_settings.json``."""
    path = _remote / "input" / "spatial_settings.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Create the spatial pipeline config file."
        )
    return json.load(open(path))


def analyze(spatial_settings: dict | None = None) -> None:
    """Run (or load) the spatial MCS, integrate against prior and posterior
    cr distributions, and produce summary + forecasts + plots under
    ``<remote>/output/results_spatial/<user>_<datetime>/``.

    All configuration comes from ``<remote>/input/spatial_settings.json``
    unless an explicit dict is passed in (handy for programmatic exploration
    without rewriting the file).

    The prior leg integrates the cached ``pf_grid`` (parametric in cr,
    uniform-cr assumption) against ``prior_pdf_per_t``. The posterior leg
    samples cr as a Kriged spatial field anchored on the observations at
    ``x = 0`` and re-evaluates the per-section LSF; system Pf is then
    consistent with the spatially-varying cr posterior.
    """
    settings_dict = spatial_settings if spatial_settings is not None else _load_spatial_settings()
    lsf_name = settings_dict["lsf_name"]
    n_samples = int(settings_dict["n_samples"])
    seed = int(settings_dict["seed"])
    # ``posterior_method`` selects the posterior-leg sampler:
    #   "field"  — current per-sample fragility interpolation (posterior_mcs)
    #   "nested" — nested-FORM tangent-hyperplane MCS (nested_mcs)
    # Both methods write to different cache files so they coexist in the same
    # signature folder.
    posterior_method = str(settings_dict.get("posterior_method", "field")).lower()
    if posterior_method not in ("field", "nested"):
        raise ValueError(
            f"Unknown posterior_method={posterior_method!r}. Use 'field' or 'nested'."
        )

    points = fragility.load_points(_remote, lsf_name)
    var_names = list(points[0]["alphas"].keys())
    L, n_sections, wall_theta, wall_rho_0, spec = covariance.resolve_config(
        var_names, settings_dict,
    )
    x = np.linspace(0.0, L, n_sections)

    # Cache and results live side-by-side under spatial_analysis/, both
    # keyed by the same setup signature (LSF, n_samples, seed, geometry,
    # spatial kernel params). Same config -> same folders; re-running with
    # an identical setup hit-caches and overwrites the results folder.
    cache_dir = _cache_dir(_remote, settings_dict)
    results_dir = _results_dir(_remote, settings_dict)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Spatial MCS — sheet-pile wall")
    print("=" * 60)
    print(f"LSF:              {lsf_name}")
    print(f"posterior method: {posterior_method}")
    print(f"L:                {L:.1f} m,  n_sections={n_sections}  (spacing {L/(n_sections-1):.1f} m)")
    print(f"wall theta:       {wall_theta:.1f} m")
    print(f"wall rho_0:       {wall_rho_0:.3f}")
    print(f"n_cr points:      {len(points)}")
    print(f"n_samples:        {n_samples}")
    print(f"seed:             {seed}")
    print(f"cache dir:        {cache_dir}")
    print(f"results dir:      {results_dir}")

    # Steps 1-5.
    data = compute_or_load_pf_grid(settings_dict)

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
    print(f"  Results: {results_dir}")

    summary = {
        "lsf_name": lsf_name,
        "n_samples": n_samples,
        "seed": seed,
        "L": L,
        "n_sections": n_sections,
        "wall_theta": wall_theta,
        "wall_rho_0": wall_rho_0,
        "corrosion_model_type": crp.get("model_type", "unknown"),
        "forecast_times": forecast_times.tolist(),
        "pf_system_t": pf_system_t.tolist(),
        "beta_system_t": [float(b) if np.isfinite(b) else None for b in beta_system_t],
        "pf_section_t": pf_section_t.tolist(),   # (n_t, n_sections)
        "x_sections": x.tolist(),
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    forecasts_dir = results_dir / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    def _jsonable_beta(arr: np.ndarray) -> list:
        """Convert a beta array to a JSON-serialisable nested list.

        ``+inf`` (zero Pf) → ``null``; ``-inf`` (Pf = 1) → ``null`` too. Use
        ``pf_*`` arrays in the same file if the exact failure rates matter.
        """
        a = np.asarray(arr, dtype=float)
        if a.ndim == 0:
            return None if not np.isfinite(a) else float(a)
        out = np.where(np.isfinite(a), a, np.nan).tolist()
        # JSON doesn't support NaN/inf — round-trip through None.
        def _walk(x):
            if isinstance(x, list):
                return [_walk(v) for v in x]
            return None if (isinstance(x, float) and not np.isfinite(x)) else x
        return _walk(out)

    prior_forecast = {
        "leg": "prior",
        "lsf_name": lsf_name,
        "n_samples": n_samples,
        "seed": seed,
        "L": L,
        "n_sections": n_sections,
        "x_sections": x.tolist(),
        "forecast_times": forecast_times.tolist(),
        "pf_section_t": pf_section_t.tolist(),                   # (n_t, n_sections)
        "pf_system_t":  pf_system_t.tolist(),                    # (n_t,)
        "beta_section_t": _jsonable_beta(beta_section_t),        # (n_t, n_sections)
        "beta_system_t":  _jsonable_beta(beta_system_t),         # (n_t,)
    }
    with open(forecasts_dir / "prior.json", "w") as f:
        json.dump(prior_forecast, f, indent=2)

    plots_dir = results_dir / "plots"
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

    # ----------------------------------------------------------------------
    # Posterior leg: per obs time, run a field-sampling MCS that anchors
    # the cr posterior at x = 0 (where obs are taken) and propagates to
    # other sections via the dedicated cr spatial kernel. System Pf is
    # exact in the model here (no uniform-cr approximation).
    # ----------------------------------------------------------------------
    theta_cr_val, rho_0_cr_val = covariance.resolve_cr_config(settings_dict)
    obs_times_sorted = sorted(
        float(k) for k in crp.get("posterior_pdf_per_obs", {}).keys()
    )

    print(f"\n{'='*60}")
    if posterior_method == "nested":
        print(f"Posterior leg — nested-FORM tangent-hyperplane MCS at x = 0")
    else:
        print(f"Posterior leg — field-sampling MCS conditioned at x = 0")
    print(f"{'='*60}")
    print(f"  cr kernel: theta = {theta_cr_val:.1f} m,  rho_0 = {rho_0_cr_val:.3f}")
    if obs_times_sorted:
        print(f"  obs scenarios: {len(obs_times_sorted)}  "
              f"({obs_times_sorted[0]:.1f} .. {obs_times_sorted[-1]:.1f})")
    else:
        print("  (no obs scenarios in file)")

    cached_post = spatial_checkpoint.try_load_posterior(
        cache_dir,
        lsf_name=lsf_name, n_samples=n_samples, seed=seed,
        n_sections=n_sections, L=L,
        wall_theta=wall_theta, wall_rho_0=wall_rho_0,
        cr_theta=theta_cr_val, cr_rho_0=rho_0_cr_val,
        cr_values=cr_values, betas=betas_form,
        obs_times=obs_times_sorted,
        method=posterior_method,
    )
    if cached_post is not None:
        posterior_results = cached_post["posterior"]
    elif obs_times_sorted:
        posterior_results: dict[str, dict] = {}
        for t_obs_str in sorted(crp["posterior_pdf_per_obs"].keys(), key=float):
            post_block = crp["posterior_pdf_per_obs"][t_obs_str]
            ft_block = np.array(
                sorted(float(t) for t in post_block.keys()), dtype=float,
            )
            if posterior_method == "nested":
                nfs, nfsys, precomp = nested_mcs.run_nested_mcs(
                    points=points, x=x, spec_basic=spec,
                    theta_cr=theta_cr_val, rho_0_cr=rho_0_cr_val,
                    cr_grid_export=cr_grid_export,
                    prior_pdf_per_t=crp["prior_pdf_per_t"],
                    posterior_pdf_per_t=post_block,
                    forecast_times=ft_block,
                    n_samples=n_samples, seed=seed,
                    desc=f"nested t_obs={float(t_obs_str):.1f}",
                )
            else:
                nfs, nfsys = posterior_mcs.run_posterior_mcs(
                    points=points, x=x, spec_basic=spec,
                    theta_cr=theta_cr_val, rho_0_cr=rho_0_cr_val,
                    cr_grid_export=cr_grid_export,
                    prior_pdf_per_t=crp["prior_pdf_per_t"],
                    posterior_pdf_per_t=post_block,
                    forecast_times=ft_block,
                    n_samples=n_samples, seed=seed,
                    desc=f"post t_obs={float(t_obs_str):.1f}",
                )
                precomp = None
            block_entry = {
                "forecast_times": ft_block.tolist(),
                "n_fail_section": nfs.tolist(),
                "n_fail_system": nfsys.tolist(),
                "pf_section": (nfs / n_samples).tolist(),
                "pf_system": (nfsys / n_samples).tolist(),
            }
            if precomp is not None:
                # Persist the full nested-FORM decomposition so plots can use
                # it on cache hits without recomputing. alpha_basic is the
                # heaviest field — (n_t, N, n_active) floats per obs — but
                # still small (~tens of KB) for realistic n_t / N / n_active.
                block_entry["nested_form"] = {
                    "beta_T":      precomp["beta_T"].tolist(),
                    "alpha_cr":    precomp["alpha_cr"].tolist(),
                    "alpha_basic": precomp["alpha_basic"].tolist(),
                    "active_vars": precomp.get("active_vars", []),
                    "xi_star":     precomp["xi_star"].tolist(),
                    "cr_star":     precomp["cr_star"].tolist(),
                    "mu_z":        precomp["mu_z"].tolist(),
                    "sigma_z":     precomp["sigma_z"].tolist(),
                }
            posterior_results[t_obs_str] = block_entry
        spatial_checkpoint.save_posterior(cache_dir, {
            "lsf_name": lsf_name,
            "method": posterior_method,
            "n_samples": int(n_samples),
            "seed": int(seed),
            "n_sections": int(n_sections),
            "L": float(L),
            "wall_theta": float(wall_theta),
            "wall_rho_0": float(wall_rho_0),
            "cr_theta": float(theta_cr_val),
            "cr_rho_0": float(rho_0_cr_val),
            "cr_values": cr_values.tolist(),
            "betas": betas_form.tolist(),
            "obs_times": obs_times_sorted,
            "posterior": posterior_results,
        }, method=posterior_method)
        print(f"  Saved posterior grid to "
              f"{spatial_checkpoint.posterior_path(cache_dir, method=posterior_method)}")
    else:
        posterior_results = {}

    if posterior_results:
        # One-line summary per obs scenario at t_end.
        for t_obs_str in sorted(posterior_results.keys(), key=float):
            block = posterior_results[t_obs_str]
            pf_sys_end = float(block["pf_system"][-1])
            t_end_block = float(block["forecast_times"][-1])
            print(f"  t_obs = {float(t_obs_str):>5.1f}:  "
                  f"Pf_sys(t={t_end_block:.1f}) = {pf_sys_end:.4e}")

        # forecasts/posterior.json — same shape as forecasts/prior.json but
        # nested by ``t_obs``. Each per-obs block contains the slice of
        # forecast times after that obs only (matches the exporter file).
        posterior_forecast: dict = {
            "leg": "posterior",
            "lsf_name": lsf_name,
            "n_samples": n_samples,
            "seed": seed,
            "L": L,
            "n_sections": n_sections,
            "x_sections": x.tolist(),
            "cr_theta": float(theta_cr_val),
            "cr_rho_0": float(rho_0_cr_val),
            "obs_times": obs_times_sorted,
            "per_obs": {},
        }
        for t_obs_str in sorted(posterior_results.keys(), key=float):
            block = posterior_results[t_obs_str]
            pf_sec = np.asarray(block["pf_section"], dtype=float)
            pf_sys = np.asarray(block["pf_system"], dtype=float)
            posterior_forecast["per_obs"][t_obs_str] = {
                "forecast_times": block["forecast_times"],
                "pf_section_t": pf_sec.tolist(),
                "pf_system_t":  pf_sys.tolist(),
                "beta_section_t": _jsonable_beta(_to_beta(pf_sec)),
                "beta_system_t":  _jsonable_beta(_to_beta(pf_sys)),
            }
        with open(forecasts_dir / "posterior.json", "w") as f:
            json.dump(posterior_forecast, f, indent=2)

        plots.pf_vs_time_posterior(
            forecast_times_prior=forecast_times,
            pf_system_prior=pf_system_t,
            posterior_results=posterior_results,
            out_path=plots_dir / "pf_vs_time_posterior.png",
        )

        # ---- System-β forecast PNG-per-obs-time + PDF + GIF, mirroring
        # run.py's beta_forecast plot but on the system (series) beta. We
        # reshape our prior/posterior arrays into the per-obs-time dict
        # shape that ``plotting.timeline.plot_beta_forecast_at_time``
        # already expects, then drive PNG/PDF/GIF rendering identically.
        prior_beta_forecast = {
            float(t): float(b)
            for t, b in zip(forecast_times, beta_system_t)
            if np.isfinite(b)
        }
        results_for_plot: dict[float, dict] = {}
        for t_obs_str in sorted(posterior_results.keys(), key=float):
            t_obs = float(t_obs_str)
            block = posterior_results[t_obs_str]
            pf_sys_block = np.asarray(block["pf_system"], dtype=float)
            ft_block = np.asarray(block["forecast_times"], dtype=float)
            beta_block = _to_beta(pf_sys_block)
            bf_post = {
                float(t): float(b)
                for t, b in zip(ft_block, beta_block)
                if np.isfinite(b)
            }
            beta_post_at_tobs = (
                bf_post[min(bf_post.keys())] if bf_post else float("nan")
            )
            beta_prior_at_tobs = prior_beta_forecast.get(
                t_obs,
                float(np.interp(t_obs, forecast_times, beta_system_t)),
            )
            results_for_plot[t_obs] = {
                "prior": {
                    "beta": beta_prior_at_tobs,
                    "beta_forecast": prior_beta_forecast,
                },
                "posterior": {
                    "beta": beta_post_at_tobs,
                    "beta_forecast": bf_post,
                },
            }

        beta_req = _settings.get("parameters", {}).get("beta_req", 2.3)
        bf_png_dir = plots_dir / "beta_forecast_system"
        bf_png_dir.mkdir(parents=True, exist_ok=True)
        for t_obs in sorted(results_for_plot.keys()):
            results_up_to_t = {
                k: v for k, v in results_for_plot.items() if k <= t_obs
            }
            fig = plot_beta_forecast_at_time(
                current_time=t_obs,
                results=results_up_to_t,
                beta_req=beta_req,
            )
            save_figure(
                fig,
                bf_png_dir / f"beta_forecast_system_t{t_obs:06.2f}.png",
            )
        collect_pngs_to_pdf(bf_png_dir, plots_dir / "beta_forecast_system.pdf")

        # ---- Spatial cr profile along the wall, per obs time. For each
        # t_obs we show the obs-conditioned cr distribution at t = t_obs
        # (the moment of observation): posterior at x = 0 sits on the
        # observation, fans out to the prior at sections beyond the cr
        # correlation length. PDF + GIF picked up downstream.
        wall_thickness = float(
            _settings.get("parameters", {}).get("wall_thickness", 9.5)
        )
        obs_error_std = float(
            _settings.get("parameters", {}).get("obs_error_std", 0.4)
        )

        # Read observation values from data.json (for the obs marker).
        data_path = _remote / "input" / "data.json"
        if data_path.exists():
            data_json = json.load(open(data_path))
            obs_value_at_time = {
                float(d["time"]): float(d["corrosion"])
                for d in data_json.values()
                if "time" in d and "corrosion" in d
            }
        else:
            obs_value_at_time = {}

        # One y-axis range shared by every cr-along-wall frame so the GIF
        # doesn't rescale between obs scenarios. Take the largest of:
        #   * prior q95 of cr at t_end (worst-case forecast scatter)
        #   * largest observed corrosion plus its 95% noise envelope
        # Multiply by 1.10 for headroom, floor at 2 mm so a near-zero
        # range doesn't squash the plot, and cap at wall_thickness so we
        # never overshoot the physical bound.
        t_end_key = f"{forecast_times[-1]:.4f}"
        prior_pdf_tend = np.asarray(crp["prior_pdf_per_t"][t_end_key])
        cdf_tend = cr_field.cdf_on_grid(prior_pdf_tend, cr_grid_export)
        cr_q95_tend = float(cr_field.inv_cdf_at(
            np.array([0.95]), cr_grid_export, cdf_tend
        ).item())
        ymax_from_prior = cr_q95_tend * wall_thickness
        if obs_value_at_time:
            ymax_from_obs = max(
                v + 1.96 * obs_error_std for v in obs_value_at_time.values()
            )
        else:
            ymax_from_obs = 0.0
        ymax_mm = max(2.0, ymax_from_prior, ymax_from_obs) * 1.10
        ymax_mm = min(ymax_mm, wall_thickness)

        cr_png_dir = plots_dir / "cr_along_wall"
        cr_png_dir.mkdir(parents=True, exist_ok=True)
        for t_obs_str in sorted(crp["posterior_pdf_per_obs"].keys(), key=float):
            post_block = crp["posterior_pdf_per_obs"][t_obs_str]
            # The posterior at the obs time is keyed by t_obs in the file.
            if t_obs_str not in post_block:
                continue
            t_obs = float(t_obs_str)
            prior_pdf_at_t = np.asarray(crp["prior_pdf_per_t"][t_obs_str])
            post_pdf_at_t = np.asarray(post_block[t_obs_str])
            prior_q, cond_q = cr_field.conditional_cr_quantiles(
                x=x, x0_idx=0,
                theta_cr=theta_cr_val, rho_0_cr=rho_0_cr_val,
                prior_pdf=prior_pdf_at_t,
                posterior_pdf=post_pdf_at_t,
                cr_grid=cr_grid_export,
                quantiles=(0.05, 0.5, 0.95),
            )
            obs_value_mm = None
            for t_data, c_mm in obs_value_at_time.items():
                if abs(t_data - t_obs) < 1e-3:
                    obs_value_mm = c_mm
                    break
            plots.cr_along_wall(
                x=x, prior_q=prior_q, cond_q=cond_q,
                t_obs=t_obs, t_at=t_obs,
                wall_thickness=wall_thickness,
                obs_value_mm=obs_value_mm,
                obs_error_std_mm=obs_error_std,
                ylim_mm=(0.0, ymax_mm),
                out_path=cr_png_dir / f"cr_along_wall_t{t_obs:06.2f}.png",
            )
        collect_pngs_to_pdf(cr_png_dir, plots_dir / "cr_along_wall.pdf")

        # ---- Per-section split-violin plot of the cr distribution
        # (prior on the left half, posterior on the right). Same y-axis
        # cap as ``cr_along_wall`` so the GIFs are directly comparable.
        violin_png_dir = plots_dir / "cr_violin"
        violin_png_dir.mkdir(parents=True, exist_ok=True)
        n_violin_samples = 5000
        violin_rng = np.random.default_rng(int(settings_dict["seed"]) + 1234)
        for t_obs_str in sorted(crp["posterior_pdf_per_obs"].keys(), key=float):
            post_block = crp["posterior_pdf_per_obs"][t_obs_str]
            if t_obs_str not in post_block:
                continue
            t_obs = float(t_obs_str)
            prior_pdf_at_t = np.asarray(crp["prior_pdf_per_t"][t_obs_str])
            post_pdf_at_t = np.asarray(post_block[t_obs_str])

            # Prior cr samples are stationary across the wall — draw once.
            F_prior = cr_field.cdf_on_grid(prior_pdf_at_t, cr_grid_export)
            u_prior = violin_rng.uniform(size=n_violin_samples)
            cr_prior_samples = cr_field.inv_cdf_at(
                u_prior, cr_grid_export, F_prior,
            )

            # Posterior cr field — Kriged conditional at every section.
            m_post, v_post = cr_field.z_moments_under_posterior(
                prior_pdf_at_t, post_pdf_at_t, cr_grid_export,
            )
            mean_factor, L_z = cr_field.kriged_chol(
                x=x, x0_idx=0,
                theta_cr=theta_cr_val, rho_0_cr=rho_0_cr_val,
                v_post=v_post,
            )
            cr_post_samples = cr_field.sample_cr_field(
                violin_rng, n_violin_samples,
                mean_factor, L_z, m_post,
                prior_pdf_at_t, cr_grid_export,
            )

            obs_value_mm = None
            for t_data, c_mm in obs_value_at_time.items():
                if abs(t_data - t_obs) < 1e-3:
                    obs_value_mm = c_mm
                    break

            plots.cr_along_wall_violin(
                x=x,
                cr_prior_samples_mm=cr_prior_samples * wall_thickness,
                cr_post_samples_mm=cr_post_samples * wall_thickness,
                t_obs=t_obs, t_at=t_obs,
                wall_thickness=wall_thickness,
                obs_value_mm=obs_value_mm,
                obs_error_std_mm=obs_error_std,
                ylim_mm=(0.0, ymax_mm),
                out_path=violin_png_dir / f"cr_violin_t{t_obs:06.2f}.png",
            )
        collect_pngs_to_pdf(violin_png_dir, plots_dir / "cr_violin.pdf")

        # ---- Nested-FORM alpha plots (only when posterior_method == "nested").
        # Per obs scenario, render two views of the (section, t) alphas:
        #   * alpha_heatmap   — RdBu heatmap per variable, one panel each, in
        #     a single figure. Shows the spatial+temporal pattern of every
        #     contribution at a glance.
        #   * alpha_lines     — alpha_v(t) lines at first/middle/last sections
        #     for direct comparison across distance from the obs anchor.
        if posterior_method == "nested" and posterior_results:
            heat_dir = plots_dir / "alpha_heatmap"
            lines_dir = plots_dir / "alpha_lines"
            heat_dir.mkdir(parents=True, exist_ok=True)
            lines_dir.mkdir(parents=True, exist_ok=True)
            # Stable per-obs section picks: ends + middle.
            section_indices = sorted(set([0, n_sections // 2, n_sections - 1]))
            for t_obs_str in sorted(posterior_results.keys(), key=float):
                nf = posterior_results[t_obs_str].get("nested_form")
                if nf is None:
                    continue
                t_obs = float(t_obs_str)
                ft = np.asarray(
                    posterior_results[t_obs_str]["forecast_times"], dtype=float,
                )
                alpha_cr_arr = np.asarray(nf["alpha_cr"], dtype=float)
                alpha_basic_arr = np.asarray(nf["alpha_basic"], dtype=float)
                active_vars = list(nf.get("active_vars", []))
                plots.alpha_heatmap(
                    forecast_times=ft, x=x,
                    alpha_cr=alpha_cr_arr,
                    alpha_basic=alpha_basic_arr,
                    active_vars=active_vars,
                    t_obs=t_obs,
                    out_path=heat_dir / f"alpha_heatmap_t{t_obs:06.2f}.png",
                )
                plots.alpha_lines_at_sections(
                    forecast_times=ft, x=x,
                    alpha_cr=alpha_cr_arr,
                    alpha_basic=alpha_basic_arr,
                    active_vars=active_vars,
                    section_indices=section_indices,
                    t_obs=t_obs,
                    out_path=lines_dir / f"alpha_lines_t{t_obs:06.2f}.png",
                )
            collect_pngs_to_pdf(heat_dir, plots_dir / "alpha_heatmap.pdf")
            collect_pngs_to_pdf(lines_dir, plots_dir / "alpha_lines.pdf")

    # GIFs from every PNG subdirectory under plots_dir (beta_forecast_system,
    # cr_along_wall, cr_violin). Same call ``run.py`` uses to bundle its
    # animations.
    make_gifs(plots_dir)

    print(f"  Plots: {plots_dir}")


if __name__ == "__main__":
    analyze()
