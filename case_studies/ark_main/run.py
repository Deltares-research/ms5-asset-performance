"""
D-SheetPiling reliability analysis pipeline.

Uses the generic FragilityPipeline from src/ with:
- Cached fragility curve (from build_fragility.py)
- C50 Bayesian updating via corrosion observations
- Corrosion model for time-dependent corrosion ratio PDFs

Usage:
    python run.py
    python run.py --lsf lsf_wall
"""

import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from src.io import get_remote_path
from src.pipeline import FragilityPipeline
from src.plotting import save_figure, collect_pngs_to_pdf, make_gifs
from src.ptk import FragilityCurveBuilder
from jpdf import JPDF
from corrosion import CorrosionModel
from plotting import (plot_beta_forecast_at_time, plot_corrosion_forecast_at_time,
                      plot_moment_forecast_at_time, plot_end_of_life)


# ---------------------------------------------------------------------------
# Paths & settings
# ---------------------------------------------------------------------------

_ENV = Path(__file__).parent / ".env"
_remote = get_remote_path(_ENV)

with open(_remote / "input" / "settings.json", "r") as f:
    _settings = json.load(f)
_config = _settings["parameters"]


# ---------------------------------------------------------------------------
# Domain-specific callables for FragilityPipeline
# ---------------------------------------------------------------------------

_corrosion_model = None


def init_corrosion_model():
    global _corrosion_model
    _corrosion_model = CorrosionModel(
        C50_mu=_config.get("C50_mu", 1.0),
        C50_std=_config.get("C50_std", 0.75),
        corrosion_rate=_config["corrosion_rate"],
        wall_thickness=_config["wall_thickness"],
        obs_error_std=_config["obs_error_std"],
        t_start=_config["t_start"],
        n_grid=_config["n_C50_grid"],
        n_corrosion_grid=_config["n_cr_grid"],
    )


def get_det_pdfs(t, use_prior, **kwargs):
    """Return {var_name: (grid, pdf)} for each deterministic variable at time t."""
    jpdf = kwargs.get("_jpdf")
    setting_at_t = kwargs.get("setting_at_t", {})
    t_obs = kwargs.get("t_obs")

    C50_pdf = jpdf.C50_prior if use_prior else jpdf.C50_pdf
    last_obs = setting_at_t.get("corrosion") if not use_prior else None
    last_obs_time = t_obs if not use_prior else None

    cr_grid, cr_pdf = _corrosion_model.corrosion_ratio_pdf(
        t=t, C50_pdf=C50_pdf,
        last_obs_time=last_obs_time, last_obs=last_obs,
    )
    return {"corrosion_rate": (cr_grid, cr_pdf)}


def do_update(jpdf, obs_times, obs_values, t):
    jpdf.update_C50(obs_times, obs_values)


def do_reset(jpdf):
    jpdf.reset_C50_to_prior()


def build_state(jpdf):
    return {
        "C50_grid": jpdf.C50_grid.tolist(),
        "C50_prior": jpdf.C50_prior.tolist(),
        "C50_posterior": jpdf.C50_pdf.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(lsf_name: str = "lsf_wall"):
    print("=" * 60)
    print("D-SheetPiling Reliability Pipeline")
    print("=" * 60)

    # Load fragility curve
    cache_dir = _remote / "output" / f"fragility_curve_{lsf_name}"
    builder = FragilityCurveBuilder.__new__(FragilityCurveBuilder)
    fc_points = builder.load(cache_dir)
    n_cr_grid = _config.get("n_corrosion_grid", _config.get("n_grid", 1000))
    print(f"Loaded fragility curve '{lsf_name}': {len(fc_points)} points")
    print(f"Integration grid: {n_cr_grid} points")

    # Initialize corrosion model
    init_corrosion_model()

    # Initialize JPDF
    jpdf = JPDF(name="dsheet", config=_config)
    jpdf.set_prior_from_settings(_remote / "input" / "settings.json")

    # Load data
    with open(_remote / "input" / "data.json", "r") as f:
        data = json.load(f)

    times = sorted([float(k) for k in data.keys()])
    forecast_interval = _config.get("forecast_interval", 1)
    t_start = int(min(times))
    t_end = int(max(times))
    forecast_times = np.array(sorted(set(
        list(range(t_start, t_end + forecast_interval, forecast_interval)) +
        [int(t) for t in times]
    )), dtype=float)

    # Create pipeline
    pipeline = FragilityPipeline(
        settings_path=_remote / "input" / "settings.json",
        performance=None,  # not used — Pf comes from fragility integration
        config=_config,
        fragility_points=fc_points,
        det_var_names=["corrosion_rate"],
        n_integration_grid=n_cr_grid,
        get_det_pdfs=get_det_pdfs,
        do_update=do_update,
        do_reset=do_reset,
        build_state=build_state,
    )
    pipeline.jpdf = jpdf
    pipeline.init_times(
        obs_times=np.array(times),
        forecast_times=forecast_times,
    )

    # Run timeline — pass jpdf via kwargs so get_det_pdfs can access it
    results = pipeline.run_timeline(data, obs_key="corrosion", _jpdf=jpdf)

    # =========================================================================
    # Save results
    # =========================================================================
    username = os.environ.get("USER", "unknown").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = _remote / f"output/results/{lsf_name}/{username}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "reliability_results.json"
    with open(results_path, "w") as f:
        json.dump({str(t): r for t, r in results.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # =========================================================================
    # Plots
    # =========================================================================
    print("\n" + "=" * 60)
    print("Generating plots")
    print("=" * 60)

    plot_dir = output_dir / "plots"

    # Extract observations for plotting
    obs_times_list, obs_corrosion_list = [], []
    for t in times:
        key = str(t) if str(t) in data else f"{t:.1f}"
        if key in data and "corrosion" in data[key]:
            obs_times_list.append(t)
            obs_corrosion_list.append(data[key]["corrosion"])

    t_first, t_last = times[0], times[-1]
    wall_moment_capacity = _config["wall_moment_capacity"]
    wall_thickness = _config["wall_thickness"]
    obs_moment_cap = [wall_moment_capacity * (1 - c / wall_thickness) for c in obs_corrosion_list]

    # Build CR forecast PDFs for corrosion/moment plots
    cr_grid_plot = np.linspace(0.0, 1.0, n_cr_grid)

    def build_cr_forecasts(t_obs):
        """Build prior/posterior CR PDF dicts for all forecast times from t_obs."""
        key = str(t_obs) if str(t_obs) in data else f"{t_obs:.1f}"
        corr_obs = data[key]["corrosion"] if key in data else None
        future = [ft for ft in forecast_times if ft >= t_obs]
        pr, po = {}, {}
        for ft in future:
            g, p = _corrosion_model.corrosion_ratio_pdf(t=ft, C50_pdf=jpdf.C50_prior)
            pr[ft] = np.interp(cr_grid_plot, g, p, left=0, right=0).tolist()
            g, p = _corrosion_model.corrosion_ratio_pdf(
                t=ft, C50_pdf=jpdf.C50_pdf, last_obs_time=t_obs, last_obs=corr_obs,
            )
            po[ft] = np.interp(cr_grid_plot, g, p, left=0, right=0).tolist()
        return pr, po

    # Beta forecast
    png_dir = plot_dir / "beta_forecast"
    png_dir.mkdir(parents=True, exist_ok=True)
    for t in times:
        results_up_to_t = {k: v for k, v in results.items() if k <= t}
        fig = plot_beta_forecast_at_time(
            current_time=t,
            results=results_up_to_t,
            beta_req=_config.get("beta_req"),
        )
        save_figure(fig, png_dir / f"beta_forecast_t{int(t):03d}.png")
    collect_pngs_to_pdf(png_dir, plot_dir / "beta_forecast.pdf")
    print(f"Beta forecast plots saved to {png_dir}")

    # Corrosion forecast
    png_dir = plot_dir / "corrosion"
    png_dir.mkdir(parents=True, exist_ok=True)
    for current_t in obs_times_list:
        cr_pr, cr_po = build_cr_forecasts(current_t)
        fig = plot_corrosion_forecast_at_time(
            cr_grid=cr_grid_plot,
            cr_forecast_prior=cr_pr,
            cr_forecast_posterior=cr_po,
            obs_times=[t for t in obs_times_list if t <= current_t],
            obs_values=[c for t, c in zip(obs_times_list, obs_corrosion_list) if t <= current_t],
            obs_error_std=_config["obs_error_std"],
            wall_thickness=wall_thickness,
            xlim=(t_first, t_last),
            ylim=(0, wall_thickness),
        )
        save_figure(fig, png_dir / f"corrosion_t{int(current_t):03d}.png")
    collect_pngs_to_pdf(png_dir, plot_dir / "corrosion.pdf")
    print(f"Corrosion forecast plots saved to {png_dir}")

    # Moment capacity forecast
    png_dir = plot_dir / "moment"
    png_dir.mkdir(parents=True, exist_ok=True)
    for current_t in obs_times_list:
        cr_pr, cr_po = build_cr_forecasts(current_t)
        fig = plot_moment_forecast_at_time(
            cr_grid=cr_grid_plot,
            cr_forecast_prior=cr_pr,
            cr_forecast_posterior=cr_po,
            wall_moment_capacity=wall_moment_capacity,
            obs_times=[t for t in obs_times_list if t <= current_t],
            obs_moment_cap=[m for t, m in zip(obs_times_list, obs_moment_cap) if t <= current_t],
            xlim=(t_first, t_last),
            ylim=(0, wall_moment_capacity * 1.1),
        )
        save_figure(fig, png_dir / f"moment_t{int(current_t):03d}.png")
    collect_pngs_to_pdf(png_dir, plot_dir / "moment.pdf")
    print(f"Moment forecast plots saved to {png_dir}")

    # End of life
    png_dir = plot_dir / "end_of_life"
    png_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_end_of_life(
        results=results,
        beta_req=_config.get("beta_req", 2.3),
        t_start=_config.get("t_start", times[0]),
    )
    save_figure(fig, png_dir / "end_of_life.png")
    collect_pngs_to_pdf(png_dir, plot_dir / "end_of_life.pdf")
    print(f"End-of-life plot saved to {png_dir}")

    # GIFs
    make_gifs(plot_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf", type=str, default="lsf_wall",
                        help="LSF name (selects fragility_curve_{lsf}/)")
    args = parser.parse_args()
    main(lsf_name=args.lsf)
