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
from src.ptk import FragilityCurveBuilder
from jpdf import JPDF
from corrosion import CorrosionModel
from plotting import save_all_plots
from build_fragility import main as build_fragility


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

    # Load fragility curve (build if missing)
    cache_dir = _remote / "output" / f"fragility_curve_{lsf_name}"
    if not cache_dir.exists() or not (cache_dir / "manifest.json").exists():
        print(f"Fragility curve not found for '{lsf_name}'. Building (dev mode)...")
        build_fragility(lsf_name=lsf_name, dev_frag=True)
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

    save_all_plots(
        results=results,
        data=data,
        config=_config,
        corrosion_model=_corrosion_model,
        jpdf=jpdf,
        forecast_times=forecast_times,
        n_cr_grid=n_cr_grid,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf", type=str, default="lsf_wall",
                        help="LSF name (selects fragility_curve_{lsf}/)")
    args = parser.parse_args()
    main(lsf_name=args.lsf)
