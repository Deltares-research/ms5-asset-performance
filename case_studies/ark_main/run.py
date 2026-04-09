"""
D-SheetPiling reliability analysis pipeline.

Uses the cached fragility curve (from build_fragility.py) and integrates
it with a time-dependent corrosion ratio PDF to compute beta(t).

The corrosion ratio PDF is obtained by marginalizing the C50 distribution
(updated via Bayesian inference from corrosion observations) through the
corrosion model.

Usage:
    python run.py
"""

import json
import os
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.integrate import trapezoid
from argparse import ArgumentParser

from src.io import get_remote_path
from src.ptk import FragilityCurveBuilder
from case_studies.ark_example.jpdf import JPDF
from case_studies.ark_example.corrosion import CorrosionModel


# ---------------------------------------------------------------------------
# Paths & settings
# ---------------------------------------------------------------------------

_ENV = Path(__file__).parent / ".env"
_remote = get_remote_path(_ENV)

with open(_remote / "input" / "settings.json", "r") as f:
    _settings = json.load(f)
_config = _settings["parameters"]


# ---------------------------------------------------------------------------
# Load fragility curve from cache
# ---------------------------------------------------------------------------

def load_fragility_curve(lsf_name: str) -> list[dict]:
    """Load the cached fragility curve points for a given LSF.

    Args:
        lsf_name: Name of the LSF function (e.g. "lsf_wall").
            Cache is read from .../fragility_curve_{lsf_name}/
    """
    cache_dir = _remote / "output" / f"fragility_curve_{lsf_name}"
    builder = FragilityCurveBuilder.__new__(FragilityCurveBuilder)
    return builder.load(cache_dir)


def integrate_fragility(
    fc_points: list[dict],
    cr_grid: np.ndarray,
    cr_pdf: np.ndarray,
) -> tuple[float, float]:
    """Integrate fragility curve against a corrosion ratio PDF.

    Interpolates the fragility Pf onto the corrosion ratio grid and
    integrates Pf(r) * f(r) dr.

    Args:
        fc_points: List of fragility point dicts.
        cr_grid: Corrosion ratio grid from the corrosion model.
        cr_pdf: PDF of corrosion ratio on that grid.

    Returns:
        Tuple of (pf, beta).
    """
    fc_r = np.array([p["point"]["corrosion_rate"] for p in fc_points])
    fc_pf = np.array([p["pf"] for p in fc_points])

    # Interpolate fragility Pf onto the corrosion ratio grid
    pf_interp = np.interp(cr_grid, fc_r, fc_pf, left=fc_pf[0], right=fc_pf[-1])

    # Integrate: Pf = integral( Pf(r) * f(r) dr )
    pf = float(trapezoid(pf_interp * cr_pdf, cr_grid))
    pf = np.clip(pf, 1e-30, 1 - 1e-10)
    beta = float(stats.norm.ppf(1 - pf))
    return pf, beta


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main(lsf_name: str = "lsf_wall"):
    print("=" * 60)
    print("D-SheetPiling Reliability Pipeline")
    print("=" * 60)

    # Load fragility curve
    fc_points = load_fragility_curve(lsf_name)
    print(f"Loaded fragility curve '{lsf_name}': {len(fc_points)} points")

    # Initialize JPDF (for C50 Bayesian updating)
    jpdf = JPDF(name="dsheet", config=_config)
    jpdf.set_prior_from_settings(_remote / "input" / "settings.json")

    # Initialize corrosion model
    corrosion_model = CorrosionModel(
        C50_mu=_config.get("C50_mu", 1.0),
        C50_std=_config.get("C50_std", 0.75),
        corrosion_rate=_config["corrosion_rate"],
        start_thickness=_config["start_thickness"],
        obs_error_std=_config["obs_error_std"],
        t_start=_config["t_start"],
        n_grid=_config["n_C50_grid"],
        n_corrosion_grid=_config["n_grid"],
    )

    # Load observation data
    with open(_remote / "input" / "data.json", "r") as f:
        data = json.load(f)

    times = sorted([float(k) for k in data.keys()])

    # Build forecast time grid
    forecast_interval = _config.get("forecast_interval", 1)
    t_start = int(min(times))
    t_end = int(max(times))
    forecast_times = list(range(t_start, t_end + forecast_interval, forecast_interval))
    forecast_times = sorted(set([float(t) for t in forecast_times] + times))

    # Collect observations
    def get_observations_up_to(t):
        obs_t, obs_v = [], []
        for key in sorted(data.keys()):
            if float(key) <= t:
                obs_t.append(float(key))
                obs_v.append(data[key]["corrosion"])
        return np.array(obs_t), np.array(obs_v)

    # Run timeline
    jpdf.reset_C50_to_prior()
    results = {}

    header = f"{'t':>6s} {'b_prior':>10s} {'b_post':>10s}"
    print(header)
    print("-" * len(header))

    for t in times:
        obs_times, obs_values = get_observations_up_to(t)

        # Bayesian update of C50
        if len(obs_times) > 0:
            jpdf.update_C50(obs_times, obs_values)

        key = str(t) if str(t) in data else f"{t:.1f}"
        corrosion_obs = data[key]["corrosion"] if key in data else None

        # Forecast at all future times
        future_times = [ft for ft in forecast_times if ft >= t]
        beta_forecast_prior = {}
        beta_forecast_posterior = {}

        for ft in future_times:
            # Prior forecast
            cr_grid_pr, cr_pdf_pr = corrosion_model.corrosion_ratio_pdf(
                t=ft, C50_pdf=jpdf.C50_prior,
            )
            pf_pr, beta_pr = integrate_fragility(fc_points, cr_grid_pr, cr_pdf_pr)
            beta_forecast_prior[ft] = beta_pr

            # Posterior forecast
            cr_grid_po, cr_pdf_po = corrosion_model.corrosion_ratio_pdf(
                t=ft, C50_pdf=jpdf.C50_pdf,
                last_obs_time=t, last_obs=corrosion_obs,
            )
            pf_po, beta_po = integrate_fragility(fc_points, cr_grid_po, cr_pdf_po)
            beta_forecast_posterior[ft] = beta_po

        t_min = min(future_times)
        results[t] = {
            "time": t,
            "prior": {
                "beta": beta_forecast_prior[t_min],
                "beta_forecast": beta_forecast_prior,
            },
            "posterior": {
                "beta": beta_forecast_posterior[t_min],
                "beta_forecast": beta_forecast_posterior,
            },
            "jpdf_state": {
                "C50_grid": jpdf.C50_grid.tolist(),
                "C50_prior": jpdf.C50_prior.tolist(),
                "C50_posterior": jpdf.C50_pdf.tolist(),
            },
        }

        print(f"{t:>6.0f} {beta_forecast_prior[t_min]:>10.2f} {beta_forecast_posterior[t_min]:>10.2f}")

    # Save results
    output_dir = _remote / "output" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "reliability_results.json"
    with open(results_path, "w") as f:
        json.dump({str(t): r for t, r in results.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf", type=str, default="lsf_wall", help="LSF name (selects fragility_curve_{lsf}/)")
    args = parser.parse_args()
    main(lsf_name=args.lsf)
