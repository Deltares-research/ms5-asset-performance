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

# Geolib reads ``geolib.env`` relative to cwd when its ``MetaData`` BaseSettings
# is first instantiated (during ``import geolib`` from src.geotechnical_models).
# Hydrate it from the case-study copy here so this script works regardless of
# the cwd it is launched from.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / "geolib.env")

# Silence geolib chatter that would interleave with progress output.
# ``geolib.utils`` emits cosmetic ``run_identification`` newline warnings; the
# base_model logger emits a one-line error before raising ``CalculationError``
# whenever DSheetPiling.exe can't be located.
import logging
logging.getLogger("geolib.utils").setLevel(logging.ERROR)
logging.getLogger("geolib.models.base_model").setLevel(logging.CRITICAL)

from src.io import get_remote_path
from src.pipeline import FragilityPipeline
from src.ptk import FragilityCurveBuilder
from models.jpdf import JPDF
from models.corrosion import CorrosionModel
from plotting import save_all_plots
from reliability.build_fragility import main as build_fragility


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
    model_type = _config.get("model_type", "power")
    _corrosion_model = CorrosionModel(
        model_type=model_type,
        # linear-mode (50-75)
        C50_mu=_config.get("C50_mu", 1.5),
        C50_std=_config.get("C50_std", 0.75),
        corrosion_rate=_config.get("corrosion_rate", 0.022),
        t_start=_config.get("t_start", 50.0),
        C50_min=_config.get("C50_min", 0.5),
        C50_max=_config.get("C50_max", 2.5),
        # power-mode (0-50)
        power_A=_config.get("power_A", 0.091),
        B_mu=_config.get("B_mu", 0.72),
        B_std=_config.get("B_std", 0.05),
        B_min=_config.get("B_min", 0.4),
        B_max=_config.get("B_max", 1.0),
        # common
        wall_thickness=_config["wall_thickness"],
        obs_error_std=_config["obs_error_std"],
        n_grid=_config.get("n_B_grid", _config.get("n_C50_grid", 100)),
        n_corrosion_grid=_config["n_cr_grid"],
    )


def get_det_pdfs(t, use_prior, **kwargs):
    """Return {var_name: (grid, pdf)} for each deterministic variable at time t."""
    jpdf = kwargs.get("_jpdf")
    setting_at_t = kwargs.get("setting_at_t", {})
    t_obs = kwargs.get("t_obs")

    param_pdf = jpdf.param_prior if use_prior else jpdf.param_pdf
    last_obs = setting_at_t.get("corrosion") if not use_prior else None
    last_obs_time = t_obs if not use_prior else None

    cr_grid, cr_pdf = _corrosion_model.corrosion_ratio_pdf(
        t=t, param_pdf=param_pdf,
        last_obs_time=last_obs_time, last_obs=last_obs,
    )
    return {"corrosion_rate": (cr_grid, cr_pdf)}


def do_update(jpdf, obs_times, obs_values, t):
    jpdf.update(obs_times, obs_values)


def do_reset(jpdf):
    jpdf.reset_to_prior()


def build_state(jpdf):
    if jpdf.model_type == "power":
        return {
            "model_type": "power",
            "B_grid": jpdf.B_grid.tolist(),
            "B_prior": jpdf.B_prior.tolist(),
            "B_posterior": jpdf.B_pdf.tolist(),
        }
    return {
        "model_type": "linear",
        "C50_grid": jpdf.C50_grid.tolist(),
        "C50_prior": jpdf.C50_prior.tolist(),
        "C50_posterior": jpdf.C50_pdf.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
        lsf_name: str = "lsf_wall",
        dev_frag: bool = False,
        model_type: str | None = None,
):
    print("=" * 60)
    print("D-SheetPiling Reliability Pipeline")
    print("=" * 60)

    # CLI override of the corrosion model_type from settings.json.
    if model_type is not None:
        if model_type not in ("power", "linear"):
            raise ValueError(f"--model-type must be 'power' or 'linear', got {model_type!r}")
        _config["model_type"] = model_type
    print(f"Corrosion model: {_config.get('model_type', 'power')}")

    # Load fragility curve (build if missing)
    cache_dir = _remote / "output" / f"fragility_curve_{lsf_name}"
    if not cache_dir.exists() or not (cache_dir / "manifest.json").exists():
        print(f"Fragility curve not found for '{lsf_name}'. Building (dev mode)...")
        build_fragility(lsf_name=lsf_name, dev_frag=dev_frag)
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
    # Anchor the forecast grid at config['t_start'] so the prior line spans the
    # full horizon. For power mode t_start = 0 → C(0) = A * 0^B = 0.
    t_start = int(_config.get("t_start", min(times)))
    t_end = int(max(times))
    forecast_times = np.array(sorted(set(
        list(range(t_start, t_end + forecast_interval, forecast_interval)) +
        times  # keep exact (float) obs times in the forecast grid
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

    # =========================================================================
    # Corrosion-rate PDF export + alpha-importance plots
    # =========================================================================
    # Export the prior/posterior cr-PDFs over time to
    # <remote>/output/cr_pdfs_<lsf>.json, then use that file to build the
    # fragility-vs-corrosion alpha decomposition (pies + per-variable lines).
    print("\n" + "=" * 60)
    print("Exporting cr-PDFs + alpha-importance plots")
    print("=" * 60)

    # ``export_cr_pdfs`` lives in the case-study ``io`` package, whose name
    # shadows the stdlib ``io`` — load it by file path to avoid the clash.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "ark_export_cr_pdfs", Path(__file__).parent / "io" / "export_cr_pdfs.py")
    _export_cr_pdfs = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_export_cr_pdfs)
    _export_cr_pdfs.main(lsf_name=lsf_name)

    from analysis.alpha_pie_cross_section import make_plots as _alpha_pie
    from analysis.alpha_lines_cross_section import make_plots as _alpha_lines
    _alpha_pie(_remote, lsf_name, results_dir=output_dir)
    _alpha_lines(_remote, lsf_name, results_dir=output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf-name", type=str, default="lsf_wall", help="LSF name (selects fragility_curve_{lsf}/)")
    parser.add_argument( "--dev-frag", action="store_true")
    parser.add_argument(
        "--model-type", type=str, choices=["power", "linear"], default=None,
        help="Corrosion model: 'power' (0-50 yr, C(t)=A*t^B, B random) or "
             "'linear' (50-75 yr, C(t)=C50*(1+r/C50_mu*(t-t_start))). "
             "Overrides 'model_type' in settings.json. Default: use settings.",
    )
    args = parser.parse_args()

    main(
        lsf_name=args.lsf_name,
        dev_frag=args.dev_frag,
        model_type=args.model_type,
    )

