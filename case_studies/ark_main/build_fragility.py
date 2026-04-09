"""
Build a fragility curve for a D-SheetPiling model as a function of corrosion.

The corrosion ratio (r in [0, 1]) is the deterministic grid variable.
All soil and wall parameters are stochastic and assessed by FORM.
If FORM doesn't converge at a grid point, importance sampling is used.

Results are cached per grid point so the computation can be stopped
and resumed at any time.

Usage:
    python build_fragility.py
    python build_fragility.py --n_grid 21 --force_rebuild
"""

import json
import os
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from dotenv import load_dotenv

from src.io import get_remote_path
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.reliability_models.dsheetpiling import build_payload, apply_payload
from src.ptk import FragilityCurveBuilder
from src.ptk.lsf import build_lsf


_ENV = Path(__file__).parent / ".env"
_SECRETS_ENV = Path(__file__).parents[2] / "secrets" / ".env"


def load_api_key() -> str | None:
    """Load D-SheetPiling API key from secrets/.env if available."""
    if _SECRETS_ENV.exists():
        load_dotenv(_SECRETS_ENV)
    return os.environ.get("DSHEET_API_KEY")


def load_settings():
    remote = get_remote_path(_ENV)
    with open(remote / "input" / "settings.json", "r") as f:
        return json.load(f)


def build_stochastic_vars(variables: list) -> dict:
    """Convert settings variable defs to FragilityCurveBuilder format."""
    stochastic = {}
    for v in variables:
        name = v["name"]
        dist = v.get("distribution_type", "normal").lower()
        defn = {"distribution": dist, "mean": v["mean"]}
        std = v.get("standard_deviation")
        if std and v["mean"] != 0:
            defn["variation"] = abs(std / v["mean"])
        elif std:
            defn["deviation"] = std
        stochastic[name] = defn
    return stochastic


def make_lsf(settings: dict):
    """Build the LSF callable and the geomodel.

    The LSF takes all stochastic soil/wall params + corrosion_rate.
    It builds a payload, applies it to the model, executes, and
    returns the safety factor.
    """
    remote = get_remote_path(_ENV)
    config = settings["parameters"]

    # Load model (with optional API key for remote execution)
    api_key = load_api_key()
    geomodel_path = remote / "input" / "model.shi"
    geomodel = DSheetPiling(str(geomodel_path), api_key=api_key)
    if api_key:
        print("Using D-SheetPiling compute API")
    else:
        print("Using local D-SheetPiling execution")

    moment_cap = config["moment_cap"]
    start_thickness = config["start_thickness"]
    var_names = [v["name"] for v in settings["variables"]]

    def lsf_body(params):
        """LSF body: build payload, apply, execute, return safety factor.

        Receives physical-domain values from ptk (not standard normal).
        """
        corrosion_rate = params.pop("corrosion_rate", 0.0)

        # Add corrosion to the flat dict so build_payload routes it
        params["corrosion"] = corrosion_rate * start_thickness
        params["start_thickness"] = start_thickness

        payload = build_payload(params, geomodel)
        apply_payload(geomodel, payload)
        geomodel.execute()

        # Safety factor with degraded capacity
        max_moment = geomodel.results.max_moment
        if isinstance(max_moment, (list, np.ndarray)):
            max_moment = max_moment[0]
        moment_cap_degraded = moment_cap * (1.0 - corrosion_rate)
        return moment_cap_degraded / (abs(max_moment) + 1e-10)

    all_var_names = var_names + ["corrosion_rate"]
    lsf = build_lsf(all_var_names, lsf_body)

    return lsf


def main(n_grid: int = 11, force_rebuild: bool = False):
    settings = load_settings()

    print("=" * 60)
    print("Building fragility curve for D-SheetPiling")
    print("=" * 60)

    lsf = make_lsf(settings)
    stochastic_vars = build_stochastic_vars(settings["variables"])

    builder = FragilityCurveBuilder(
        lsf=lsf,
        stochastic_vars=stochastic_vars,
        deterministic_vars=["corrosion_rate"],
        form_params={
            "relaxation_factor": 0.15,
            "maximum_iterations": 100,
            "variation_coefficient": 0.05,
            "step_size": 0.05,
        },
    )

    grid = {"corrosion_rate": np.linspace(0.0, 1.0, n_grid)}
    cache_dir = get_remote_path(_ENV) / "output" / "fragility_curve"

    print(f"Grid: {n_grid} points, cache: {cache_dir}\n")

    fc = builder.build(
        grid=grid,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        verbose=True,
    )

    # Summary
    print(f"\n{'r':>8s} {'beta':>8s} {'Pf':>12s} {'method':>8s} {'ok':>5s}")
    print("-" * 45)
    for fp in fc.fragility_points:
        print(f"{fp.point[0]:>8.3f} {fp.beta:>8.3f} {fp.pf:>12.3e} {fp.method:>8s} {str(fp.convergence):>5s}")

    fc.save(cache_dir / "fragility_curve.json")
    print(f"\nSaved to {cache_dir / 'fragility_curve.json'}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_grid", type=int, default=11)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()
    main(n_grid=args.n_grid, force_rebuild=args.force_rebuild)
