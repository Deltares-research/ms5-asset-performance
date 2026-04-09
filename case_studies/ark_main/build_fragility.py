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
import tempfile
import numpy as np
from copy import deepcopy
from pathlib import Path
from argparse import ArgumentParser
from dotenv import load_dotenv

from src.io import get_remote_path
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.ptk import FragilityCurveBuilder


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ENV = Path(__file__).parent / ".env"
_SECRETS_ENV = Path(__file__).parents[2] / "secrets" / ".env"
WORK_DIR = Path(tempfile.mkdtemp(prefix="dsheet_fc_")).resolve()


def load_api_key() -> str | None:
    """Load D-SheetPiling API key from secrets/.env if available."""
    if _SECRETS_ENV.exists():
        load_dotenv(_SECRETS_ENV)
    return os.environ.get("DSHEET_API_KEY")


def load_settings() -> dict:
    remote = get_remote_path(_ENV)
    with open(remote / "input" / "settings.json", "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model & LSF
# ---------------------------------------------------------------------------

# Load model once — shared across all LSF evaluations
_settings = load_settings()
_config = _settings["parameters"]
_api_key = load_api_key()
_remote = get_remote_path(_ENV)
_geomodel_path = _remote / "input" / "model.shi"
_base_model = DSheetPiling(str(_geomodel_path), api_key=_api_key)

MOMENT_CAP = _config["moment_cap"]
START_THICKNESS = _config["start_thickness"]


def lsf(
    Klei_soilphi,
    Klei_soilcohesion,
    Klei_soilcurkb1,
    Zand_soilphi,
    Zand_soilcurkb1,
    Zandvast_soilphi,
    Zandvast_soilcurkb1,
    Zandlos_soilphi,
    Zandlos_soilcurkb1,
    Wall_SheetPilingElementEI,
    corrosion_rate,
):
    """Limit state function: g = M_capacity(r) - |M_max|

    Parameters
    ----------
    Klei_soilphi : float
        Friction angle of clay layer [deg].
    Klei_soilcohesion : float
        Cohesion of clay layer [kPa].
    Klei_soilcurkb1 : float
        Subgrade reaction modulus of clay [kN/m3].
    Zand_soilphi : float
        Friction angle of sand layer [deg].
    Zand_soilcurkb1 : float
        Subgrade reaction modulus of sand [kN/m3].
    Zandvast_soilphi : float
        Friction angle of dense sand layer [deg].
    Zandvast_soilcurkb1 : float
        Subgrade reaction modulus of dense sand [kN/m3].
    Zandlos_soilphi : float
        Friction angle of loose sand layer [deg].
    Zandlos_soilcurkb1 : float
        Subgrade reaction modulus of loose sand [kN/m3].
    Wall_SheetPilingElementEI : float
        Elastic stiffness of the sheet pile [kNm2/m].
    corrosion_rate : float
        Corrosion ratio [-], 0 (intact) to 1 (fully corroded).
    """
    # Degraded section properties
    factor = 1.0 - corrosion_rate
    m_capacity = MOMENT_CAP * factor
    ei = Wall_SheetPilingElementEI * factor

    # Deep-copy and update model
    model = deepcopy(_base_model)

    # Update soil parameters
    model.update_soils({
        "Klei": {"soilphi": Klei_soilphi, "soilcohesion": Klei_soilcohesion, "soilcurkb1": Klei_soilcurkb1},
        "Zand": {"soilphi": Zand_soilphi, "soilcurkb1": Zand_soilcurkb1},
        "Zandvast": {"soilphi": Zandvast_soilphi, "soilcurkb1": Zandvast_soilcurkb1},
        "Zandlos": {"soilphi": Zandlos_soilphi, "soilcurkb1": Zandlos_soilcurkb1},
    })

    # Update wall stiffness (degraded by corrosion)
    model.update_wall({"SheetPilingElementEI": ei})

    # Execute
    model.execute()

    # Limit state: capacity - demand
    max_moment = model.results.max_moment
    if isinstance(max_moment, (list, np.ndarray)):
        max_moment = max_moment[0]

    return m_capacity - abs(max_moment)


# ---------------------------------------------------------------------------
# Stochastic variable definitions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_grid: int = 11, force_rebuild: bool = False):
    print("=" * 60)
    print("Building fragility curve for D-SheetPiling")
    print("=" * 60)
    print(f"API: {'yes' if _api_key else 'no (local)'}")
    print(f"Work dir: {WORK_DIR}")

    stochastic_vars = build_stochastic_vars(_settings["variables"])

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
    cache_dir = _remote / "output" / "fragility_curve"

    print(f"Grid: {n_grid} points, cache: {cache_dir}\n")

    results = builder.build(
        grid=grid,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        verbose=True,
    )

    # Summary
    print(f"\n{'r':>8s} {'beta':>8s} {'Pf':>12s} {'method':>8s} {'ok':>5s}")
    print("-" * 45)
    for r in results:
        cr = r["point"]["corrosion_rate"]
        print(f"{cr:>8.3f} {r['beta']:>8.3f} {r['pf']:>12.3e} {r['method']:>8s} {str(r['convergence']):>5s}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_grid", type=int, default=11)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()
    main(n_grid=args.n_grid, force_rebuild=args.force_rebuild)
