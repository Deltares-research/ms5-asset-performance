"""Force importance sampling at selected cr indices.

PTK's FragilityCurveBuilder runs FORM first and only falls back to IS when
FORM reports `is_converged == False`. For lsf_wall, FORM at cr >= 0.3 stalls
into a false-converged state (small/negative beta with `is_converged=True`),
so the auto-fallback never triggers. This script bypasses that by setting up
the same PTK project as build_fragility.py and explicitly running IS for a
chosen index range, overwriting the corresponding point_NNNN.json files.

Usage:
    runfile('rebuild_is.py')                                # default: cr >= 0.3
    runfile('rebuild_is.py', args='--lsf-name lsf_wall --indices 3,4,5,6,7,8')
"""

import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from argparse import ArgumentParser

# Allow `from reliability.…` / `from src.…` when run as a script from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.io import get_remote_path
from reliability.build_fragility import (
    init_model, _settings, _config, _remote,
    LSF_REGISTRY, build_stochastic_vars, wrap_lsf_with_deterministic,
)


def run_is_at(project, ptk, cr_value: float, index: int) -> dict:
    """Force IS at corrosion_rate = cr_value and return a serializable result."""
    project.variables["corrosion_rate"].distribution = ptk.DistributionType.deterministic
    project.variables["corrosion_rate"].mean = float(cr_value)

    project.settings.reliability_method = ptk.ReliabilityMethod.importance_sampling
    print(f"  [{index:04d}] IS at corrosion_rate={cr_value:.4g} ...", end="", flush=True)
    project.run()
    dp = project.design_point

    pf = float(dp.probability_failure)
    beta = float(dp.reliability_index)
    print(f" beta={beta:.3f}, Pf={pf:.3e}")

    det_names = {"corrosion_rate"}
    return {
        "index": int(index),
        "point": {"corrosion_rate": float(cr_value)},
        "pf": pf,
        "beta": beta,
        "logpf": math.log(pf) if pf > 0 else -math.inf,
        "convergence": bool(dp.is_converged),
        "method": "importance_sampling",
        "design_point": {
            a.variable.name: float(a.x) for a in dp.alphas
            if a.variable.name not in det_names
        },
        "alphas": {
            a.variable.name: float(a.alpha) for a in dp.alphas
            if a.variable.name not in det_names
        },
    }


def main(lsf_name: str = "lsf_wall", indices: list[int] | None = None,
         use_api: bool = False) -> None:
    if lsf_name not in LSF_REGISTRY:
        raise ValueError(f"Unknown LSF '{lsf_name}'. Available: {list(LSF_REGISTRY.keys())}")

    cache_dir = _remote / "output" / f"fragility_curve_{lsf_name}"
    if not (cache_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"No manifest at {cache_dir}. Run build_fragility.py first."
        )

    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)
    cr_grid = manifest["grid"]["corrosion_rate"]
    n_total = len(cr_grid)

    if indices is None:
        # Default: any index whose cr >= 0.3
        indices = [i for i, cr in enumerate(cr_grid) if cr >= 0.3]
    print(f"Forcing IS at indices {indices} (cr = {[cr_grid[i] for i in indices]})")

    # Setup PTK project the same way build_fragility does.
    init_model(use_api=use_api)
    lsf_fn = wrap_lsf_with_deterministic(LSF_REGISTRY[lsf_name], _settings["variables"])
    stochastic_vars = build_stochastic_vars(_settings["variables"])

    import probabilistic_library as ptk
    project = ptk.ReliabilityProject()
    project.model = lsf_fn

    dist_map = {
        "normal": ptk.DistributionType.normal,
        "log_normal": ptk.DistributionType.log_normal,
        "lognormal": ptk.DistributionType.log_normal,
        "uniform": ptk.DistributionType.uniform,
        "beta": ptk.DistributionType.beta,
        "gumbel": ptk.DistributionType.gumbel,
    }
    for name, defn in stochastic_vars.items():
        project.variables[name].distribution = dist_map.get(
            defn.get("distribution", "normal").lower(), ptk.DistributionType.normal
        )
        for attr in ("mean", "deviation", "variation", "minimum", "maximum", "shape", "shape_b"):
            if attr in defn:
                setattr(project.variables[name], attr, defn[attr])

    # corrosion_rate stays deterministic, set per-point inside run_is_at
    project.variables["corrosion_rate"].distribution = ptk.DistributionType.deterministic
    project.variables["corrosion_rate"].mean = 0.0

    # Run IS for each requested index, overwrite the cached point_NNNN.json
    for i in indices:
        cr_val = cr_grid[i]
        result = run_is_at(project, ptk, cr_val, i)
        out = cache_dir / f"point_{i:04d}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"           wrote {out.name}")

    print(f"\nDone. Re-load fragility cache to see updated curve at indices {indices}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf-name", type=str, default="lsf_wall")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices (default: all i where cr >= 0.3)")
    parser.add_argument("--use-api", action="store_true")
    args = parser.parse_args()
    idx = None if args.indices is None else [int(x) for x in args.indices.split(",")]
    main(lsf_name=args.lsf_name, indices=idx, use_api=args.use_api)
