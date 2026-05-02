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
from time import time
from src.io import get_remote_path
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.reliability_models.dsheetpiling import build_payload, apply_payload
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

# Load settings once — model creation deferred to init_model()
_settings = load_settings()
_config = _settings["parameters"]
_remote = get_remote_path(_ENV)
_geomodel_path = _remote / "input" / "model.shi"
_base_model = None

WALL_MOMENT_CAPACITY = _config["wall_moment_capacity"]
ANCHOR_CAPACITY = _config["anchor_capacity"]
WALL_THICKNESS = _config["wall_thickness"]


def init_model(use_api: bool = False) -> None:
    """Create the base D-SheetPiling model (called once before LSF evaluations)."""
    global _base_model
    api_key = load_api_key() if use_api else None
    _base_model = DSheetPiling(str(_geomodel_path), api_key=api_key)
    print(f"Execution: {'API' if api_key else 'local'}")


def lsf_wall(
    Klei_soilphi,
    Klei_soilcohesion,
    Klei_soilcurkb1,
    Klei_soilgamdry,
    Klei_soilgamwet,
    Zand_soilphi,
    Zand_soilcurkb1,
    Zand_soilgamdry,
    Zand_soilgamwet,
    Zandvast_soilphi,
    Zandvast_soilcurkb1,
    Zandvast_soilgamdry,
    Zandvast_soilgamwet,
    Zandlos_soilphi,
    Zandlos_soilcurkb1,
    Zandlos_soilgamdry,
    Zandlos_soilgamwet,
    Wall_SheetPilingElementEI,
    model_factor_M,
    model_factor_F,
    phreatic_level,
    canal_level,
    uniform_load_left,
    corrosion_rate,
):
    """Limit state function: g = M_capacity(r) - theta_M * |M_max|

    model_factor_M : multiplies wall-moment demand (lognormal, mean 1).
    model_factor_F : accepted for signature compatibility (unused here).
    """
    # Degraded section properties
    factor = 1.0 - corrosion_rate
    m_capacity = WALL_MOMENT_CAPACITY * factor

    # Deep-copy and update model via payload
    model = deepcopy(_base_model)
    params = {k: v for k, v in locals().items()
              if k not in ("corrosion_rate", "factor", "m_capacity", "model",
                           "model_factor_M", "model_factor_F")}
    params["Wall_SheetPilingElementEI"] = Wall_SheetPilingElementEI * factor  # degraded EI
    payload = build_payload(params, model)
    apply_payload(model, payload)

    try:
        # Execute
        model.execute()

        # Limit state: capacity / (model uncertainty * demand) - 1
        # (g > 0 safe, g < 0 failure)
        max_moment = model.results.max_moment
        if isinstance(max_moment, (list, np.ndarray)):
            max_moment = max_moment[0]
        g = m_capacity / (model_factor_M * abs(max_moment)) - 1
        _internals = {"factor", "m_capacity", "model", "params", "payload",
                      "max_moment", "g"}
        inputs = [(k, v) for k, v in locals().items() if k not in _internals]
        # print(f"\nTime:{time():.1f} | LSF={g:.3f} | {inputs}")
        return g
    except Exception as e:
        _internals = {"factor", "m_capacity", "model", "params", "payload",
                      "max_moment", "g"}
        inputs = [(k, v) for k, v in locals().items() if k not in _internals]
        print(f"LSF crashed at: {inputs} err={e}")
        return -99999.

def lsf_anchor(
    Klei_soilphi,
    Klei_soilcohesion,
    Klei_soilcurkb1,
    Klei_soilgamdry,
    Klei_soilgamwet,
    Zand_soilphi,
    Zand_soilcurkb1,
    Zand_soilgamdry,
    Zand_soilgamwet,
    Zandvast_soilphi,
    Zandvast_soilcurkb1,
    Zandvast_soilgamdry,
    Zandvast_soilgamwet,
    Zandlos_soilphi,
    Zandlos_soilcurkb1,
    Zandlos_soilgamdry,
    Zandlos_soilgamwet,
    Wall_SheetPilingElementEI,
    model_factor_M,
    model_factor_F,
    phreatic_level,
    canal_level,
    uniform_load_left,
    corrosion_rate,
):
    """Limit state function: g = F_yield(r) - theta_F * |F_anchor|

    model_factor_F : multiplies anchor-force demand (lognormal, mean 1).
    model_factor_M : accepted for signature compatibility (unused here).
    """
    factor = 1.0 - corrosion_rate
    f_yield = ANCHOR_CAPACITY

    # Deep-copy and update model via payload
    model = deepcopy(_base_model)
    params = {k: v for k, v in locals().items()
              if k not in ("corrosion_rate", "factor", "f_yield", "model",
                           "model_factor_M", "model_factor_F")}
    params["Wall_SheetPilingElementEI"] = Wall_SheetPilingElementEI * factor
    payload = build_payload(params, model)
    apply_payload(model, payload)

    try:
        model.execute()

        anchor_force = model.results.anchor_force
        if isinstance(anchor_force, (list, np.ndarray)):
            anchor_force = anchor_force[0]

        return f_yield / (model_factor_F * abs(anchor_force)) - 1
    except Exception as e:
        print(f"LSF crashed at: {[(k, v) for k, v in locals().items() if k != 'model'][:5]}... err={e}")
        return -99999.


def lsf_wall_anchor(
    Klei_soilphi,
    Klei_soilcohesion,
    Klei_soilcurkb1,
    Klei_soilgamdry,
    Klei_soilgamwet,
    Zand_soilphi,
    Zand_soilcurkb1,
    Zand_soilgamdry,
    Zand_soilgamwet,
    Zandvast_soilphi,
    Zandvast_soilcurkb1,
    Zandvast_soilgamdry,
    Zandvast_soilgamwet,
    Zandlos_soilphi,
    Zandlos_soilcurkb1,
    Zandlos_soilgamdry,
    Zandlos_soilgamwet,
    Wall_SheetPilingElementEI,
    model_factor_M,
    model_factor_F,
    phreatic_level,
    canal_level,
    uniform_load_left,
    corrosion_rate,
    return_separate: bool = False,
):
    """Limit state function for combined wall + anchor (normalized form).

    g_wall   = M_capacity / (theta_M * |M_max|)   - 1
    g_anchor = F_yield    / (theta_F * |F_anchor|) - 1

    model_factor_M : model uncertainty on wall moments (lognormal, mean 1).
    model_factor_F : model uncertainty on anchor force (lognormal, mean 1).
    """
    # Degraded section properties
    factor = 1.0 - corrosion_rate
    m_capacity = WALL_MOMENT_CAPACITY * factor
    f_yield = ANCHOR_CAPACITY

    # Deep-copy and update model via payload
    model = deepcopy(_base_model)
    params = {k: v for k, v in locals().items()
              if k not in ("corrosion_rate", "factor", "m_capacity", "f_yield", "model",
                           "model_factor_M", "model_factor_F", "return_separate")}
    params["Wall_SheetPilingElementEI"] = Wall_SheetPilingElementEI * factor  # degraded EI
    payload = build_payload(params, model)
    apply_payload(model, payload)

    try:
        # Execute
        model.execute()

        max_moment = model.results.max_moment
        if isinstance(max_moment, (list, np.ndarray)):
            max_moment = max_moment[0]

        anchor_force = model.results.anchor_force
        if isinstance(anchor_force, (list, np.ndarray)):
            anchor_force = anchor_force[0]

        g_wall = m_capacity / (abs(max_moment) * model_factor_M) - 1
        g_anchor = f_yield / (abs(anchor_force) * model_factor_F) - 1

        if return_separate:
            return g_wall, g_anchor
        return min(g_wall, g_anchor)

    except Exception as e:

        print(f"LSF crashed at: {[(k, v) for k, v in locals().items() if k != 'model'][:5]}... err={e}")
        if return_separate:
            return -99999., -99999.
        else:
            return -99999.


# ---------------------------------------------------------------------------
# LSF registry — add new LSFs here
# ---------------------------------------------------------------------------

LSF_REGISTRY = {
    "lsf_wall": lsf_wall,
    "lsf_anchor": lsf_anchor,
    "lsf_wall_anchor": lsf_wall_anchor,
}


_DETERMINISTIC_TYPES = ("deterministic", "constant", "fixed")


def _is_deterministic(v: dict) -> bool:
    return v.get("distribution_type", "normal").lower() in _DETERMINISTIC_TYPES


def build_stochastic_vars(variables: list) -> dict:
    """Convert settings variable defs to FragilityCurveBuilder format.

    Variables flagged as deterministic in settings.json are excluded — they
    must be bound separately via ``wrap_lsf_with_deterministic``.
    """
    stochastic = {}
    for v in variables:
        if _is_deterministic(v):
            continue
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


def wrap_lsf_with_deterministic(lsf_fn, variables: list):
    """Pre-bind all deterministic variables to their means.

    PTK calls the LSF positionally based on the inspected signature: it
    samples the stochastic vars, passes the grid value for ``corrosion_rate``,
    and falls back to 0 for any signature param it doesn't recognise (which
    is how ``Wall_SheetPilingElementEI`` was silently zeroed before this
    wrapper existed).

    The wrapper exposes the same signature as ``lsf_fn`` (so PTK still
    inspects 24 params), then overrides the deterministic positions with
    their means before delegating. PTK still passes through 24 values per
    call, but only the 9 stochastic dimensions are perturbed by FD because
    only those appear in ``stochastic_vars`` — the rest are fixed.
    """
    import inspect

    deterministic_kwargs = {v["name"]: v["mean"] for v in variables if _is_deterministic(v)}
    sig = inspect.signature(lsf_fn)
    param_names = list(sig.parameters.keys())

    def wrapped(*args, **kwargs):
        # Map positional -> kwargs so we can override uniformly.
        bound = dict(zip(param_names, args))
        bound.update(kwargs)
        # Deterministic preset wins over whatever PTK supplied (0 / mean / etc.)
        bound.update(deterministic_kwargs)
        return lsf_fn(**bound)

    wrapped.__signature__ = sig
    wrapped.__name__ = f"{lsf_fn.__name__}_form_wrapper"
    wrapped.__wrapped__ = lsf_fn
    wrapped.deterministic_kwargs = deterministic_kwargs
    return wrapped


class WarmStartFragilityBuilder(FragilityCurveBuilder):
    """FragilityCurveBuilder with explicit warm-start across grid points.

    Captures the converged design point after each grid point and pushes
    it into the project's stochastic variables before the next run, so
    FORM starts from a near-optimal point instead of the median.

    Also forces ``start_method = sensitivity_search`` (uses prior gradient
    info) when available, falling back gracefully to ``fixed_value``.
    """

    def _setup_project(self):
        project = super()._setup_project()
        # fixed_value uses each variable's design_value as the FORM seed;
        # sensitivity_search burns extra LSF calls on an upfront exploration
        # which is wasted when consecutive grid points share a design point.
        try:
            import probabilistic_library as ptk
            project.settings.start_method = ptk.StartMethod.fixed_value
        except (AttributeError, ImportError):
            pass
        self._last_design = {}
        return project

    def _compute_point(self, project, point, index, verbose=True):
        # Push previous design point into stochastic variables (warm start)
        if self._last_design and verbose:
            print(f"  [{index:04d}] warm-start from prev design point", flush=True)
        for name, x_val in self._last_design.items():
            try:
                project.variables[name].design_value = float(x_val)
            except Exception:
                pass

        result = super()._compute_point(project, point, index, verbose=verbose)

        # Capture this point's design values (in x-space) for the next call
        if result.get("convergence", False) and result.get("method") == "form":
            self._last_design = dict(result.get("design_point", {}))
        return result

    def build(self, grid, cache_dir, force_rebuild=False, verbose=True):
        """Override to seed _last_design from the latest cached point.

        Without this, restarting after partial completion would re-do a cold
        start at the first uncached point, defeating the warm-start.
        """
        from pathlib import Path
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        points = self._build_grid_points(grid)
        n_total = len(points)
        if force_rebuild:
            completed = set()
        else:
            manifest = self._load_manifest(cache_dir)
            completed = set(manifest["completed_indices"]) if manifest else set()
        if verbose:
            print(f"Fragility curve: {n_total} points, {len(completed)} cached, "
                  f"{n_total - len(completed)} to compute")
        project = self._setup_project()

        for i, point in enumerate(points):
            if i in completed:
                # Seed warm-start from this cached point so the first
                # uncached point can warm-start from the latest cached one
                cached = self._load_point(cache_dir, i)
                if cached and cached.get("convergence") and cached.get("method") == "form":
                    self._last_design = dict(cached.get("design_point", {}))
                continue
            result = self._compute_point(project, point, i, verbose=verbose)
            self._save_point(result, cache_dir, i)
            completed.add(i)
            self._save_manifest(cache_dir, grid, sorted(completed))
        return self.load(cache_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dev_fragility(lsf_name: str, n_cr_grid: int, use_api: bool) -> list[dict]:
    """Build a fast development fragility curve.

    Only computes FORM at cr=0. The rest is extrapolated:
        beta(cr) = beta(0) * (1 - cr)
        Pf(cr) = Phi(-beta(cr))

    All extrapolated points are marked with method='development'.
    """
    from scipy.stats import norm as sp_norm

    init_model(use_api=use_api)
    lsf_fn = LSF_REGISTRY[lsf_name]
    lsf_fn = wrap_lsf_with_deterministic(lsf_fn, _settings["variables"])
    stochastic_vars = build_stochastic_vars(_settings["variables"])

    builder = WarmStartFragilityBuilder(
        lsf=lsf_fn,
        stochastic_vars=stochastic_vars,
        deterministic_vars=["corrosion_rate"],
        form_params={
            "relaxation_factor": 0.3,
            "maximum_iterations": 30,
            "variation_coefficient": 0.20,
            "step_size": 0.10,
        },
    )

    # Only compute cr=0
    cache_dir = _remote / "output" / f"fragility_curve_{lsf_name}_dev"
    print(f"DEV MODE: computing FORM only at cr=0, extrapolating the rest")

    results_cr0 = builder.build(
        grid={"corrosion_rate": np.array([0.0])},
        cache_dir=cache_dir,
        force_rebuild=False,
        verbose=True,
    )

    beta_0 = results_cr0[0]["beta"]
    cr_grid = np.linspace(0.0, 1.0, n_cr_grid)

    results = []
    for i, cr in enumerate(cr_grid):
        if cr == 0.0:
            results.append(results_cr0[0])
        else:
            beta_cr = beta_0 * (1.0 - cr)
            pf_cr = float(sp_norm.cdf(-beta_cr))
            results.append({
                "index": i,
                "point": {"corrosion_rate": float(cr)},
                "pf": pf_cr,
                "beta": beta_cr,
                "logpf": float(np.log(pf_cr)) if pf_cr > 0 else -np.inf,
                "convergence": True,
                "method": "development",
                "design_point": {},
                "alphas": {},
            })

    return results


def main(lsf_name: str = "lsf_wall", use_api: bool = False, force_rebuild: bool = False, dev_frag: bool = False):
    if lsf_name not in LSF_REGISTRY:
        raise ValueError(f"Unknown LSF '{lsf_name}'. Available: {list(LSF_REGISTRY.keys())}")

    n_cr_grid = _config.get("n_fc_grid", 10)

    print("=" * 60)
    print("Building fragility curve for D-SheetPiling")
    print("=" * 60)

    if dev_frag:
        results = build_dev_fragility(lsf_name, n_cr_grid, use_api)
        # Save to the standard cache location so run.py can load it
        cache_dir = _remote / "output" / f"fragility_curve_{lsf_name}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            path = cache_dir / f"point_{r['index']:04d}.json"
            with open(path, "w") as f:
                json.dump(r, f, indent=2)
        # Write manifest
        manifest = {
            "lsf_name": lsf_name,
            "deterministic_vars": ["corrosion_rate"],
            "stochastic_vars": build_stochastic_vars(_settings["variables"]),
            "grid": {"corrosion_rate": np.linspace(0.0, 1.0, n_cr_grid).tolist()},
            "form_params": {},
            "n_total": n_cr_grid,
            "n_completed": n_cr_grid,
            "completed_indices": list(range(n_cr_grid)),
        }
        with open(cache_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    else:
        init_model(use_api=use_api)
        lsf_fn = LSF_REGISTRY[lsf_name]
        lsf_fn = wrap_lsf_with_deterministic(lsf_fn, _settings["variables"])
        stochastic_vars = build_stochastic_vars(_settings["variables"])

        builder = WarmStartFragilityBuilder(
            lsf=lsf_fn,
            stochastic_vars=stochastic_vars,
            deterministic_vars=["corrosion_rate"],
            form_params={
                "relaxation_factor": 0.5,
                "maximum_iterations": 30,
                "variation_coefficient": 0.20,
                "step_size": 0.10,
            },
        )

        grid = {"corrosion_rate": np.linspace(0.0, 1.0, n_cr_grid)}
        cache_dir = _remote / "output" / f"fragility_curve_{lsf_name}"

        print(f"LSF: {lsf_name}")
        print(f"Grid: {n_cr_grid} points from 0.0 to 1.0")
        print(f"Cache: {cache_dir}\n")

        results = builder.build(
            grid=grid,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild,
            verbose=True,
        )

    # Summary
    print(f"\n{'r':>8s} {'beta':>8s} {'Pf':>12s} {'method':>12s} {'ok':>5s}")
    print("-" * 50)
    for r in results:
        cr = r["point"]["corrosion_rate"]
        print(f"{cr:>8.3f} {r['beta']:>8.3f} {r['pf']:>12.3e} {r['method']:>12s} {str(r['convergence']):>5s}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf", type=str, default="lsf_wall",
                        help=f"LSF to use. Available: {list(LSF_REGISTRY.keys())}")
    parser.add_argument("--use_api", action="store_true", help="Use the D-SheetPiling compute API")
    parser.add_argument("--force_rebuild", action="store_true", help="Recompute all points")
    parser.add_argument("--dev_frag", action="store_true",
                        help="Fast dev mode: FORM at cr=0 only, extrapolate rest as beta(cr)=beta(0)*(1-cr)")
    args = parser.parse_args()
    main(lsf_name=args.lsf, use_api=args.use_api, force_rebuild=args.force_rebuild, dev_frag=args.dev_frag)
