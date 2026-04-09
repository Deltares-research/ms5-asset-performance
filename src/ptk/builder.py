"""
Fragility curve builder with per-point caching and FORM/IS fallback.

Supports N deterministic variables forming a grid. Each grid point is
computed independently and cached as a plain JSON dict, allowing
long-running jobs to be stopped and resumed.

Results are plain dicts — no custom data classes. Each point is::

    {
        "index": 0,
        "point": {"corrosion_rate": 0.1},
        "pf": 1.2e-4,
        "beta": 3.67,
        "logpf": -9.03,
        "convergence": true,
        "method": "form",
        "design_point": {"phi_sand": 22.1, "su_clay": 16.3},
        "alphas": {"phi_sand": -0.82, "su_clay": -0.57}
    }
"""

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import trapezoid


class FragilityCurveBuilder:
    """Build a fragility curve over an N-D grid of deterministic variables.

    For each grid point:
    1. Set deterministic variables to constant (deterministic) values.
    2. Run FORM on the remaining stochastic variables.
    3. If FORM doesn't converge, fall back to importance sampling.
    4. Cache the result as a plain JSON dict.

    Usage::

        builder = FragilityCurveBuilder(
            lsf=my_lsf,
            stochastic_vars={
                "phi_sand": {"distribution": "normal", "mean": 25, "variation": 0.10},
            },
            deterministic_vars=["corrosion_rate"],
        )
        results = builder.build(
            grid={"corrosion_rate": np.linspace(0, 1, 11)},
            cache_dir=Path("cache/fragility"),
        )
        # results is a list of dicts
    """

    def __init__(
        self,
        lsf: Callable,
        stochastic_vars: Dict[str, Dict[str, Any]],
        deterministic_vars: List[str],
        form_params: Optional[Dict[str, float]] = None,
    ) -> None:
        self.lsf = lsf
        self.stochastic_vars = stochastic_vars
        self.deterministic_vars = deterministic_vars
        self.form_params = form_params or {
            "relaxation_factor": 0.15,
            "maximum_iterations": 100,
            "variation_coefficient": 0.05,
            "step_size": 0.05,
        }

    # ------------------------------------------------------------------
    # ptk project setup
    # ------------------------------------------------------------------

    def _setup_project(self):
        """Create and configure a ptk ReliabilityProject."""
        import probabilistic_library as ptk
        self._ptk = ptk

        project = ptk.ReliabilityProject()
        project.model = self.lsf

        dist_map = {
            "normal": ptk.DistributionType.normal,
            "log_normal": ptk.DistributionType.log_normal,
            "lognormal": ptk.DistributionType.log_normal,
            "uniform": ptk.DistributionType.uniform,
            "beta": ptk.DistributionType.beta,
            "gumbel": ptk.DistributionType.gumbel,
        }

        for name, defn in self.stochastic_vars.items():
            dist_type = defn.get("distribution", "normal").lower()
            project.variables[name].distribution = dist_map.get(
                dist_type, ptk.DistributionType.normal
            )
            for attr in ("mean", "deviation", "variation", "minimum", "maximum", "shape", "shape_b"):
                if attr in defn:
                    setattr(project.variables[name], attr, defn[attr])

        # Deterministic variables — set as constants, overwritten per point
        for name in self.deterministic_vars:
            project.variables[name].distribution = ptk.DistributionType.deterministic
            project.variables[name].mean = 0.0

        project.settings.reliability_method = ptk.ReliabilityMethod.form
        for key, val in self.form_params.items():
            setattr(project.settings, key, val)

        return project

    # ------------------------------------------------------------------
    # Single point computation
    # ------------------------------------------------------------------

    def _compute_point(
        self, project, point: Dict[str, float], index: int, verbose: bool = True
    ) -> dict:
        """Compute one fragility point. Returns a plain dict.

        Sets each deterministic variable to a constant value, runs FORM,
        falls back to IS if FORM doesn't converge.
        """
        point_str = ", ".join(f"{k}={v:.4g}" for k, v in point.items())

        # Fix deterministic variables to their grid values
        for name, val in point.items():
            project.variables[name].distribution = self._ptk.DistributionType.deterministic
            project.variables[name].mean = float(val)

        # FORM
        if verbose:
            print(f"  [{index:04d}] FORM at {point_str} ...", end="", flush=True)

        project.settings.reliability_method = self._ptk.ReliabilityMethod.form
        project.run()
        dp = project.design_point

        if dp.is_converged:
            method = "form"
            if verbose:
                print(f" converged. beta={dp.reliability_index:.3f}, Pf={dp.probability_failure:.3e}")
        else:
            # Importance sampling fallback
            if verbose:
                print(f" not converged. Running IS ...", end="", flush=True)
            project.settings.reliability_method = self._ptk.ReliabilityMethod.importance_sampling
            project.run()
            dp = project.design_point
            method = "importance_sampling"
            if verbose:
                print(f" beta={dp.reliability_index:.3f}, Pf={dp.probability_failure:.3e}")

        pf = dp.probability_failure
        det_names = set(self.deterministic_vars)

        return {
            "index": index,
            "point": point,
            "pf": pf,
            "beta": dp.reliability_index,
            "logpf": math.log(pf) if pf > 0 else -math.inf,
            "convergence": dp.is_converged,
            "method": method,
            "design_point": {
                a.variable.name: a.x for a in dp.alphas
                if a.variable.name not in det_names
            },
            "alphas": {
                a.variable.name: a.alpha for a in dp.alphas
                if a.variable.name not in det_names
            },
        }

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _save_point(result: dict, cache_dir: Path, index: int) -> None:
        path = cache_dir / f"point_{index:04d}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    @staticmethod
    def _load_point(cache_dir: Path, index: int) -> Optional[dict]:
        path = cache_dir / f"point_{index:04d}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def _save_manifest(self, cache_dir: Path, grid: Dict[str, list], completed: List[int]) -> None:
        manifest = {
            "deterministic_vars": self.deterministic_vars,
            "stochastic_vars": self.stochastic_vars,
            "grid": {k: [float(x) for x in v] for k, v in grid.items()},
            "form_params": self.form_params,
            "n_total": int(np.prod([len(v) for v in grid.values()])),
            "n_completed": len(completed),
            "completed_indices": sorted(completed),
        }
        with open(cache_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    @staticmethod
    def _load_manifest(cache_dir: Path) -> Optional[dict]:
        path = cache_dir / "manifest.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_grid_points(grid: Dict[str, NDArray]) -> List[Dict[str, float]]:
        names = list(grid.keys())
        arrays = [grid[n] for n in names]
        meshes = np.meshgrid(*arrays, indexing="ij")
        flat = np.column_stack([m.ravel() for m in meshes])
        return [{n: float(row[i]) for i, n in enumerate(names)} for row in flat]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        grid: Dict[str, NDArray],
        cache_dir: Path,
        force_rebuild: bool = False,
        verbose: bool = True,
    ) -> List[dict]:
        """Build fragility curve over N-D grid with per-point caching.

        Args:
            grid: Dict mapping deterministic var names to 1D arrays.
            cache_dir: Directory for per-point JSON cache files.
            force_rebuild: Recompute all points.
            verbose: Print progress.

        Returns:
            List of result dicts, one per grid point.
        """
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
                continue
            result = self._compute_point(project, point, i, verbose=verbose)
            self._save_point(result, cache_dir, i)
            completed.add(i)
            self._save_manifest(cache_dir, grid, sorted(completed))

        return self.load(cache_dir)

    def load(self, cache_dir: Path) -> List[dict]:
        """Load fragility curve results from cached point files.

        Args:
            cache_dir: Directory containing point_NNNN.json files.

        Returns:
            List of result dicts, sorted by index.
        """
        cache_dir = Path(cache_dir)
        manifest = self._load_manifest(cache_dir)
        if manifest is None:
            raise FileNotFoundError(f"No manifest.json in {cache_dir}")

        results = []
        for i in sorted(manifest["completed_indices"]):
            pt = self._load_point(cache_dir, i)
            if pt is not None and not math.isnan(pt["pf"]):
                results.append(pt)
        return results

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    @staticmethod
    def integrate(
        results: List[dict],
        distributions: Dict[str, Any],
        det_var_names: List[str],
    ) -> Tuple[float, float]:
        """Integrate fragility results over distributions.

        Marginalizes out the deterministic variables by integrating
        Pf(x) * f(x) over the grid using the trapezoidal rule.

        Args:
            results: List of result dicts from build() or load().
            distributions: Dict mapping variable name to a frozen
                scipy.stats distribution (e.g. ``stats.beta(a, b)``).
            det_var_names: Ordered list of deterministic variable names.

        Returns:
            Tuple of (pf, beta).
        """
        points = np.array([[r["point"][n] for n in det_var_names] for r in results])
        logpfs = np.array([r["logpf"] for r in results])

        log_pdf = np.zeros(len(points))
        for i, name in enumerate(det_var_names):
            if name in distributions:
                log_pdf += distributions[name].logpdf(points[:, i])

        integrand = np.exp(logpfs + log_pdf)

        # Reshape to N-D for multi-dim trapezoid
        grids = [np.sort(np.unique(points[:, i])) for i in range(points.shape[1])]
        shapes = tuple(len(g) for g in grids)
        integrand_nd = integrand.reshape(shapes)

        for i in reversed(range(len(grids))):
            integrand_nd = trapezoid(integrand_nd, grids[i], axis=i)

        pf = float(np.clip(integrand_nd, 1e-30, 1 - 1e-10))
        beta = float(stats.norm.ppf(1 - pf))
        return pf, beta
