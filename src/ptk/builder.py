"""
Fragility curve builder with per-point caching and FORM/IS fallback.

Supports N deterministic variables forming a grid. Each grid point is
computed independently and cached to disk, allowing long-running jobs
to be stopped and resumed.
"""

import json
import math
import itertools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import trapezoid

from .fragility import FragilityPoint, FragilityCurve


class FragilityCurveBuilder:
    """Build a fragility curve over an N-D grid of deterministic variables.

    For each grid point:
    1. Set deterministic variables to fixed values.
    2. Run FORM on the remaining stochastic variables.
    3. If FORM doesn't converge, fall back to importance sampling.
    4. Cache the result to disk.

    Usage::

        builder = FragilityCurveBuilder(
            lsf=my_lsf,
            stochastic_vars={
                "phi_sand": {"distribution": "normal", "mean": 25, "variation": 0.10},
                "su_clay":  {"distribution": "log_normal", "mean": 20, "variation": 0.10},
            },
            deterministic_vars=["corrosion_rate"],
        )
        fc = builder.build(
            grid={"corrosion_rate": np.linspace(0, 1, 11)},
            cache_dir=Path("cache/fragility"),
        )
    """

    def __init__(
        self,
        lsf: Callable,
        stochastic_vars: Dict[str, Dict[str, Any]],
        deterministic_vars: List[str],
        form_params: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the builder.

        Args:
            lsf: Limit state function callable. Arguments must include all
                stochastic + deterministic variable names.
            stochastic_vars: Dict of stochastic variable definitions::

                {"var_name": {"distribution": "normal", "mean": 0, "deviation": 1, ...}}

                Supported keys per variable: distribution, mean, deviation,
                variation, minimum, maximum, shape, shape_b.
            deterministic_vars: Names of variables that form the grid.
            form_params: FORM solver settings. Keys: relaxation_factor,
                maximum_iterations, variation_coefficient, step_size.
        """
        self.lsf = lsf
        self.stochastic_vars = stochastic_vars
        self.deterministic_vars = deterministic_vars
        self.form_params = form_params or {
            "relaxation_factor": 0.15,
            "maximum_iterations": 100,
            "variation_coefficient": 0.05,
            "step_size": 0.05,
        }

    def _setup_project(self):
        """Create and configure a ptk ReliabilityProject."""
        import probabilistic_library as ptk
        self._ptk = ptk
        project = self._ptk.ReliabilityProject()
        project.model = self.lsf

        # Configure stochastic variables
        dist_map = {
            "normal": self._ptk.DistributionType.normal,
            "log_normal": self._ptk.DistributionType.log_normal,
            "lognormal": self._ptk.DistributionType.log_normal,
            "uniform": self._ptk.DistributionType.uniform,
            "beta": self._ptk.DistributionType.beta,
            "gumbel": self._ptk.DistributionType.gumbel,
        }

        for name, defn in self.stochastic_vars.items():
            dist_type = defn.get("distribution", "normal").lower()
            project.variables[name].distribution = dist_map.get(
                dist_type, self._ptk.DistributionType.normal
            )
            for attr in ("mean", "deviation", "variation", "minimum", "maximum", "shape", "shape_b"):
                if attr in defn:
                    setattr(project.variables[name], attr, defn[attr])

        # Configure deterministic variables (placeholder — overwritten per point)
        for name in self.deterministic_vars:
            project.variables[name].distribution = self._ptk.DistributionType.deterministic
            project.variables[name].mean = 0.0

        # FORM settings
        project.settings.reliability_method = self._ptk.ReliabilityMethod.form
        for key, val in self.form_params.items():
            setattr(project.settings, key, val)

        return project

    def _compute_point(
        self,
        project,
        point: Dict[str, float],
        index: int,
        verbose: bool = True,
    ) -> FragilityPoint:
        """Compute one fragility point: FORM first, IS fallback.

        Args:
            project: Configured ptk project.
            point: Dict mapping deterministic var names to values.
            index: Grid point index (for logging).
            verbose: Print progress.

        Returns:
            FragilityPoint with results.
        """
        point_values = [point[name] for name in self.deterministic_vars]
        point_str = ", ".join(f"{k}={v:.4g}" for k, v in point.items())

        # Set deterministic values
        for name, val in point.items():
            project.variables[name].distribution = self._ptk.DistributionType.deterministic
            project.variables[name].mean = float(val)

        # Try FORM
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
            # Fall back to importance sampling
            if verbose:
                print(f" not converged. Running IS ...", end="", flush=True)

            project.settings.reliability_method = self._ptk.ReliabilityMethod.importance_sampling
            project.run()
            dp = project.design_point
            method = "importance_sampling"

            if verbose:
                print(f" beta={dp.reliability_index:.3f}, Pf={dp.probability_failure:.3e}")

        pf = dp.probability_failure
        logpf = math.log(pf) if pf > 0 else -np.inf

        return FragilityPoint(
            point=point_values,
            pf=pf,
            beta=dp.reliability_index,
            design_point={
                a.variable.name: a.x for a in dp.alphas
                if a.variable.name not in self.deterministic_vars
            },
            alphas={
                a.variable.name: a.alpha for a in dp.alphas
                if a.variable.name not in self.deterministic_vars
            },
            logpf=logpf,
            convergence=dp.is_converged,
            method=method,
        )

    def _save_point(self, fp: FragilityPoint, cache_dir: Path, index: int) -> None:
        """Save a single fragility point to disk."""
        path = cache_dir / f"point_{index:04d}.json"
        with open(path, "w") as f:
            json.dump(fp._asdict(), f, indent=2)

    def _load_point(self, cache_dir: Path, index: int) -> Optional[FragilityPoint]:
        """Load a single fragility point from disk. Returns None if not found."""
        path = cache_dir / f"point_{index:04d}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return FragilityPoint(**data)

    def _save_manifest(self, cache_dir: Path, grid: Dict[str, list], completed: List[int]) -> None:
        """Save manifest with grid definition and completion status."""
        manifest = {
            "deterministic_vars": self.deterministic_vars,
            "stochastic_vars": {k: v for k, v in self.stochastic_vars.items()},
            "grid": {k: [float(x) for x in v] for k, v in grid.items()},
            "form_params": self.form_params,
            "n_total": int(np.prod([len(v) for v in grid.values()])),
            "n_completed": len(completed),
            "completed_indices": sorted(completed),
        }
        with open(cache_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def _load_manifest(self, cache_dir: Path) -> Optional[Dict]:
        """Load manifest from cache dir."""
        path = cache_dir / "manifest.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def _build_grid_points(self, grid: Dict[str, NDArray]) -> List[Dict[str, float]]:
        """Build flat list of grid point dicts from N-D grid."""
        names = list(grid.keys())
        arrays = [grid[name] for name in names]
        meshes = np.meshgrid(*arrays, indexing="ij")
        flat = np.column_stack([m.ravel() for m in meshes])
        return [{name: float(row[i]) for i, name in enumerate(names)} for row in flat]

    def build(
        self,
        grid: Dict[str, NDArray],
        cache_dir: Path,
        force_rebuild: bool = False,
        verbose: bool = True,
    ) -> FragilityCurve:
        """Build fragility curve over N-D grid with per-point caching.

        Args:
            grid: Dict mapping deterministic var names to 1D arrays.
            cache_dir: Directory for per-point JSON cache files.
            force_rebuild: If True, recompute all points.
            verbose: Print progress messages.

        Returns:
            FragilityCurve with all computed points.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        points = self._build_grid_points(grid)
        n_total = len(points)

        # Check existing cache
        if force_rebuild:
            completed = set()
        else:
            manifest = self._load_manifest(cache_dir)
            completed = set(manifest["completed_indices"]) if manifest else set()

        if verbose:
            n_skip = len(completed)
            n_remaining = n_total - n_skip
            print(f"Fragility curve: {n_total} points, {n_skip} cached, {n_remaining} to compute")

        # Setup ptk project
        project = self._setup_project()

        # Compute missing points
        for i, point in enumerate(points):
            if i in completed:
                continue

            fp = self._compute_point(project, point, i, verbose=verbose)
            self._save_point(fp, cache_dir, i)
            completed.add(i)
            self._save_manifest(cache_dir, grid, sorted(completed))

        # Compile all points
        return self.load(cache_dir)

    def load(self, cache_dir: Path) -> FragilityCurve:
        """Load a fragility curve from cached point files.

        Args:
            cache_dir: Directory containing point_NNNN.json files.

        Returns:
            FragilityCurve with all valid points.
        """
        cache_dir = Path(cache_dir)
        manifest = self._load_manifest(cache_dir)
        if manifest is None:
            raise FileNotFoundError(f"No manifest.json in {cache_dir}")

        fps = []
        for i in sorted(manifest["completed_indices"]):
            fp = self._load_point(cache_dir, i)
            if fp is not None and not math.isnan(fp.pf):
                fps.append(fp)

        if not fps:
            raise ValueError("No valid fragility points found in cache.")

        return FragilityCurve(fragility_points=fps)

    @staticmethod
    def integrate(
        fragility_curve: FragilityCurve,
        distributions: Dict[str, Any],
        det_var_names: List[str],
    ) -> Tuple[float, float]:
        """Integrate fragility curve over distributions of deterministic variables.

        Marginalizes out the deterministic variables by integrating
        Pf(x) * f(x) over the grid using the trapezoidal rule.

        Args:
            fragility_curve: Computed fragility curve.
            distributions: Dict mapping variable name to scipy.stats distribution
                (frozen, e.g. ``stats.beta(a, b)``).
            det_var_names: Ordered list of deterministic variable names
                (must match the order in fragility_curve.points columns).

        Returns:
            Tuple of (pf, beta).
        """
        points = fragility_curve.points
        logpfs = fragility_curve.logpfs

        # Compute log-PDF of the joint distribution at each point
        log_pdf = np.zeros(len(points))
        for i, name in enumerate(det_var_names):
            if name in distributions:
                log_pdf += distributions[name].logpdf(points[:, i])

        # Pf * f(x) in log space
        log_integrand = logpfs + log_pdf
        integrand = np.exp(log_integrand)

        # Integrate over each dimension
        # Reshape to N-D grid for multi-dim trapezoid
        grids = []
        for i in range(points.shape[1]):
            grids.append(np.sort(np.unique(points[:, i])))

        shapes = tuple(len(g) for g in grids)
        integrand_nd = integrand.reshape(shapes)

        for i in reversed(range(len(grids))):
            integrand_nd = trapezoid(integrand_nd, grids[i], axis=i)

        pf = float(integrand_nd)
        pf = np.clip(pf, 1e-30, 1 - 1e-10)
        beta = float(stats.norm.ppf(1 - pf))

        return pf, beta
