"""
Fragility-based reliability pipeline.

Integrates a pre-computed fragility curve (list of dicts) with time-varying
PDFs of the deterministic variables to compute Pf at each time step.

The pipeline is generic: the user provides:
- A fragility curve (list of point dicts from FragilityCurveBuilder)
- A callable ``get_det_pdfs(t, use_prior, **kwargs)`` that returns the PDF
  of each deterministic variable at time t
- A callable ``do_bayesian_update(jpdf, obs_times, obs_values, t)``
- A callable ``reset_posterior(jpdf)``
- A callable ``build_jpdf_state(jpdf)`` for storing JPDF snapshots
"""

import numpy as np
from scipy import stats as st
from scipy.integrate import trapezoid
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List, Callable

from .base import BasePipeline


class FragilityPipeline(BasePipeline):
    """Pipeline that integrates a fragility curve with time-varying PDFs.

    The fragility curve is a list of dicts (from FragilityCurveBuilder).
    At each forecast time, the user-provided ``get_det_pdfs`` callable
    returns ``{var_name: (grid, pdf)}`` for each deterministic variable.
    The pipeline interpolates the fragility Pf onto these grids and
    integrates ``Pf(x) * f(x) dx`` over all deterministic dimensions.

    Usage::

        pipeline = FragilityPipeline(
            settings_path=...,
            performance=...,
            config=config,
            fragility_points=fc_points,
            det_var_names=["corrosion_rate"],
            get_det_pdfs=my_pdf_fn,
            do_update=my_update_fn,
            do_reset=my_reset_fn,
            build_state=my_state_fn,
        )
        pipeline.init_times(obs_times, forecast_times)
        results = pipeline.run_timeline(setting)
    """

    def __init__(
        self,
        settings_path: Path | str,
        performance,
        config: Optional[Dict[str, Any]] = None,
        obs_error: float = None,
        fragility_points: Optional[List[dict]] = None,
        det_var_names: Optional[List[str]] = None,
        n_integration_grid: int = 1000,
        get_det_pdfs: Optional[Callable] = None,
        do_update: Optional[Callable] = None,
        do_reset: Optional[Callable] = None,
        build_state: Optional[Callable] = None,
    ) -> None:
        """Initialize the fragility pipeline.

        Args:
            settings_path: Path to the JSON settings file.
            performance: Performance function instance.
            config: Parameters dict.
            obs_error: Override for observation error std.
            fragility_points: List of fragility point dicts.
            det_var_names: Ordered names of deterministic variables
                (must match the keys in each point's "point" dict).
            n_integration_grid: Number of grid points per variable for
                interpolation and integration.
            get_det_pdfs: Callable ``(t, use_prior, **kwargs) ->
                {var_name: (grid, pdf)}``. Returns the PDF of each
                deterministic variable at time t.
            do_update: Callable ``(jpdf, obs_times, obs_values, t) -> None``.
                Performs Bayesian update on the JPDF.
            do_reset: Callable ``(jpdf) -> None``. Resets JPDF to prior.
            build_state: Callable ``(jpdf) -> dict``. Builds JPDF state
                snapshot for results storage.
        """
        self.config = config or {}
        _obs_error = obs_error or self.config.get("obs_error_std", self.config.get("obs_error", 0.1))
        super().__init__(settings_path, performance, _obs_error)

        self.fragility_points = fragility_points or []
        self.det_var_names = det_var_names or []
        self.n_integration_grid = n_integration_grid

        # User-provided callables
        self._get_det_pdfs = get_det_pdfs
        self._do_update = do_update
        self._do_reset = do_reset
        self._build_state = build_state

        # Pre-extract fragility grid for fast interpolation
        self._fc_grids = {}  # {var_name: sorted array of grid values}
        self._fc_pf = None
        if self.fragility_points:
            self._prepare_fragility()

    def _prepare_fragility(self) -> None:
        """Pre-extract grids and Pf from fragility points for interpolation."""
        for name in self.det_var_names:
            vals = sorted(set(p["point"][name] for p in self.fragility_points))
            self._fc_grids[name] = np.array(vals)
        self._fc_pf = np.array([p["pf"] for p in self.fragility_points])

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------

    def setup(self, seed: int = 42, **kwargs) -> None:
        if self.jpdf is None:
            raise ValueError("Set self.jpdf before calling setup().")

    def _reset_posterior(self) -> None:
        if self._do_reset is not None:
            self._do_reset(self.jpdf)

    def _do_bayesian_update(
        self, obs_times: NDArray, obs_values: NDArray, t: float, **kwargs
    ) -> None:
        if self._do_update is not None:
            self._do_update(self.jpdf, obs_times, obs_values, t)

    def compute_pf_at_time(
        self, forecast_time: float, use_prior: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """Compute Pf by integrating fragility curve against deterministic PDFs.

        Interpolates both the fragility Pf and each deterministic variable's
        PDF onto a common grid, then integrates over all dimensions.

        Args:
            forecast_time: Time at which to evaluate.
            use_prior: Use prior (True) or posterior (False).
            **kwargs: Forwarded to get_det_pdfs.

        Returns:
            Dict with pf, beta.
        """
        if self._get_det_pdfs is None:
            raise ValueError("Set get_det_pdfs callable.")
        if not self.fragility_points:
            raise ValueError("Set fragility_points.")

        # Get PDFs of each deterministic variable at this time
        det_pdfs = self._get_det_pdfs(forecast_time, use_prior, **kwargs)

        # Build common integration grid per variable
        n = self.n_integration_grid
        integration_grids = {}
        for name in self.det_var_names:
            lo = self._fc_grids[name][0]
            hi = self._fc_grids[name][-1]
            integration_grids[name] = np.linspace(lo, hi, n)

        # For 1D: straightforward interpolation
        if len(self.det_var_names) == 1:
            name = self.det_var_names[0]
            grid = integration_grids[name]

            # Interpolate fragility Pf
            pf_interp = np.interp(grid, self._fc_grids[name], self._fc_pf,
                                  left=self._fc_pf[0], right=self._fc_pf[-1])

            # Interpolate variable PDF
            src_grid, src_pdf = det_pdfs[name]
            pdf_interp = np.interp(grid, src_grid, src_pdf, left=0, right=0)

            pf = float(trapezoid(pf_interp * pdf_interp, grid))
        else:
            # N-D: build meshgrid, interpolate fragility + joint PDF, integrate
            grids = [integration_grids[name] for name in self.det_var_names]
            meshes = np.meshgrid(*grids, indexing="ij")
            flat_points = np.column_stack([m.ravel() for m in meshes])

            # Interpolate fragility Pf at each mesh point
            # Use nearest-neighbor from cached points
            fc_points_arr = np.array([[p["point"][name] for name in self.det_var_names]
                                      for p in self.fragility_points])
            from scipy.interpolate import LinearNDInterpolator
            interp_fn = LinearNDInterpolator(fc_points_arr, self._fc_pf, fill_value=0.0)
            pf_mesh = interp_fn(flat_points).reshape([len(g) for g in grids])

            # Joint PDF = product of independent marginal PDFs
            pdf_mesh = np.ones_like(pf_mesh)
            for i, name in enumerate(self.det_var_names):
                src_grid, src_pdf = det_pdfs[name]
                marginal = np.interp(grids[i], src_grid, src_pdf, left=0, right=0)
                shape = [1] * len(self.det_var_names)
                shape[i] = len(grids[i])
                pdf_mesh *= marginal.reshape(shape)

            integrand = pf_mesh * pdf_mesh
            for i in reversed(range(len(grids))):
                integrand = trapezoid(integrand, grids[i], axis=i)
            pf = float(integrand)

        pf = float(np.clip(pf, 1e-30, 1 - 1e-10))
        beta = float(st.norm.ppf(1 - pf))

        return {"pf": pf, "beta": beta}

    def _build_step_result(
        self,
        t: float,
        obs_times: NDArray,
        obs_values: NDArray,
        forecast_prior: Dict[float, Dict[str, Any]],
        forecast_posterior: Dict[float, Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        t_min = min(forecast_prior)
        result = {
            "time": t,
            "obs_times": obs_times.tolist(),
            "obs_values": obs_values.tolist(),
            "prior": {
                "beta": forecast_prior[t_min]["beta"],
                "beta_forecast": {ft: r["beta"] for ft, r in forecast_prior.items()},
            },
            "posterior": {
                "beta": forecast_posterior[t_min]["beta"],
                "beta_forecast": {ft: r["beta"] for ft, r in forecast_posterior.items()},
            },
        }

        if self._build_state is not None:
            result["jpdf_state"] = self._build_state(self.jpdf)

        return result

    def _get_forecast_times(self, t: float) -> List[float]:
        """Only forecast future times (ft >= t)."""
        return [ft for ft in self.forecast_times.tolist() if ft >= t]

    def run_timeline(
        self,
        setting: Dict[str, Any],
        obs_key: str = "corrosion",
        verbose: bool = True,
        **kwargs,
    ) -> Dict[float, Dict[str, Any]]:
        """Run reliability analysis over timeline.

        Extracts observations from a nested setting dict and runs the
        Bayesian update + forecast loop.

        Args:
            setting: Dict keyed by time, each value a dict with at least
                ``obs_key`` field for the observation value.
            obs_key: Key in each setting entry for the observation value.
            verbose: Print progress.
            **kwargs: Forwarded to compute_pf_at_time.

        Returns:
            Results dict keyed by observation time.
        """
        self.results = {}
        times = sorted([float(k) for k in setting.keys()])
        analysis_times = set(self.forecast_times.tolist()) | set(times)

        def get_observations_up_to(t):
            obs_t, obs_v = [], []
            for key in sorted(setting.keys()):
                if float(key) <= t:
                    obs_t.append(float(key))
                    obs_v.append(setting[key][obs_key])
            return np.array(obs_t), np.array(obs_v)

        self._reset_posterior()

        if verbose:
            print(f"{'t':>8s}{'b_prior':>10s}{'b_post':>10s}")
            print("-" * 28)

        for t in times:
            obs_times_arr, obs_values_arr = get_observations_up_to(t)

            if len(obs_times_arr) > 0:
                self._do_bayesian_update(obs_times_arr, obs_values_arr, t, **kwargs)

            key = str(t) if str(t) in setting else f"{t:.1f}"
            step_kwargs = {**kwargs, "setting_at_t": setting.get(key, {}), "t_obs": t}

            future_times = [ft for ft in sorted(analysis_times) if ft >= t]

            forecast_prior = {}
            forecast_posterior = {}
            for ft in future_times:
                forecast_prior[ft] = self.compute_pf_at_time(
                    ft, use_prior=True, **step_kwargs,
                )
                forecast_posterior[ft] = self.compute_pf_at_time(
                    ft, use_prior=False, **step_kwargs,
                )

            self.results[t] = self._build_step_result(
                t=t,
                obs_times=obs_times_arr,
                obs_values=obs_values_arr,
                forecast_prior=forecast_prior,
                forecast_posterior=forecast_posterior,
                **kwargs,
            )

            if verbose:
                t_min = min(forecast_prior)
                bp = forecast_prior[t_min]["beta"]
                bq = forecast_posterior[t_min]["beta"]
                print(f"{t:>8.0f}{bp:>10.2f}{bq:>10.2f}")

        return self.results
