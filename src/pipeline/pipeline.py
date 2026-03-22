"""
Generic Bayesian reliability pipeline.

Orchestrates the full analysis workflow:
1. Setup — initialize JPDF and performance function from specs.
2. Model evaluation — pre-evaluate a model on the parameter space via a
   user-supplied callable.
3. Timeline — iterate over observation times, update the posterior via Bayes,
   compute failure probability and output PDFs at each step.

Supports two analysis modes:
- ``"semi-analytical"``: grid-based N-D integration.
- ``"sample-based"``: importance sampling with weighted particles.
"""

import json
import numpy as np
from scipy import stats as st
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, Callable, List

from ..jpdf import JPDF
from ..performance import BasePerformance


class ReliabilityPipeline:
    """Bayesian reliability analysis pipeline.

    Manages the joint PDF of random variables, pre-evaluates a model on the
    parameter space, performs sequential Bayesian updating as observations
    arrive, and computes failure probabilities and output PDFs at each step.

    The pipeline is domain-agnostic: the physical model is supplied as a
    callable ``model_fn``, and JPDF variable names are mapped to model
    parameters via ``var_map``.
    """

    def __init__(
        self,
        specs_path: Path | str,
        performance: BasePerformance,
        analysis_method: str = "semi-analytical",
        n_samples: int = 100_000,
        obs_error: float = 0.1,
    ) -> None:
        """Initialize the pipeline.

        Args:
            specs_path: Path to the JSON specifications file.
            performance: Performance (limit-state) function instance.
            analysis_method: "semi-analytical" or "sample-based".
            n_samples: Number of IS samples (sample-based mode only).
            obs_error: Standard deviation of observation measurement error.
        """
        self.specs_path = Path(specs_path)
        self.performance = performance
        self.analysis_method = analysis_method
        self.n_samples = n_samples
        self.obs_error = obs_error

        # Components (set during setup / init_model_output)
        self.jpdf: Optional[JPDF] = None
        self.output_at_obs_times: Optional[NDArray] = None
        self.output_residual: Optional[NDArray] = None
        self.output_forecast: Optional[NDArray] = None
        self.output_grid: Optional[NDArray] = None
        self.residual_grid: Optional[NDArray] = None
        self.obs_times: Optional[NDArray] = None
        self.forecast_times: Optional[NDArray] = None
        self.results: Dict[str, Any] = {}
        self.W_posterior_per_t: Dict[float, NDArray] = {}

    @property
    def is_sample_based(self) -> bool:
        return self.analysis_method == "sample-based"

    # ------------------------------------------------------------------
    # Step 1: Setup
    # ------------------------------------------------------------------

    def setup(self, seed: int = 42) -> None:
        """Initialize the JPDF from specs and optionally draw IS samples.

        Args:
            seed: Random seed for importance sampling.
        """
        self.jpdf = JPDF(name="pipeline")
        self.jpdf.set_prior_from_specs(self.specs_path)

        if self.is_sample_based:
            self.jpdf.init_prior_samples(
                n_samples=self.n_samples, covar_IS=2.0, seed=seed
            )

    # ------------------------------------------------------------------
    # Step 2: Model evaluation
    # ------------------------------------------------------------------

    def init_times(
        self,
        obs_times: NDArray,
        forecast_times: NDArray,
    ) -> None:
        """Set observation and forecast time arrays.

        Args:
            obs_times: 1D array of observation times.
            forecast_times: 1D array of forecast evaluation times.
        """
        self.obs_times = obs_times
        self.forecast_times = np.sort(np.unique(
            np.concatenate([forecast_times, obs_times])
        ))

    def init_model_output(
        self,
        model_fn: Callable,
        var_map: Dict[str, str],
        model_kwargs: Dict[str, Any] = None,
        cache_dir: Optional[Path] = None,
        force_rebuild: bool = False,
    ) -> None:
        """Pre-evaluate the model on the parameter space for all time steps.

        Args:
            model_fn: Callable with signature
                ``model_fn(t=..., **var_arrays, grid_based=..., **model_kwargs)``.
                Must return an array with time as the last axis.
            var_map: Dict mapping JPDF variable names to model parameter names,
                e.g. ``{"CR": "CR", "k": "k"}``.
            model_kwargs: Additional keyword arguments passed to model_fn.
            cache_dir: Directory for .npy caches. None disables caching.
            force_rebuild: If True, recompute even if cache exists.
        """
        model_kwargs = model_kwargs or {}

        cache_obs = cache_dir / "output_obs.npy" if cache_dir else None
        cache_fcast = cache_dir / "output_forecast.npy" if cache_dir else None

        if (
            not force_rebuild
            and cache_obs and cache_obs.exists()
            and cache_fcast and cache_fcast.exists()
        ):
            output_obs = np.load(cache_obs)
            output_forecast = np.load(cache_fcast)

            if output_obs.shape[-1] != len(self.obs_times):
                raise ValueError(
                    f"Cached output has {output_obs.shape[-1]} times, "
                    f"analysis needs {len(self.obs_times)}."
                )
        else:
            # Build parameter arrays from JPDF
            if self.is_sample_based:
                engine_vars = {
                    eng: self.jpdf.get_samples(jpdf)
                    for jpdf, eng in var_map.items()
                }
            else:
                engine_vars = {
                    eng: self.jpdf.grids[jpdf]
                    for jpdf, eng in var_map.items()
                }

            output_obs = model_fn(
                t=self.obs_times,
                **engine_vars,
                grid_based=not self.is_sample_based,
                **model_kwargs,
            )
            output_forecast = model_fn(
                t=self.forecast_times,
                **engine_vars,
                grid_based=not self.is_sample_based,
                **model_kwargs,
            )

            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(cache_obs, output_obs)
                np.save(cache_fcast, output_forecast)

        # Downcast large arrays
        for arr_name in ("output_obs", "output_forecast"):
            arr = locals()[arr_name]
            if arr.nbytes / 1e6 >= 100:
                locals()[arr_name] = arr.astype(np.float32)

        self.output_at_obs_times = output_obs
        self.output_residual = output_forecast[..., -1] - output_forecast[..., -2]
        self.output_forecast = output_forecast

        # Build 1D grids for output PDF binning
        n_grid = 1_001
        out_min = min(np.nanmin(self.output_at_obs_times), np.nanmin(self.output_forecast))
        out_max = max(np.nanmax(self.output_at_obs_times), np.nanmax(self.output_forecast))
        self.output_grid = np.sort(np.unique(np.append(
            np.linspace(out_min, out_max, n_grid), 0
        )))

        res_min = np.nanmin(self.output_residual)
        res_max = np.nanmax(self.output_residual)
        self.residual_grid = np.sort(np.unique(np.append(
            np.linspace(res_min, res_max, n_grid), 0
        )))

    # ------------------------------------------------------------------
    # Step 3: Output PDF and Pf
    # ------------------------------------------------------------------

    def get_output_pdf(
        self,
        output: NDArray,
        use_prior: bool = False,
        grid: NDArray = None,
    ) -> Tuple[NDArray, NDArray]:
        """Compute 1D output PDF from the joint parameter distribution.

        Grid-based: bins probability mass from the N-D grid into a 1D
        histogram. Sample-based: weighted histogram.

        Args:
            output: Model output values. Shape matches the grid for
                grid-based, or (n_samples,) for sample-based.
            use_prior: If True, use prior weights/PDF.
            grid: 1D bin edges. Defaults to self.output_grid.

        Returns:
            Tuple of (grid_centers, pdf).
        """
        grid = grid if grid is not None else self.output_grid

        if self.is_sample_based:
            weights = self.jpdf.W_prior_samples if use_prior else self.jpdf.W_samples
            hist, _ = np.histogram(output, bins=grid, weights=weights)
        else:
            pdf = self.jpdf.get_prior() if use_prior else self.jpdf.pdf
            var_grids = [self.jpdf.grids[name] for name in self.jpdf.variable_names]
            diffs = [np.diff(g) for g in var_grids]

            # N-D cell center averaging
            pdf_centers = pdf.copy()
            output_centers = output.copy()
            for ax in range(self.jpdf.nvar):
                n = pdf_centers.shape[ax]
                pdf_centers = (
                    np.take(pdf_centers, range(n - 1), axis=ax)
                    + np.take(pdf_centers, range(1, n), axis=ax)
                ) / 2
                output_centers = (
                    np.take(output_centers, range(n - 1), axis=ax)
                    + np.take(output_centers, range(1, n), axis=ax)
                ) / 2

            # Probability mass
            prob_mass = pdf_centers
            for i, d in enumerate(diffs):
                shape = [1] * self.jpdf.nvar
                shape[i] = len(d)
                prob_mass = prob_mass * d.reshape(shape)

            s_flat = output_centers.flatten()
            pm_flat = prob_mass.flatten()
            valid = np.isfinite(s_flat) & np.isfinite(pm_flat)
            idx = np.digitize(s_flat[valid], bins=grid) - 1
            idx = np.clip(idx, 0, len(grid) - 2)
            hist = np.zeros(len(grid) - 1)
            np.add.at(hist, idx, pm_flat[valid])

        ds = np.diff(grid)
        out_pdf = hist / ds
        grid_centers = (grid[:-1] + grid[1:]) / 2
        integral = np.trapezoid(out_pdf, grid_centers)
        if integral > 0:
            out_pdf /= integral

        return grid_centers, out_pdf

    def compute_pf_at_time(
        self, forecast_time: float, use_prior: bool = False
    ) -> Dict[str, Any]:
        """Compute failure probability, beta, and output PDF at a forecast time.

        Args:
            forecast_time: The time at which to evaluate.
            use_prior: If True, use the prior PDF.

        Returns:
            Dict with keys "pf", "beta", "output_grid", "output_pdf".
        """
        if self.is_sample_based:
            weights = self.jpdf.W_prior_samples if use_prior else self.jpdf.W_samples
            pf = self.performance.failure_probability(
                x=self.output_residual, weights=weights,
            )
        else:
            pf = self.performance.failure_probability(
                x=self.output_residual,
                pdf=self.jpdf.get_prior() if use_prior else self.jpdf.pdf,
                grids=[self.jpdf.grids[name] for name in self.jpdf.variable_names],
            )

        pf_clipped = np.clip(pf, 1e-10, 1 - 1e-10)
        beta = st.norm.ppf(1 - pf_clipped)

        output_at_t = self.output_forecast[
            ..., self.forecast_times == forecast_time
        ].squeeze()
        output_grid, output_pdf = self.get_output_pdf(
            output=output_at_t, use_prior=use_prior,
        )

        return {
            "pf": float(pf),
            "beta": float(beta),
            "output_grid": output_grid.tolist(),
            "output_pdf": output_pdf.tolist(),
        }

    # ------------------------------------------------------------------
    # Step 4: Timeline
    # ------------------------------------------------------------------

    def run_timeline(
        self,
        obs_values: NDArray,
        verbose: bool = True,
    ) -> Dict[float, Dict[str, Any]]:
        """Run the sequential Bayesian analysis over all observation times.

        For each observation time:
        1. Collects all observations up to that time.
        2. Performs a full Bayesian update with all available observations.
        3. Computes prior and posterior failure probabilities and output PDFs.
        4. Stores the JPDF state for visualization.

        Args:
            obs_values: 1D array of observed values, one per obs_time.
            verbose: If True, print progress table.

        Returns:
            Dict mapping observation times to result dicts.
        """
        self.results = {}
        self.W_posterior_per_t = {}

        self.jpdf.reset_to_priors()

        if verbose:
            header = f"{'t':>8s}{'b_prior':>10s}{'b_post':>10s}"
            print(header)
            print("-" * len(header))

        for idx_t, t in enumerate(self.obs_times.tolist()):
            mask = self.obs_times <= t
            obs_up_to = obs_values[mask]

            # Bayesian update
            if len(obs_up_to) > 0:
                model_at_obs = self.output_at_obs_times[..., mask]
                self.jpdf.update(obs_up_to, model_at_obs, obs_error=self.obs_error)

            if self.is_sample_based:
                self.W_posterior_per_t[t] = self.jpdf.W_samples.copy()

            # Pf forecast at all times
            forecast_prior = {}
            forecast_posterior = {}
            for pt in self.forecast_times.tolist():
                forecast_prior[pt] = self.compute_pf_at_time(pt, use_prior=True)
                forecast_posterior[pt] = self.compute_pf_at_time(pt, use_prior=False)

            # Residual output PDFs
            res_grid_prior, res_pdf_prior = self.get_output_pdf(
                self.output_residual, use_prior=True, grid=self.residual_grid,
            )
            res_grid_post, res_pdf_post = self.get_output_pdf(
                self.output_residual, use_prior=False, grid=self.residual_grid,
            )

            t_min = min(forecast_prior)
            self.results[t] = {
                "time": t,
                "obs_times": self.obs_times[mask].tolist(),
                "obs_values": obs_up_to.tolist(),
                "prior": {
                    "pf": forecast_prior[t_min]["pf"],
                    "beta": forecast_prior[t_min]["beta"],
                    "pf_forecast": {pt: r["pf"] for pt, r in forecast_prior.items()},
                    "beta_forecast": {pt: r["beta"] for pt, r in forecast_prior.items()},
                    "output_grid": {pt: r["output_grid"] for pt, r in forecast_prior.items()},
                    "output_pdf": {pt: r["output_pdf"] for pt, r in forecast_prior.items()},
                },
                "posterior": {
                    "pf": forecast_posterior[t_min]["pf"],
                    "beta": forecast_posterior[t_min]["beta"],
                    "pf_forecast": {pt: r["pf"] for pt, r in forecast_posterior.items()},
                    "beta_forecast": {pt: r["beta"] for pt, r in forecast_posterior.items()},
                    "output_grid": {pt: r["output_grid"] for pt, r in forecast_posterior.items()},
                    "output_pdf": {pt: r["output_pdf"] for pt, r in forecast_posterior.items()},
                },
                "residual": {
                    "prior_grid": res_grid_prior.tolist(),
                    "prior_pdf": res_pdf_prior.tolist(),
                    "posterior_grid": res_grid_post.tolist(),
                    "posterior_pdf": res_pdf_post.tolist(),
                },
                "jpdf_state": self._build_jpdf_state(obs_up_to, model_at_obs),
            }

            if verbose:
                bp = forecast_prior[t_min]["beta"]
                bq = forecast_posterior[t_min]["beta"]
                print(f"{t:>8.0f}{bp:>10.2f}{bq:>10.2f}")

        return self.results

    def _build_jpdf_state(
        self, obs_values: NDArray, model_at_obs: NDArray
    ) -> Dict[str, Any]:
        """Build JPDF state dict for results storage and plotting.

        Args:
            obs_values: Observed values used for updating.
            model_at_obs: Model output at observation times.

        Returns:
            Dict with per-variable grids, prior/posterior marginals, and
            (grid-based only) joint arrays.
        """
        state = {}
        if self.is_sample_based:
            for name in self.jpdf.variable_names:
                samples = self.jpdf.get_samples(name)
                grid = self.jpdf.grids[name]
                hist_prior, _ = np.histogram(
                    samples, bins=grid, weights=self.jpdf.W_prior_samples
                )
                hist_post, _ = np.histogram(
                    samples, bins=grid, weights=self.jpdf.W_samples
                )
                dg = np.diff(grid)
                centers = (grid[:-1] + grid[1:]) / 2
                state[f"{name}_grid"] = centers.tolist()
                state[f"{name}_prior"] = (hist_prior / dg).tolist()
                state[f"{name}_posterior"] = (hist_post / dg).tolist()
        else:
            for name in self.jpdf.variable_names:
                state[f"{name}_grid"] = self.jpdf.grids[name].tolist()
                state[f"{name}_prior"] = self.jpdf.priors[name].tolist()
                state[f"{name}_posterior"] = self.jpdf.marginals[name].tolist()
            state["prior"] = self.jpdf.get_prior().tolist()
            state["loglikes"] = self.jpdf.get_loglikes(
                obs_values, model_at_obs, self.obs_error
            ).tolist()
            state["posterior"] = self.jpdf.pdf.tolist()
        return state
