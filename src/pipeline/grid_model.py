"""
Grid/sample-based reliability pipeline.

Pre-evaluates a model on the JPDF parameter space (grid or IS samples),
then performs sequential Bayesian updating of the full joint distribution
and computes Pf by integrating indicator x PDF or via weighted sums.
"""

import numpy as np
from scipy import stats as st
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, Callable, List

from ..jpdf import JPDF
from .base import BasePipeline


class GridModelPipeline(BasePipeline):
    """Pipeline that pre-evaluates a model on the JPDF parameter space.

    Supports both grid-based (semi-analytical) and sample-based (importance
    sampling) modes.

    Usage::

        pipeline = GridModelPipeline(
            settings_path="settings.json",
            performance=my_performance,
            config=config_dict,          # auto-extracts analysis_method, n_samples, obs_error
        )
        pipeline.setup(seed=42)
        pipeline.init_model_output(model_fn=my_model, model_kwargs={...})
        results = pipeline.run(obs_values=obs_values)
    """

    def __init__(
        self,
        settings_path: Path | str,
        performance,
        config: Optional[Dict[str, Any]] = None,
        analysis_method: str = None,
        n_samples: int = None,
        obs_error: float = None,
    ) -> None:
        """Initialize the grid model pipeline.

        Config values are read from the ``config`` dict if provided.
        Explicit keyword arguments override config values.

        Args:
            settings_path: Path to the JSON settings file.
            performance: Performance (limit-state) function instance.
            config: Parameters dict (e.g. from settings["parameters"]).
                Reads analysis_method, n_samples, obs_error, forecast_interval,
                end_time automatically.
            analysis_method: Override for analysis method.
            n_samples: Override for number of IS samples.
            obs_error: Override for observation error std.
        """
        self.config = config or {}
        _obs_error = obs_error or self.config.get("obs_error", 0.1)
        super().__init__(settings_path, performance, _obs_error)

        self.analysis_method = analysis_method or self.config.get("analysis_method", "semi-analytical")
        self.n_samples = n_samples or self.config.get("n_samples", 100_000)

        # Model output arrays (set by init_model_output)
        self.output_at_obs_times: Optional[NDArray] = None
        self.output_residual: Optional[NDArray] = None
        self.output_forecast: Optional[NDArray] = None
        self.output_grid: Optional[NDArray] = None
        self.residual_grid: Optional[NDArray] = None
        self.W_posterior_per_t: Dict[float, NDArray] = {}

    @property
    def is_sample_based(self) -> bool:
        return self.analysis_method == "sample-based"

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------

    def setup(self, seed: int = 42, **kwargs) -> None:
        """Initialize JPDF from settings and optionally draw IS samples."""
        self.jpdf = JPDF(name="pipeline")
        self.jpdf.set_variables(self.settings_path)
        self.jpdf.set_prior_from_settings()

        if self.is_sample_based:
            self.jpdf.init_prior_samples(
                n_samples=self.n_samples, covar_IS=2.0, seed=seed
            )

    def _reset_posterior(self) -> None:
        self.jpdf.reset_to_priors()
        self.W_posterior_per_t = {}

    def _do_bayesian_update(
        self, obs_times: NDArray, obs_values: NDArray, t: float, **kwargs
    ) -> None:
        mask = self.obs_times <= t
        model_at_obs = self.output_at_obs_times[..., mask]
        self.jpdf.update(obs_values, model_at_obs, obs_error=self.obs_error)
        if self.is_sample_based:
            self.W_posterior_per_t[t] = self.jpdf.W_samples.copy()

    def compute_pf_at_time(
        self, forecast_time: float, use_prior: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """Compute Pf, beta, and output PDF at a forecast time."""
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

    def _build_step_result(
        self,
        t: float,
        obs_times: NDArray,
        obs_values: NDArray,
        forecast_prior: Dict[float, Dict[str, Any]],
        forecast_posterior: Dict[float, Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        mask = self.obs_times <= t
        model_at_obs = self.output_at_obs_times[..., mask]

        # Residual output PDFs
        res_grid_prior, res_pdf_prior = self.get_output_pdf(
            self.output_residual, use_prior=True, grid=self.residual_grid,
        )
        res_grid_post, res_pdf_post = self.get_output_pdf(
            self.output_residual, use_prior=False, grid=self.residual_grid,
        )

        t_min = min(forecast_prior)
        return {
            "time": t,
            "obs_times": obs_times.tolist(),
            "obs_values": obs_values.tolist(),
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
            "jpdf_state": self._build_jpdf_state(obs_values, model_at_obs),
        }

    # ------------------------------------------------------------------
    # Convenience: setup + run in one call
    # ------------------------------------------------------------------

    def run(
        self,
        obs_times: NDArray,
        obs_values: NDArray,
        model_fn: Callable,
        var_map: Optional[Dict[str, str]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        forecast_times: Optional[NDArray] = None,
        cache_dir: Optional[Path] = None,
        force_rebuild: bool = False,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict[float, Dict[str, Any]]:
        """Run the full pipeline: setup, model evaluation, timeline.

        Combines setup(), init_times(), init_model_output(), and
        run_timeline() into a single call for convenience.

        Args:
            obs_times: Observation times.
            obs_values: Observed values (one per obs_time).
            model_fn: Model callable.
            var_map: JPDF-to-model variable name mapping. Defaults to
                identity mapping from JPDF variable names.
            model_kwargs: Extra kwargs for model_fn.
            forecast_times: Forecast time array. If None, built from
                config keys "forecast_interval" and "end_time".
            cache_dir: Cache directory for model output.
            force_rebuild: Force recomputation of cached output.
            seed: Random seed.
            verbose: Print progress.

        Returns:
            Results dict.
        """
        self.setup(seed=seed)

        # Default var_map: identity
        if var_map is None:
            var_map = {name: name for name in self.jpdf.variable_names}

        # Default forecast_times from config
        if forecast_times is None:
            interval = self.config.get("forecast_interval", 10)
            end_time = self.config.get("end_time", obs_times[-1])
            forecast_times = np.arange(0, end_time + interval, interval)

        self.init_times(obs_times=obs_times, forecast_times=forecast_times)

        self.init_model_output(
            model_fn=model_fn,
            var_map=var_map,
            model_kwargs=model_kwargs,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild,
        )

        return self.run_timeline(obs_values=obs_values, verbose=verbose)

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def init_model_output(
        self,
        model_fn: Callable,
        var_map: Optional[Dict[str, str]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[Path] = None,
        force_rebuild: bool = False,
    ) -> None:
        """Pre-evaluate the model on the parameter space for all time steps.

        Args:
            model_fn: Callable with signature
                ``model_fn(t=..., **var_arrays, grid_based=..., **model_kwargs)``.
            var_map: JPDF-to-model variable name mapping. Defaults to
                identity mapping.
            model_kwargs: Additional keyword arguments passed to model_fn.
            cache_dir: Directory for .npy caches.
            force_rebuild: If True, recompute even if cache exists.
        """
        if var_map is None:
            var_map = {name: name for name in self.jpdf.variable_names}
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
                t=self.obs_times, **engine_vars,
                grid_based=not self.is_sample_based, **model_kwargs,
            )
            output_forecast = model_fn(
                t=self.forecast_times, **engine_vars,
                grid_based=not self.is_sample_based, **model_kwargs,
            )

            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(cache_obs, output_obs)
                np.save(cache_fcast, output_forecast)

        # Downcast large arrays
        if output_obs.nbytes / 1e6 >= 100:
            output_obs = output_obs.astype(np.float32)
        if output_forecast.nbytes / 1e6 >= 100:
            output_forecast = output_forecast.astype(np.float32)

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
    # Output PDF
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
            output: Model output values.
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

    # ------------------------------------------------------------------
    # JPDF state
    # ------------------------------------------------------------------

    def _build_jpdf_state(
        self, obs_values: NDArray, model_at_obs: NDArray
    ) -> Dict[str, Any]:
        """Build JPDF state snapshot for results and plotting."""
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
