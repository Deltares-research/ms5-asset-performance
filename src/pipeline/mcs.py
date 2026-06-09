"""
Monte Carlo reliability pipeline.

Concrete sibling of :class:`FragilityPipeline` and :class:`GridModelPipeline`.
Computes Pf at each forecast time by drawing samples from a (possibly
spatial) JPDF and evaluating an LSF on those samples.

The pipeline owns nothing domain-specific. Two strategy slots do all the
heavy lifting:

* ``sampler`` — draws ``n_samples`` realisations from the JPDF at a given
  forecast time, under either the prior or the posterior. For the spatial
  workflow this returns full ``(n_samples, n_locs, n_params)`` arrays.
* ``evaluator`` — given those samples, returns the LSF value per
  (sample, location). Failure is ``g < 0``. For the alphas/nested-FORM
  workflow this is a tangent-hyperplane evaluation parameterised by the
  1-D FORM design point in cr.

The base class's :meth:`run_timeline` loop already does:

1. ``_reset_posterior`` — reset the JPDF (delegated below).
2. For each obs time, ``_do_bayesian_update`` (delegated below).
3. For each forecast time, ``compute_pf_at_time`` under both prior and
   posterior.
4. ``_build_step_result`` to assemble the per-step output dict.

Where the spatial/Bayesian propagation lives
--------------------------------------------

Obs at location 0 -> Bayesian update -> projection to all other locs is
the JPDF's job, not the pipeline's. The pipeline calls
``self.jpdf.update_at_location(obs_times, obs_values, obs_loc_index, t)``
and the JPDF takes care of the discrete Bayes / IS reweight at the obs
loc and propagates via the spatial Kriging kernel to the remaining locs.

This concentrates the "what does an observation mean for the rest of the
wall" logic in one file (the extended JPDF), independent of which
sampler/evaluator the MCS pipeline is wired with.

Composition example (spatial alphas leg)
----------------------------------------

::

    from src.jpdf import SpatialJPDF            # JPDF with spatial cov
    from src.pipeline import MCSPipeline
    from spatial.samplers import NatafSpatialSampler
    from spatial.evaluators import AlphasTangentLSF

    jpdf = SpatialJPDF(name="dsheet")
    jpdf.set_variables(settings_path)
    jpdf.set_spatial_kernel(per_variable_kernels)     # (theta, rho_0) per var
    jpdf.set_locations(x_sections)
    jpdf.set_prior_from_settings()

    # 1-D FORM in cr, per forecast time -> (beta_T, alpha_cr, alpha_basic)(t).
    # Reuses the cached fragility curve; no D-Sheet calls in the inner loop.
    evaluator = AlphasTangentLSF.from_fragility(
        fragility_points, forecast_times,
    )

    sampler = NatafSpatialSampler(jpdf)            # samples cr field + u-fields

    pipeline = MCSPipeline(
        settings_path=settings_path,
        sampler=sampler,
        evaluator=evaluator,
        n_samples=100_000, seed=42,
    )
    pipeline.jpdf = jpdf
    pipeline.init_times(obs_times, forecast_times)
    results = pipeline.run_timeline(obs_values)

Drop-in for the per-section direct-MCS workflow (`run_mc.py`-style):
the sampler draws directly from the JPDF marginals (no spatial cov), the
evaluator calls the actual ``lsf_wall_anchor`` per sample. Same pipeline,
different strategy classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy import stats as st

from .base import BasePipeline


# ----------------------------------------------------------------------
# Strategy protocols
# ----------------------------------------------------------------------

@runtime_checkable
class Sampler(Protocol):
    """Draws MC realisations from the JPDF at a forecast time.

    The returned object is opaque to the pipeline — only the evaluator
    needs to know its concrete shape. For spatial work it will typically
    be a dict ``{var_name: np.ndarray of shape (n_samples, n_locs)}``
    plus any auxiliary fields the evaluator needs (e.g. the cr-axis xi
    used by alphas-mode).
    """

    def draw(
        self,
        forecast_time: float,
        n_samples: int,
        use_prior: bool,
        seed: int,
        **kwargs: Any,
    ) -> Any:
        """Return a sample batch at ``forecast_time``.

        Implementations should either pull from ``jpdf.priors`` /
        ``jpdf.marginals`` (grid-based) or from a fresh / cached
        Cholesky factor (sample-based). For spatial JPDFs this includes
        per-location correlated draws via the JPDF's spatial kernel.
        """
        ...


@runtime_checkable
class LSFEvaluator(Protocol):
    """Evaluates the limit-state function on a sample batch.

    Convention: ``g < 0`` is failure. Shape of the returned array is
    ``(n_samples, n_locs)``; for non-spatial work ``n_locs == 1`` so the
    same downstream tally code works in both cases.
    """

    def evaluate(
        self,
        samples: Any,
        forecast_time: float,
        use_prior: bool,
        **kwargs: Any,
    ) -> NDArray:
        """Return ``g`` with shape ``(n_samples, n_locs)``."""
        ...


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

class MCSPipeline(BasePipeline):
    """Sequential Bayesian reliability via Monte Carlo sampling.

    Delegates sample drawing to a ``Sampler``, LSF evaluation to an
    ``LSFEvaluator``, and Bayesian update + reset to the JPDF. The
    pipeline itself only knows the time loop and the failure tally.

    The Pf computed per forecast time is a *system* Pf
    ``P(exists loc i: g_i < 0)`` plus a per-location Pf vector
    ``P(g_i < 0)``. Both are returned from
    :meth:`compute_pf_at_time`; the system Pf is what populates the
    standard ``"beta"`` field expected by :meth:`BasePipeline._print_step`.
    """

    def __init__(
        self,
        settings_path: Path | str,
        sampler: Sampler,
        evaluator: LSFEvaluator,
        config: Optional[Dict[str, Any]] = None,
        n_samples: int = 10_000,
        seed: int = 42,
        obs_error: float = None,
    ) -> None:
        """Build the MCS pipeline.

        Args:
            settings_path: Path to the JSON settings file.
            sampler: Strategy that draws ``n_samples`` realisations at a
                given forecast time.
            evaluator: Strategy that computes ``g(samples, t)``.
            config: Parameters dict (e.g. ``settings["parameters"]``).
            n_samples: MC sample count per forecast-time evaluation.
            seed: RNG seed; the sampler is expected to derive per-call
                seeds from this for prior/posterior reproducibility.
            obs_error: Override for observation error std.
        """
        self.config = config or {}
        _obs_error = obs_error or self.config.get(
            "obs_error_std", self.config.get("obs_error", 0.1),
        )
        # BasePipeline requires a ``performance`` arg; the MCS pipeline
        # carries the LSF in its ``evaluator`` instead, so we pass None
        # and ignore the slot. (BasePipeline never invokes it directly.)
        super().__init__(settings_path, performance=None, obs_error=_obs_error)

        self.sampler = sampler
        self.evaluator = evaluator
        self.n_samples = int(n_samples)
        self.seed = int(seed)

    # ------------------------------------------------------------------
    # Abstract-hook implementations
    # ------------------------------------------------------------------

    def setup(self, seed: int = 42, **kwargs: Any) -> None:
        if self.jpdf is None:
            raise ValueError(
                "MCSPipeline requires a JPDF. Set `pipeline.jpdf = ...` "
                "before calling setup()."
            )
        self.seed = int(seed)

    def _reset_posterior(self) -> None:
        """Drop the JPDF posterior back to the prior.

        For a spatial JPDF this also clears any cached per-location
        posterior moments / Kriging weights so the next forecast pulls
        the unconditional cr field.
        """
        if self.jpdf is None:
            return
        self.jpdf.reset_to_priors()

    def _do_bayesian_update(
        self,
        obs_times: NDArray,
        obs_values: NDArray,
        t: float,
        **kwargs: Any,
    ) -> None:
        """Hand the observation set to the JPDF and let it propagate.

        Spatial propagation (obs at loc 0 -> Krige to all other locs)
        lives in the JPDF's update method, not here. This keeps the
        update logic in one place regardless of which sampler /
        evaluator is wired into the pipeline.

        Expected JPDF surface (sketch):

            jpdf.update_at_location(
                obs_times=obs_times,
                obs_values=obs_values,
                obs_loc_index=kwargs.get("obs_loc_index", 0),
                obs_error=self.obs_error,
                t=t,
            )

        Falls back to the existing ``jpdf.update(obs_values,
        model_output, obs_error)`` signature when no spatial-aware
        update is available (i.e. for non-spatial JPDFs).
        """
        if self.jpdf is None:
            return
        if hasattr(self.jpdf, "update_at_location"):
            self.jpdf.update_at_location(
                obs_times=obs_times,
                obs_values=obs_values,
                obs_loc_index=kwargs.get("obs_loc_index", 0),
                obs_error=self.obs_error,
                t=t,
            )
        else:
            # Non-spatial fallback — caller is responsible for the
            # ``model_output`` array shape.
            model_output = kwargs.get("model_output")
            if model_output is not None:
                self.jpdf.update(
                    obs_values=obs_values,
                    model_output=model_output,
                    obs_error=self.obs_error,
                )

    def compute_pf_at_time(
        self,
        forecast_time: float,
        use_prior: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Draw samples, evaluate LSF, tally failures.

        Returns a dict containing at least:

        * ``pf`` / ``beta`` — system-level Pf and reliability index
          (these are what :meth:`BasePipeline._print_step` reads).
        * ``pf_section`` — per-location Pf array, shape ``(n_locs,)``.
        * ``beta_section`` — per-location beta array, shape ``(n_locs,)``.
        * ``n_fail_section`` / ``n_fail_system`` — raw counts for CI.
        * ``n_samples`` — for downstream CI estimation.
        """
        # Per-call seed: prior and posterior at the same forecast time
        # share the same field draws so cross-leg comparisons are
        # variance-reduced. (Common random numbers.)
        per_call_seed = self.seed + int(round(forecast_time * 1_000))

        samples = self.sampler.draw(
            forecast_time=forecast_time,
            n_samples=self.n_samples,
            use_prior=use_prior,
            seed=per_call_seed,
            **kwargs,
        )

        g = self.evaluator.evaluate(
            samples=samples,
            forecast_time=forecast_time,
            use_prior=use_prior,
            **kwargs,
        )
        g = np.asarray(g)
        if g.ndim == 1:                            # non-spatial -> add loc axis
            g = g[:, None]

        fail = g < 0
        n_fail_section = fail.sum(axis=0)                       # (n_locs,)
        n_fail_system = int(fail.any(axis=1).sum())             # scalar

        n = self.n_samples
        pf_section = n_fail_section / n
        pf_system = n_fail_system / n

        # Map to (clipped) beta so the print/result schema matches the
        # fragility/grid-model pipelines.
        def _beta(pf: NDArray | float) -> NDArray | float:
            arr = np.atleast_1d(pf).astype(float)
            arr = np.clip(arr, 1e-30, 1 - 1e-15)
            beta = st.norm.ppf(1.0 - arr)
            return beta if arr.size > 1 else float(beta[0])

        return {
            # BasePipeline schema
            "pf": float(pf_system),
            "beta": float(_beta(pf_system)),
            # Spatial extras
            "pf_section": pf_section.tolist(),
            "beta_section": np.atleast_1d(_beta(pf_section)).tolist(),
            "n_fail_section": n_fail_section.tolist(),
            "n_fail_system": n_fail_system,
            "n_samples": n,
        }

    def _build_step_result(
        self,
        t: float,
        obs_times: NDArray,
        obs_values: NDArray,
        forecast_prior: Dict[float, Dict[str, Any]],
        forecast_posterior: Dict[float, Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Assemble the per-obs-time step result.

        Mirrors :meth:`FragilityPipeline._build_step_result` so existing
        consumers (forecast.json schema, plot drivers) keep working;
        adds ``pf_forecast`` + ``pf_section_forecast`` blocks since
        spatial work needs the per-location series.
        """
        t_min = min(forecast_prior)

        def _series(d: Dict[float, Dict[str, Any]], key: str) -> Dict[float, Any]:
            return {ft: r[key] for ft, r in d.items()}

        result = {
            "time": float(t),
            "obs_times": obs_times.tolist(),
            "obs_values": obs_values.tolist(),
            "prior": {
                "beta": forecast_prior[t_min]["beta"],
                "beta_forecast": _series(forecast_prior, "beta"),
                "pf_forecast": _series(forecast_prior, "pf"),
                "beta_section_forecast": _series(forecast_prior, "beta_section"),
                "pf_section_forecast": _series(forecast_prior, "pf_section"),
            },
            "posterior": {
                "beta": forecast_posterior[t_min]["beta"],
                "beta_forecast": _series(forecast_posterior, "beta"),
                "pf_forecast": _series(forecast_posterior, "pf"),
                "beta_section_forecast": _series(forecast_posterior, "beta_section"),
                "pf_section_forecast": _series(forecast_posterior, "pf_section"),
            },
            "n_samples": self.n_samples,
        }

        if hasattr(self.jpdf, "build_state"):
            result["jpdf_state"] = self.jpdf.build_state()
        return result

    # ------------------------------------------------------------------
    # Convenience: forecast only future times (matches FragilityPipeline)
    # ------------------------------------------------------------------

    def _get_forecast_times(self, t: float) -> List[float]:
        return [ft for ft in self.forecast_times.tolist() if ft >= t]
