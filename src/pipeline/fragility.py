"""
Fragility-based reliability pipeline.

Builds a fragility surface (Pf vs degradation parameter), then performs
sequential Bayesian updating of a degradation parameter (e.g. C50) and
computes Pf by integrating the fragility curve against the degradation
ratio PDF at each time step.

This is the "ark" pattern — applicable to case studies where Pf is
computed via fragility integration rather than direct model evaluation.
"""

import numpy as np
from scipy import stats as st
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List

from .base import BasePipeline


class FragilityPipeline(BasePipeline):
    """Pipeline that integrates a pre-computed fragility surface with a
    time-varying parameter PDF to compute Pf.

    The JPDF is expected to be a domain-specific subclass that provides
    methods like ``update_C50``, ``reset_C50_to_prior``, ``get_C50_stats``,
    and ``C50_grid``/``C50_pdf``/``C50_prior`` attributes.

    The corrosion_model and fragility_surface must be set externally
    (by the case study script) before calling ``run_timeline``.
    """

    def __init__(
        self,
        settings_path: Path | str,
        performance,
        obs_error: float = 0.1,
    ) -> None:
        super().__init__(settings_path, performance, obs_error)

        # Domain components (set externally by the case study)
        self.corrosion_model = None
        self.fragility_surface = None

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------

    def setup(self, seed: int = 42, **kwargs) -> None:
        """Initialize JPDF from settings.

        The JPDF instance must be set on self.jpdf by the case study before
        calling setup(), since it requires a domain-specific subclass.
        If self.jpdf is already set, this just loads the settings. Otherwise
        it raises.
        """
        if self.jpdf is None:
            raise ValueError(
                "Set self.jpdf to a domain-specific JPDF subclass before calling setup()."
            )
        self.jpdf.set_prior_from_settings(self.settings_path)

    def _reset_posterior(self) -> None:
        self.jpdf.reset_C50_to_prior()

    def _do_bayesian_update(
        self, obs_times: NDArray, obs_values: NDArray, t: float, **kwargs
    ) -> None:
        self.jpdf.update_C50(obs_times, obs_values)

    def compute_pf_at_time(
        self,
        forecast_time: float,
        use_prior: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute Pf at a forecast time via fragility integration.

        Args:
            forecast_time: Time at which to evaluate.
            use_prior: Use prior C50 PDF (True) or posterior (False).
            **kwargs: Must include:
                - moment_survived (float): proven moment capacity threshold.
                - last_obs_time (float, optional): time of last observation.
                - last_obs (float, optional): last observed corrosion value.

        Returns:
            Dict with pf, beta, C50_stats, cr_pdf.
        """
        if self.fragility_surface is None:
            raise ValueError("Set fragility_surface before computing Pf.")

        moment_survived = kwargs.get("moment_survived", 0.0)
        last_obs_time = kwargs.get("last_obs_time", None)
        last_obs = kwargs.get("last_obs", None)

        C50_pdf = self.jpdf.C50_pdf if not use_prior else self.jpdf.C50_prior

        cr_grid, cr_pdf = self.get_corrosion_ratio_pdf(
            forecast_time, C50_pdf, last_obs_time, last_obs
        )

        fragility = self.fragility_surface.get_curve_at(moment_survived)

        cr_pdf_interp = np.interp(
            fragility.corrosion_ratios, cr_grid, cr_pdf, left=0, right=0
        )
        norm = np.trapezoid(cr_pdf_interp, fragility.corrosion_ratios)
        if norm > 0:
            cr_pdf_interp /= norm

        pf, beta = self.performance.pf_from_fragility(fragility, cr_pdf_interp)

        return {
            "pf": pf,
            "beta": beta,
            "time": forecast_time,
            "moment_survived": moment_survived,
            "C50_stats": self.jpdf.get_C50_stats(),
            "cr_pdf": cr_pdf_interp.tolist(),
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
        forecast_posterior_proven = kwargs.get("forecast_posterior_proven", {})

        t_min = min(forecast_prior)
        cr_grid_prior, _ = self.get_corrosion_ratio_pdf(t, self.jpdf.C50_prior)

        result = {
            "time": t,
            "obs_times": obs_times.tolist(),
            "obs_values": obs_values.tolist(),
            "prior": {
                "beta": forecast_prior[t_min]["beta"],
                "beta_forecast": {ft: r["beta"] for ft, r in forecast_prior.items()},
                "cr_forecast": {ft: r["cr_pdf"] for ft, r in forecast_prior.items()},
            },
            "posterior": {
                "beta": forecast_posterior[t_min]["beta"],
                "beta_forecast": {ft: r["beta"] for ft, r in forecast_posterior.items()},
                "cr_forecast": {ft: r["cr_pdf"] for ft, r in forecast_posterior.items()},
            },
            "jpdf_state": {
                "C50_grid": self.jpdf.C50_grid.tolist(),
                "C50_prior": self.jpdf.C50_prior.tolist(),
                "C50_posterior": self.jpdf.C50_pdf.tolist(),
                "cr_grid": cr_grid_prior.tolist(),
            },
        }

        if forecast_posterior_proven:
            t_min_proven = min(forecast_posterior_proven)
            result["posterior_proven_strength"] = {
                "beta": forecast_posterior_proven[t_min_proven]["beta"],
                "beta_forecast": {ft: r["beta"] for ft, r in forecast_posterior_proven.items()},
                "cr_forecast": {ft: r["cr_pdf"] for ft, r in forecast_posterior_proven.items()},
            }

        return result

    def _get_forecast_times(self, t: float) -> List[float]:
        """Only forecast future times (ft >= t)."""
        return [ft for ft in self.forecast_times.tolist() if ft >= t]

    # ------------------------------------------------------------------
    # Fragility-specific public methods
    # ------------------------------------------------------------------

    def get_corrosion_ratio_pdf(
        self,
        t: float,
        C50_pdf: Optional[NDArray] = None,
        last_obs_time: Optional[float] = None,
        last_obs: Optional[float] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Compute corrosion ratio PDF at time t given C50 distribution.

        Args:
            t: Time [years].
            C50_pdf: PDF over C50 grid. Uses jpdf.C50_pdf if None.
            last_obs_time: Time of last observation (for conditional forecast).
            last_obs: Last observed corrosion value.

        Returns:
            Tuple of (corrosion_ratio_grid, pdf).
        """
        if self.corrosion_model is None:
            raise ValueError("Set corrosion_model before computing corrosion ratio PDF.")

        if C50_pdf is None:
            C50_pdf = self.jpdf.C50_pdf

        return self.corrosion_model.corrosion_ratio_pdf(
            t=t, C50_pdf=C50_pdf,
            last_obs_time=last_obs_time, last_obs=last_obs,
        )

    def run_timeline(
        self,
        setting: Dict[str, Any],
        verbose: bool = True,
        **kwargs,
    ) -> Dict[float, Dict[str, Any]]:
        """Run reliability analysis over timeline.

        Overrides the base run_timeline because the ark pattern extracts
        observations from a nested setting dict (not a flat obs_values array)
        and needs per-step context (moment_survived, corrosion_obs).

        Args:
            setting: Case study setting dict with time-series data.
                Each entry has "corrosion" and optionally "moment_survived".
            verbose: Print progress.

        Returns:
            Results dict keyed by observation time.
        """
        if self.fragility_surface is None:
            raise ValueError("Set fragility_surface before running timeline.")

        self.results = {}
        times = sorted([float(k) for k in setting.keys()])

        # Build full analysis time grid
        analysis_times = set(self.forecast_times.tolist()) | set(times)

        # Extract survived moments from setting
        survived_times = []
        survived_moments = []
        for time_key in sorted(setting.keys()):
            ms = setting[time_key].get("moment_survived")
            if ms is not None:
                survived_times.append(float(time_key))
                survived_moments.append(ms)

        def interpolate_moment_survived(ft: float, t_obs: float) -> float:
            st_ = [s for s, m in zip(survived_times, survived_moments) if s <= t_obs]
            sm = [m for s, m in zip(survived_times, survived_moments) if s <= t_obs]
            if not st_:
                return 0.0
            return float(np.interp(ft, st_, sm, left=sm[0], right=sm[-1]))

        def get_observations_up_to(t: float):
            obs_t, obs_v = [], []
            for time_key in sorted(setting.keys()):
                time = float(time_key)
                if time <= t:
                    obs_t.append(time)
                    obs_v.append(setting[time_key]["corrosion"])
            return np.array(obs_t), np.array(obs_v)

        self._reset_posterior()

        if verbose:
            header = f"{'t':>8s}{'b_prior':>10s}{'b_post':>10s}{'b_proven':>10s}"
            print(header)
            print("-" * len(header))

        for t in times:
            obs_times, obs_values = get_observations_up_to(t)

            if len(obs_times) > 0:
                self._do_bayesian_update(obs_times, obs_values, t)

            key = str(t) if str(t) in setting else f"{t:.1f}"
            corrosion_obs = setting[key]["corrosion"] if key in setting else None

            future_times = [ft for ft in sorted(analysis_times) if ft >= t]

            forecast_prior = {}
            forecast_posterior = {}
            forecast_posterior_proven = {}

            for ft in future_times:
                ms_at_ft = interpolate_moment_survived(ft, t)

                forecast_prior[ft] = self.compute_pf_at_time(
                    ft, use_prior=True,
                    moment_survived=0.0,
                    last_obs_time=None, last_obs=None,
                )
                forecast_posterior[ft] = self.compute_pf_at_time(
                    ft, use_prior=False,
                    moment_survived=0.0,
                    last_obs_time=t, last_obs=corrosion_obs,
                )
                forecast_posterior_proven[ft] = self.compute_pf_at_time(
                    ft, use_prior=False,
                    moment_survived=ms_at_ft,
                    last_obs_time=t, last_obs=corrosion_obs,
                )

            self.results[t] = self._build_step_result(
                t=t,
                obs_times=obs_times,
                obs_values=obs_values,
                forecast_prior=forecast_prior,
                forecast_posterior=forecast_posterior,
                forecast_posterior_proven=forecast_posterior_proven,
            )

            if verbose:
                t_min = min(forecast_prior)
                bp = forecast_prior[t_min]["beta"]
                bq = forecast_posterior[t_min]["beta"]
                bps = forecast_posterior_proven[t_min]["beta"]
                print(f"{t:>8.0f}{bp:>10.2f}{bq:>10.2f}{bps:>10.2f}")

        return self.results
