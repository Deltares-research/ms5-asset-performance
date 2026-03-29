"""
Abstract base class for sequential Bayesian reliability pipelines.

Provides the template-method `run_timeline` loop and delegates domain-specific
operations (Bayesian update, Pf computation, result assembly) to abstract hooks
that subclasses implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from numpy.typing import NDArray
from scipy import stats as st

from ..jpdf import JPDF
from ..performance import BasePerformance


class BasePipeline(ABC):
    """Abstract base for sequential Bayesian reliability pipelines.

    Owns the JPDF, performance function, time arrays, and results dict.
    Subclasses provide the Bayesian update logic, Pf computation, and
    result assembly via abstract hooks.
    """

    def __init__(
        self,
        specs_path: Path | str,
        performance: BasePerformance,
        obs_error: float = 0.1,
    ) -> None:
        """Initialize the base pipeline.

        Args:
            specs_path: Path to the JSON specifications file.
            performance: Performance (limit-state) function instance.
            obs_error: Standard deviation of observation measurement error.
        """
        self.specs_path = Path(specs_path)
        self.performance = performance
        self.obs_error = obs_error

        self.jpdf: Optional[JPDF] = None
        self.obs_times: Optional[NDArray] = None
        self.forecast_times: Optional[NDArray] = None
        self.results: Dict[float, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Concrete: time management
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

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self, seed: int = 42, **kwargs) -> None:
        """Initialize JPDF and any domain-specific components."""
        ...

    @abstractmethod
    def _reset_posterior(self) -> None:
        """Reset the JPDF posterior to the prior before the timeline loop."""
        ...

    @abstractmethod
    def _do_bayesian_update(
        self,
        obs_times: NDArray,
        obs_values: NDArray,
        t: float,
        **kwargs,
    ) -> None:
        """Perform Bayesian update given observations up to time t.

        Mutates self.jpdf in place.
        """
        ...

    @abstractmethod
    def compute_pf_at_time(
        self,
        forecast_time: float,
        use_prior: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute Pf and beta at a single forecast time.

        Returns:
            Dict with at least "pf" and "beta" keys.
        """
        ...

    @abstractmethod
    def _build_step_result(
        self,
        t: float,
        obs_times: NDArray,
        obs_values: NDArray,
        forecast_prior: Dict[float, Dict[str, Any]],
        forecast_posterior: Dict[float, Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Assemble the full result dict for one observation step.

        Subclasses add domain-specific fields (output PDFs, corrosion
        ratio PDFs, posterior_proven_strength, jpdf_state, etc.).
        """
        ...

    # ------------------------------------------------------------------
    # Concrete: timeline loop (template method)
    # ------------------------------------------------------------------

    def run_timeline(
        self,
        obs_values: NDArray,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[float, Dict[str, Any]]:
        """Run sequential Bayesian analysis over observation times.

        For each observation time:
        1. Collect observations up to t.
        2. Bayesian update (delegated to subclass).
        3. Compute prior and posterior Pf forecasts.
        4. Build and store result dict.

        Args:
            obs_values: 1D array of observed values, one per obs_time.
            verbose: Print progress table.
            **kwargs: Forwarded to subclass hooks.

        Returns:
            Dict mapping observation time to result dict.
        """
        self.results = {}
        self._reset_posterior()

        if verbose:
            self._print_header()

        for t in self.obs_times.tolist():
            mask = self.obs_times <= t
            obs_times_up_to = self.obs_times[mask]
            obs_values_up_to = obs_values[mask]

            # Bayesian update
            if len(obs_values_up_to) > 0:
                self._do_bayesian_update(
                    obs_times_up_to, obs_values_up_to, t, **kwargs
                )

            # Pf forecast
            forecast_prior = {}
            forecast_posterior = {}
            for ft in self._get_forecast_times(t):
                forecast_prior[ft] = self.compute_pf_at_time(
                    ft, use_prior=True, **kwargs
                )
                forecast_posterior[ft] = self.compute_pf_at_time(
                    ft, use_prior=False, **kwargs
                )

            # Assemble result
            self.results[t] = self._build_step_result(
                t=t,
                obs_times=obs_times_up_to,
                obs_values=obs_values_up_to,
                forecast_prior=forecast_prior,
                forecast_posterior=forecast_posterior,
                **kwargs,
            )

            if verbose:
                self._print_step(t, forecast_prior, forecast_posterior)

        return self.results

    # ------------------------------------------------------------------
    # Overridable helpers
    # ------------------------------------------------------------------

    def _get_forecast_times(self, t: float) -> List[float]:
        """Return forecast times to evaluate at step t.

        Default: all forecast_times. Override to filter (e.g. only ft >= t).
        """
        return self.forecast_times.tolist()

    def _print_header(self) -> None:
        header = f"{'t':>8s}{'b_prior':>10s}{'b_post':>10s}"
        print(header)
        print("-" * len(header))

    def _print_step(
        self,
        t: float,
        forecast_prior: Dict[float, Dict[str, Any]],
        forecast_posterior: Dict[float, Dict[str, Any]],
    ) -> None:
        t_min = min(forecast_prior)
        bp = forecast_prior[t_min]["beta"]
        bq = forecast_posterior[t_min]["beta"]
        print(f"{t:>8.0f}{bp:>10.2f}{bq:>10.2f}")
