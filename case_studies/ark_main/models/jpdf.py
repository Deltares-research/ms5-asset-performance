"""
JPDF for D-Sheet piling case study.

Extends src.JPDF with the random corrosion-model parameter as a 1D numerical
PDF on a grid, supporting Bayesian updating from corrosion observations.
The active random parameter depends on the corrosion model_type:

- "power" (default, 0-50 yr): exponent B of C(t) = A * t^B
- "linear" (50-75 yr):        anchor C50 of C(t) = C50 * (1 + r/C50_mu * (t - t_start))

The pipeline addresses the random parameter generically through
``param_grid`` / ``param_prior`` / ``param_pdf``; the legacy ``C50_*`` and the
new ``B_*`` attributes alias onto these depending on model_type.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import scipy.stats as st
from numpy.typing import NDArray

from src import JPDF as BaseJPDF


class JPDF(BaseJPDF):

    def __init__(self, name: str = "", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.config = config or {}
        self.model_type: str = self.config.get("model_type", "power")
        if self.model_type not in ("power", "linear"):
            raise ValueError(f"model_type must be 'power' or 'linear', got {self.model_type!r}")

        # Generic param-grid view (set by init_*_prior, used by the pipeline).
        self.param_grid: Optional[NDArray] = None
        self.param_prior: Optional[NDArray] = None
        self.param_pdf: Optional[NDArray] = None

        # Mode-specific aliases. Only one set is non-None at a time.
        self.C50_grid: Optional[NDArray] = None
        self.C50_prior: Optional[NDArray] = None
        self.C50_pdf: Optional[NDArray] = None

        self.B_grid: Optional[NDArray] = None
        self.B_prior: Optional[NDArray] = None
        self.B_pdf: Optional[NDArray] = None

    # ------------------------------------------------------------------
    # Settings entry point
    # ------------------------------------------------------------------

    def set_prior_from_settings(self, filepath: Path | str) -> None:
        # Soil/load variables in settings.json drive FORM/MC, not this JPDF —
        # we only carry the corrosion-model random parameter (B or C50). Skip
        # the parent's set_variables (which doesn't know "deterministic" /
        # "gumbel" / "gumbel_min").
        with open(filepath, "r") as f:
            settings = json.load(f)
        params = settings.get("parameters", {})
        self.model_type = params.get("model_type", self.model_type)

        if self.model_type == "linear":
            self.init_C50_prior(
                C50_mu=params.get("C50_mu", 1.5),
                C50_std=params.get("C50_std", 0.75),
                C50_min=params.get("C50_min", 0.5),
                C50_max=params.get("C50_max", 2.5),
            )
        else:
            self.init_B_prior(
                B_mu=params.get("B_mu", 0.72),
                B_std=params.get("B_std", 0.05),
                B_min=params.get("B_min", 0.4),
                B_max=params.get("B_max", 1.0),
            )

    # ------------------------------------------------------------------
    # Prior initialisation (per mode)
    # ------------------------------------------------------------------

    def init_C50_prior(
        self,
        C50_mu: float,
        C50_std: float,
        C50_min: float = 0.5,
        C50_max: float = 2.5,
    ) -> None:
        n_grid = self.config.get("n_C50_grid", 100)
        self.C50_grid = np.linspace(C50_min, C50_max, n_grid)
        a = (C50_min - C50_mu) / C50_std
        b = (C50_max - C50_mu) / C50_std
        self.C50_prior = st.truncnorm.pdf(self.C50_grid, a, b, loc=C50_mu, scale=C50_std)
        self.C50_prior /= np.trapezoid(self.C50_prior, self.C50_grid)
        self.C50_pdf = self.C50_prior.copy()

        self.param_grid = self.C50_grid
        self.param_prior = self.C50_prior
        self.param_pdf = self.C50_pdf

    def init_B_prior(
        self,
        B_mu: float,
        B_std: float,
        B_min: float = 0.4,
        B_max: float = 1.0,
    ) -> None:
        n_grid = self.config.get("n_B_grid", self.config.get("n_C50_grid", 100))
        self.B_grid = np.linspace(B_min, B_max, n_grid)
        a = (B_min - B_mu) / B_std
        b = (B_max - B_mu) / B_std
        self.B_prior = st.truncnorm.pdf(self.B_grid, a, b, loc=B_mu, scale=B_std)
        self.B_prior /= np.trapezoid(self.B_prior, self.B_grid)
        self.B_pdf = self.B_prior.copy()

        self.param_grid = self.B_grid
        self.param_prior = self.B_prior
        self.param_pdf = self.B_pdf

    # ------------------------------------------------------------------
    # Reset / update (generic + per-mode)
    # ------------------------------------------------------------------

    def reset_to_prior(self) -> None:
        if self.model_type == "linear":
            self.reset_C50_to_prior()
        else:
            self.reset_B_to_prior()

    def reset_C50_to_prior(self) -> None:
        if self.C50_prior is not None:
            self.C50_pdf = self.C50_prior.copy()
            self.param_pdf = self.C50_pdf

    def reset_B_to_prior(self) -> None:
        if self.B_prior is not None:
            self.B_pdf = self.B_prior.copy()
            self.param_pdf = self.B_pdf

    def update(self, obs_times: NDArray, obs_values: NDArray) -> None:
        if self.model_type == "linear":
            self.update_C50(obs_times, obs_values)
        else:
            self.update_B(obs_times, obs_values)

    def update_C50(self, obs_times: NDArray, obs_values: NDArray) -> None:
        """Bayesian update of C50 given linear-model corrosion observations."""
        C50_mu = self.config.get("C50_mu", 1.5)
        corrosion_rate = self.config.get("corrosion_rate", 0.022)
        obs_error_std = self.config.get("obs_error_std", 0.4)
        t_ref = self.config.get("t_start", 50.0)

        log_prior = np.log(self.C50_prior + 1e-10)

        C50_grid = self.C50_grid[:, np.newaxis]
        obs_times = np.asarray(obs_times)[np.newaxis, :]
        obs_values = np.asarray(obs_values)

        corr_mu = C50_grid * (1 + corrosion_rate / C50_mu * (obs_times - t_ref))
        deviations = (obs_values - corr_mu) / obs_error_std
        deviations = np.concatenate(
            (deviations[:, 0][:, np.newaxis], np.diff(deviations, axis=1)), axis=1
        )

        loglike = st.norm.logpdf(deviations).sum(axis=1)
        log_post = log_prior + loglike
        post = np.exp(log_post - log_post.max())
        post /= np.trapezoid(post, C50_grid.squeeze())
        self.C50_pdf = post.copy()
        self.param_pdf = self.C50_pdf

    def update_B(self, obs_times: NDArray, obs_values: NDArray) -> None:
        """Bayesian update of B given power-law corrosion observations.

        Likelihood:
            obs_i = A * t_i^B + epsilon_i,    epsilon_i iid N(0, sigma_obs^2)

        Standardised residuals are differenced before summing the log-pdf
        so that perfectly correlated noise across the timeline (which is not
        the model here) wouldn't be double-counted; this mirrors the C50
        update path for parity.
        """
        power_A = self.config.get("power_A", 0.091)
        obs_error_std = self.config.get("obs_error_std", 0.4)

        log_prior = np.log(self.B_prior + 1e-10)

        B_grid = self.B_grid[:, np.newaxis]
        obs_times = np.asarray(obs_times)[np.newaxis, :]
        obs_values = np.asarray(obs_values)

        corr_mu = power_A * np.power(np.maximum(obs_times, 0.0), B_grid)
        deviations = (obs_values - corr_mu) / obs_error_std
        deviations = np.concatenate(
            (deviations[:, 0][:, np.newaxis], np.diff(deviations, axis=1)), axis=1
        )

        loglike = st.norm.logpdf(deviations).sum(axis=1)
        log_post = log_prior + loglike
        post = np.exp(log_post - log_post.max())
        post /= np.trapezoid(post, B_grid.squeeze())
        self.B_pdf = post.copy()
        self.param_pdf = self.B_pdf

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def get_param_stats(self) -> Dict[str, float]:
        if self.param_pdf is None or self.param_grid is None:
            return {}
        grid = self.param_grid
        pdf = self.param_pdf
        mean = np.trapezoid(grid * pdf, grid)
        var = np.trapezoid((grid - mean) ** 2 * pdf, grid)
        cdf = np.cumsum(pdf) * np.diff(grid, prepend=grid[0])
        cdf /= cdf[-1]
        return {
            "mean": mean, "std": np.sqrt(var),
            "q05": np.interp(0.05, cdf, grid),
            "median": np.interp(0.50, cdf, grid),
            "q95": np.interp(0.95, cdf, grid),
        }

    def get_C50_stats(self) -> Dict[str, float]:
        if self.model_type != "linear":
            return {}
        return self.get_param_stats()

    def get_B_stats(self) -> Dict[str, float]:
        if self.model_type != "power":
            return {}
        return self.get_param_stats()
