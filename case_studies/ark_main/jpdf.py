"""
JPDF for D-Sheet piling case study.

Extends src.JPDF with C50 as a 1D numerical PDF on a grid,
supporting Bayesian updating from corrosion observations.
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

        self.C50_grid: Optional[NDArray] = None
        self.C50_prior: Optional[NDArray] = None
        self.C50_pdf: Optional[NDArray] = None

    def set_prior_from_settings(self, filepath: Path | str) -> None:
        """Load variables and initialize C50 prior from settings."""
        self.set_variables(filepath)

        with open(filepath, "r") as f:
            settings = json.load(f)
        params = settings.get("parameters", {})
        self.init_C50_prior(params.get("C50_mu", 1.0), params.get("C50_std", 0.75))

    def init_C50_prior(self, C50_mu: float, C50_std: float) -> None:
        n_grid = self.config.get("n_C50_grid", 100)
        self.C50_grid = np.linspace(0.5, 2.5, n_grid)

        a = (self.C50_grid.min() - C50_mu) / C50_std
        b = (self.C50_grid.max() - C50_mu) / C50_std
        self.C50_prior = st.truncnorm.pdf(self.C50_grid, a, b, loc=C50_mu, scale=C50_std)
        self.C50_pdf = self.C50_prior.copy()

    def reset_C50_to_prior(self) -> None:
        if self.C50_prior is not None:
            self.C50_pdf = self.C50_prior.copy()

    def update_C50(self, obs_times: NDArray, obs_values: NDArray) -> None:
        """Bayesian update of C50 given corrosion observations."""
        C50_mu = self.config.get("C50_mu", 1.0)
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

    def get_C50_stats(self) -> Dict[str, float]:
        if self.C50_pdf is None or self.C50_grid is None:
            return {}
        mean = np.trapezoid(self.C50_grid * self.C50_pdf, self.C50_grid)
        var = np.trapezoid((self.C50_grid - mean) ** 2 * self.C50_pdf, self.C50_grid)
        cdf = np.cumsum(self.C50_pdf) * np.diff(self.C50_grid, prepend=self.C50_grid[0])
        cdf /= cdf[-1]
        return {
            "mean": mean, "std": np.sqrt(var),
            "q05": np.interp(0.05, cdf, self.C50_grid),
            "median": np.interp(0.50, cdf, self.C50_grid),
            "q95": np.interp(0.95, cdf, self.C50_grid),
        }
