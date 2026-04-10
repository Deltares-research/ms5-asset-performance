"""
Corrosion model for sheet pile wall degradation.

Models corrosion progression over time using:
- C50: corrosion thickness at reference age
- Linear growth rate
- Truncated normal distribution (bounded by wall thickness)
"""

import numpy as np
from scipy import stats
from numpy.typing import NDArray
from typing import Tuple, Optional


class CorrosionModel:

    def __init__(
        self,
        C50_mu: float = 1.0,
        C50_std: float = 0.75,
        corrosion_rate: float = 0.022,
        wall_thickness: float = 9.5,
        obs_error_std: float = 0.4,
        t_start: float = 50.0,
        n_grid: int = 100,
        n_corrosion_grid: int = 1000,
    ):
        self.C50_mu = C50_mu
        self.C50_std = C50_std
        self.corrosion_rate = corrosion_rate
        self.wall_thickness = wall_thickness
        self.obs_error_std = obs_error_std
        self.t_start = t_start
        self.n_grid = n_grid
        self.n_corrosion_grid = n_corrosion_grid

        self.C50_grid = np.linspace(0.5, 2.5, n_grid)
        a = (self.C50_grid.min() - C50_mu) / C50_std
        b = (self.C50_grid.max() - C50_mu) / C50_std
        self.C50_prior = stats.truncnorm.pdf(self.C50_grid, a, b, loc=C50_mu, scale=C50_std)
        self.C50_prior /= np.trapezoid(self.C50_prior, self.C50_grid)

    def mean_corrosion(self, t, C50):
        return np.atleast_1d(C50) * (1 + self.corrosion_rate / self.C50_mu * (np.atleast_1d(t) - self.t_start))

    def corrosion_params(self, t, C50):
        mu = self.mean_corrosion(t, C50)
        scale = np.maximum(mu * 0.5, 1e-10)
        a = (0 - mu) / scale
        b = (self.wall_thickness - mu) / scale
        return mu, scale, a, b

    def corrosion_ratio_pdf(
        self,
        t: float,
        C50_pdf: NDArray = None,
        last_obs_time: Optional[float] = None,
        last_obs: Optional[float] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Compute PDF of corrosion ratio at time t, integrating over C50."""
        if C50_pdf is None:
            C50_pdf = self.C50_prior

        corrosion_grid = np.linspace(0, self.wall_thickness, self.n_corrosion_grid)
        ratio_grid = corrosion_grid / self.wall_thickness

        C50_grid = self.C50_grid[:, np.newaxis, np.newaxis]

        if last_obs is not None:
            d_time = t - last_obs_time
            d_mu = C50_grid * self.corrosion_rate / self.C50_mu * d_time
            mu = last_obs + d_mu
            scale = self.obs_error_std + d_mu * 0.5
            a = (0 - mu) / scale
            b = (self.wall_thickness - mu) / scale
        else:
            mu, scale, a, b = self.corrosion_params(t, C50_grid)

        corrosion_pdf = stats.truncnorm.pdf(corrosion_grid, a, b, loc=mu, scale=scale)
        corrosion_pdf *= C50_pdf[:, np.newaxis, np.newaxis]
        pdf = np.trapezoid(corrosion_pdf, self.C50_grid, axis=0)
        pdf /= np.trapezoid(pdf, corrosion_grid, axis=-1)

        ratio_pdf = pdf * self.wall_thickness

        return ratio_grid, ratio_pdf.squeeze()
