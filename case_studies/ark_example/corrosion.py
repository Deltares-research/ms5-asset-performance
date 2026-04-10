"""
Corrosion model for sheet pile wall degradation.

Models corrosion progression over time using:
- C50: corrosion thickness at reference age (50 years)
- Linear growth rate
- Truncated normal distribution (bounded by wall thickness)
- Bayesian updating based on observations
"""

import numpy as np
from scipy import stats
from numpy.typing import NDArray
from typing import Tuple, Optional


class CorrosionModel:
    """
    Corrosion progression model for sheet pile walls.

    The model assumes corrosion thickness follows:
        C(t) = C50 * (1 + rate/C50_mu * (t - t_start))

    where C50 is the corrosion at reference time t_start (default 50 years).

    Args:
        C50_mu: Mean of C50 prior distribution [mm].
        C50_std: Std of C50 prior distribution [mm].
        corrosion_rate: Annual corrosion rate coefficient.
        wall_thickness: Initial wall thickness [mm].
        obs_error_std: Observation error standard deviation [mm].
        t_start: Reference time for C50 [years].
        n_grid: Grid size for C50 discretization.
        n_corrosion_grid: Grid size for corrosion discretization.
    """

    def __init__(
        self,
        C50_mu: float = 1.5,
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

        # Initialize C50 grid and prior
        self._init_prior()

    def _init_prior(self) -> None:
        """Initialize C50 grid and prior distribution."""
        self.C50_grid = np.linspace(0.5, 2.5, self.n_grid)

        # Truncated normal prior for C50
        a_clip = (self.C50_grid.min() - self.C50_mu) / self.C50_std
        b_clip = (self.C50_grid.max() - self.C50_mu) / self.C50_std
        self.C50_prior = stats.truncnorm.pdf(self.C50_grid, a_clip, b_clip, loc=self.C50_mu, scale=self.C50_std)
        self.C50_prior /= np.trapezoid(self.C50_prior, self.C50_grid)

    def mean_corrosion(self, t: float | NDArray, C50: float | NDArray) -> NDArray:
        """
        Calculate mean corrosion thickness at time t.

        Args:
            t: Time [years].
            C50: Corrosion at reference time [mm].

        Returns:
            Mean corrosion thickness [mm].
        """
        t = np.atleast_1d(t)
        C50 = np.atleast_1d(C50)
        return C50 * (1 + self.corrosion_rate / 1.5 * (t - self.t_start))

    def corrosion_params(
        self, t: float | NDArray, C50: float | NDArray
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Get truncated normal distribution parameters for corrosion at time t.

        Args:
            t: Time [years].
            C50: Corrosion at reference time [mm].

        Returns:
            Tuple of (mu, scale, a, b) for truncnorm distribution.
        """
        mu = self.mean_corrosion(t, C50)
        scale = np.maximum(mu * 0.5, 1e-10)  # CoV = 0.5, avoid zero
        a = (0 - mu) / scale  # Lower truncation
        b = (self.wall_thickness - mu) / scale  # Upper truncation
        return mu, scale, a, b

    def sample_corrosion(
        self,
        t: float | NDArray,
        C50: float | NDArray,
        n_samples: int = 1,
        seed: int | None = None,
    ) -> NDArray:
        """
        Sample corrosion thickness at time t.

        Args:
            t: Time [years].
            C50: Corrosion at reference time [mm].
            n_samples: Number of samples.
            seed: Random seed.

        Returns:
            Corrosion samples [mm].
        """
        if seed is not None:
            np.random.seed(seed)

        mu, scale, a, b = self.corrosion_params(t, C50)
        return stats.truncnorm.rvs(a, b, loc=mu, scale=scale, size=n_samples)

    def generate_observations(
        self,
        times: NDArray,
        C50: float,
        seed: int | None = None,
    ) -> NDArray:
        """
        Generate synthetic corrosion observations over time.

        Adds observation error to true corrosion values.

        Args:
            times: Observation times [years].
            C50: True C50 value [mm].
            seed: Random seed.

        Returns:
            Observed corrosion values [mm].
        """
        if seed is not None:
            np.random.seed(seed)

        true_corrosion = self.mean_corrosion(times, C50)
        noise = self.obs_error_std * np.random.randn(len(times))
        observations = true_corrosion + noise
        return np.clip(observations, 0, self.wall_thickness)

    def corrosion_ratio(self, corrosion: float | NDArray) -> NDArray:
        """
        Convert corrosion thickness to ratio of start thickness.

        Args:
            corrosion: Corrosion thickness [mm].

        Returns:
            Corrosion ratio [-].
        """
        return np.asarray(corrosion) / self.wall_thickness

    def update_C50_posterior(
        self,
        obs_times: NDArray,
        obs_values: NDArray,
        prior: NDArray | None = None,
    ) -> NDArray:
        """
        Update C50 posterior given corrosion observations.

        Uses Bayesian updating with likelihood based on observation model.

        Args:
            obs_times: Times of observations [years].
            obs_values: Observed corrosion values [mm].
            prior: Prior PDF over C50_grid. Defaults to initial prior.

        Returns:
            Posterior PDF over C50_grid.
        """
        if prior is None:
            prior = self.C50_prior.copy()

        log_prior = np.log(prior + 1e-10)

        # Compute likelihood for each C50 value
        C50_grid = self.C50_grid[:, np.newaxis]  # (n_grid, 1)
        obs_times = np.asarray(obs_times)[np.newaxis, :]  # (1, n_obs)
        obs_values = np.asarray(obs_values)  # (n_obs,)

        # Expected corrosion at each time for each C50
        mu = C50_grid * (1 + self.corrosion_rate / self.C50_mu * (obs_times - self.t_start))

        # Compute normalized deviations
        deviations = (obs_values - mu) / self.obs_error_std

        # Use differences for sequential updating (avoids double-counting)
        deviations_diff = np.concatenate(
            [deviations[:, 0:1], np.diff(deviations, axis=1)], axis=1
        )

        # Log-likelihood
        log_likelihood = stats.norm.logpdf(deviations_diff).sum(axis=1)

        # Posterior
        log_posterior = log_prior + log_likelihood
        posterior = np.exp(log_posterior - log_posterior.max())  # Numerical stability
        posterior /= np.trapezoid(posterior, self.C50_grid)

        return posterior

    def corrosion_ratio_pdf(
        self,
        t: float,
        C50_pdf: NDArray | None = None,
        last_obs_time: Optional[float] = None,
        last_obs: Optional[float] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute PDF of corrosion ratio at time t.

        Integrates over C50 uncertainty.

        Args:
            t: Time [years].
            C50_pdf: PDF over C50_grid. Defaults to prior.

        Returns:
            Tuple of (corrosion_ratio_grid, pdf).
        """
        if C50_pdf is None:
            C50_pdf = self.C50_prior

        # Create corrosion grid
        corrosion_grid = np.linspace(0, self.wall_thickness, self.n_corrosion_grid)
        ratio_grid = corrosion_grid / self.wall_thickness

        # For each C50, compute PDF of corrosion
        C50_grid = self.C50_grid[:, np.newaxis, np.newaxis]

        if last_obs:
            d_time = t - last_obs_time
            d_mu = C50_grid * self.corrosion_rate / self.C50_mu * d_time
            mu = last_obs + d_mu
            scale = self.obs_error_std + d_mu * 0.5
            a = (0 - mu) / scale
            b = (self.wall_thickness - mu) / scale
        else:
            mu, scale, a, b = self.corrosion_params(t, C50_grid)

        # PDF at each corrosion value for each C50
        corrosion_pdf = stats.truncnorm.pdf(corrosion_grid, a, b, loc=mu, scale=scale)

        # Weight by C50 PDF and integrate
        corrosion_pdf *= C50_pdf[:, np.newaxis, np.newaxis]
        pdf = np.trapezoid(corrosion_pdf, self.C50_grid, axis=0)

        # Normalize
        pdf /= np.trapezoid(pdf, corrosion_grid, axis=-1)

        # Transform to ratio PDF
        ratio_pdf = pdf * 1 / (1 / self.wall_thickness)  # Scaling of PDF from corrosion to corrosion rate

        return ratio_grid, ratio_pdf.squeeze()
