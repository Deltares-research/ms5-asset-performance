"""
Corrosion model for sheet pile wall degradation.

Two model types are supported:

- "power" (default, 0-50 yr): bi-logarithmic / power law
      C(t) = A * t^B
  with A fixed (calibrated, typically against NEN 6766:2023 tabulated values)
  and B the random parameter (truncated normal prior).

- "linear" (50-75 yr): legacy model anchored at t_start
      C(t) = C50 * (1 + corrosion_rate / C50_mu * (t - t_start))
  with C50 the random parameter (truncated normal prior).

In both cases the random parameter (B or C50) plays the same role inside the
pipeline: it is the 1D quantity carried on a grid in the JPDF and updated
with corrosion observations via Bayesian inference. All downstream code
(corrosion ratio PDFs, observation generation, posterior updating) uses a
common ``param_*`` interface.
"""

import numpy as np
from scipy import stats
from numpy.typing import NDArray
from typing import Tuple, Optional


class CorrosionModel:

    def __init__(
        self,
        # selector
        model_type: str = "power",
        # linear-mode params
        C50_mu: float = 1.5,
        C50_std: float = 0.75,
        corrosion_rate: float = 0.022,
        t_start: float = 50.0,
        C50_min: float = 0.5,
        C50_max: float = 2.5,
        # power-mode params
        power_A: float = 0.091,
        B_mu: float = 0.72,
        B_std: float = 0.05,
        B_min: float = 0.4,
        B_max: float = 1.0,
        # common
        wall_thickness: float = 9.5,
        obs_error_std: float = 0.4,
        n_grid: int = 100,
        n_corrosion_grid: int = 1000,
    ):
        if model_type not in ("power", "linear"):
            raise ValueError(f"model_type must be 'power' or 'linear', got {model_type!r}")
        self.model_type = model_type

        # linear params
        self.C50_mu = C50_mu
        self.C50_std = C50_std
        self.corrosion_rate = corrosion_rate
        self.t_start = t_start

        # power params
        self.power_A = power_A
        self.B_mu = B_mu
        self.B_std = B_std

        # common
        self.wall_thickness = wall_thickness
        self.obs_error_std = obs_error_std
        self.n_grid = n_grid
        self.n_corrosion_grid = n_corrosion_grid

        if model_type == "linear":
            self.C50_grid = np.linspace(C50_min, C50_max, n_grid)
            a = (C50_min - C50_mu) / C50_std
            b = (C50_max - C50_mu) / C50_std
            self.C50_prior = stats.truncnorm.pdf(self.C50_grid, a, b, loc=C50_mu, scale=C50_std)
            self.C50_prior /= np.trapezoid(self.C50_prior, self.C50_grid)
            self.param_grid = self.C50_grid
            self.param_prior = self.C50_prior
            self.B_grid = None
            self.B_prior = None
        else:
            self.B_grid = np.linspace(B_min, B_max, n_grid)
            a = (B_min - B_mu) / B_std
            b = (B_max - B_mu) / B_std
            self.B_prior = stats.truncnorm.pdf(self.B_grid, a, b, loc=B_mu, scale=B_std)
            self.B_prior /= np.trapezoid(self.B_prior, self.B_grid)
            self.param_grid = self.B_grid
            self.param_prior = self.B_prior
            self.C50_grid = None
            self.C50_prior = None

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def mean_corrosion(self, t, param):
        """Mean corrosion thickness at time t given the random parameter.

        param is C50 in linear mode, B in power mode. Broadcasts naturally
        across param-grid and time-grid axes.
        """
        t = np.atleast_1d(t).astype(float)
        param = np.atleast_1d(param).astype(float)
        if self.model_type == "linear":
            return param * (1 + self.corrosion_rate / self.C50_mu * (t - self.t_start))
        # power: C(t) = A * t^B
        # use log/exp to keep broadcast clean for both scalar and vector args
        return self.power_A * np.power(np.maximum(t, 0.0), param)

    def corrosion_params(self, t, param):
        """Truncated-normal parameters for C(t) | param.

        Returns mu, scale, and (a, b) standardized truncation points so the
        result can be fed straight into scipy.stats.truncnorm.
        """
        mu = self.mean_corrosion(t, param)
        scale = np.maximum(mu * 0.5, 1e-10)
        a = (0 - mu) / scale
        b = (self.wall_thickness - mu) / scale
        return mu, scale, a, b

    def sample_corrosion(self, t, param, n_samples=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        mu, scale, a, b = self.corrosion_params(t, param)
        return stats.truncnorm.rvs(a, b, loc=mu, scale=scale, size=n_samples)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def generate_observations(self, times, param_true, seed=None):
        """Generate synthetic corrosion observations for a given true param.

        ``param_true`` is the true value of C50 (linear) or B (power).
        Adds Gaussian observation noise with std = ``obs_error_std`` and
        clips to [0, wall_thickness].
        """
        if seed is not None:
            np.random.seed(seed)
        times = np.atleast_1d(times).astype(float)
        true_corrosion = self.mean_corrosion(times, param_true).reshape(-1)
        noise = self.obs_error_std * np.random.randn(times.size)
        return np.clip(true_corrosion + noise, 0, self.wall_thickness)

    # ------------------------------------------------------------------
    # Bayesian updating of the random parameter
    # ------------------------------------------------------------------

    def update_param_posterior(self, obs_times, obs_values, prior=None):
        """Bayesian update of the random parameter from corrosion observations.

        Likelihood is built on the *first observation + successive
        differences* of standardized residuals, which gives the right
        sequential weight when observations along a single timeline are
        fully correlated (same realisation of the random parameter, only
        i.i.d. observation noise).
        """
        if prior is None:
            prior = self.param_prior.copy()

        log_prior = np.log(prior + 1e-10)

        param_grid = self.param_grid[:, np.newaxis]                 # (n_grid, 1)
        obs_times = np.asarray(obs_times)[np.newaxis, :]            # (1, n_obs)
        obs_values = np.asarray(obs_values)                         # (n_obs,)

        if self.model_type == "linear":
            mu = param_grid * (1 + self.corrosion_rate / self.C50_mu * (obs_times - self.t_start))
        else:
            mu = self.power_A * np.power(np.maximum(obs_times, 0.0), param_grid)

        deviations = (obs_values - mu) / self.obs_error_std
        deviations_diff = np.concatenate(
            [deviations[:, 0:1], np.diff(deviations, axis=1)], axis=1
        )

        log_likelihood = stats.norm.logpdf(deviations_diff).sum(axis=1)

        log_posterior = log_prior + log_likelihood
        posterior = np.exp(log_posterior - log_posterior.max())
        posterior /= np.trapezoid(posterior, self.param_grid)
        return posterior

    # Back-compat alias for the linear-only API.
    def update_C50_posterior(self, obs_times, obs_values, prior=None):
        if self.model_type != "linear":
            raise RuntimeError("update_C50_posterior requires model_type='linear'")
        return self.update_param_posterior(obs_times, obs_values, prior=prior)

    # ------------------------------------------------------------------
    # Corrosion-ratio PDF (marginalised over the random parameter)
    # ------------------------------------------------------------------

    def corrosion_ratio_pdf(
        self,
        t: float,
        param_pdf: NDArray | None = None,
        last_obs_time: Optional[float] = None,
        last_obs: Optional[float] = None,
        # back-compat: callers that still pass C50_pdf
        C50_pdf: NDArray | None = None,
    ) -> Tuple[NDArray, NDArray]:
        """Compute PDF of the corrosion *ratio* at time t.

        ``param_pdf`` is the current PDF of the random parameter on
        ``self.param_grid`` (B in power mode, C50 in linear mode).
        Defaults to the prior. ``C50_pdf`` is accepted as an alias for
        backward compatibility.

        If ``last_obs`` is given, the forecast is anchored on the last
        observation: mean grows by the model's incremental corrosion from
        ``last_obs_time`` to ``t``, and the std grows from ``obs_error_std``.
        """
        if param_pdf is None:
            param_pdf = C50_pdf
        if param_pdf is None:
            param_pdf = self.param_prior

        corrosion_grid = np.linspace(0, self.wall_thickness, self.n_corrosion_grid)
        ratio_grid = corrosion_grid / self.wall_thickness

        param_grid = self.param_grid[:, np.newaxis, np.newaxis]

        if last_obs is not None:
            # incremental mean from t_obs to t, dispatched per model
            if self.model_type == "linear":
                d_time = t - last_obs_time
                d_mu = param_grid * self.corrosion_rate / self.C50_mu * d_time
            else:
                t_now = max(float(t), 0.0)
                t_obs = max(float(last_obs_time), 0.0)
                d_mu = self.power_A * (np.power(t_now, param_grid) - np.power(t_obs, param_grid))
            mu = last_obs + d_mu
            scale = self.obs_error_std + d_mu * 0.5
            a = (0 - mu) / scale
            b = (self.wall_thickness - mu) / scale
        else:
            mu, scale, a, b = self.corrosion_params(t, param_grid)

        corrosion_pdf = stats.truncnorm.pdf(corrosion_grid, a, b, loc=mu, scale=scale)
        corrosion_pdf *= param_pdf[:, np.newaxis, np.newaxis]
        pdf = np.trapezoid(corrosion_pdf, self.param_grid, axis=0)

        # normalise over corrosion_grid
        norm = np.trapezoid(pdf, corrosion_grid, axis=-1)
        pdf = pdf / np.where(norm > 0, norm, 1.0)

        ratio_pdf = pdf * self.wall_thickness  # change of variable c -> c / d0

        return ratio_grid, ratio_pdf.squeeze()
