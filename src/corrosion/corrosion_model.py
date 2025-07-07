import numpy as np
from numpy.typing import NDArray
from scipy.stats import truncnorm, norm
from typing import List, Tuple, Dict, Optional, Type, NamedTuple
from dataclasses import dataclass


class CorrosionModel:
    """
    According to EC:
    C_t = C50 * (1 + 0.022 * (t - 50))    {50 <= t <= 100}                          {1}
    C50 ~ TruncN(1.5[mm], (1.5*0.5)**2 [mm**2], a=0, b=start_thickness)            {2}
    {1} + {2} -->
    mu = 1.5 * (1 + 0.022 / 1.5 * (t - 50))
    std = (1.5 * (1 + 0.022 / 1.5 * (t - 50)) * 0.5 [mm**2]
    C_t ~ TruncN(mu, std ** 2, , a=0, b=start_thickness)  {3}

    "param" is the mean of corrosion distribution (as a function of time).
    """

    def __init__(
            self,
            n_grid: int = 100,
            corrosion_rate: float = 0.022,
            start_thickness: float = 9.5,
            C50_mu: float = 1.5,
            C50_std: float = 1.5 * 0.5,
            obs_error_std: float = .1
    ) -> None:
        self.corrosion_rate = corrosion_rate
        self.start_thickness = start_thickness
        self.C50_mu = C50_mu
        self.C50_std = C50_std
        self.obs_error_std = obs_error_std
        self.C50_grid = np.linspace(.5, 2.5, n_grid)
        self.C50_prior = truncnorm(
            loc=self.C50_mu,
            scale=self.C50_std,
            a=(self.C50_grid.min()-self.C50_mu)/self.C50_std,
            b=(self.C50_grid.max()-self.C50_mu)/self.C50_std
        ).pdf(self.C50_grid)

    def corrosion_model_params(self, times, C50: float=1.5):
        if isinstance(C50, float):
            C50 = np.array([C50])
        C50 = C50[..., np.newaxis]
        mu = C50 * (1 + self.corrosion_rate / 1.5 * (times - 50))
        scale = mu * 0.5
        lower_trunc = (0 - mu) / scale
        upper_trunc = (self.start_thickness - mu) / scale
        return mu, scale, lower_trunc, upper_trunc

    def corrosion_distribution(self, t: NDArray[np.float32], C50: float=1.5) -> NDArray[np.float32]:
        """
        Distribution of corrosion x at time t for rate C50 according to EC model.
        :t: Time
        :param: Mean of corrosion distribution (as a function of time)
        :return:
        """
        mu, scale, lower_trunc, upper_trunc = self.corrosion_model_params(t, C50)
        corrosion_dist = truncnorm(loc=mu, scale=scale, a=lower_trunc, b=upper_trunc)
        return corrosion_dist

    def prob(self, x: NDArray[np.float32], t: NDArray[np.float32], C50: float=1.5) -> NDArray[np.float32]:
        """
        PDF of corrosion x at time t for rate C50 according to EC model.
        :param param: Mean of corrosion distribution (as a function of time)
        :return:
        """
        mu, scale, lower_trunc, upper_trunc = self.corrosion_model_params(t)
        corrosion_pdf = truncnorm(loc=mu, scale=scale, a=lower_trunc, b=upper_trunc).pdf(x)
        return corrosion_pdf

    def generate_observations(self, obs_time: NDArray[np.float32] | float | int, C50: NDArray[np.float32] | float=1.5,
                              obs_error_std: float=0.1, n_timelines: int = 1, seed: int = 42) -> NDArray[np.float32]:
        """
        Corrosion observations over a timeline are fully correlated -> there is only one truly random variable for
        for generating observations: C50

        Observation randomness is added to enable meaningful Bayesian updating.

        C_t_obs ~ N(C_t, sigma_obs**2)
        :param rng:
        :param obs_times:
        :param C50:
        :param n_timelines:
        :return:
        """

        if isinstance(obs_time, float) or isinstance(obs_time, int): obs_time = np.asarray([obs_time])
        if isinstance(C50, float): C50 = np.asarray([C50])

        n_times = obs_time.size
        obs_time = obs_time[:, np.newaxis, np.newaxis]
        obs_mean = C50[np.newaxis, np.newaxis, :] * (1 + self.corrosion_rate / 1.5 * (obs_time - 50))
        np.random.seed(seed)
        obs_error = np.random.normal(loc=0, scale=1, size=(n_times, n_timelines, C50.size))
        obs_error = np.cumsum(obs_error, axis=0)
        obs = obs_mean + obs_error * obs_error_std

        return obs.squeeze()

    def bayesian_updating(self, obs, obs_times, C50_prior_pdf, C50_grid):
        """
        Corrosion observations over a timeline are fully correlated -> there is only one truly random variable for
        generating observations: C50 -> There is one truly random observation

        Observation uncertainty is added in observation generation to allow for meaningful Bayesian updating.

        :param obs:
        :param obs_times:
        :return:
        """

        log_prior = np.log(C50_prior_pdf)

        C50_grid = C50_grid[:, np.newaxis]

        obs_times = obs_times[np.newaxis, :]

        C_mu = C50_grid * (1 + self.corrosion_rate / self.C50_mu * (obs_times - 50))
        C_deviations = (obs - C_mu) / self.obs_error_std
        C_deviations = np.concatenate((C_deviations[:, 0][:, np.newaxis], np.diff(C_deviations, axis=1)), axis=1)

        loglikes = norm(loc=0, scale=1).logpdf(C_deviations)

        loglike = loglikes.sum(axis=1)

        log_post = log_prior + loglike
        post = np.exp(log_post)
        post /= np.trapezoid(post, C50_grid.squeeze())

        return post


if __name__ == "__main__":

    pass

