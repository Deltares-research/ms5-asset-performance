import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Optional, Type, NamedTuple
from dataclasses import dataclass
from src.bayesian_updating.ERADist import ERADist


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

    def __init__(self, corrosion_rate: float = 0.022, start_thickness: float = 9.5) -> None:
        self.corrosion_rate = corrosion_rate
        self.start_thickness = start_thickness

    def corrosion_model_params(self, C50: float=1.5):
        C50 = C50[..., np.newaxis]
        mu = C50 * (1 + self.corrosion_rate / 1.5 * (times - 50))
        scale = mu * 0.5
        lower_trunc = (0 - mu) / scale
        upper_trunc = (self.start_thickness - mu) / scale
        return mu, scale, lower_trunc, upper_trunc

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

    # def bayesian_updating(self, obs: OBS, obs_times: OBS_TIME, forecast: bool = False) -> POST_PDF:
    #     """
    #     Corrosion observations over a timeline are fully correlated -> there is only one truly random variable for
    #     for generating observations: C50 -> There is one truly random observation
    #
    #     Observation uncertainty is added in observation generation to allow for meaningful Bayesian updating.
    #
    #     :param obs:
    #     :param obs_times:
    #     :return:
    #     """
    #
    #     log_prior = np.log(self.config.C50_prior_pdf)
    #
    #     C50 = self.config.C50_grid
    #     C50 = C50[:, np.newaxis, np.newaxis, np.newaxis]
    #
    #     obs_times = obs_times[np.newaxis, :, np.newaxis, np.newaxis]
    #
    #     C_mu = C50 * (1 + self.config.corrosion_rate / self.config.C50_mu * (obs_times - 50))
    #     C_deviations = (obs - C_mu) / self.config.obs_error_std
    #     C_deviations = C_deviations.squeeze()
    #     C_deviations = np.concatenate((C_deviations[:, 0, ...][:, np.newaxis, ...], np.diff(C_deviations, axis=1)),
    #                                   axis=1)
    #
    #     loglikes = norm(loc=0, scale=1).logpdf(C_deviations)
    #
    #     if not forecast:
    #
    #         loglike = loglikes.sum(axis=1)
    #
    #         log_post = log_prior[..., np.newaxis, np.newaxis] + loglike
    #         post = np.exp(log_post)
    #         post /= trapz(post, self.config.C50_grid, axis=0)[np.newaxis, ...]
    #
    #     else:
    #
    #         loglike = loglikes.cumsum(axis=1)
    #
    #         log_post = log_prior[..., np.newaxis, np.newaxis, np.newaxis] + loglike
    #         post = np.exp(log_post)
    #         post /= trapz(post, self.config.C50_grid, axis=0)[np.newaxis, ...]
    #
    #     """To xarray"""
    #     if self.to_xarray:
    #         if not forecast:
    #             coords = dict(
    #                 C50_grid=self.config.C50_grid,
    #                 mc=np.arange(1, obs.shape[1] + 1),
    #                 C50=np.arange(1, obs.shape[2] + 1)
    #             )
    #             post = xr.DataArray(data=post, dims=['C50_grid', 'mc', 'C50'], coords=coords)
    #         else:
    #             coords = dict(
    #                 C50_grid=self.config.C50_grid,
    #                 obs_time=obs_times,
    #                 mc=np.arange(1, obs.shape[1] + 1),
    #                 C50=np.arange(1, obs.shape[2] + 1)
    #             )
    #             post = xr.DataArray(data=post, dims=['C50_grid', 'obs_times', 'mc', 'C50'], coords=coords)
    #
    #     return post


class CorrosionModelSimple:

    """
    According to Jongbloed curves detailed in Report:
    Richtlijn bewezen sterkte damwanden en kademuren.
    Corrosion model with simple parameters:
    a ~ N(mean_a, sigma_a)
    b ~ N(mean_b, sigma_b)
    """    
    def get_corrosion_rate_at_t(self, samples_a: np.ndarray, cur_t: int, b: float = 0.57):
        return (samples_a * cur_t** b) / 10
    

if __name__ == "__main__":

    pass

