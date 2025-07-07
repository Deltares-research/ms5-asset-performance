import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from pathlib import Path
import json
from copy import deepcopy
from main.case_study_2025.reliability.moment_calculation.chebysev_moments import FoSCalculator as ChebysevFoS
from main.case_study_2025.reliability.chebysev_reliability import moment_mcs
from dataclasses import dataclass, field, asdict
from typing import Type, Optional
import matplotlib.pyplot as plt


if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


@dataclass
class TimelineParameters:
    setting: dict
    n_mcs: int = 100_000
    start_thickness: float = 9.5
    EI_start: float = 30_000.
    moment_cap_start: float = 30.
    moment_survived: float = 0.
    water_lvl: float = -1.
    water_lvl: float = -1.
    C50_mu: float = 1.5
    corrosion_rate: float = 0.022
    obs_error_std: float = .1
    times: list = field(init=False)

    def __post_init__(self):
        self.times = [float(time) for time in self.setting.keys()]


@dataclass
class TimelineRunner:
    time: float = 0.
    timestep: int = -1
    start_thickness: float = 9.5
    EI_start: float = 30_000.
    moment_cap_start: float = 30.
    moment_survived: float = 0.
    water_lvl: float = -1.
    C50_grid: list = None
    C50_prior: list = None
    C50_posterior: Optional[list] = None
    moment_cap: float = field(init=False)
    corrosion_obs_times: list = field(init=False)
    corrosion_obs: list = field(init=False)

    def time_step(self, time):
        self.timestep += 1
        self.time = time

    def update_moment_cap(self, data):
        moment_survived = max(self.moment_survived, data["max_moment"])
        moment_cap = max(moment_survived, self.moment_cap_start)
        self.moment_survived = moment_survived
        self.moment_cap = moment_cap

    def read_corrosion_data(self, corrosion_obs_times, corrosion_obs):
        self.corrosion_obs_times = corrosion_obs_times.tolist()
        self.corrosion_obs = corrosion_obs.tolist()

    def update_C50(self, C50_mu, corrosion_rate, obs_error_std):

        log_prior = np.log(self.C50_prior)

        C50_grid = np.array(self.C50_grid)[:, np.newaxis]

        corrosion_obs_times = np.array(self.corrosion_obs_times)[np.newaxis, :]
        corrosion_obs = np.array(self.corrosion_obs)

        C_mu = C50_grid * (1 + corrosion_rate / C50_mu * (corrosion_obs_times - 50))
        C_deviations = (corrosion_obs - C_mu) / obs_error_std
        C_deviations = np.concatenate((C_deviations[:, 0][:, np.newaxis], np.diff(C_deviations, axis=1)), axis=1)

        loglikes = stats.norm(loc=0, scale=1).logpdf(C_deviations)

        loglike = loglikes.sum(axis=1)

        log_post = log_prior + loglike
        post = np.exp(log_post)
        post /= np.trapezoid(post, C50_grid.squeeze())

        self.C50_posterior = post.tolist()

    def step(self, time, params):

        self.time_step(time)

        self.update_moment_cap(params.setting[time])

        corrosion_obs_times, corrosion_obs = collect_corrosion_data(time, params.setting)
        self.read_corrosion_data(corrosion_obs_times, corrosion_obs)

        self.update_C50(
            C50_mu=params.C50_mu,
            corrosion_rate=params.corrosion_rate,
            obs_error_std=params.obs_error_std
        )

    def finish_step(self):
        self.C50_prior = deepcopy(self.C50_posterior)

    def log(self, path):
        if not isinstance(path, Path): path = Path(path)
        runner_dict = asdict(self)
        with open(path/f"runnerlog_time_{self.time:.0f}.json", "w") as f:
            json.dump(runner_dict, f, indent=4)


def collect_corrosion_data(time, data):
    corrosion_obs_times = np.array([float(key) for key in data.keys() if float(key) <= time])
    corrosion_obs = np.array([val["corrosion"] for (key, val) in data.items() if float(key) <= time])
    return corrosion_obs_times, corrosion_obs


class PfCalculator:
    def __init__(self, C50_grid, params, corrosion_model, fos_calculator, mcs_samples_path):
        self.C50_grid = np.asarray(C50_grid)
        self.params = params
        self.corrosion_model = corrosion_model
        self.fos_calculator = fos_calculator
        self.n_mcs = self.params.n_mcs
        self.load_data(mcs_samples_path)
        self.corrosion_ratio_sample = self.sample_corrosion_ratios(self.C50_grid, self.params.times)

    def sample_corrosion_ratios(self, C50, times):

        C50 = np.array(C50)
        times = np.array(times)

        mu, scale, lower_trunc, upper_trunc = self.corrosion_model.corrosion_model_params(times, C50)
        mu = np.expand_dims(mu, axis=-1)
        scale = np.expand_dims(scale, axis=-1)
        lower_trunc = np.expand_dims(lower_trunc, axis=-1)
        upper_trunc = np.expand_dims(upper_trunc, axis=-1)
        corrosion_dist = stats.truncnorm(loc=mu, scale=scale, a=lower_trunc, b=upper_trunc)

        np.random.seed(42)
        corrosion_sample = corrosion_dist.rvs(size=(C50.size, times.size, self.n_mcs))
        corrosion_ratio_sample = corrosion_sample / self.params.start_thickness

        return  corrosion_ratio_sample

    def load_data(self, path):
        mcs_samples = np.load(path)
        mcs_samples = mcs_samples[:self.n_mcs]
        mcs_samples = mcs_samples[:, :-1]  # Remove EI column
        self.mcs_samples_torch = torch.from_numpy(mcs_samples.astype(np.float32)).to(device=device)

    def moment_mcs(self, water_lvl, EI):

        samples = torch.column_stack((
            self.mcs_samples_torch,
            torch.from_numpy(EI.astype(np.float32)).to(device=device),
            torch.from_numpy(water_lvl.astype(np.float32)).to(device=device)
        ))

        dataset = TensorDataset(samples)
        loader = DataLoader(dataset, batch_size=1_000, shuffle=True)

        max_moments = []
        for (x,) in loader:
            moments = self.fos_calculator.moments(x)
            max_moments.append(np.abs(moments).max(axis=-1))

        max_moments = np.concatenate(max_moments)

        return max_moments

    def calculate_max_moments_C50(self, C50, runner):

        C50_idx = np.where(self.C50_grid==C50)[0].item()

        corrosion_ratio_sample = self.corrosion_ratio_sample[C50_idx, runner.timestep]

        EI_sample = self.params.EI_start * (1 - corrosion_ratio_sample)

        water_lvl_sample = np.array([self.params.water_lvl] * self.params.n_mcs)

        max_moment_sample = self.moment_mcs(water_lvl_sample, EI_sample)

        return max_moment_sample

    def calculate_max_moments(self, runner):

        max_moments = np.zeros((np.array(self.C50_grid).size, self.n_mcs))

        for i, C50 in enumerate(self.C50_grid):

            mm = self.calculate_max_moments_C50(C50, runner)

            max_moments[i] = mm

        return max_moments


def load_chebysev_calculator(path, x_path):

    with open(x_path, "r") as f: x = np.array(json.load(f))

    n_points = len(x)
    wall_props = (1e+4, 0, x, None)

    fos_calculator = ChebysevFoS(
        n_points=n_points,
        wall_props=wall_props,
        x=x,
        model_path=path / "torch_weights.pth",
        scaler_x_path=path / "scaler_x.joblib",
        scaler_y_path=path / "scaler_y.joblib",
        device=device
    )

    return fos_calculator


def plot_errorbar(x, xerr, y, color="b", whiskersize=0.1):
    plt.scatter(x, y, c=color)
    plt.hlines(y, xmin=min(xerr), xmax=max(xerr), colors=color)
    plt.vlines(min(xerr), ymin=y-whiskersize/2, ymax=y+whiskersize/2, colors=color)
    plt.vlines(max(xerr), ymin=y-whiskersize/2, ymax=y+whiskersize/2, colors=color)


def plot_fos_hist(fos, path=None, modelfit="lognormal", ci_alpha=0.05):

    fig = plt.figure()

    pf_mcs = np.mean(fos<1)
    q_mcs = np.quantile(fos, [ci_alpha/2, 1-ci_alpha/2])
    bins = 80 if fos.size >= 1_000 else 50
    plt.hist(fos, bins=80, density=True, color="b", alpha=0.6, edgecolor="k", linewidth=.5, label="MCS ${P}_{f}$ = "+f"{pf_mcs*100:.1e}")
    plot_errorbar(x=fos.mean(), xerr=q_mcs, y=0.7, color="b", whiskersize=0.1)

    if modelfit == "lognormal":
        x_grid = np.linspace(fos.min(), fos.max(), 1_000)
        shape, loc, scale = stats.lognorm.fit(fos)
        pdf_fitted = stats.lognorm.pdf(x_grid, shape, loc, scale)
        expectation_fit = np.trapezoid(pdf_fitted*x_grid, x_grid)
        q_fit = stats.lognorm.ppf([0.025, 0.975], shape, loc, scale)
        pf_fit = stats.lognorm.cdf(1, shape, loc, scale)
        plt.plot(x_grid, pdf_fitted, c="r", label="Fit ${P}_{f}$ = "+f"{pf_fit*100:.1e}")
        plot_errorbar(x=expectation_fit, xerr=q_fit, y=0.5, color="r", whiskersize=0.1)

    plt.axvline(1, c="k", label="Safety margin")
    plt.xlabel("FoS [-]", fontsize=12)
    plt.ylabel("Density [-]", fontsize=12)
    plt.legend(fontsize=12)
    plt.close()

    if path is not None: fig.savefig(path)


if __name__ == "__main__":

    pass

