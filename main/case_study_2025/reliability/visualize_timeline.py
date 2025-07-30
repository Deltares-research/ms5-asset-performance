import json
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from main.case_study_2025.reliability.utils import *
from src.corrosion.corrosion_model import CorrosionModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def read_log(log_path, time):
    path = log_path / f"time_{time}.json"
    with open(path, "r") as f:
        log = json.load(f)
    return log


def plot_pf(log):

    fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharex=True, sharey=True)

    ax = axs[0]

    data = log["theoretical"]["posterior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    pf_forecast_posterior = [float(val) for val in data["pf_forecast"].values()]
    ax.plot(times_forecast, pf_forecast_posterior, c="r", label="Factual forecast")

    data = log["theoretical"]["prior"]
    times_prior = [float(key) for key in data["pf_forecast"].keys()]
    pf_forecast_prior = [float(val) for val in data["pf_forecast"].values()]
    ax.plot(times_forecast, pf_forecast_posterior, c="b", label="Counterfactual forecast")

    ax.set_xlabel("Time [yr]", fontsize=12)
    ax.set_ylabel("${P}_{f}$ [-]", fontsize=12)
    ax.set_yscale('log')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.legend(fontsize=10)
    ax.set_title("Theoretical moment capacity", fontsize=12)

    ax = axs[1]

    data = log["survived"]["posterior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    pf_forecast_posterior = [float(val) for val in data["pf_forecast"].values()]
    ax.plot(times_forecast, pf_forecast_posterior, c="r", label="Factual forecast")

    data = log["theoretical"]["prior"]
    times_prior = [float(key) for key in data["pf_forecast"].keys()]
    pf_forecast_prior = [float(val) for val in data["pf_forecast"].values()]
    ax.plot(times_forecast, pf_forecast_posterior, c="b", label="Counterfactual forecast")

    ax.set_xlabel("Time [yr]", fontsize=12)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.legend(fontsize=10)
    ax.set_title("Empirical moment capacity", fontsize=12)

    return fig


def plot_corrosion_ratio(log, params, a=0.05):

    setting = params.setting

    all_times = list(setting.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    data = log["theoretical"]["posterior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    corrosion_ratio_grid = np.asarray(data["corrosion_ratio_grid"])
    corrosion_ratio_pdf = np.asarray(data["corrosion_ratio_pdf"])
    corrosion_ratio_cdf = cumulative_trapezoid(corrosion_ratio_pdf, corrosion_ratio_grid, axis=-1)
    corrosion_ratio_mean = np.trapezoid(corrosion_ratio_pdf*corrosion_ratio_grid[np.newaxis, :], corrosion_ratio_grid, axis=-1)
    corrosion_ratio_quantiles = np.stack((
                            corrosion_ratio_grid[np.argmin(np.abs(corrosion_ratio_cdf-a), axis=-1)],
                            corrosion_ratio_grid[np.argmin(np.abs(corrosion_ratio_cdf-(1-a)), axis=-1)]
    ))

    ax.fill_between(times_forecast, corrosion_ratio_quantiles[0], corrosion_ratio_quantiles[1], color="r", alpha=0.3)
    ax.plot(times_forecast, corrosion_ratio_quantiles[0], color="r", linewidth=0.2)
    ax.plot(times_forecast, corrosion_ratio_quantiles[1], color="r", linewidth=0.2)
    ax.plot(times_forecast, corrosion_ratio_mean, color="r", label="Factual forecast")

    data = log["theoretical"]["prior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    corrosion_ratio_grid = np.asarray(data["corrosion_ratio_grid"])
    corrosion_ratio_pdf = np.asarray(data["corrosion_ratio_pdf"])
    corrosion_ratio_cdf = cumulative_trapezoid(corrosion_ratio_pdf, corrosion_ratio_grid, axis=-1)
    corrosion_ratio_mean = np.trapezoid(corrosion_ratio_pdf*corrosion_ratio_grid[np.newaxis, :], corrosion_ratio_grid, axis=-1)
    corrosion_ratio_quantiles = np.stack((
                            corrosion_ratio_grid[np.argmin(np.abs(corrosion_ratio_cdf-a), axis=-1)],
                            corrosion_ratio_grid[np.argmin(np.abs(corrosion_ratio_cdf-(1-a)), axis=-1)]
    ))
    
    ax.fill_between(times_forecast, corrosion_ratio_quantiles[0], corrosion_ratio_quantiles[1], color="b", alpha=0.3)
    ax.plot(times_forecast, corrosion_ratio_quantiles[0], color="b", linewidth=0.2)
    ax.plot(times_forecast, corrosion_ratio_quantiles[1], color="b", linewidth=0.2)
    ax.plot(times_forecast, corrosion_ratio_mean, color="b", label="Counterfactual forecast")

    current_time = times_forecast[0]
    past_times = [time for time in all_times if time <= current_time]
    corrosion_ratio_obs = [val["corrosion_ratio"] for (key, val) in setting.items() if key <= current_time]
    corrosion_ratio_obs_error = params.obs_error_std
    ax.scatter(past_times, corrosion_ratio_obs, color="k", label="Observations")
    ax.errorbar(x=past_times, y=corrosion_ratio_obs, yerr=corrosion_ratio_obs_error, fmt='none', c="k")

    ax.axvline(current_time, c="k", linestyle="--")

    ax.set_xlabel("Time [yr]", fontsize=12)
    ax.set_ylabel("Corrosion ratio [-]", fontsize=12)
    ax.set_xlim(min(all_times), max(all_times))
    ax.set_ylim(0, 1)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.legend(fontsize=10)

    return fig


def plot_moment(log, params, a=0.05):

    setting = params.setting

    all_times = list(setting.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    data = log["theoretical"]["posterior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    moment_cap_effective = data["moment_cap_effective"]
    corrosion_ratio_grid = np.asarray(data["corrosion_ratio_grid"])
    moment_cap = moment_cap_effective * (1 - np.array(corrosion_ratio_grid))

    corrosion_ratio_pdf = np.asarray(data["corrosion_ratio_pdf"])
    corrosion_ratio_cdf = cumulative_trapezoid(corrosion_ratio_pdf, corrosion_ratio_grid, axis=-1)
    corrosion_ratio_mean = np.trapezoid(corrosion_ratio_pdf * corrosion_ratio_grid[np.newaxis, :], corrosion_ratio_grid,
                                        axis=-1)
    corrosion_ratio_quantiles = np.stack((
        corrosion_ratio_grid[np.argmin(np.abs(corrosion_ratio_cdf - a), axis=-1)],
        corrosion_ratio_grid[np.argmin(np.abs(corrosion_ratio_cdf - (1 - a)), axis=-1)]
    ))

    ax.fill_between(times_forecast, corrosion_ratio_quantiles[0], corrosion_ratio_quantiles[1], color="r", alpha=0.3)
    ax.plot(times_forecast, corrosion_ratio_quantiles[0], color="r", linewidth=0.2)
    ax.plot(times_forecast, corrosion_ratio_quantiles[1], color="r", linewidth=0.2)
    ax.plot(times_forecast, corrosion_ratio_mean, color="r", label="Factual forecast")

    data = log["theoretical"]["prior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    corrosion_ratio_grid = np.asarray(data["corrosion_ratio_grid"])
    corrosion_ratio_pdf = np.asarray(data["corrosion_ratio_pdf"])
    corrosion_ratio_cdf = cumulative_trapezoid(corrosion_ratio_pdf, corrosion_ratio_grid, axis=-1)
    corrosion_ratio_mean = np.trapezoid(corrosion_ratio_pdf * corrosion_ratio_grid[np.newaxis, :], corrosion_ratio_grid,
                                        axis=-1)
    corrosion_ratio_quantiles = np.stack((
        corrosion_ratio_grid[np.argmin(np.abs(corrosion_ratio_cdf - a), axis=-1)],
        corrosion_ratio_grid[np.argmin(np.abs(corrosion_ratio_cdf - (1 - a)), axis=-1)]
    ))

    ax.fill_between(times_forecast, corrosion_ratio_quantiles[0], corrosion_ratio_quantiles[1], color="b", alpha=0.3)
    ax.plot(times_forecast, corrosion_ratio_quantiles[0], color="b", linewidth=0.2)
    ax.plot(times_forecast, corrosion_ratio_quantiles[1], color="b", linewidth=0.2)
    ax.plot(times_forecast, corrosion_ratio_mean, color="b", label="Counterfactual forecast")

    current_time = times_forecast[0]
    past_times = [time for time in all_times if time <= current_time]
    corrosion_ratio_obs = [val["corrosion_ratio"] for (key, val) in setting.items() if key <= current_time]
    corrosion_ratio_obs_error = params.obs_error_std
    ax.scatter(past_times, corrosion_ratio_obs, color="k", label="Observations")
    ax.errorbar(x=past_times, y=corrosion_ratio_obs, yerr=corrosion_ratio_obs_error, fmt='none', c="k")

    ax.axvline(current_time, c="k", linestyle="--")

    ax.set_xlabel("Time [yr]", fontsize=12)
    ax.set_ylabel("Corrosion ratio [-]", fontsize=12)
    ax.set_xlim(min(all_times), max(all_times))
    ax.set_ylim(0, 1)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.legend(fontsize=10)

    return fig

if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    setting_path = SCRIPT_DIR / "data/setting/case_study.json"
    z_path = SCRIPT_DIR / "data/setting/z.json"
    mcs_samples_path = SCRIPT_DIR / f"data/mc_samples_normal_100000000.npy"
    results_path = SCRIPT_DIR / "results/reliability_timeline"
    plots_path = results_path / "plots"
    log_path = results_path / "runner_log"

    with open(setting_path, "r") as f:
        setting_data = json.load(f)
    setting_data = {float(key): val for (key, val) in setting_data.items()}

    params = TimelineParameters(setting=setting_data)

    figs = []
    all_times = params.times
    for i, time in enumerate(tqdm(params.setting.keys(), desc="Running time step")):

        log = read_log(log_path, int(time))

        # fig = plot_pf(log)
        # fig = plot_corrosion_ratio(log, params)
        fig = plot_moment(log, params)
        fig.savefig("dummy.png")




        pass

    pp = PdfPages(plots_path/"plots.pdf")
    [pp.savefig(fig) for fig in figs]
    pp.close()