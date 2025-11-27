import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm
from main.case_study_2025.reliability.utils import *
from src.corrosion.corrosion_model import CorrosionModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


"""
Visualization utilities for reliability timeline analysis of D-SheetPiling comparing only the prior (no Bayesian
updating or conditioning to measurements) vs the posterior (conditioning to measurements, Bayesian updating of corrosion
and moment).

This script:
- Reads logged reliability results from runner logs.
- Produces plots for:
    * Probability of failure (Pf) forecasts
    * Corrosion progression
    * Corrosion ratio forecasts
    * Moment capacity degradation
    * Scatterplots of random variables vs. reliability outcomes
- Exports figures to PDF files under `results/reliability_timeline/plots`.

Functions return matplotlib Figure objects for flexible use.
"""


def read_log(log_path: Path, time: float) -> Dict[str, Dict[str, Any]]:
    """
    Read log data for a given time step.

    Args:
        log_path (Path): Path to the log directory (runner_log/<time>).
        time (float): Current time step.

    Returns:
        dict: Nested dict of log data for ["theoretical"]["prior"/"posterior"]
              and ["survived"]["prior"/"posterior"].
    """
    log = {}
    for cap_type in ["theoretical", "survived"]:
        log[cap_type] = {}
        for pdf_type in ["prior", "posterior"]:
            path = log_path / f"{cap_type}/{pdf_type}"
            with open(path / "data.json", "r") as f:
                log[cap_type][pdf_type] = json.load(f)
    return log


def plot_beta(log: Dict[str, Any], beta_req: int = 2.) -> plt.Figure:
    """
    Plot reliability index forecasts for theoretical and empirical capacity.

    Args:
        log (dict): Log dictionary of prior and posterior.
        beta_req (int): Reliability index requirement.

    Returns:
        matplotlib.figure.Figure: The generated Pf plot.
    """

    max_beta = 3.

    fig = plt.figure(figsize=(8, 4))
    colors = ["b", "r"]
    for i, (key, data) in enumerate(log.items()):
        times_forecast = [float(key) for key in data["pf_forecast"].keys()]
        pf_forecast = [float(val) for val in data["pf_forecast"].values()]
        beta_forecast = [norm.ppf(1 - pf) for pf in pf_forecast]
        beta_forecast = [max_beta if np.isinf(beta) else beta for beta in beta_forecast]
        plt.plot(times_forecast, beta_forecast, c=colors[i], label=key.title())

    plt.axhline(beta_req, c="k", linestyle="--", label="Requirement")
    plt.xlabel("Forecast time [yr]", fontsize=12)
    plt.ylabel("${β}$ [-]", fontsize=12)
    plt.xlim(50, 75)
    plt.ylim(.5, max_beta)
    plt.legend(fontsize=12)
    plt.grid()

    return fig


def plot_corrosion(log: Dict[str, Any], params: TimelineParameters, alpha: float = 0.05) -> plt.Figure:
    """
    Plot corrosion depth progression (prior vs posterior forecast).

    Args:
        log (dict): Log dictionary for current time.
        params (TimelineParameters): Simulation parameters (with settings, thickness, etc.).
        alpha (float, optional): Confidence level (1-alpha). Defaults to 0.05.

    Returns:
        matplotlib.figure.Figure: The generated corrosion plot.
    """
    setting = params.setting

    all_times = list(setting.keys())

    fig = plt.figure(figsize=(8, 4))
    colors = ["b", "r"]
    for i, (key, data) in enumerate(log.items()):
        times_forecast = [float(key) for key in data["pf_forecast"].keys()]
        corrosion_grid = np.asarray(data["corrosion_grid"])
        corrosion_pdf = np.asarray(data["corrosion_pdf"])
        corrosion_cdf = cumulative_trapezoid(corrosion_pdf, corrosion_grid, axis=-1)
        corrosion_mean = np.trapezoid(corrosion_pdf * corrosion_grid[np.newaxis, :], corrosion_grid, axis=-1)
        corrosion_quantiles = np.stack((
            corrosion_grid[np.argmin(np.abs(corrosion_cdf - alpha), axis=-1)],
            corrosion_grid[np.argmin(np.abs(corrosion_cdf - (1 - alpha)), axis=-1)]
        ))
        plt.fill_between(times_forecast, corrosion_quantiles[0], corrosion_quantiles[1], color=colors[i], alpha=0.3)
        plt.plot(times_forecast, corrosion_quantiles[0], color=colors[i], linewidth=0.2)
        plt.plot(times_forecast, corrosion_quantiles[1], color=colors[i], linewidth=0.2)
        plt.plot(times_forecast, corrosion_mean, color=colors[i], label=key.title())

        if key == "posterior":
            current_time = times_forecast[0]
            past_times = [time for time in all_times if time <= current_time]
            corrosion_obs = [val["corrosion"] for (key, val) in setting.items() if key <= current_time]
            corrosion_obs_error = params.obs_error_std
            plt.errorbar(x=past_times, y=corrosion_obs, yerr=corrosion_obs_error * norm.ppf(1 - alpha), fmt='o', c="k",
                        capsize=3, label="Observations")

    plt.xlabel("Forecast time [yr]", fontsize=12)
    plt.ylabel("Corrosion [mm]", fontsize=12)
    plt.xlim(50, 75)
    plt.ylim(0, 9.5)
    plt.legend(fontsize=12)
    plt.grid()

    return fig


def plot_moment(log: Dict[str, Any], params: TimelineParameters, alpha: float = 0.005) -> plt.Figure:
    """
    Plot bending moment capacity degradation (prior vs posterior).

    Args:
        log (dict): Log dictionary for current time.
        params (TimelineParameters): Simulation parameters.
        alpha (float, optional): Confidence level (1-alpha). Defaults to 0.05.

    Returns:
        matplotlib.figure.Figure: The generated moment plot.
    """
    setting = params.setting

    all_times = list(setting.keys())

    fig = plt.figure(figsize=(8, 4))
    colors = ["b", "r"]
    for i, (key, data) in enumerate(log.items()):
        times_forecast = [float(key) for key in data["pf_forecast"].keys()]
        moment_cap = data["moment_cap_start"]
        corrosion_ratio_grid = np.asarray(data["corrosion_ratio_grid"])
        corrosion_ratio_pdf = np.asarray(data["corrosion_ratio_pdf"])
        moment_survived = data["moment_survived"]
        moment_cap_grid = moment_cap * (1 - corrosion_ratio_grid) + 1e-3
        moment_cap_pdf = corrosion_ratio_pdf * (1 / moment_cap)  # Variable change
        moment_cap_grid = np.flip(moment_cap_grid)
        moment_cap_pdf_truncated = np.flip(moment_cap_pdf, axis=-1).copy()
        moment_cap_pdf_truncated = np.where(moment_cap_grid <= moment_survived, 0., moment_cap_pdf_truncated)
        moment_cap_pdf_truncated /= np.trapezoid(moment_cap_pdf_truncated, moment_cap_grid, axis=-1)[:, None]

        moment_cap_mean = np.trapezoid(moment_cap_pdf_truncated * moment_cap_grid, moment_cap_grid, axis=-1)
        moment_cap_cdf_truncated = cumulative_trapezoid(moment_cap_pdf_truncated, moment_cap_grid, axis=-1)
        moment_cap_quantiles =  np.stack((
            moment_cap_grid[np.argmin(np.abs(np.where(moment_cap_cdf_truncated-alpha<0, np.inf, moment_cap_cdf_truncated-alpha)), axis=-1)],
            moment_cap_grid[np.argmin(np.abs(moment_cap_cdf_truncated - (1 - alpha)), axis=-1)]
        ))
        plt.fill_between(times_forecast, moment_cap_quantiles[0], moment_cap_quantiles[1], color=colors[i], alpha=0.3)
        plt.plot(times_forecast, moment_cap_quantiles[0], color=colors[i], linewidth=0.2)
        plt.plot(times_forecast, moment_cap_quantiles[1], color=colors[i], linewidth=0.2)
        plt.plot(times_forecast, moment_cap_mean, color=colors[i], label=key.title())

        if key == "posterior":
            current_time = times_forecast[0]
            past_times = [time for time in all_times if time <= current_time]
            corrosion_ratio_obs = [val["corrosion_ratio"] for (key, val) in setting.items() if key <= current_time]
            corrosion_ratio_obs_error = params.obs_error_std
            moment_cap_effective = data["moment_cap_effective"]
            plt.scatter(x=past_times, y=moment_cap_effective * (1 - np.array(corrosion_ratio_obs)), color="k",
                       label="Observations")

    plt.axhline(log["posterior"]["moment_survived"], c="g", label="Survived moment")
    plt.xlabel("Forecast time [yr]", fontsize=12)
    plt.ylabel("Moment capacity [kNm]", fontsize=12)
    plt.xlim(50, 75)
    plt.ylim(100, 800)
    plt.legend(fontsize=12)
    plt.grid()

    return fig


def main() -> None:
    """
    Main execution:
    - Reads case study settings.
    - Iterates through all time steps.
    - Reads logs and generates plots for Pf, corrosion, corrosion ratio, and moment capacity.
    - Exports figures into PDF reports under `results/reliability_timeline/plots`.
    """
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    setting_path = SCRIPT_DIR / "data/case_study.json"
    mcs_samples_path = SCRIPT_DIR / f"data/surrogate_data.csv"
    moment_path = SCRIPT_DIR / "train/results/surrogate/mlp_moment/lr_1.0e-04_epochs_100000"
    results_path = SCRIPT_DIR / "results/reliability_timeline"
    log_path = results_path / "runner_log"
    plots_path = results_path / "plots_prior_posterior"
    plots_path.mkdir(parents=True, exist_ok=True)

    with open(setting_path, "r") as f:
        setting_data = json.load(f)
    setting_data = {float(key): val for (key, val) in setting_data.items()}

    params = TimelineParameters(setting=setting_data)

    pf_figs = []
    beta_figs = []
    corrosion_figs = []
    corrosion_ratio_figs = []
    moment_figs = []
    scatter_figs = []
    all_times = params.times

    for i, time in enumerate(tqdm(params.setting.keys(), desc="Running time step")):

        if time > 51:
            continue


        with open(log_path/"prior/data.json", "r") as f:
            log_prior = json.load(f)

        log_posterior = read_log(log_path / f"{time}", int(time))

        log = {
            "prior": log_prior,
            "posterior": log_posterior["survived"]["posterior"],
        }

        fig = plot_beta(log)
        fig.suptitle(f"Time={time}")
        beta_figs.append(fig)

        fig = plot_corrosion(log, params)
        fig.suptitle(f"Time={time}")
        corrosion_figs.append(fig)

        fig = plot_moment(log, params)
        fig.suptitle(f"Time={time}")
        moment_figs.append(fig)

    pp = PdfPages(plots_path / "beta_plots.pdf")
    [pp.savefig(fig) for fig in beta_figs]
    pp.close()

    pp = PdfPages(plots_path / "corrosion_plot.pdf")
    [pp.savefig(fig) for fig in corrosion_figs]
    pp.close()

    pp = PdfPages(plots_path / "moment_plots.pdf")
    [pp.savefig(fig) for fig in moment_figs]
    pp.close()


if __name__ == "__main__":

    main()

