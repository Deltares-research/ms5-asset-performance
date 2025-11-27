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
Visualization utilities for reliability timeline analysis of D-SheetPiling.

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
            log[cap_type][pdf_type]["fos"] = np.load(path / "fos.npy")
            log[cap_type][pdf_type]["survival"] = np.load(path / "survival.npy")
    return log


def plot_pf(log: Dict[str, Any], current_time: float) -> plt.Figure:
    """
    Plot probability of failure (Pf) forecasts for theoretical and empirical capacity.

    Args:
        log (dict): Log dictionary returned by `read_log`.
        current_time (float): Current time step.

    Returns:
        matplotlib.figure.Figure: The generated Pf plot.
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=False)

    ax = axs[0]

    data = log["theoretical"]["posterior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    pf_forecast_posterior = [float(val) for val in data["pf_forecast"].values()]
    ax.plot(times_forecast, pf_forecast_posterior, c="r", label="Posterior forecast")

    data = log["theoretical"]["prior"]
    times_prior = [float(key) for key in data["pf_forecast"].keys()]
    pf_forecast_prior = [float(val) for val in data["pf_forecast"].values()]
    ax.plot(times_forecast, pf_forecast_prior, c="b", label="Prior forecast")

    ax.annotate("", xy=(current_time, 0), xytext=(current_time, 0.1), arrowprops=dict(arrowstyle='->', color="k", linewidth=3))
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
    ax.plot(times_forecast, pf_forecast_posterior, c="r", label="Posterior forecast")

    data = log["survived"]["prior"]
    times_prior = [float(key) for key in data["pf_forecast"].keys()]
    pf_forecast_prior = [float(val) for val in data["pf_forecast"].values()]
    ax.plot(times_forecast, pf_forecast_prior, c="b", label="Prior forecast")
    
    ax.axvline(current_time, c="k", linestyle="--", linewidth=0.5)
    ax.annotate("", xy=(current_time, 0), xytext=(current_time, 0.1), arrowprops=dict(arrowstyle='->', color="k", linewidth=3))
    ax.set_xlabel("Time [yr]", fontsize=12)
    ax.set_ylabel("${P}_{f}$ [-]", fontsize=12)
    ax.set_yscale('log')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.legend(fontsize=10)
    ax.set_title("Empirical moment capacity", fontsize=12)

    return fig


def plot_beta(log: Dict[str, Any], current_time: float, beta_req: int = 2.) -> plt.Figure:
    """
    Plot reliability index forecasts for theoretical and empirical capacity.

    Args:
        log (dict): Log dictionary returned by `read_log`.
        current_time (float): Current time step.
        beta_req (int): Reliability index requirement.

    Returns:
        matplotlib.figure.Figure: The generated Pf plot.
    """

    labels = {
        "theoretical_prior": "A",
        "theoretical_posterior": "B",
        "survived_prior": "C",
        "survived_posterior": "D"
    }

    colors = {
        "A": "b",
        "B": "r",
        "C": "m",
        "D": "g",
    }

    max_beta = 3.

    fig = plt.figure(figsize=(8, 4))

    for (key, label) in labels.items():

        proven_str_update, corrosion_update = key.split("_")
        data = log[proven_str_update][corrosion_update]

        times_forecast = [float(key) for key in data["pf_forecast"].keys()]
        pf_forecast_posterior = [float(val) for val in data["pf_forecast"].values()]
        beta_forecast_posterior = [norm.ppf(1 - pf) for pf in pf_forecast_posterior]
        beta_forecast_posterior = [max_beta if np.isinf(beta) else beta for beta in beta_forecast_posterior]
        plt.plot(times_forecast, beta_forecast_posterior, c=colors[label], label=label)

    plt.axvline(beta_req, c="k", linestyle="--", label="Requirement")

    plt.xlabel("Forecast time [yr]", fontsize=12)
    plt.ylabel("${β}$ [-]", fontsize=12)
    plt.xlim(50, 80)
    plt.ylim(1., max_beta)
    plt.legend(fontsize=12)
    plt.grid()
    fig.suptitle(f"Current time: {current_time:.0f}", fontsize=14)

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

    fig, ax = plt.subplots(figsize=(8, 4))

    data = log["theoretical"]["prior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    corrosion_grid = np.asarray(data["corrosion_grid"])
    corrosion_pdf = np.asarray(data["corrosion_pdf"])
    corrosion_cdf = cumulative_trapezoid(corrosion_pdf, corrosion_grid, axis=-1)
    corrosion_mean = np.trapezoid(corrosion_pdf * corrosion_grid[np.newaxis, :], corrosion_grid, axis=-1)
    corrosion_quantiles = np.stack((
        corrosion_grid[np.argmin(np.abs(corrosion_cdf - alpha), axis=-1)],
        corrosion_grid[np.argmin(np.abs(corrosion_cdf - (1 - alpha)), axis=-1)]
    ))

    ax.fill_between(times_forecast, corrosion_quantiles[0], corrosion_quantiles[1], color="b", alpha=0.3)
    ax.plot(times_forecast, corrosion_quantiles[0], color="b", linewidth=0.2)
    ax.plot(times_forecast, corrosion_quantiles[1], color="b", linewidth=0.2)
    ax.plot(times_forecast, corrosion_mean, color="b", label="Prior forecast")

    data = log["theoretical"]["posterior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    corrosion_grid = np.asarray(data["corrosion_grid"])
    corrosion_pdf = np.asarray(data["corrosion_pdf"])
    corrosion_cdf = cumulative_trapezoid(corrosion_pdf, corrosion_grid, axis=-1)
    corrosion_mean = np.trapezoid(corrosion_pdf*corrosion_grid[np.newaxis, :], corrosion_grid, axis=-1)
    corrosion_quantiles = np.stack((
                            corrosion_grid[np.argmin(np.abs(corrosion_cdf-alpha), axis=-1)],
                            corrosion_grid[np.argmin(np.abs(corrosion_cdf-(1-alpha)), axis=-1)]
    ))

    ax.fill_between(times_forecast, corrosion_quantiles[0], corrosion_quantiles[1], color="r", alpha=0.3)
    ax.plot(times_forecast, corrosion_quantiles[0], color="r", linewidth=0.2)
    ax.plot(times_forecast, corrosion_quantiles[1], color="r", linewidth=0.2)
    ax.plot(times_forecast, corrosion_mean, color="r", label="Posterior forecast")

    current_time = times_forecast[0]
    past_times = [time for time in all_times if time <= current_time]
    corrosion_obs = [val["corrosion"] for (key, val) in setting.items() if key <= current_time]
    corrosion_obs_error = params.obs_error_std
    ax.errorbar(x=past_times, y=corrosion_obs, yerr=corrosion_obs_error*norm.ppf(1-alpha), fmt='o', c="k", capsize=3, label="Observations")
    
    ax.axvline(current_time, c="k", linestyle="--", linewidth=0.5)
    ax.annotate("", xy=(current_time, 0), xytext=(current_time, 0.1), arrowprops=dict(arrowstyle='->', color="k", linewidth=3))
    ax.set_xlabel("Time [yr]", fontsize=12)
    ax.set_ylabel("Corrosion [mm]", fontsize=12)
    ax.set_xlim(min(all_times), max(all_times))
    ax.set_ylim(0, params.start_thickness)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.legend(fontsize=10)

    return fig


def plot_corrosion_ratio(log: Dict[str, Any], params: TimelineParameters, alpha: float = 0.05) -> plt.Figure:
    """
    Plot corrosion ratio progression (prior vs posterior forecast).

    Args:
        log (dict): Log dictionary for current time.
        params (TimelineParameters): Simulation parameters.
        alpha (float, optional): Confidence level (1-alpha). Defaults to 0.05.

    Returns:
        matplotlib.figure.Figure: The generated corrosion ratio plot.
    """
    setting = params.setting

    all_times = list(setting.keys())

    fig, ax = plt.subplots(figsize=(8, 4))

    data = log["theoretical"]["prior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    corrosion_ratio_grid = np.asarray(data["corrosion_ratio_grid"])
    corrosion_ratio_pdf = np.asarray(data["corrosion_ratio_pdf"])
    corrosion_ratio_cdf = cumulative_trapezoid(corrosion_ratio_pdf, corrosion_ratio_grid, axis=-1)
    corrosion_ratio_mean = np.trapezoid(corrosion_ratio_pdf * corrosion_ratio_grid[np.newaxis, :], corrosion_ratio_grid,
                                        axis=-1)
    corrosion_ratio_quantiles = np.stack((
        corrosion_ratio_grid[np.argmin(np.abs(corrosion_ratio_cdf - alpha), axis=-1)],
        corrosion_ratio_grid[np.argmin(np.abs(corrosion_ratio_cdf - (1 - alpha)), axis=-1)]
    ))

    ax.fill_between(times_forecast, corrosion_ratio_quantiles[0], corrosion_ratio_quantiles[1], color="b", alpha=0.3)
    ax.plot(times_forecast, corrosion_ratio_quantiles[0], color="b", linewidth=0.2)
    ax.plot(times_forecast, corrosion_ratio_quantiles[1], color="b", linewidth=0.2)
    ax.plot(times_forecast, corrosion_ratio_mean, color="b", label="Prior forecast")

    data = log["theoretical"]["posterior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    corrosion_ratio_grid = np.asarray(data["corrosion_ratio_grid"])
    corrosion_ratio_pdf = np.asarray(data["corrosion_ratio_pdf"])
    corrosion_ratio_cdf = cumulative_trapezoid(corrosion_ratio_pdf, corrosion_ratio_grid, axis=-1)
    corrosion_ratio_mean = np.trapezoid(corrosion_ratio_pdf*corrosion_ratio_grid[np.newaxis, :], corrosion_ratio_grid, axis=-1)
    corrosion_ratio_quantiles = np.stack((
                            corrosion_ratio_grid[np.argmin(np.abs(corrosion_ratio_cdf-alpha), axis=-1)],
                            corrosion_ratio_grid[np.argmin(np.abs(corrosion_ratio_cdf-(1-alpha)), axis=-1)]
    ))

    ax.fill_between(times_forecast, corrosion_ratio_quantiles[0], corrosion_ratio_quantiles[1], color="r", alpha=0.3)
    ax.plot(times_forecast, corrosion_ratio_quantiles[0], color="r", linewidth=0.2)
    ax.plot(times_forecast, corrosion_ratio_quantiles[1], color="r", linewidth=0.2)
    ax.plot(times_forecast, corrosion_ratio_mean, color="r", label="Posterior forecast")

    current_time = times_forecast[0]
    past_times = [time for time in all_times if time <= current_time]
    corrosion_ratio_obs = [val["corrosion_ratio"] for (key, val) in setting.items() if key <= current_time]
    corrosion_ratio_obs_error = params.obs_error_std / params.start_thickness
    ax.errorbar(x=past_times, y=corrosion_ratio_obs, yerr=corrosion_ratio_obs_error*norm.ppf(1-alpha), fmt='o', c="k", capsize=3, label="Observations")
    
    ax.axvline(current_time, c="k", linestyle="--", linewidth=0.5)
    ax.annotate("", xy=(current_time, 0), xytext=(current_time, 0.1), arrowprops=dict(arrowstyle='->', color="k", linewidth=3))
    ax.set_xlabel("Time [yr]", fontsize=12)
    ax.set_ylabel("Corrosion ratio [-]", fontsize=12)
    ax.set_xlim(min(all_times), max(all_times))
    ax.set_ylim(0, 1)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.legend(fontsize=10)

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

    fig, ax = plt.subplots(figsize=(8, 4))

    data = log["survived"]["posterior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    moment_cap_effective = data["moment_cap_effective"]
    corrosion_ratio_grid = np.asarray(data["corrosion_ratio_grid"])
    corrosion_ratio_pdf = np.asarray(data["corrosion_ratio_pdf"])
    moment_cap = data["moment_cap_start"]
    time_survived = data["time_survived"]
    moment_survived = data["moment_survived"]

    moment_cap_grid = moment_cap * (1 - corrosion_ratio_grid) + 1e-3
    moment_cap_pdf = corrosion_ratio_pdf * (1 / moment_cap)  # Variable change
    moment_cap_grid = np.flip(moment_cap_grid)
    moment_cap_pdf_truncated = np.flip(moment_cap_pdf, axis=-1).copy()
    moment_cap_pdf_truncated = np.where(moment_cap_grid <= moment_survived, 0., moment_cap_pdf_truncated)
    moment_cap_pdf_truncated /= np.trapezoid(moment_cap_pdf_truncated, moment_cap_grid, axis=-1)[:, None]

    moment_cap_mean = np.trapezoid(moment_cap_pdf_truncated*moment_cap_grid, moment_cap_grid, axis=-1)
    moment_cap_cdf_truncated = cumulative_trapezoid(moment_cap_pdf_truncated, moment_cap_grid, axis=-1)
    moment_cap_quantiles = np.stack((
                            moment_cap_grid[np.argmin(np.abs(moment_cap_cdf_truncated-alpha), axis=-1)],
                            moment_cap_grid[np.argmin(np.abs(moment_cap_cdf_truncated-(1-alpha)), axis=-1)]
    ))

    ax.fill_between(times_forecast, moment_cap_quantiles[0], moment_cap_quantiles[1], color="r", alpha=0.3)
    ax.plot(times_forecast, moment_cap_quantiles[0], color="r", linewidth=0.2)
    ax.plot(times_forecast, moment_cap_quantiles[1], color="r", linewidth=0.2)
    ax.plot(times_forecast, moment_cap_mean, color="r", label="Posterior forecast")

    data = log["survived"]["prior"]
    times_forecast = [float(key) for key in data["pf_forecast"].keys()]
    moment_cap_effective = data["moment_cap_effective"]
    corrosion_ratio_grid = np.asarray(data["corrosion_ratio_grid"])
    corrosion_ratio_pdf = np.asarray(data["corrosion_ratio_pdf"])
    moment_cap = data["moment_cap_start"]
    time_survived = data["time_survived"]
    moment_survived = data["moment_survived"]

    moment_cap_grid = moment_cap * (1 - corrosion_ratio_grid) + 1e-3
    moment_cap_pdf = corrosion_ratio_pdf * (1 / moment_cap)  # Variable change
    moment_cap_grid = np.flip(moment_cap_grid)
    moment_cap_pdf_truncated = np.flip(moment_cap_pdf, axis=-1).copy()
    moment_cap_pdf_truncated = np.where(moment_cap_grid <= moment_survived, 0., moment_cap_pdf_truncated)
    moment_cap_pdf_truncated /= np.trapezoid(moment_cap_pdf_truncated, moment_cap_grid, axis=-1)[:, None]

    moment_cap_mean = np.trapezoid(moment_cap_pdf_truncated * moment_cap_grid, moment_cap_grid, axis=-1)
    moment_cap_cdf_truncated = cumulative_trapezoid(moment_cap_pdf_truncated, moment_cap_grid, axis=-1)
    moment_cap_quantiles = np.stack((
        moment_cap_grid[np.argmin(np.abs(moment_cap_cdf_truncated - alpha), axis=-1)],
        moment_cap_grid[np.argmin(np.abs(moment_cap_cdf_truncated - (1 - alpha)), axis=-1)]
    ))

    ax.fill_between(times_forecast, moment_cap_quantiles[0], moment_cap_quantiles[1], color="b", alpha=0.3)
    ax.plot(times_forecast, moment_cap_quantiles[0], color="b", linewidth=0.2)
    ax.plot(times_forecast, moment_cap_quantiles[1], color="b", linewidth=0.2)
    ax.plot(times_forecast, moment_cap_mean, color="b", label="Prior forecast")

    current_time = times_forecast[0]
    past_times = [time for time in all_times if time <= current_time]
    corrosion_ratio_obs = [val["corrosion_ratio"] for (key, val) in setting.items() if key <= current_time]
    corrosion_ratio_obs_error = params.obs_error_std
    ax.scatter(x=past_times, y=moment_cap_effective*(1-np.array(corrosion_ratio_obs)), color="k", label="Observations")
    ax.scatter([time_survived], [moment_survived], marker="x", color="g", label="Survived moment")
    ax.axhline(moment_survived, c="g", linestyle="--", linewidth=1)

    ax.axvline(current_time, c="k", linestyle="--", linewidth=0.5)
    ax.annotate("", xy=(current_time, 0), xytext=(current_time, 80), arrowprops=dict(arrowstyle='->', color="k", linewidth=3))
    ax.set_xlabel("Time [yr]", fontsize=12)
    ax.set_ylabel("Moment capacity [kNm]", fontsize=12)
    ax.set_xlim(min(all_times), max(all_times))
    ax.set_ylim(400, 700)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.legend(fontsize=10, loc="lower right")

    return fig


def plot_scatter(log: Dict[str, Any], mcs_data: np.ndarray, rvs: List[str] = ["Klei_soilphi", "Corrosion_ratio"]) -> plt.Figure:
    """
    Scatter plot of random variables vs. reliability outcomes.

    Args:
        log (dict): Log dictionary for current time.
        mcs_data (np.ndarray): Monte Carlo samples (input features).
        rvs (List[str], optional): Variables to plot on x and y axes. Defaults to ["Klei_soilphi", "Corrosion_ratio"].

    Returns:
        matplotlib.figure.Figure: The generated scatter plot.
    """
    def make_df(data, mcs_data):
        fos = data["fos"]
        survival = data["survival"].flatten()
        moment_survived = data["moment_survived"]
        moment_cap = data["moment_cap_effective"]
        cols_keep = [
            'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1', 'Zand_soilphi', 'Zand_soilcurkb1',
            'Zandvast_soilphi',
            'Zandvast_soilcurkb1', 'Zandlos_soilphi', 'Zandlos_soilcurkb1'
        ]

        mcs_data = np.repeat(mcs_data[:, :-1], fos.shape[0], axis=0)
        df = pd.DataFrame(
            data=np.hstack((mcs_data,  np.repeat(np.array(data["corrosion_ratio_grid"]), fos.shape[-1])[:, np.newaxis], fos.flatten()[:, np.newaxis])),
            columns=cols_keep + ["Corrosion_ratio", "FoS"]
        )
        df["Survival"] = moment_cap * (1-df["Corrosion_ratio"]) >= moment_survived

        df["hue"] = ""
        df.loc[df["FoS"]*df["Survival"] == 1, "hue"] = "Safety_Survival"
        df.loc[df["FoS"]*(1-df["Survival"]) == 1, "hue"] = "Safety_NonSurvival"
        df.loc[1-df["FoS"]==1, "hue"] = "Failure"

        return df

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 12))
    i = 0
    for cap_type in ["theoretical", "survived"]:
        for pdf_type in ["prior", "posterior"]:
            df = make_df(log[cap_type][pdf_type], mcs_data)
            np.random.seed(42)
            # idx = np.random.randint(low=0, high=mcs_data.shape[0]-1, size=1_000)
            # df = df.iloc[idx]
            sns.scatterplot(data=df, x=rvs[0], y=rvs[1], hue="hue",  ax=axs.flatten()[i])
            axs.flatten()[i].set(title=f"{cap_type}-{pdf_type}")
            i += 1

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
    plots_path = results_path / "plots"
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

        # # if not time % 10 == 0:
        # if time > 50:
        #     continue

        log = read_log(log_path / f"{time}", int(time))

        # fig = plot_pf(log, float(time))
        # fig.suptitle(f"Time={time}")
        # pf_figs.append(fig)

        fig = plot_beta(log, float(time))
        fig.suptitle(f"Time={time}")
        beta_figs.append(fig)

        fig = plot_corrosion(log, params)
        fig.suptitle(f"Time={time}")
        corrosion_figs.append(fig)

        fig = plot_corrosion_ratio(log, params)
        fig.suptitle(f"Time={time}")
        corrosion_ratio_figs.append(fig)

        fig = plot_moment(log, params)
        fig.suptitle(f"Time={time}")
        moment_figs.append(fig)


    # pp = PdfPages(plots_path / "pf_plots.pdf")
    # [pp.savefig(fig) for fig in pf_figs]
    # pp.close()

    pp = PdfPages(plots_path / "beta_plots.pdf")
    [pp.savefig(fig) for fig in beta_figs]
    pp.close()

    pp = PdfPages(plots_path / "corrosion_plots.pdf")
    [pp.savefig(fig) for fig in corrosion_figs]
    pp.close()

    pp = PdfPages(plots_path / "corrosion_ratio_plots.pdf")
    [pp.savefig(fig) for fig in corrosion_ratio_figs]
    pp.close()

    pp = PdfPages(plots_path / "moment_plots.pdf")
    [pp.savefig(fig) for fig in moment_figs]
    pp.close()


if __name__ == "__main__":

    main()

