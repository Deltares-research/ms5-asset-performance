import json
from pathlib import Path
import numpy as np
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

        fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharex=True, sharey=True)

        ax = axs[0]

        data = log["theoretical"]["posterior"]
        times_posterior = [float(key) for key in data["pf_forecast"].keys()]
        pf_forecast_posterior = [float(val) for val in data["pf_forecast"].values()]
        ax.plot(times_posterior, pf_forecast_posterior, c="r", label="Factual forecast")

        data = log["theoretical"]["prior"]
        times_prior = [float(key) for key in data["pf_forecast"].keys()]
        pf_forecast_prior = [float(val) for val in data["pf_forecast"].values()]
        ax.plot(times_posterior, pf_forecast_posterior, c="b", label="Counterfactual forecast")

        ax.set_xlabel("Time [yr]", fontsize=12)
        ax.set_ylabel("${P}_{f}$ [-]", fontsize=12)
        ax.set_yscale('log')
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax.legend(fontsize=10)
        ax.set_title("Theoretical moment capacity", fontsize=12)

        ax = axs[1]

        data = log["survived"]["posterior"]
        times_posterior = [float(key) for key in data["pf_forecast"].keys()]
        pf_forecast_posterior = [float(val) for val in data["pf_forecast"].values()]
        ax.plot(times_posterior, pf_forecast_posterior, c="r", label="Factual forecast")

        data = log["theoretical"]["prior"]
        times_prior = [float(key) for key in data["pf_forecast"].keys()]
        pf_forecast_prior = [float(val) for val in data["pf_forecast"].values()]
        ax.plot(times_posterior, pf_forecast_posterior, c="b", label="Counterfactual forecast")

        ax.set_xlabel("Time [yr]", fontsize=12)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax.legend(fontsize=10)
        ax.set_title("Empirical moment capacity", fontsize=12)

        fig.savefig("dummy.png")

        pass

    pp = PdfPages(plots_path/"plots.pdf")
    [pp.savefig(fig) for fig in figs]
    pp.close()