import os
import numpy as np
from pathlib import Path
import pandas as pd
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.rvs.state import MvnRV, GaussianState
from src.reliability_models.dsheetpiling.lsf import package_lsf
from src.reliability_models.dsheetpiling.reliability import ReliabilityFragilityCurve
from utils import parse_parameter_dist
import matplotlib.pyplot as plt


def time_pf(rm, df, times):

    df = df.loc[df["times"].isin(times)].copy()

    corrosions = 1 - times / 365 / 75
    # corrosions = np.ones_like(times)
    wall_EI_start = 4.5003e+04
    wall_EIs = corrosions * wall_EI_start

    water_lvls = df.loc[:, "water_lvl"].values

    points = np.column_stack((wall_EIs, water_lvls))

    pfs = rm.pf_point(points)

    df["corrosion"] = corrosions
    df["pf"] = pfs

    return df


if __name__ == "__main__":

    output_path = Path(r"../../results/ptk")

    soil_states, other_states = parse_parameter_dist(r"../../data/parameter_distributions.csv")
    water_state = MvnRV(mus=[-0.8], stds=[0.2], names=["water_lvl"])
    state = GaussianState(rvs=soil_states+other_states+[water_state])

    geomodel_path = os.environ["MODEL_PATH"]
    form_path = os.environ["FORM_PATH"]
    geomodel = DSheetPiling(geomodel_path, form_path)

    performance_config = ("max_moment", lambda x: 40. / (x[0] + 1e-5))
    form_params = (0.15, 30, 0.02)
    lsf = package_lsf(geomodel, state, performance_config, True)

    fc_savedir = r"../../results/ptk/fragility_curve.json"
    rm = ReliabilityFragilityCurve(lsf, state, "form", form_params, integration_rv_names=["Wall_SheetPilingElementEI", "water_lvl"])
    rm.load_fragility(fc_savedir)

    years = 25
    water_period_yr = 5
    times = np.linspace(0, years*365, 1_000)
    water_lvls = -0.8 + 0.3 * np.sin(times/365*2*np.pi/water_period_yr)
    df = pd.DataFrame(data=np.column_stack((times, water_lvls)), columns=["times", "water_lvl"])

    times_calc = df["times"].values
    df = time_pf(rm, df, times_calc)
    df["times_yr"] = df["times"] / 365

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    ax = axs[0]
    ax2 = ax.twinx()
    ax.plot(df["times_yr"], df["water_lvl"], c="b", label="Water level")
    ax2.plot(df["times_yr"], df["pf"], c="r", label="${P}_{f}$")
    ax.set_xlabel("Time [yr]", fontsize=14)
    ax.set_ylabel("Water level [+mNAP]", fontsize=14)
    ax2.set_ylabel("${P}_{f}$ [-]", fontsize=14)
    ax2.set_yscale('log')
    ax.grid()

    ax = axs[1]
    ax2 = ax.twinx()
    ax.plot(df["times_yr"], df["corrosion"], c="g", label="Corrosion ratio")
    ax2.plot(df["times_yr"], df["pf"], c="r", label="${P}_{f}$")
    ax.set_xlabel("Time [yr]", fontsize=14)
    ax.set_ylabel("Water level [+mNAP]", fontsize=14)
    ax2.set_ylabel("${P}_{f}$ [-]", fontsize=14)
    ax2.set_yscale('log')
    ax.grid()

    plt.tight_layout()
    fig.savefig(output_path/"timeline_pf.png")

