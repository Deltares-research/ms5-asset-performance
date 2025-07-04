from pathlib import Path
import numpy as np
from scipy.stats import truncnorm
from main.case_study_2025.reliability.utils import *
from src.corrosion.corrosion_model import CorrosionModel
from tqdm import tqdm


if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    setting_path = SCRIPT_DIR / "data/setting/case_study.json"
    z_path = SCRIPT_DIR / "data/setting/z.json"
    mcs_samples_path = SCRIPT_DIR / f"data/mc_samples_normal_100000000.npy"
    chebysev_path = SCRIPT_DIR / "train/results/srg/chebysev/lr_1.0e-05_epochs_100000_fullprofile_True"

    n_mcs = 100_000
    fos_calculator = load_chebysev_calculator(chebysev_path, z_path)

    corrosion_model = CorrosionModel()

    start_thickness = corrosion_model.start_thickness
    EI_start = 30000
    moment_cap_start = 40
    moment_survived = 0
    water_lvl = -1

    C50_grid = np.linspace(.5, 2.5, 10)
    C50_dist = truncnorm(loc=1.5, scale=1., a=(C50_grid.min()-1.5)/1., b=(C50_grid.max()-1.5)/1.)
    C50_prior = C50_dist.pdf(C50_grid)

    with open(setting_path, "r") as f:
        setting_data = json.load(f)

    for time, data in tqdm(setting_data.items(), desc="Running time step"):

        time = float(time)

        # if time == 50: continue
        if time > 50: break

        corrosion_obs = np.array([val["corrosion"] for (key, val) in setting_data.items() if float(key)<=time])
        corrosion_obs_times = np.array([float(key) for key in setting_data.keys() if float(key)<=time])
        moment_survived = max(moment_survived, data["max_moment"])

        C50_posterior = corrosion_model.bayesian_updating(
            corrosion_obs,
            corrosion_obs_times,
            C50_prior,
            C50_grid
        )

        pfs = np.zeros_like(C50_grid)
        for i, C50 in enumerate(C50_grid):
            pf = C50_pf(
                C50,
                time,
                corrosion_model,
                fos_calculator,
                n_mcs,
                start_thickness,
                moment_cap_start,
                EI_start,
                water_lvl,
                mcs_samples_path,
                moment_survived
            )
            pfs[i] = pf

        prior_pf = pfs * C50_prior
        prior_pf /= np.trapezoid(prior_pf, C50_grid)

        posterior_pf = pfs * C50_posterior
        posterior_pf /= np.trapezoid(posterior_pf, C50_grid)

        C50_prior = C50_posterior.copy()


