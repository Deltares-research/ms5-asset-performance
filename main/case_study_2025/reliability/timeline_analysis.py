import json
from pathlib import Path
import numpy as np
from scipy.stats import norm, truncnorm
from main.case_study_2025.reliability.utils import *
from src.corrosion.corrosion_model import CorrosionModel
from tqdm import tqdm


if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    setting_path = SCRIPT_DIR / "data/setting/case_study.json"
    z_path = SCRIPT_DIR / "data/setting/z.json"
    mcs_samples_path = SCRIPT_DIR / f"data/mc_samples_normal_100000000.npy"
    chebysev_path = SCRIPT_DIR / "train/results/srg/chebysev/lr_1.0e-05_epochs_100000_fullprofile_True"
    results_path = SCRIPT_DIR / "results/reliability_timeline"
    results_path.mkdir(parents=True, exist_ok=True)

    with open(setting_path, "r") as f:
        setting_data = json.load(f)
    setting_data = {float(key): val for (key, val) in setting_data.items()}

    fos_calculator = load_chebysev_calculator(chebysev_path, z_path)

    params = TimelineParameters(setting=setting_data)

    corrosion_model = CorrosionModel(
        n_grid=10,
        C50_mu=params.C50_mu,
        corrosion_rate=params.corrosion_rate,
        obs_error_std=params.obs_error_std,
        start_thickness=params.start_thickness
    )

    runner = TimelineRunner(
        start_thickness = params.start_thickness,
        EI_start = params.EI_start,
        moment_cap_start = params.moment_cap_start,
        moment_survived = params.moment_survived,
        water_lvl = params.water_lvl,
        C50_grid=corrosion_model.C50_grid.tolist(),
        C50_prior=corrosion_model.C50_prior.tolist()
    )

    pf_calculator = PfCalculator(runner.C50_grid, params, corrosion_model, fos_calculator, mcs_samples_path)

    results = {}
    for time, data in tqdm(params.setting.items(), desc="Running time step"):

        time = float(time)

        if time > 50: break

        runner.step(time, params)

        runner.log(results_path)

        max_moments = pf_calculator.calculate_max_moments(runner)

        pfs_all = []
        for cap_type in ["theoretical", "survived"]:

            if cap_type == "theoretical":
                moment_cap_actual = runner.moment_cap_start
            elif cap_type == "survived":
                moment_cap_actual = runner.moment_cap

            for pdf_type in ["prior", "posterior"]:

                if pdf_type == "prior":
                    C50_pdf = np.array(runner.C50_prior)
                elif pdf_type == "posterior":
                    C50_pdf = np.array(runner.C50_posterior)

                fos = moment_cap_sample / moment_sample
                fos = fos[moment_cap_sample >= moment_survived]
                pf = np.mean(fos >= 1)

                pf = pfs * C50_pdf
                pf /= np.trapezoid(pf, C50_grid)
                beta = norm.ppf(1-pf)

                pfs_all.append({
                    "moment_cap_type": cap_type,
                    "C50_dist_type": pdf_type,
                    "pf": pf.tolist(),
                    "beta": beta.tolist(),
                })

        # with open(results_path/"timeline_results.json", "w") as f:
        #     json.dump(results, f)

        runner.finish_step()

