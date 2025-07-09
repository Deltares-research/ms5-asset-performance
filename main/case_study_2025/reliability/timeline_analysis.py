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
    pflog_path = results_path / "pf_logs"
    pflog_path.mkdir(parents=True, exist_ok=True)

    with open(setting_path, "r") as f:
        setting_data = json.load(f)
    setting_data = {float(key): val for (key, val) in setting_data.items()}

    fos_calculator = load_chebysev_calculator(chebysev_path, z_path)

    params = TimelineParameters(setting=setting_data)

    corrosion_model = CorrosionModel(
        n_grid=100,
        C50_mu=params.C50_mu,
        corrosion_rate=params.corrosion_rate,
        obs_error_std=params.obs_error_std,
        start_thickness=params.start_thickness
    )

    pf_calculator = PfCalculator(1_000, params, corrosion_model, fos_calculator, mcs_samples_path)

    pf_calculator.calculate_max_moments(results_path)

    runner = TimelineRunner(
        time=params.times[0],
        start_thickness = params.start_thickness,
        EI_start = params.EI_start,
        moment_cap_start = params.moment_cap_start,
        moment_survived = params.moment_survived,
        water_lvl = params.water_lvl,
        corrosion_rate = params.corrosion_rate,
        corrosion_ratio_grid=pf_calculator.corrosion_ratio_grid.tolist(),
        C50_grid=corrosion_model.C50_grid.tolist(),
        C50_prior=corrosion_model.C50_prior.tolist()
    )

    results = {}
    for time, data in tqdm(params.setting.items(), desc="Running time step"):

        time = float(time)

        if time > 50: break

        runner.step(time, params)

        times = [iter_time for iter_time in params.times if iter_time >= time]
        corrosion_ratio_prior = runner.update_corrosion_ratio_pdf("prior", times)
        corrosion_ratio_posterior = runner.update_corrosion_ratio_pdf("posterior", times)

        pfs = {}
        for cap_type in ["theoretical", "survived"]:

            if cap_type == "theoretical":
                moment_cap_actual = runner.moment_cap_start
            elif cap_type == "survived":
                if runner.moment_survived > runner.moment_cap_start:
                    moment_cap_actual = runner.moment_survived
                    time_survived = runner.time_survived
                else:
                    continue

            moment_cap = moment_cap_actual * (1 - np.array(runner.corrosion_ratio_grid))
            # TODO: moment cap corrosion when survived

            pfs[cap_type] = {}
            for pdf_type in ["prior", "posterior"]:

                if pdf_type == "prior":
                    corrosion_ratio_pdf = corrosion_ratio_prior.copy()
                elif pdf_type == "posterior":
                    corrosion_ratio_pdf = corrosion_ratio_posterior.copy()

                pf = pf_calculator.get_pf(moment_cap, corrosion_ratio_prior, runner.corrosion_ratio_grid)
                beta = norm.ppf(1-pf)

                pfs[cap_type][pdf_type] = {
                    "moment_cap_type": cap_type,
                    "C50_dist_type": pdf_type,
                    "moment_cap": moment_cap_actual,
                    "corrosion_ratio_pdf": corrosion_ratio_pdf.tolist(),
                    "corrosion_ratio_pdf": corrosion_ratio_pdf.tolist(),
                    "pf": pf.tolist(),
                    "beta": beta.tolist(),
                }

        runner.log(results_path)

        with open(pflog_path/f"time_{time:.0f}.json", "w") as f:
            json.dump(pfs, f, indent=4)

        runner.finish_step()

