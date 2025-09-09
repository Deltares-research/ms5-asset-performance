import json
from pathlib import Path
import numpy as np
from main.case_study_2025.reliability.utils import *
from src.corrosion.corrosion_model import CorrosionModel
from tqdm import tqdm
from argparse import ArgumentParser


def main(n_mcs_samples=10_000_000, n_corrosion_grid=100, n_corrosion_ratio_grid=1_000, n_mcs=1_000_000):

    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    setting_path = SCRIPT_DIR / "data/case_study.json"
    mcs_samples_path = SCRIPT_DIR / f"data/mc_samples_normal_{n_mcs_samples}.npy"
    moment_model_path = SCRIPT_DIR / f"results/surrogate/mlp_moment/lr_1.0e-04_epochs_10000"
    results_path = SCRIPT_DIR / "results/reliability_timeline"
    results_path.mkdir(parents=True, exist_ok=True)

    with open(setting_path, "r") as f:
        setting_data = json.load(f)
    setting_data = {float(key): val for (key, val) in setting_data.items()}

    moment_calculator = load_moment_calculator(moment_model_path)

    params = TimelineParameters(setting=setting_data, n_mcs=n_mcs)

    corrosion_model = CorrosionModel(
        n_grid=n_corrosion_grid,
        C50_mu=params.C50_mu,
        corrosion_rate=params.corrosion_rate,
        obs_error_std=params.obs_error_std,
        start_thickness=params.start_thickness
    )

    pf_calculator = PfCalculator(n_corrosion_ratio_grid, params, corrosion_model, moment_calculator, mcs_samples_path)
    pf_calculator.calculate_max_moments(results_path)

    runner = TimelineRunner(
        time=params.times[0],
        start_thickness = params.start_thickness,
        moment_cap_start = params.moment_cap_start,
        water_lvl = params.water_lvl,
        corrosion_rate = params.corrosion_rate,
        obs_error_std=params.obs_error_std,
        corrosion_ratio_grid=pf_calculator.corrosion_ratio_grid.tolist(),
        C50_grid=corrosion_model.C50_grid.tolist(),
        C50_prior=corrosion_model.C50_prior.tolist(),
        C50_prior_fixed=corrosion_model.C50_prior.tolist(),
    )

    results = {}
    for time, data in tqdm(params.setting.items(), desc="Running time step"):

        time = float(time)

        runner.step(time, params)

        results = pf_calculator.calculate(params, runner)

        runner.log(time, results, results_path)

        runner.finish_step()


if __name__ == "__main__":

    main()

