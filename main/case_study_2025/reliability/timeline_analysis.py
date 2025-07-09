import json
from pathlib import Path
import numpy as np
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
        moment_cap_start = params.moment_cap_start,
        water_lvl = params.water_lvl,
        corrosion_rate = params.corrosion_rate,
        obs_error_std=params.obs_error_std,
        corrosion_ratio_grid=pf_calculator.corrosion_ratio_grid.tolist(),
        C50_grid=corrosion_model.C50_grid.tolist(),
        C50_prior=corrosion_model.C50_prior.tolist(),
    )

    results = {}
    for time, data in tqdm(params.setting.items(), desc="Running time step"):

        time = float(time)

        if time > 50: break

        runner.step(time, params)

        pfs = pf_calculator.get_pfs(params, runner)
        runner.read_pfs(pfs)

        runner.log(results_path)

        runner.finish_step()

