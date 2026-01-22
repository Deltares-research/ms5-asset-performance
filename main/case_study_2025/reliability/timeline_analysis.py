import json
from pathlib import Path
import numpy as np
from main.case_study_2025.reliability.utils import *
from src.corrosion.corrosion_model import CorrosionModel
from tqdm import tqdm
from argparse import ArgumentParser


"""
Reliability timeline analysis for D-SheetPiling with corrosion effects.

This script:
- Loads a case study configuration (`case_study.json`).
- Uses surrogate models (MLP) for moment prediction.
- Simulates corrosion progression over time.
- Calculates failure probabilities (Pf) with Monte Carlo Simulation (MCS).
- Logs results per time step for reliability assessment.

Outputs are stored under `results/reliability_timeline/`.
"""


def main(
    n_mcs_samples: int = 10_000_000,
    n_corrosion_grid: int = 100,
    n_corrosion_ratio_grid: int = 1_000,
    n_mcs: int = 1_000_000,
) -> None:
    """
    Run reliability timeline analysis for the case study.

    Steps:
        1. Load case study settings and surrogate moment model.
        2. Initialize corrosion and probability of failure (Pf) calculators.
        3. For each time step:
            - Update system state.
            - Compute failure probability.
            - Log results to output directory.

    Args:
        n_mcs_samples (int, optional): Number of Monte Carlo samples for parameter sampling.
            Defaults to 10,000,000.
        n_corrosion_grid (int, optional): Grid size for corrosion discretization.
            Defaults to 100.
        n_corrosion_ratio_grid (int, optional): Grid size for corrosion ratio discretization.
            Defaults to 1,000.
        n_mcs (int, optional): Number of Monte Carlo simulations for reliability estimation.
            Defaults to 1,000,000.

    Saves:
        - Reliability results per time step into `results/reliability_timeline/`.
    """
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    setting_path = SCRIPT_DIR / "data/case_study.json"
    mcs_samples_path = SCRIPT_DIR / f"data/mc_samples_normal.npy"
    moment_model_path = SCRIPT_DIR / f"results/surrogate/mlp_moment"
    results_path = SCRIPT_DIR / "results/reliability_timeline"
    results_path.mkdir(parents=True, exist_ok=True)

    with open(setting_path, "r") as f:
        setting_data = json.load(f)
    setting_data = {float(key): val for (key, val) in setting_data.items()}

    moment_calculator = load_moment_calculator(moment_model_path)

    params = TimelineParameters(setting=setting_data, n_mcs=n_mcs, moment_cap_start=750., obs_error_std=.4)

    C50_mu = 1.  # Manual adjustment for more optimistic corrosion measurements.
    params.C50_mu = C50_mu

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

    prior_results = pf_calculator.calculate_prior(params, runner)
    runner.log_prior(prior_results, results_path)

    results = {}
    for time, data in tqdm(params.setting.items(), desc="Running time step"):

        time = float(time)

        runner.step(time, params)

        results = pf_calculator.calculate(params, runner)

        runner.log(time, results, results_path)

        runner.finish_step()


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--n-grid", type=int, default=1_000)
    args = parser.parse_args()

    main(n_corrosion_ratio_grid=args.n_grid)

