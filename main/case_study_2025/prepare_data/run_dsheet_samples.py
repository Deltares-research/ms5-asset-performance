import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from src.geotechnical_models.dsheetpiling.model import DSheetPiling, DSheetPilingResults
from src.reliability_models.dsheetpiling.lsf import *
from pathlib import Path
import json
from tqdm import tqdm
from typing import List, Dict
from argparse import ArgumentParser


"""
Surrogate data generator for D-SheetPiling model simulations.

This script:
- Loads surrogate random variable samples.
- Runs the D-SheetPiling geotechnical model on a subset of samples.
- Extracts displacements and bending moments.
- Saves results as both JSON (`surrogate_data.json`) and CSV (`surrogate_data.csv`).

Intended for building surrogate models of structural response under uncertainty.
"""


def run_model(rvs: List[float], model: DSheetPiling, rv_names: List[str]) -> DSheetPilingResults:
    """
    Run a D-SheetPiling model simulation for a single set of random variables.

    Args:
        rvs (np.ndarray): 1D array of random variable values.
        model (DSheetPiling): Initialized D-SheetPiling model instance.
        rv_names (List[str]): Names of the random variables in order.

    Returns:
        DSheetPilingResults: Results of the model run (displacements, moments, etc.).
    """
    params = {name: rv for (name, rv) in zip(rv_names, rvs)}
    soil_data = unpack_soil_params(params, list(model.soils.keys()))
    water_data = unpack_water_params(params, [lvl.name for lvl in model.water.water_lvls])
    wall_data = unpack_wall_data(params, model.wall._asdict())
    model.update_soils(soil_data)
    model.update_water(water_data)
    model.update_wall(wall_data)
    model.execute()
    return model.results


def sample_disp(
    rv_sample: np.ndarray, rv_names: List[str], model: DSheetPiling
) -> Tuple[List[float], List[float]]:
    """
    Evaluate the D-SheetPiling model for multiple random variable samples.

    Args:
        rv_sample (np.ndarray): 2D array of random variable samples (N x d).
        rv_names (List[str]): Names of the random variables in order.
        model (DSheetPiling): Initialized D-SheetPiling model instance.

    Returns:
        Tuple[List[float], List[float]]:
            - Displacements at the first node for all runs.
            - Moments at the first node for all runs.
    """
    disp_sample = []
    moment_sample = []

    for i_run, rvs in enumerate(tqdm(rv_sample, desc="Calculating sample")):
        results = run_model(rvs, model, rv_names)
        disp_sample.append(results.displacement[0])
        moment_sample.append(results.moment[0])

    return disp_sample, moment_sample


def main(n_samples_to_use: int = 1_000) -> None:
    """
    Run D-SheetPiling simulations for surrogate samples and save results.

    Args:
        n_samples_to_use (int, optional): Number of surrogate samples to evaluate.
            Defaults to 1,000.

    Saves:
        - `surrogate_data.json`: Simulation results in JSON format.
        - `surrogate_data.csv`: Tabular results with random variables, displacements, and moments.
    """
    path = Path(__file__).parent

    load_dotenv(path.parents[2] / ".env")

    geomodel_path = os.environ["DSHEET_MODEL_PATH"]

    data_path = path.parent / "data/surrogate_samples_uniform.npy"
    result_path = path.parent / "data"
    result_path.mkdir(exist_ok=True, parents=True)

    rv_names = [
        'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1', 'Zand_soilphi', 'Zand_soilcurkb1', 'Zandvast_soilphi',
        'Zandvast_soilcurkb1', 'Zandlos_soilphi', 'Zandlos_soilcurkb1', 'Wall_SheetPilingElementEI', 'water_lvl'
    ]

    samples = np.load(data_path)[:n_samples_to_use]
    samples = np.unique(samples, axis=0)

    geomodel = DSheetPiling(geomodel_path)

    disp_sample, moment_sample = sample_disp(samples, rv_names, geomodel)

    results = {
        "sample": samples.tolist(),
        "displacement": disp_sample,
        "moment": moment_sample,
    }

    with open(result_path / "surrogate_data.json", "w") as f:
        json.dump(results, f, indent=4)

    data = np.c_[np.array(samples), np.array(disp_sample), np.array(moment_sample)]
    df_srg = pd.DataFrame(
        data=data,
        columns=rv_names + [f"disp_{i}" for i in range(1, len(disp_sample[0]) + 1)] + [f"moment_{i}" for i in range(1, len(moment_sample[0]) + 1)]
    )

    df_srg.to_csv(result_path / "surrogate_data.csv")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--n_samples_to_use", type=int, default=1_000)
    args = parser.parse_args()

    main(n_samples_to_use=args.n_samples_to_use)

