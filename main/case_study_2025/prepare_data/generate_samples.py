import numpy as np
import pandas as pd
from pathlib import Path
# from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
import os
from tqdm import tqdm
import json

os.getenv('geolib.env')

import sys
sys.path.append(r'C:\Users\eijnden\OneDrive - Stichting Deltares\Desktop\MS5_2\ms5-asset-performance')

from src.geotechnical_models.dsheetpiling.model import DSheetPiling, DSheetPilingResults
from src.reliability_models.dsheetpiling.lsf import unpack_soil_params, unpack_water_params, unpack_wall_data



def main(n_mc_samples: int, n_srg_samples: int) -> None:
    """
    Generate Monte Carlo and surrogate samples for model input parameters
    and save them as NumPy `.npy` files.

    Args:
        n_mc_samples (int): Number of Monte Carlo samples to generate.
        n_srg_samples (int): Number of surrogate (uniform) samples to generate.

    Saves:
        - `mc_samples_normal_<n_mc_samples>.npy` with normally distributed samples.
        - `surrogate_samples_uniform_<n_srg_samples>.npy` with uniform samples.
    """
    path = Path(__file__).parent

    data_path = path.parent / "data/parameter_distributions.csv"
    df = pd.read_csv(data_path)

<<<<<<< HEAD
    print(df)

=======
    rv_names = [
        'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1', 'Zand_soilphi', 'Zand_soilcurkb1', 'Zandvast_soilphi',
        'Zandvast_soilcurkb1', 'Zandlos_soilphi', 'Zandlos_soilcurkb1', 'Wall_SheetPilingElementEI'
    ]

    df = df.loc[df["parameter"].isin(rv_names)].reset_index(drop=True)
>>>>>>> dev/case_study

    means = df["mean"].values[np.newaxis, :]
    stds = df["std"].values[np.newaxis, :]
    lower = df["lower"].values[np.newaxis, :]
    upper = df["upper"].values[np.newaxis, :]

    np.random.seed(42)
    mc_samples = means + stds * np.random.randn(n_mc_samples, len(df))
    mc_samples = np.clip(mc_samples, lower, upper)
    water_lvls = -1. * np.ones((n_mc_samples, 1))
    mc_samples = np.hstack((mc_samples, water_lvls))
    np.save(path.parent / f"data/mc_samples_normal_{n_mc_samples}.npy", mc_samples)

    np.random.seed(43)
    uniform_samples = np.random.uniform(size=(n_srg_samples, len(df)))
    srg_samples = means - 6 * stds + (6 * stds - (-6 * stds)) * uniform_samples
    srg_samples = np.clip(srg_samples, lower, upper)
    water_lvls = -1. * np.ones((n_srg_samples, 1))
    srg_samples = np.hstack((srg_samples, water_lvls))
    np.save(path.parent / f"data/surrogate_samples_uniform_{n_srg_samples}.npy", srg_samples)


<<<<<<< HEAD

def run_model(rvs: List[float], model: DSheetPiling, rv_names: List[str]) -> Any:
    """
    Run the D-SheetPiling geotechnical model with a single set of random variables.

    Args:
        rvs (np.ndarray): Array of sampled parameter values (1D).
        model (DSheetPiling): Initialized D-SheetPiling model instance.
        rv_names (List[str]): Names of the random variables in order.

    Returns:
        DSheetPilingResults: Simulation results object containing displacements,
        bending moments, etc.
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
    Run multiple model simulations to generate displacement and moment samples.

    Args:
        rv_sample (np.ndarray): 2D array of random variable samples (N x d).
        rv_names (List[str]): Names of the random variables in order.
        model (DSheetPiling): Initialized D-SheetPiling model instance.

    Returns:
        Tuple[List[float], List[float]]:
            - Displacement samples (first node per run).
            - Moment samples (first node per run).
    """
    disp_sample = []
    moment_sample = []

    for i_run, rvs in enumerate(tqdm(rv_sample, desc="Calculating sample")):
        results = run_model(rvs, model, rv_names)
        disp_sample.append(results.displacement[0])
        moment_sample.append(results.moment[0])

    return disp_sample, moment_sample


def calculate(n_samples_to_use: int = 1_000) -> None:
    """
    Generate surrogate model data by evaluating the D-SheetPiling model
    on a subset of surrogate samples.

    Args:
        n_samples_to_use (int, optional): Number of surrogate samples to evaluate.
            Defaults to 1,000.

    Saves:
        - `surrogate_data.json`: Raw simulation results (samples, displacements, moments).
        - `surrogate_data.csv`: Tabular format with random variables + outputs.
    """

    path = Path(__file__).parent

    load_dotenv(path.parents[2] / ".env")

    geomodel_path = os.environ["DSHEET_MODEL_PATH"]
    data_path = path.parent / "data/surrogate_samples_uniform_1000.npy"
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
=======
# def run_model(rvs: List[float], model: DSheetPiling, rv_names: List[str]) -> Any:
#     """
#     Run the D-SheetPiling geotechnical model with a single set of random variables.
#
#     Args:
#         rvs (np.ndarray): Array of sampled parameter values (1D).
#         model (DSheetPiling): Initialized D-SheetPiling model instance.
#         rv_names (List[str]): Names of the random variables in order.
#
#     Returns:
#         DSheetPilingResults: Simulation results object containing displacements,
#         bending moments, etc.
#     """
#     params = {name: rv for (name, rv) in zip(rv_names, rvs)}
#     soil_data = unpack_soil_params(params, list(model.soils.keys()))
#     water_data = unpack_water_params(params, [lvl.name for lvl in model.water.water_lvls])
#     wall_data = unpack_wall_data(params, model.wall._asdict())
#     model.update_soils(soil_data)
#     model.update_water(water_data)
#     model.update_wall(wall_data)
#     model.execute()
#     return model.results
#
#
# def sample_disp(
#     rv_sample: np.ndarray, rv_names: List[str], model: DSheetPiling
# ) -> Tuple[List[float], List[float]]:
#     """
#     Run multiple model simulations to generate displacement and moment samples.
#
#     Args:
#         rv_sample (np.ndarray): 2D array of random variable samples (N x d).
#         rv_names (List[str]): Names of the random variables in order.
#         model (DSheetPiling): Initialized D-SheetPiling model instance.
#
#     Returns:
#         Tuple[List[float], List[float]]:
#             - Displacement samples (first node per run).
#             - Moment samples (first node per run).
#     """
#     disp_sample = []
#     moment_sample = []
#
#     for i_run, rvs in enumerate(tqdm(rv_sample, desc="Calculating sample")):
#         results = run_model(rvs, model, rv_names)
#         disp_sample.append(results.displacement[0])
#         moment_sample.append(results.moment[0])
#
#     return disp_sample, moment_sample
#
#
# def calculate(n_samples_to_use: int = 1_000) -> None:
#     """
#     Generate surrogate model data by evaluating the D-SheetPiling model
#     on a subset of surrogate samples.
#
#     Args:
#         n_samples_to_use (int, optional): Number of surrogate samples to evaluate.
#             Defaults to 1,000.
#
#     Saves:
#         - `surrogate_data.json`: Raw simulation results (samples, displacements, moments).
#         - `surrogate_data.csv`: Tabular format with random variables + outputs.
#     """
#
#     path = Path(__file__).parent
#
#     load_dotenv(path.parents[2] / ".env")
#
#     geomodel_path = os.environ["DSHEET_MODEL_PATH"]
#     data_path = path.parent / "data/srg_samples_uniform_100000.npy"
#     result_path = path.parent / "data"
#     result_path.mkdir(exist_ok=True, parents=True)
#
#     rv_names = [
#         'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1', 'Zand_soilphi', 'Zand_soilcurkb1', 'Zandvast_soilphi',
#         'Zandvast_soilcurkb1', 'Zandlos_soilphi', 'Zandlos_soilcurkb1', 'Wall_SheetPilingElementEI', 'water_lvl'
#     ]
#
#     samples = np.load(data_path)[:n_samples_to_use]
#     samples = np.unique(samples, axis=0)
#
#     geomodel = DSheetPiling(geomodel_path)
#
#     disp_sample, moment_sample = sample_disp(samples, rv_names, geomodel)
#
#     results = {
#         "sample": samples.tolist(),
#         "displacement": disp_sample,
#         "moment": moment_sample,
#     }
#
#     with open(result_path / "surrogate_data.json", "w") as f:
#         json.dump(results, f, indent=4)
#
#     data = np.c_[np.array(samples), np.array(disp_sample), np.array(moment_sample)]
#     df_srg = pd.DataFrame(
#         data=data,
#         columns=rv_names + [f"disp_{i}" for i in range(1, len(disp_sample[0]) + 1)] + [f"moment_{i}" for i in range(1, len(moment_sample[0]) + 1)]
#     )
#
#     df_srg.to_csv(result_path / "surrogate_data.csv")
>>>>>>> dev/case_study


if __name__ == "__main__":

    parser = ArgumentParser()
<<<<<<< HEAD
    try: 
        parser.add_argument("--n_mc_samples", type=int, default=10_000_000)
        parser.add_argument("--n_srg_samples", type=int, default=1_000)
        args = parser.parse_args()
    except:
        print('argument parser error')
        args = parser.parse_args()
        args.n_mc_samples = 10_000_000   
        args.n_srg_samples = 1000

    draw(args.n_mc_samples, args.n_srg_samples)
    calculate(n_samples_to_use=args.n_srg_samples)

    #except:     
    #    pass



=======
    parser.add_argument("--n_mc_samples", type=int, default=10_000_000)
    parser.add_argument("--n_srg_samples", type=int, default=100_000)
    args = parser.parse_args()

    main(args.n_mc_samples, args.n_srg_samples)
    # draw(args.n_mc_samples, args.n_srg_samples)
    # calculate(n_samples_to_use=args.n_srg_samples)
>>>>>>> dev/case_study
