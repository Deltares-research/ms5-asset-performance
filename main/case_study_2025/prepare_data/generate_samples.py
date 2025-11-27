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

    rv_names = [
        'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1', 'Zand_soilphi', 'Zand_soilcurkb1', 'Zandvast_soilphi',
        'Zandvast_soilcurkb1', 'Zandlos_soilphi', 'Zandlos_soilcurkb1', 'Wall_SheetPilingElementEI'
    ]

    df = df.loc[df["parameter"].isin(rv_names)].reset_index(drop=True)

    means = df["mean"].values[np.newaxis, :]
    stds = df["std"].values[np.newaxis, :]
    lower = df["lower"].values[np.newaxis, :]
    upper = df["upper"].values[np.newaxis, :]

    np.random.seed(42)
    mc_samples = means + stds * np.random.randn(n_mc_samples, len(df))
    mc_samples = np.clip(mc_samples, lower, upper)
    water_lvls = -1. * np.ones((n_mc_samples, 1))
    mc_samples = np.hstack((mc_samples, water_lvls))
    np.save(path.parent / f"data/mc_samples_normal.npy", mc_samples)

    np.random.seed(43)
    uniform_samples = np.random.uniform(size=(n_srg_samples, len(df)))
    srg_samples = means - 6 * stds + (6 * stds - (-6 * stds)) * uniform_samples
    srg_samples = np.clip(srg_samples, lower, upper)
    water_lvls = -1. * np.ones((n_srg_samples, 1))
    srg_samples = np.hstack((srg_samples, water_lvls))
    np.save(path.parent / f"data/surrogate_samples_uniform.npy", srg_samples)




if __name__ == "__main__":
    parser.add_argument("--n_mc_samples", type=int, default=10_000_000)
    parser.add_argument("--n_srg_samples", type=int, default=100_000)
    args = parser.parse_args()

    main(args.n_mc_samples, args.n_srg_samples)
