import numpy as np
import torch
import pandas as pd
from scipy.stats import norm
from pathlib import Path


if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).resolve().parent

    n = 100_000_000
    seed = 42

    param_dist_path = SCRIPT_DIR / "../data/parameter_distributions.csv"
    output_path = SCRIPT_DIR.parent / f"data/mc_samples_normal_{n}.npy"

    parameter_dists = pd.read_csv(param_dist_path)
    parameter_dists = parameter_dists.set_index(parameter_dists["parameter"], drop=True)
    cols_keep = [
        'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1', 'Zand_soilphi', 'Zand_soilcurkb1', 'Zandvast_soilphi',
        'Zandvast_soilcurkb1', 'Zandlos_soilphi', 'Zandlos_soilcurkb1', 'Wall_SheetPilingElementEI'
    ]
    parameter_dists = parameter_dists.loc[cols_keep]
    mus = parameter_dists["mean"].values
    sigmas = parameter_dists["std"].values
    dist = norm(loc=mus, scale=sigmas)

    np.random.seed(seed)
    samples = dist.rvs((n, len(cols_keep)))

    np.save(output_path, samples)

