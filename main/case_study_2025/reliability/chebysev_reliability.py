import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import joblib
from tqdm import tqdm
from main.case_study_2025.train.srg.chebysev_train import Chebysev, MinMaxScaler
from main.case_study_2025.reliability.moment_calculation.chebysev_moments import FoSCalculator
from main.case_study_2025.reliability.utils import *
import matplotlib.pyplot as plt


if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("✅ Using MPS (Metal) backend")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA backend")
else:
    device = torch.device("cpu")
    print("⚠️ MPS and CUDA not available — using CPU")


def reliability(n_mcs=100_000, plot=True, fit=None):

    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    chebysev_path = SCRIPT_DIR / "train/results/srg/chebysev/lr_1.0e-05_epochs_100000_fullprofile_True"
    mcs_samples_path = SCRIPT_DIR / f"data/mc_samples_normal_100000000.npy"
    result_path = SCRIPT_DIR / f"results/mcs/fos_sample"
    result_path.mkdir(parents=True, exist_ok=True)
    plot_path = SCRIPT_DIR / f"results/mcs/plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    mcs_samples = np.load(mcs_samples_path)
    water_lvl = -1
    water_lvl = np.repeat(np.array(water_lvl), mcs_samples.shape[0])
    mcs_samples = np.column_stack((mcs_samples, water_lvl))
    mcs_samples_torch = torch.from_numpy(mcs_samples.astype(np.float32)).to(device=device)
    mcs_samples_torch = mcs_samples_torch[:n_samples]

    dataset = TensorDataset(mcs_samples_torch)
    loader = DataLoader(dataset, batch_size=1_000, shuffle=True)

    n_points = 156
    moment_cap = 40
    wall_locs = np.linspace(0, 10, n_points)
    monitoring_locs = np.linspace(0, 10, n_points)
    wall_props = (1e+4, moment_cap, wall_locs, monitoring_locs)

    fos_calculator = FoSCalculator(
        n_points=n_points,
        wall_props=wall_props,
        model_path=chebysev_path/"torch_weights.pth",
        scaler_x_path=chebysev_path/"scaler_x.joblib",
        scaler_y_path=chebysev_path/"scaler_y.joblib",
        device=device
    )

    fos = []
    for (x,) in tqdm(loader):
        f = fos_calculator.fos(x)
        fos.append(f)
    fos = np.stack(fos).flatten()
    np.save(result_path/f"fos_mcs_{n_samples}", fos)

    plot_fos_hist(fos, plot_path/f"fos_histogram_{n_samples}.png", modelfit="lognormal", ci_alpha=0.05)


if __name__ == "__main__":

    pass

