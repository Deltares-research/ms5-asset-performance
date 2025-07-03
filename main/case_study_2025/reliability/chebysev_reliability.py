import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import joblib
from tqdm import tqdm
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


def reliability(
        fos_calculator,
        mcs_samples_path,
        water_lvl,
        EI,
        n_mcs=100_000,
        plot=True,
        fit=None
):

    result_path = SCRIPT_DIR / f"results/mcs/fos_sample"
    result_path.mkdir(parents=True, exist_ok=True)

    plot_path = SCRIPT_DIR / f"results/mcs/plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    if isinstance(water_lvl, float):
        water_lvl = np.repeat(water_lvl, n_mcs)

    if isinstance(EI, float):
        EI = np.repeat(EI, n_mcs)

    mcs_samples = np.load(mcs_samples_path)
    mcs_samples = mcs_samples[:, :-1]  # Remove EI column
    mcs_samples = np.column_stack((mcs_samples, EI, water_lvl))
    mcs_samples_torch = torch.from_numpy(mcs_samples.astype(np.float32)).to(device=device)
    mcs_samples_torch = mcs_samples_torch[:n_samples]

    dataset = TensorDataset(mcs_samples_torch)
    loader = DataLoader(dataset, batch_size=1_000, shuffle=True)

    fos = []
    for (x,) in tqdm(loader):
        f = fos_calculator.fos(x)
        fos.append(f)
    fos = np.stack(fos).flatten()
    np.save(result_path/f"fos_mcs_{n_samples}", fos)

    plot_fos_hist(fos, plot_path/f"fos_histogram_{n_samples}.png", modelfit="lognormal", ci_alpha=0.05)


if __name__ == "__main__":

    pass

