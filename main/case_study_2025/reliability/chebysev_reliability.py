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


def moment_mcs(
        fos_calculator,
        mcs_samples_path,
        water_lvl,
        EI,
        n_mcs=100_000
):

    if isinstance(water_lvl, float) or isinstance(water_lvl, int):
        water_lvl = np.repeat(water_lvl, n_mcs)

    if isinstance(EI, float) or isinstance(EI, int):
        EI = np.repeat(EI, n_mcs)

    mcs_samples = np.load(mcs_samples_path)
    mcs_samples = mcs_samples[:n_mcs]
    mcs_samples = mcs_samples[:, :-1]  # Remove EI column
    mcs_samples = np.column_stack((mcs_samples, EI, water_lvl))
    mcs_samples_torch = torch.from_numpy(mcs_samples.astype(np.float32)).to(device=device)

    dataset = TensorDataset(mcs_samples_torch)
    loader = DataLoader(dataset, batch_size=1_000, shuffle=True)

    moments = []
    for (x,) in loader:
        m = fos_calculator.moments(x)
        moments.append(m)
    moments = np.stack(moments)
    moments = moments.reshape(-1, moments.shape[-1])

    max_moments = np.abs(moments).max(axis=1)

    return max_moments


if __name__ == "__main__":

    pass

