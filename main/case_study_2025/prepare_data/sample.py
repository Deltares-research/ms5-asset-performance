import numpy as np
import pandas as pd
from pathlib import Path


if __name__ == "__main__":

    path = Path(__file__).parent

    data_path = path.parents[1] / "data/parameter_distributions.csv"
    df = pd.read_csv(data_path)

    n_mc_samples = 10_000_000
    np.random.seed(42)
    means = df["mean"].values[np.newaxis, :]
    stds = df["std"].values[np.newaxis, :]
    mc_samples = means + stds * np.random.randn(n_mc_samples, len(df))
    water_lvls = -1. * np.zeros((n_mc_samples, 1))
    mc_samples = np.hstack((mc_samples, water_lvls))
    np.save(path.parents[1] / f"data/mc_samples_normal_{n_mc_samples}.npy", mc_samples)

    n_srg_samples = 100_000
    np.random.seed(42)
    means = df["mean"].values[np.newaxis, :]
    stds = df["std"].values[np.newaxis, :]
    uniform_samples = np.random.uniform(size=(n_srg_samples, len(df)))
    srg_samples = means - 6 * stds + (6 * stds - (-6 * stds)) * uniform_samples
    srg_samples = np.clip(srg_samples, 5, 55)
    water_lvls = -1. * np.zeros((n_srg_samples, 1))
    srg_samples = np.hstack((srg_samples, water_lvls))
    np.save(path.parents[1] / f"data/srg_samples_uniform_{n_srg_samples}.npy", srg_samples)