import numpy as np
import pandas as pd
from pathlib import Path


if __name__ == "__main__":

    path = Path(__file__).parent

    data_path = path.parent / "data/parameter_distributions.csv"
    df = pd.read_csv(data_path)

    means = df["mean"].values[np.newaxis, :]
    stds = df["std"].values[np.newaxis, :]
    lower = df["lower"].values[np.newaxis, :]
    upper = df["upper"].values[np.newaxis, :]

    n_mc_samples = 10_000_000
    np.random.seed(42)
    mc_samples = means + stds * np.random.randn(n_mc_samples, len(df))
    mc_samples = np.clip(mc_samples, lower, upper)
    water_lvls = -1. * np.ones((n_mc_samples, 1))
    mc_samples = np.hstack((mc_samples, water_lvls))
    np.save(path.parent / f"data/mc_samples_normal_{n_mc_samples}.npy", mc_samples)

    n_srg_samples = 100_000
    np.random.seed(43)
    uniform_samples = np.random.uniform(size=(n_srg_samples, len(df)))
    srg_samples = means - 6 * stds + (6 * stds - (-6 * stds)) * uniform_samples
    srg_samples = np.clip(srg_samples, lower, upper)
    water_lvls = -1. * np.ones((n_srg_samples, 1))
    srg_samples = np.hstack((srg_samples, water_lvls))
    np.save(path.parent / f"data/srg_samples_uniform_{n_srg_samples}.npy", srg_samples)