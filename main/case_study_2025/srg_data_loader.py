import numpy as np
import pandas as pd


def load_data(path):

    df = pd.read_csv(path)

    X_cols = [col for col in df.columns if col.split("_")[0] != "disp" and col != "index"]
    X_cols = [col for col in X_cols if "soilcurko2" not in col and "soilcurko3" not in col]
    X = df[X_cols].values

    idx_locs = list(range(1, 151, 10))
    y_cols = [col for col in df.columns if col.split("_")[0] == "disp" and int(col.split("_")[-1]) in idx_locs]
    y = df[y_cols].values

    # Remove extreme outliers which probably correspond to numerical error in DSheetPiling
    # quartiles = np.quantile(y, [0.25, 0.75], axis=0)
    # iqr = np.diff(quartiles, axis=0).squeeze()
    # bnd_distance = 0.1
    # bounds = quartiles + np.array([-bnd_distance, +bnd_distance])[:, np.newaxis] * iqr[np.newaxis, :]
    # outliers = np.all(np.logical_and(bounds[0]<=y, y<=bounds[1]), axis=1)

    max_disp = 50
    outliers = np.any(np.abs(y) >= max_disp, axis=1)

    X, y = X[~outliers], y[~outliers]

    return X, y