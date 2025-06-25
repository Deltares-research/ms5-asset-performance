import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.interpolate import UnivariateSpline


def load_data(path, step=10):

    df = pd.read_csv(path)

    X_cols = [col for col in df.columns if col.split("_")[0] != "disp" and col != "index"]
    X_cols = [col for col in X_cols if "soilcurko2" not in col and "soilcurko3" not in col and "soilcurkb2" not in col and "soilcurkb3" not in col and "soilcurko1" not in col]
    X = df[X_cols].values
    print(X.shape)

    idx_locs = list(range(1, 151, step))
    y_cols = [col for col in df.columns if col.split("_")[0] == "disp" and int(col.split("_")[-1]) in idx_locs]
    y = df[y_cols].values

    # check which row of y contains nan values
    y_nan = np.isnan(y)
    row_has_nan = np.any(y_nan, axis=1)  # Check if any value in each row is NaN
    # remove rows of X and y that contain nan values
    X = X[~row_has_nan]
    y = y[~row_has_nan]


    # Remove extreme outliers which probably correspond to numerical error in DSheetPiling
    # quartiles = np.quantile(y, [0.25, 0.75], axis=0)
    # iqr = np.diff(quartiles, axis=0).squeeze()
    # bnd_distance = 0.1
    # bounds = quartiles + np.array([-bnd_distance, +bnd_distance])[:, np.newaxis] * iqr[np.newaxis, :]
    # outliers = np.all(np.logical_and(bounds[0]<=y, y<=bounds[1]), axis=1)

    max_disp = 100
    outliers = np.any(np.abs(y) >= max_disp, axis=1)

    X, y = X[~outliers], y[~outliers]

    return X, y


def plot_predictions(y_train, y_test, y_hat_train, y_hat_test, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    figs = []
    zipped = zip(y_train.T, y_hat_train.T, y_test.T, y_hat_test.T)
    for i_point, (y_t_train, y_p_train, y_t_test, y_p_test) in enumerate(zipped):
        fig = plt.figure()
        fig.suptitle(f"Point #{i_point + 1:d} along wall", fontsize=14)
        plt.scatter(y_t_train, y_p_train, marker='x', c='b', label="Train")
        plt.scatter(y_t_test, y_p_test, marker='x', c='r', label="Test")
        plt.axline((0, 0), slope=1, c='k')
        plt.plot([y_t_train.min(), y_t_train.min()], [y_t_train.max(), y_t_train.max()], c='k')
        plt.xlabel('Observation', fontsize=12)
        plt.ylabel('Prediction', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        plt.close()
        figs.append(fig)
    pp = PdfPages(path)
    [pp.savefig(fig) for fig in figs]
    pp.close()

def plot_wall_displacements(x, y, y_hat, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    fig = plt.figure(figsize=(8, 8))
    plt.axvline(0, color="k", linewidth=2, label="Starting position")
    locs = np.arange(1, y.shape[-1] + 1)
    ci = np.quantile(y, (0.025, 0.975), axis=0)
    plt.fill_betweenx(locs, ci[0], ci[1], color="b", alpha=0.3, label="Observation 95% CI")
    plt.plot(y.mean(axis=0), locs, c="b", label="Mean observations")
    for i, y_hat_depth in enumerate(y_hat.T):
        loc = i + 1
        label = "Prediction 95% CI" if i == y_hat.shape[-1] - 1 else None
        mean = y_hat_depth.mean()
        ci = np.quantile(y_hat_depth, [0.025, 0.975])
        xerr = np.abs(mean - ci)
        plt.errorbar([mean], [loc], xerr=xerr[:, np.newaxis], fmt='o', color="r", capsize=5, label=label)
    plt.xlabel('Moment [kNm]', fontsize=12)
    plt.ylabel('Point # along wall', fontsize=12)
    plt.legend(fontsize=10)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.close()

    if path.suffix == ".png":
        fig.savefig(path)
    else:
        pp = PdfPages(path)
        pp.savefig(fig)
        pp.close()

def plot_wall_moments(x, true_displacements, predicted_displacements, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    # Create wall_locations to match the number of displacement points
    n_displacement_points = true_displacements.shape[-1]
    wall_locations = np.linspace(0, 10.4, n_displacement_points)
    true_moments = moments(true_displacements, x[:, -2], wall_locations)
    pred_moments = moments(predicted_displacements, x[:, -2], wall_locations)

    fig = plt.figure(figsize=(8, 8))
    plt.axvline(0, color="k", linewidth=2, label="Starting position")
    locs = np.arange(1, true_moments.shape[-1] + 1)
    ci = np.quantile(true_moments, (0.025, 0.975), axis=0)
    plt.fill_betweenx(locs, ci[0], ci[1], color="b", alpha=0.3, label="Observation 95% CI")
    plt.plot(true_moments.mean(axis=0), locs, c="b", label="Mean observations")
    n_points = pred_moments.shape[-1]
    for i, y_hat_depth in enumerate(pred_moments.T):
        # if i in [0,1,2,n_points-3,n_points-2,n_points-1]:
            # continue
        loc = i + 1
        label = "Prediction 95% CI" if i == pred_moments.shape[-1] - 1 else None
        mean = y_hat_depth.mean()
        if abs(mean) > 300:
            continue
        ci = np.quantile(y_hat_depth, [0.025, 0.975])
        xerr = np.abs(mean - ci)
        plt.errorbar([mean], [loc], xerr=xerr[:, np.newaxis], fmt='o', color="r", capsize=5, label=label)
    plt.xlabel('Moment [kNm]', fontsize=12)
    plt.ylabel('Point # along wall', fontsize=12)
    plt.legend(fontsize=10)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.close()

    if path.suffix == ".png":
        fig.savefig(path)
    else:
        pp = PdfPages(path)
        pp.savefig(fig)
        pp.close()


def plot_losses(losses, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    fig = plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, losses.size + 1), losses, c="b")
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss [${mm}^{2}$]', fontsize=12)
    plt.yscale("log")
    plt.grid()
    plt.close()

    fig.savefig(path)


def plot_violins(x, y, y_hat, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    df = pd.DataFrame(data=np.concatenate((y.flatten(), y_hat.flatten())), columns=["Moment [kNm]"])
    df["Type"] = ["Observation"] * y.size + ["Prediction"] * y_hat.size
    df["Type"] = pd.Categorical(df["Type"], categories=["Observation", "Prediction"], ordered=True)
    df["Location"] = np.repeat(np.tile(np.arange(1, y.shape[-1] + 1), y.shape[0]), 2)

    df["x_offset"] = df["Location"].copy().astype(float)
    df.loc[df["Type"] == "Prediction", "x_offset"] += 0.2
    df.loc[df["Type"] == "Observation", "x_offset"] -= 0.2

    plt.figure(figsize=(14, 8))
    sns.violinplot(data=df, x="x_offset", y="Moment [kNm]", hue="Type", split=True, dodge=False, inner=None, cut=0,
                   density_norm="width")
    plt.xlabel("Location")
    labels = np.unique(df["Location"])
    ticks = np.linspace(df["x_offset"].min(), df["x_offset"].max(), len(labels))
    #    plt.xticks(ticks=ticks, labels=labels)
    plt.xticks([], [])
    plt.savefig(path, dpi=600, bbox_inches="tight")


def plot(inference, model, x_train, x_test, y_train, y_test, scaler_x, scaler_y, path, losses=None):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat_train = inference(model, x_train, scaler_x, scaler_y)
    y_hat_test = inference(model, x_test, scaler_x, scaler_y)

    path.mkdir(parents=True, exist_ok=True)

    plot_predictions(y_train, y_test, y_hat_train, y_hat_test, path / "predictions.pdf")
    plot_wall_displacements(x_train, y_train, y_hat_train, path / "wall_displacement_train.png")
    plot_wall_displacements(x_test, y_test, y_hat_test, path / "wall_displacement_test.png")
    plot_wall_moments(x_train, y_train, y_hat_train, path / "wall_moments_train.png")
    plot_wall_moments(x_test, y_test, y_hat_test, path / "wall_moments_test.png")
    plot_violins(x_test, y_test, y_hat_test, path / "violins.png")

    if losses is not None: plot_losses(losses, path / "losses.png")

def plot_without_inference(x_train, x_test, y_train, y_test, y_test_hat, path, y_hat_train=None, losses=None):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    path.mkdir(parents=True, exist_ok=True)
    if y_hat_train is not None:
        plot_predictions(y_train, y_test, y_hat_train, y_test_hat, path / "predictions.pdf")
        plot_wall_displacements(x_train, y_train, y_hat_train, path / "wall_displacement_train.png")
        plot_wall_moments(x_train, y_train, y_hat_train, path / "wall_moments_train.png")
    plot_wall_displacements(x_test, y_test, y_test_hat, path / "wall_displacement_test.png")
    plot_wall_moments(x_test, y_test, y_test_hat, path / "wall_moments_test.png")
    plot_violins(x_test, y_test, y_test_hat, path / "violins.png")

    if losses is not None: plot_losses(losses, path / "losses.png")


def curvature(displacements, dLs):
    """
    Calculate the curvature of the displacements
    """

    displacements = displacements.copy() / 1_000

    #TODO: Fix moment estimation using simple double derivative calculation
    x = np.cumsum(dLs)
    dy2_dx2 = np.zeros_like(displacements)
    
    # Handle 2D input where each row contains displacements
    if displacements.ndim == 2:
        for i, disp_row in enumerate(displacements):
            spline = UnivariateSpline(x, disp_row, s=1e-8)
            dy2_dx2[i, :] = spline.derivative(n=2)(x)
    else:
        # Handle 3D input (original nested loop structure)
        for i, disp_chain in enumerate(displacements):
            for j, disp_chain_sample in enumerate(disp_chain):
                spline = UnivariateSpline(x, disp_chain_sample, s=1e-8)
                dy2_dx2[i, j] = spline.derivative(n=2)(x)

    return dy2_dx2


def moments(displacements, EI, wall_locs):
    """
    Calculate the moments of the displacements
    """

    # EI, _, wall_locs, monitoring_locs = wall_props

    _, keep_idx = np.unique(wall_locs, return_index=True)
    keep_idx = np.sort(keep_idx)
    wall_locs = wall_locs[keep_idx]

    dLs = np.diff(wall_locs)
    dLs = np.concatenate([[dLs[0]], (dLs[:-1] + dLs[1:])/2, [dLs[-1]]])
    dLs = np.abs(dLs)

    dy2_dx2 = curvature(displacements, dLs)

    # Reshape EI for proper broadcasting with dy2_dx2
    EI_reshaped = EI.reshape(-1, 1)  # Convert from (n_samples,) to (n_samples, 1)
    moments = - EI_reshaped * dy2_dx2  # Minus for proper sign in moment convention
    return moments

if __name__ == "__main__":
    pass

