import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pathlib import Path


def load_data(path, full_profile=False):

    df = pd.read_csv(path)

    X_cols = [col for col in df.columns if col.split("_")[0] != "disp" and col != "index"]
    X_cols = [col for col in X_cols if "soilcurko2" not in col and "soilcurko3" not in col]
    X = df[X_cols].values

    idx_locs = list(range(1, 151, 10))
    if full_profile:
        y_cols = [col for col in df.columns if col.split("_")[0] == "disp"]
    else:
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


def plot_predictions(inference, model, x_train, x_test, y_train, y_test, scaler_x, scaler_y, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat_train = inference(model, x_train, scaler_x, scaler_y)
    y_hat_test = inference(model, x_test, scaler_x, scaler_y)

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


def plot_wall(inference, model, x, y, scaler_x, scaler_y, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat = inference(model, x, scaler_x, scaler_y)

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
    plt.xlabel('Displacement [mm]', fontsize=12)
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


def plot_violins(inference, model, x, y, scaler_x, scaler_y, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat = inference(model, x, scaler_x, scaler_y)

    df = pd.DataFrame(data=np.concatenate((y.flatten(), y_hat.flatten())), columns=["Displacement [mm]"])
    df["Type"] = ["Observation"] * y.size + ["Prediction"] * y_hat.size
    df["Type"] = pd.Categorical(df["Type"], categories=["Observation", "Prediction"], ordered=True)
    df["Location"] = np.repeat(np.tile(np.arange(1, y.shape[-1] + 1), y.shape[0]), 2)

    df["x_offset"] = df["Location"].copy().astype(float)
    df.loc[df["Type"] == "Prediction", "x_offset"] += 0.2
    df.loc[df["Type"] == "Observation", "x_offset"] -= 0.2

    plt.figure(figsize=(14, 8))
    sns.violinplot(data=df, x="x_offset", y="Displacement [mm]", hue="Type", split=True, dodge=False, inner=None, cut=0,
                   density_norm="width")
    plt.xlabel("Location")
    labels = np.unique(df["Location"])
    ticks = np.linspace(df["x_offset"].min(), df["x_offset"].max(), len(labels))
    #    plt.xticks(ticks=ticks, labels=labels)
    plt.xticks([], [])
    plt.savefig(path, dpi=600, bbox_inches="tight")


def plot(inference, model, x_train, x_test, y_train, y_test, scaler_x, scaler_y, path, losses=None):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    path.mkdir(parents=True, exist_ok=True)

    plot_predictions(inference, model, x_train, x_test, y_train, y_test, scaler_x, scaler_y, path / "predictions.pdf")
    plot_wall(inference, model, x_train, y_train, scaler_x, scaler_y, path / "wall.png")
    plot_wall(inference, model, x_test, y_test, scaler_x, scaler_y, path / "wall_test.png")
    plot_violins(inference, model, x_test, y_test, scaler_x, scaler_y, path / "violins.png")

    if losses is not None: plot_losses(losses, path / "losses.png")


if __name__ == "__main__":

    pass

