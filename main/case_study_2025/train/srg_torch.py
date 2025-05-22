import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from srg_utils import load_data

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Tuple, Optional
from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import joblib
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train(model, X, y, epochs, lr):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_torch = torch.tensor(X, dtype=torch.float32, device=device)
    y_torch = torch.tensor(y, dtype=torch.float32, device=device)

    epoch_losses = []
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        optimizer.zero_grad()
        preds = model(x_torch)
        loss = criterion(preds, y_torch)
        loss.backward()
        optimizer.step()
        epoch_loss = loss.item()
        epoch_losses.append(epoch_loss)

    epoch_losses = np.asarray(epoch_losses)

    return model, epoch_losses


def inference(model, x, scaler_x, scaler_y, device=None):
    if device is None:
        device = next(model.parameters()).device
    x = scaler_x.transform(x)
    x_torch = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_hat = model(x_torch)
    y_hat = y_hat.detach().cpu().numpy()
    y_hat = scaler_y.inverse_transform(y_hat)
    return y_hat


def plot_predictions(model, x_train, x_test, y_train, y_test, scaler_x, scaler_y, path):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat_train = inference(model, x_train, scaler_x, scaler_y)
    y_hat_test = inference(model, x_test, scaler_x, scaler_y)

    figs = []
    zipped = zip(y_train.T, y_hat_train.T, y_test.T, y_hat_test.T)
    for i_point, (y_t_train, y_p_train, y_t_test, y_p_test) in enumerate(zipped):
        fig = plt.figure()
        fig.suptitle(f"Point #{i_point+1:d} along wall", fontsize=14)
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


def plot_wall(model, x, y, scaler_x, scaler_y, path):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat = inference(model, x, scaler_x, scaler_y)

    fig = plt.figure(figsize=(8, 8))
    plt.axvline(0, color="k", linewidth=2, label="Starting position")
    locs = np.arange(1, y.shape[-1]+1)
    ci = np.quantile(y, (0.025, 0.975), axis=0)
    plt.fill_betweenx(locs, ci[0], ci[1], color="b", alpha=0.3, label="Observation 95% CI")
    plt.plot(y.mean(axis=0), locs, c="b", label="Mean observations")
    for i, y_hat_depth in enumerate(y_hat.T):
        loc = i + 1
        label = "Prediction 95% CI" if i == y_hat.shape[-1]-1 else None
        mean = y_hat_depth.mean()
        ci = np.quantile(y_hat_depth, [0.025, 0.975])
        xerr = np.abs(mean-ci)
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


def plot_wall_error(model, x, y, scaler_x, scaler_y, path):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat = inference(model, x, scaler_x, scaler_y)
    residuals = y - y_hat
    st_residuals = residuals / np.abs(y+1e-8) * 100

    fig = plt.figure()
    for r in st_residuals:
        plt.plot(r, np.arange(1, y_hat.shape[1]+1), c='b', alpha=0.3)
    plt.xlabel('Error [%]', fontsize=12)
    plt.ylabel('Point # along wall', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.close()

    if path.suffix == ".png":
        fig.savefig(path)
    else:
        pp = PdfPages(path)
        pp.savefig(fig)
        pp.close()


def plot_variables(model, x, y, scaler_x, scaler_y, path):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat = inference(model, x, scaler_x, scaler_y)
    residuals = y - y_hat
    st_residuals = residuals / np.abs(y+1e-8) * 100

    figs = []
    for i_point, r in enumerate(st_residuals.T):
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
        fig.suptitle(f"Point #{i_point+1:d} along wall", fontsize=14)
        sc = axs[0].scatter(x[:, 0], r, c=x[:, 1])
        cbar = fig.colorbar(sc, ax=axs[0])
        cbar.set_label("Cohesion [kPa]", fontsize=10, labelpad=10)
        axs[0].set_xlabel('Phi [deg]', fontsize=12)
        axs[0].set_ylabel('Error [%]', fontsize=12)
        sc = axs[1].scatter(x[:, 1], r, c=x[:, 0])
        cbar = fig.colorbar(sc, ax=axs[1])
        cbar.set_label("Phi [deg]", fontsize=10, labelpad=10)
        axs[1].set_xlabel('Cohesion [kPa]', fontsize=12)
        axs[1].set_ylabel('Error [%]', fontsize=12)
        axs[0].set_ylim(residuals.min(), residuals.max())
        axs[1].set_ylim(residuals.min(), residuals.max())
        axs[0].grid()
        axs[1].grid()
        plt.close()
        figs.append(fig)
    pp = PdfPages(path)
    [pp.savefig(fig) for fig in figs]
    pp.close()


def plot_losses(losses, path):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    fig = plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, losses.size+1), losses, c="b")
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss [${mm}^{2}$]', fontsize=12)
    plt.yscale("log")
    plt.grid()
    plt.close()

    fig.savefig(path)


def plot_violins(model, x, y, scaler_x, scaler_y, path):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat = inference(model, x, scaler_x, scaler_y)

    df = pd.DataFrame(data=np.concatenate((y.flatten(), y_hat.flatten())), columns=["Displacement [mm]"])
    df["Type"] = ["Observation"] * y.size + ["Prediction"] * y_hat.size
    df["Type"] = pd.Categorical(df["Type"], categories=["Observation", "Prediction"], ordered=True)
    df["Location"] = np.repeat(np.tile(np.arange(1, y.shape[-1]+1), y.shape[0]), 2)
    
    df["x_offset"] = df["Location"].copy().astype(float)
    df.loc[df["Type"] == "Prediction", "x_offset"] += 0.2
    df.loc[df["Type"] == "Observation", "x_offset"] -= 0.2
    
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=df, x="x_offset", y="Displacement [mm]", hue="Type", split=True, dodge=False, inner=None, cut=0, density_norm="width")
    plt.xlabel("Location")
    labels = np.unique(df["Location"])
    ticks = np.linspace(df["x_offset"].min(), df["x_offset"].max(), len(labels))
#    plt.xticks(ticks=ticks, labels=labels)
    plt.xticks([], [])
    plt.savefig(path, dpi=600, bbox_inches="tight")


def plot(model, x_train, x_test, y_train, y_test, scaler_x, scaler_y, path, losses=None):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    path.mkdir(parents=True, exist_ok=True)

    plot_predictions(model, x_train, x_test, y_train, y_test, scaler_x, scaler_y, path/"predictions.pdf")
    plot_wall(model, x_train, y_train, scaler_x, scaler_y, path/"wall.png")
    plot_wall(model, x_test, y_test, scaler_x, scaler_y, path/"wall_test.png")
#    plot_wall_error(model, x_train, y_train, scaler_x, scaler_y, path/"wall_error.png")
#    plot_variables(model, x_train, y_train, scaler_x, scaler_y, path/"variables.pdf")
    plot_violins(model, x_test, y_test, scaler_x, scaler_y, path/"violins.png")

    if losses is not None: plot_losses(losses, path/"losses.png")


if __name__ == "__main__":

    # path = os.environ["SRG_DATA_PATH"]
    data_path = r"../data/srg_data_20250520_094244.csv"

    path = Path(Path(data_path).as_posix())

    X, y = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) backend")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    model = MLP(
        input_dim=X.shape[-1],
        hidden_dims=[512, 256, 128, 64, 32],
        output_dim=y.shape[-1]
    ).to(device)

    torch.manual_seed(42)

    retrain = True
    # retrain = False

    if retrain:

        model, epoch_losses = train(model, X_train_scaled, y_train_scaled, lr=1e-4, epochs=10_000)

        torch.save(model.state_dict(), r"results/torch_weights.pth")
        joblib.dump(scaler_x, r"results/scaler_x.joblib")
        joblib.dump(scaler_y, r"results/scaler_y.joblib")

    else:

        model.load_state_dict(torch.load(r"results/torch_weights.pth", map_location=device))
        model.to(device)
        scaler_x = joblib.load(r"results/scaler_x.joblib")
        scaler_y = joblib.load(r"results/scaler_y.joblib")

        epoch_losses = None

    model.eval()

    y_hat = inference(model, X_test, scaler_x, scaler_y)
    rmse = mean_squared_error(y_hat, y_test)

    plot(model, X_train, X_test, y_train, y_test, scaler_x, scaler_y, r'figures/srg_torch', epoch_losses)

