import os
import numpy as np
import pickle
import joblib
import arviz as az
import torch
import xarray as xr
from pathlib import Path
import json
from typing import Optional
from scipy.signal import savgol_filter
from numpy.polynomial.chebyshev import chebvander
from main.case_study_2025.train.srg.chebysev_train import Chebysev, MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class FoSCalculator:

    def __init__(self, n_points, wall_props, model_path, scaler_x_path, scaler_y_path, posterior_path=None, device=None):

        x = np.linspace(0, 10, n_points)
        x = np.cumsum(x)

        if not isinstance(model_path, Path): model_path = Path(Path(model_path).as_posix())
        self.model = Chebysev(
            input_dim=11,
            hidden_dims=[1024, 512, 256, 128, 64, 32],
            x=x,
            degree=10
        )

        if device is not None:
            self.model = self.model.to(device)

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        if not isinstance(scaler_x_path, Path): scaler_x_path = Path(Path(scaler_x_path).as_posix())
        self.scaler_x = joblib.load(scaler_x_path)

        if not isinstance(scaler_y_path, Path): scaler_y_path = Path(Path(scaler_y_path).as_posix())
        self.scaler_y = joblib.load(scaler_y_path)

        if posterior_path is not None:
            if not isinstance(posterior_path, Path): posterior_path = Path(Path(posterior_path).as_posix())
            self.idata = az.from_netcdf(posterior_path)

        self.wall_props = wall_props

    def inference(self, X):
        if isinstance(X, xr.DataArray): X = X.values
        X_scaled = self.scaler_x.transform(X)
        X_scaled_tensor = torch.from_numpy(X_scaled).float()
        y_scaled = self.model(X_scaled_tensor)
        y_scaled = y_scaled.detach().numpy()
        y = self.scaler_y.inverse_transform(y_scaled)
        return y

    def displacement(self, x):

        with torch.no_grad():
            displacements = self.model(x, return_coeffs=False)
        curvatures = curvatures.detach().numpy()

        return displacements

    def moments(self, x):

        EI, _, wall_locs, monitoring_locs = self.wall_props

        with torch.no_grad():
            coeffs = self.model(x, return_coeffs=True)
        curvatures = coeffs @ self.model.basis_der
        curvatures = curvatures.cpu().numpy()
        curvatures /= (1_000)  # Convert [mm/m^2] displacements to [1/m].
        moments = - EI * curvatures  # Minus for proper sign in moment convention

        return moments

    def fos(self, x):

        _, moment_cap, _, _ = self.wall_props

        moments = self.moments(x)

        fos = moment_cap / moments.max(axis=-1)

        return fos

    def plot_moments(self, displacements, moments, path):

        _, moment_cap, wall_locs, monitoring_locs = self.wall_props
        # _, keep_idx = np.unique(wall_locs, return_index=True)
        # keep_idx = np.sort(keep_idx)
        # wall_locs = wall_locs[keep_idx]

        x = wall_locs

        fig, axs = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(12, 6))

        displacements = displacements.reshape(-1, displacements.shape[-1])
        displacements_mean = displacements.mean(axis=0)
        displacements_ci = np.quantile(displacements, [0.05, 0.95], axis=0)

        axs[0].fill_betweenx(x, displacements_ci[0], displacements_ci[1], color="b", alpha=0.6, label="90% CI")
        axs[0].plot(displacements_mean, x, c="b", label="Mean model")
        axs[0].set_xlabel("Displacement [mm]", fontsize=14)
        axs[0].set_ylabel("Depth along wall [m]", fontsize=14)
        axs[0].invert_yaxis()
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(handles[::-1], labels[::-1], fontsize=10)
        axs[0].grid()

        moments = moments.reshape(-1, moments.shape[-1])
        moments_mean = moments.mean(axis=0)
        moments_ci = np.quantile(moments, [0.05, 0.95], axis=0)

        # fos = self.idata.posterior_prediction.fos.values
        fos = moments.max(axis=1) / moment_cap
        pf = np.mean(fos < 1)

        axs[1].axvline(moment_cap, c="k", linestyle="--", label="Capacity")
        axs[1].fill_betweenx(x, moments_ci[0], moments_ci[1], color="r", alpha=0.6, label="90% CI")
        axs[1].plot(moments_mean, x, c="r", label="Mean model")
        axs[1].set_xlabel("Moment [kNm]", fontsize=14)
        handles, labels = axs[1].get_legend_handles_labels()
        axs[1].legend(handles[::-1], labels[::-1], fontsize=10)
        axs[1].grid()
        axs[1].set_title("Posterior\n" + "${P}_{f}$=" + f"{pf:.2e}", fontsize=14)

        plt.tight_layout()
        plt.close()

        path = Path(Path(path).as_posix())
        fig.savefig(path / "moments.png")


if __name__ == "__main__":

    pass

