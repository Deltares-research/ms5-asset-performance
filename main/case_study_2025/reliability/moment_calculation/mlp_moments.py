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
from scipy.interpolate import UnivariateSpline
from main.case_study_2025.train.srg.mlp_train import MLP, MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class FoSCalculator:

    def __init__(self, n_points, wall_props, model_path, posterior_path, scaler_x_path, scaler_y_path):

        if not isinstance(model_path, Path): model_path = Path(Path(model_path).as_posix())
        self.model = MLP(
            input_dim=11,
            hidden_dims=[1024, 512, 256, 128, 64, 32],
            output_dim=n_points
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        if not isinstance(scaler_x_path, Path): scaler_x_path = Path(Path(scaler_x_path).as_posix())
        self.scaler_x = joblib.load(scaler_x_path)

        if not isinstance(scaler_y_path, Path): scaler_y_path = Path(Path(scaler_y_path).as_posix())
        self.scaler_y = joblib.load(scaler_y_path)

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

    def _curvature(self, displacements, dLs):

        displacements = displacements.copy() / 1_000

        padding = (
            (0, 0),  # no padding on axis 0
            (2, 2),  # pad two columns on axis 1 -> double derivative -> moments have the shape of displacements
        )
        displacements_padded = np.pad(displacements.copy(), pad_width=padding, mode='edge')
        dy2_dx2 = np.diff(displacements_padded, n=2, axis=-1)[..., 1:-1] / (dLs ** 2 + 1e-6)

        # # TODO: Fix moment estimation using simple double derivative calculation
        # x = np.cumsum(dLs)
        # dy2_dx2 = np.zeros_like(displacements)
        # for i, disp_chain in enumerate(displacements):
        #     for j, disp_chain_sample in enumerate(disp_chain):
        #         spline = UnivariateSpline(x, disp_chain_sample, s=1e-8)
        #         dy2_dx2[i, j] = spline.derivative(n=2)(x)

        return dy2_dx2

    def moments(self, displacements):

        EI, _, wall_locs, monitoring_locs = self.wall_props

        _, keep_idx = np.unique(wall_locs, return_index=True)
        keep_idx = np.sort(keep_idx)
        wall_locs = wall_locs[keep_idx]

        dLs = np.abs(np.diff(wall_locs))
        dLs = np.append(dLs[0], dLs)

        dy2_dx2 = self._curvature(displacements, dLs)

        moments = - EI * dy2_dx2  # Minus for proper sign in moment convention

        return moments

    def _fos(self, displacements, wall_props):

        _, moment_cap, _, _ = wall_props

        moments = self.moments(displacements, wall_props)

        fos = moment_cap / moments.max(axis=-1)

        return fos

    def plot_moments(self, moments, path):

        _, moment_cap, wall_locs, monitoring_locs = self.wall_props
        # _, keep_idx = np.unique(wall_locs, return_index=True)
        # keep_idx = np.sort(keep_idx)
        # wall_locs = wall_locs[keep_idx]

        x = wall_locs

        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))

        # moments = self.idata.prior_prediction.moments.values
        # moments = moments.reshape(-1, moments.shape[-1])
        # moments_mean = moments.mean(axis=0)
        # moments_ci = np.quantile(moments, [0.05, 0.95], axis=0)
        #
        # fos = self.idata.prior_prediction.fos.values
        # pf = np.mean(fos < 1)
        #
        # axs[0].axvline(moment_cap, c="k", linestyle="--", label="Capacity")
        # axs[0].fill_betweenx(x, moments_ci[0], moments_ci[1], color="b", alpha=0.6, label="90% CI")
        # axs[0].plot(moments_mean, x, c="b", label="Mean model")
        # axs[0].set_xlabel("Moment [kNm]", fontsize=14)
        # axs[0].set_ylabel("Depth along wall [m]", fontsize=14)
        # # axs[0].invert_yaxis()
        # handles, labels = axs[0].get_legend_handles_labels()
        # axs[0].legend(handles[::-1], labels[::-1], fontsize=10)
        # axs[0].grid()
        # axs[0].set_title("Prior\n" + "${P}_{f}$=" + f"{pf:.2e}", fontsize=14)

        # moments = self.idata.posterior_prediction.moments.values
        moments = moments.reshape(-1, moments.shape[-1])
        moments_mean = moments.mean(axis=0)
        moments_ci = np.quantile(moments, [0.05, 0.95], axis=0)

        # fos = self.idata.posterior_prediction.fos.values
        fos = np.ones(moments.shape[0])
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

    # def plot_moments(self, path):
    #
    #     (y_data, wall_props) = self.data
    #     _, moment_cap, wall_locs, monitoring_locs = wall_props
    #     _, keep_idx = np.unique(wall_locs, return_index=True)
    #     keep_idx = np.sort(keep_idx)
    #     wall_locs = wall_locs[keep_idx]
    #
    #     x = wall_locs
    #
    #     fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
    #
    #     moments = self.idata.prior_prediction.moments.values
    #     moments = moments.reshape(-1, moments.shape[-1])
    #     moments_mean = moments.mean(axis=0)
    #     moments_ci = np.quantile(moments, [0.05, 0.95], axis=0)
    #
    #     fos = self.idata.prior_prediction.fos.values
    #     pf = np.mean(fos < 1)
    #
    #     axs[0].axvline(moment_cap, c="k", linestyle="--", label="Capacity")
    #     axs[0].fill_betweenx(x, moments_ci[0], moments_ci[1], color="b", alpha=0.6, label="90% CI")
    #     axs[0].plot(moments_mean, x, c="b", label="Mean model")
    #     axs[0].set_xlabel("Moment [kNm]", fontsize=14)
    #     axs[0].set_ylabel("Depth along wall [m]", fontsize=14)
    #     # axs[0].invert_yaxis()
    #     handles, labels = axs[0].get_legend_handles_labels()
    #     axs[0].legend(handles[::-1], labels[::-1], fontsize=10)
    #     axs[0].grid()
    #     axs[0].set_title("Prior\n" + "${P}_{f}$=" + f"{pf:.2e}", fontsize=14)
    #
    #     moments = self.idata.posterior_prediction.moments.values
    #     moments = moments.reshape(-1, moments.shape[-1])
    #     moments_mean = moments.mean(axis=0)
    #     moments_ci = np.quantile(moments, [0.05, 0.95], axis=0)
    #
    #     fos = self.idata.posterior_prediction.fos.values
    #     pf = np.mean(fos < 1)
    #
    #     axs[1].axvline(moment_cap, c="k", linestyle="--", label="Capacity")
    #     axs[1].fill_betweenx(x, moments_ci[0], moments_ci[1], color="r", alpha=0.6, label="90% CI")
    #     axs[1].plot(moments_mean, x, c="r", label="Mean model")
    #     axs[1].set_xlabel("Moment [kNm]", fontsize=14)
    #     handles, labels = axs[1].get_legend_handles_labels()
    #     axs[1].legend(handles[::-1], labels[::-1], fontsize=10)
    #     axs[1].grid()
    #     axs[1].set_title("Posterior\n" + "${P}_{f}$=" + f"{pf:.2e}", fontsize=14)
    #
    #     plt.tight_layout()
    #     plt.close()
    #
    #     path = Path(Path(path).as_posix())
    #     fig.savefig(path / "moments.png")


if __name__ == "__main__":

    pass

