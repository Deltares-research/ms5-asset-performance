import os
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import xarray as xr
import arviz as az
from pathlib import Path
import json
from typing import Optional

import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

import pymc as pm
from surrogate_mlp import NeuralNetwork
from bdax.flaxtraining import Trainer
from bdax.jaxpymc import Sampler

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class InferenceModel(Sampler):

    def set_inference_model(self):

        (obs, _) = self.data

        corr_mat = np.load(r"results/corr_mat.npy")
        # jitter = 1e-4
        # corr_mat += np.eye(corr_mat.shape[0]) * jitter

        with pm.Model() as model:

            phi = pm.Normal("phi", mu=35, sigma=12)
            cohesion = pm.Normal("cohesion", mu=15, sigma=5)

            x = pm.Deterministic("x", pm.math.stack((phi, cohesion)))
            y_hat = pm.Deterministic("y", self.custom_op(x[np.newaxis, :]).squeeze())
            sigma = pm.HalfNormal("sigma")

            z = pm.Normal("z", y_hat, sigma=sigma, observed=obs)

            # #Infer with correlated observations
            # sigmas = pt.ones(obs.size) * sigma
            # cov = pm.Deterministic("cov", pt.outer(sigmas, sigmas)*corr_mat)
            # chol = pm.Deterministic("chol", pt.linalg.cholesky(cov))
            # z = pm.MvNormal("z", y_hat, cov=cov, observed=obs)

        return model

    def predict(self, idata: az.InferenceData, type: str = "posterior") -> az.InferenceData:

        (obs, wall_props) = self.data

        if type == "posterior":
            x = idata.posterior.x.values

        if type == "prior":
            x = idata.prior.x.values

        # Use surrogate for all points along wall for smoother prediction plots.
        model = NeuralNetwork(150)
        trainer = Trainer(model)
        trainer.load(r"results/mlp_surrogate_150locs.pkl")
        y_hat = np.asarray(trainer.predict(x)).reshape(x.shape[0], x.shape[1], -1)

        # y_hat = np.asarray(self.nn_model_fn(x)).reshape(x.shape[0], x.shape[1], -1)
        moments = self._moments(y_hat, wall_props)
        fos = self._fos(y_hat, wall_props)
        data_pred = {"y_prediction": y_hat, "moments": moments, "fos": fos}
        idata_pred = az.convert_to_inference_data(data_pred, group="prediction")

        return idata_pred

    def _curvature(self, displacements, dLs):
        padding = (
            (0, 0),  # no padding on axis 0
            (0, 0),  # no padding on axis 1
            (2, 2),  # pad two columns on axis 2 -> double derivative -> moments have the shape of displacements
        )
        displacements_padded = np.pad(displacements, pad_width=padding, mode='edge')

        dy2_dx2 = np.diff(displacements_padded, n=2, axis=-1)[..., 1:-1] / dLs ** 2

        return dy2_dx2

    def _moments(self, displacements, wall_props):

        EI, _, dLs = wall_props

        dy2_dx2 = self._curvature(displacements, dLs)
        dy2_dx2 /= 1_000  # [mm] -> [m]

        moments = - EI * dy2_dx2  # Minus for proper sign in moment convention

        return moments

    def _fos(self, displacements, wall_props):
        
        _, moment_cap, _ = wall_props
        
        moments = self._moments(displacements, wall_props)
        
        fos = moment_cap / moments.max(axis=-1)

        return fos

    def plot_model(self, path):

        (y_data, wall_props) = self.data
        _, moment_cap, dLs = wall_props
        depths = np.cumsum(dLs)

        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))

        y_hat = self.idata.prior_prediction.y_prediction.values
        y_hat = y_hat.reshape(-1, y_hat.shape[-1])
        y_hat_mean = y_hat.mean(axis=0)
        y_hat_ci = np.quantile(y_hat, [0.05, 0.95], axis=0)
        y = self.idata.prior_predictive.z.values
        y = y.reshape(-1, y.shape[-1])
        y_pi = np.quantile(y, [0.05, 0.95], axis=0)

        x_data = np.linspace(0, depths.max(), y_data.shape[-1])
        x_hat = depths

        axs[0].fill_betweenx(x_data, y_pi[0], y_pi[1], color="b", alpha=0.3, label="90% PI")
        axs[0].fill_betweenx(x_hat, y_hat_ci[0], y_hat_ci[1], color="b", alpha=0.6, label="90% CI")
        axs[0].plot(y_hat_mean, x_hat, c="b", label="Mean model")
        axs[0].scatter(y_data, x_data, c="k", marker="x", label="Data")
        axs[0].set_xlabel("Displacement [mm]", fontsize=14)
        axs[0].set_ylabel("# of point along wall", fontsize=14)
        axs[0].invert_yaxis()
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(handles[::-1], labels[::-1], fontsize=10)
        axs[0].grid()
        axs[0].set_title("Prior", fontsize=14)

        y_hat = self.idata.posterior_prediction.y_prediction.values
        y_hat = y_hat.reshape(-1, y_hat.shape[-1])
        y_hat_mean = y_hat.mean(axis=0)
        y_hat_ci = np.quantile(y_hat, [0.05, 0.95], axis=0)
        y = self.idata.posterior_predictive.z.values
        y = y.reshape(-1, y.shape[-1])
        y_pi = np.quantile(y, [0.05, 0.95], axis=0)

        x_data = np.linspace(0, depths.max(), y_data.shape[-1])
        x_hat = depths

        axs[1].fill_betweenx(x_data, y_pi[0], y_pi[1], color="r", alpha=0.3, label="90% PI")
        axs[1].fill_betweenx(x_hat, y_hat_ci[0], y_hat_ci[1], color="r", alpha=0.6, label="90% CI")
        axs[1].plot(y_hat_mean, x_hat, c="r", label="Mean model")
        axs[1].scatter(y_data, x_data, c="k", marker="x", label="Data")
        axs[1].set_xlabel("Displacement [mm]", fontsize=14)
        handles, labels = axs[1].get_legend_handles_labels()
        axs[1].legend(handles[::-1], labels[::-1], fontsize=10)
        axs[1].grid()
        axs[1].set_title("Posterior", fontsize=14)

        plt.tight_layout()
        plt.close()

        path = Path(Path(path).as_posix())
        fig.savefig(path/"model.png")

        # pp = PdfPages(path/"model.pdf")
        # pp.savefig(fig)
        # pp.close()

    def plot_moments(self, path):

        (y_data, wall_props) = self.data
        _, moment_cap, dLs = wall_props
        depths = np.cumsum(dLs)

        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))

        moments = self.idata.prior_prediction.moments.values
        moments = moments.reshape(-1, moments.shape[-1])
        moments_mean = moments.mean(axis=0)
        moments_ci = np.quantile(moments, [0.05, 0.95], axis=0)

        fos = self.idata.prior_prediction.fos.values
        pf = np.mean(fos<1)

        axs[0].axvline(moment_cap, c="k", linestyle="--", label="Capacity")
        axs[0].fill_betweenx(depths, moments_ci[0], moments_ci[1], color="b", alpha=0.6, label="90% CI")
        axs[0].plot(moments_mean, depths, c="b", label="Mean model")
        axs[0].set_xlabel("Moment [kNm]", fontsize=14)
        axs[0].set_ylabel("# of point along wall", fontsize=14)
        axs[0].invert_yaxis()
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(handles[::-1], labels[::-1], fontsize=10)
        axs[0].grid()
        axs[0].set_title("Prior\n"+"${P}_{f}$="+f"{pf:.2e}", fontsize=14)

        moments = self.idata.posterior_prediction.moments.values
        moments = moments.reshape(-1, moments.shape[-1])
        moments_mean = moments.mean(axis=0)
        moments_ci = np.quantile(moments, [0.05, 0.95], axis=0)

        fos = self.idata.posterior_prediction.fos.values
        pf = np.mean(fos<1)

        axs[1].axvline(moment_cap, c="k", linestyle="--", label="Capacity")
        axs[1].fill_betweenx(depths, moments_ci[0], moments_ci[1], color="r", alpha=0.6, label="90% CI")
        axs[1].plot(moments_mean, depths, c="r", label="Mean model")
        axs[1].set_xlabel("Moment [kNm]", fontsize=14)
        handles, labels = axs[1].get_legend_handles_labels()
        axs[1].legend(handles[::-1], labels[::-1], fontsize=10)
        axs[1].grid()
        axs[1].set_title("Posterior\n"+"${P}_{f}$="+f"{pf:.2e}", fontsize=14)

        plt.tight_layout()
        plt.close()

        path = Path(Path(path).as_posix())
        fig.savefig(path/"moments.png")

        # pp = PdfPages(path/"moments.png")
        # pp.savefig(fig)
        # pp.close()

    def plot_fos(self, path):
        
        fos_prior = self.idata.prior_prediction.fos.values.flatten()
        pf_prior = np.mean(fos_prior<1)

        fos_posterior = self.idata.posterior_prediction.fos.values.flatten()
        pf_posterior = np.mean(fos_posterior<1)
        
        fig = plt.figure()

        label = "Prior\n" + "${P}_{f}$ = " + f"{pf_prior:.2e}"
        mean = fos_prior.mean()
        ci = np.quantile(fos_prior, [0.025, 0.975])
        counts, _, _ = plt.hist(fos_prior, bins=100, density=True, color="b", alpha=0.6, label=label)
        plt.errorbar(mean, 0.55*max(counts), xerr=[[mean - ci.min()], [ci.max() - mean]], fmt='o', color='b', capsize=5)

        label = "Posterior\n" + "${P}_{f}$ = " + f"{pf_posterior:.2e}"
        mean = fos_posterior.mean()
        ci = np.quantile(fos_posterior, [0.025, 0.975])
        counts, _, _ = plt.hist(fos_posterior, bins=100, density=True, color="r", alpha=0.6, label=label)
        plt.errorbar(mean, 0.55*max(counts), xerr=[[mean - ci.min()], [ci.max() - mean]], fmt='o', color='r', capsize=5)

        plt.xlabel("FoS [-]", fontsize=14)
        plt.ylabel("Density [-]", fontsize=14)
        plt.legend(fontsize=10)

        path = Path(Path(path).as_posix())
        fig.savefig(path/"fos.png")


def create_trainer(idx: Optional[int] = None):

    data_path = r"results/sample_1000_unpooled.json"
    data_path = Path(Path(data_path).as_posix())
    with open(data_path, "r") as f: data = json.load(f)
    y = data["displacement"]
    y = [[item if item is not None else np.nan for item in row] for row in y]
    y = np.asarray(y)

    if idx is not None: y = y[..., idx]

    model = NeuralNetwork(y.shape[-1])
    trainer = Trainer(model)
    trainer.load(r"results/mlp_surrogate.pkl")

    return trainer


def load_data(path: str | Path, use_noisy: bool = True, idx: Optional[int] = None):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    with open(path, "r") as f: data = json.load(f)

    if use_noisy:
        obs = np.asarray(data["displacement_noisy"]).squeeze()
    else:
        obs = np.asarray(data["displacement"]).squeeze()

    if idx is not None: obs = obs[..., idx]

    true_params = {
        "phi": np.unique(data["Klei_soilphi"]),
        "cohesion": np.unique(data["Klei_soilcohesion"])
    }

    return (obs,), true_params


if __name__ == "__main__":

    points_idx = np.arange(0, 150, 10)  # idx of locations along wall where measurements are collected
    # wall_props = (1e+4, 15., np.ones(points_idx.size) * 0.1)
    wall_props = (1e+4, 15., np.ones(150) * 0.1)

    data_path = r"results/sample.json"
    data, true_params = load_data(data_path, use_noisy=True, idx=points_idx)
    data = (
        data[0][0],  # Use monitoring of one wall cross-section
        wall_props
    )

    trainer = create_trainer(points_idx)
    sampler = InferenceModel(trainer.model, trainer.params, data)
    # sampler.sample(n_chains=4, n_samples=1_000, n_warmup=1_000, target_accept=0.95, path=r"results/idata_mlp.netcdf")
    sampler.load_idata(path=r"results/idata_mlp.netcdf")
    # sampler.predict(sampler.idata, "posterior")

    plot_path = Path(Path(r"figures/pymc_mlp").as_posix())
    plot_path.mkdir(parents=True, exist_ok=True)

    sampler.plot_trace(plot_path/"trace.pdf")
    sampler.plot_posterior(plot_path/"posterior.pdf", ref_vals=true_params)
    sampler.plot_model(plot_path)
    sampler.plot_moments(plot_path)
    sampler.plot_fos(plot_path)

