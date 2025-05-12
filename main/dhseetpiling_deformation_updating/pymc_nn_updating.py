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
from surrogate_nn import NeuralNetwork
from bdax.flaxtraining import Trainer
from bdax.jaxpymc import Sampler

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class InferenceModel(Sampler):

    def set_inference_model(self):

        (obs,) = self.data

        corr_mat = np.load(r"results/corr_mat.npy")
        # jitter = 1e-4
        # corr_mat += np.eye(corr_mat.shape[0]) * jitter

        with pm.Model() as model:

            phi = pm.Normal("phi", mu=30, sigma=3)
            cohesion = pm.Normal("cohesion", mu=10, sigma=1)

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

        (obs,) = self.data

        if type == "posterior":
            x = idata.posterior.x.values

        if type == "prior":
            x = idata.prior.x.values

        y_hat = np.asarray(self.nn_model_fn(x)).reshape(x.shape[0], x.shape[1], -1)
        idata_pred = az.convert_to_inference_data({"prediction": y_hat}, group="prediction")

        return idata_pred

    def plot_model(self, path):

        y_data = self.data[0]
        x = np.arange(0, y_data.shape[-1])

        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 4))

        y_hat = self.idata.prior_prediction.prediction.values.reshape(-1, x.size)
        y_hat_mean = y_hat.mean(axis=0)
        y_hat_ci = np.quantile(y_hat, [0.05, 0.95], axis=0)
        y = self.idata.prior_predictive.z.values.reshape(-1, x.size)
        y_pi = np.quantile(y, [0.05, 0.95], axis=0)

        axs[0].scatter(y_data, x, c="k", marker="x", label="Data")
        axs[0].plot(y_hat_mean, x, c="b", label="Mean model")
        axs[0].fill_betweenx(x, y_hat_ci[0], y_hat_ci[1], color="b", alpha=0.6, label="90% CI")
        axs[0].fill_betweenx(x, y_pi[0], y_pi[1], color="b", alpha=0.3, label="90% PI")
        axs[0].set_xlabel("Displacement [mm]", fontsize=14)
        axs[0].set_ylabel("# of point along wall", fontsize=14)
        axs[0].invert_xaxis()
        axs[0].legend(fontsize=8, title="Prior")
        axs[0].grid()
        axs[1].set_title("Prior", fontsize=14)

        y_hat = self.idata.posterior_prediction.prediction.values.reshape(-1, x.size)
        y_hat_mean = y_hat.mean(axis=0)
        y_hat_ci = np.quantile(y_hat, [0.05, 0.95], axis=0)
        y = self.idata.posterior_predictive.z.values.reshape(-1, x.size)
        y_pi = np.quantile(y, [0.05, 0.95], axis=0)

        axs[1].scatter(y_data, x, c="k", marker="x", label="Data")
        axs[1].plot(y_hat_mean, x, c="r", label="Mean model")
        axs[1].fill_betweenx(x, y_hat_ci[0], y_hat_ci[1], color="r", alpha=0.6, label="90% CI")
        axs[1].fill_betweenx(x, y_pi[0], y_pi[1], color="r", alpha=0.3, label="90% PI")
        axs[1].set_xlabel("Displacement [mm]", fontsize=14)
        axs[1].set_ylabel("# of point along wall", fontsize=14)
        axs[1].invert_xaxis()
        axs[1].legend(fontsize=8, title="Posterior")
        axs[1].grid()
        axs[1].set_title("Posterior", fontsize=14)

        plt.close()

        pp = PdfPages(path)
        pp.savefig(fig)
        pp.close()


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
    trainer.load(r"results/nn_surrogate.pkl")

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

    points_idx = np.arange(0, 150, 10)  # idx of locations along wall where measuremnts are collected

    data_path = r"results/sample.json"
    data, true_params = load_data(data_path, use_noisy=True, idx=points_idx)
    data = (data[0][0],)

    trainer = create_trainer(points_idx)
    sampler = InferenceModel(trainer.model, trainer.params, data)
    sampler.sample(n_chains=4, n_samples=1_000, n_warmup=1_000, path=r"results/idata_nn.netcdf")
    # sampler.load_idata(path=r"results/idata_nn.netcdf")

    sampler.plot_trace(r"figures/nn_pymc_traceplot.pdf")
    sampler.plot_posterior(r"figures/nn_pymc_posteriorplot.pdf", ref_vals=true_params)
    sampler.plot_model(r"figures/nn_pymc_modelplot.pdf")

