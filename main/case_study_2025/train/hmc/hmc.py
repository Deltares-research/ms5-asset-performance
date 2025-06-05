import os
import json
import pymc as pm
import arviz as az
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
from pathlib import Path
from main.case_study_2025.train.srg.mlp_train import MLP, MinMaxScaler
import matplotlib.pyplot as plt


class TorchModelLogLike(Op):
    itypes = [pt.dvector]  # Input type (1D vector)
    otypes = [pt.dscalar]  # Output type (scalar log-likelihood)

    def __init__(self, torch_model, y_obs):
        self.torch_model = torch_model
        self.y_obs = torch.tensor(y_obs, dtype=torch.float32)

    def perform(self, node, inputs, outputs):
        x_input, = inputs
        x_tensor = torch.tensor(x_input, dtype=torch.float32, requires_grad=True)
        y_pred = self.torch_model(x_tensor)
        # Assume Gaussian likelihood
        logp = -0.5 * torch.sum((y_pred - self.y_obs)**2 / 0.1**2 + torch.log(2 * torch.pi * 0.1**2))
        outputs[0][0] = np.array(logp.item(), dtype=np.float64)

    def grad(self, inputs, output_grads):
        x_input, = inputs
        x_array = np.array(x_input, dtype=np.float32, ndmin=1).squeeze()
        x_tensor = torch.tensor(x_array, requires_grad=True)
        y_pred = self.torch_model(x_tensor)
        logp = -0.5 * torch.sum((y_pred - self.y_obs)**2 / 0.1**2 + torch.log(2 * torch.pi * 0.1**2))
        logp.backward()
        grad = x_tensor.grad.detach().numpy().astype(np.float64)
        return [output_grads[0] * pt.as_tensor_variable(grad)]


def extract_mlp_weights(model):
    weights = []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            weights.append((W, b))
    return weights


def mlp_forward_pt(x, weights):
    h = x
    for i, (W, b) in enumerate(weights):
        W_pt = pt.constant(W.astype("float32"))
        b_pt = pt.constant(b.astype("float32"))
        h = pt.dot(h, W_pt.T) + b_pt
        if i < len(weights) - 1:
            h = pt.switch(h > 0, h, 0)
    return h


if __name__  == "__main__":

    SCRIPT_DIR = Path(__file__).resolve().parent

    torch_path = SCRIPT_DIR / "../results/srg/mlp/lr_1.0e-06_epochs_100000/torch_weights.pth"
    param_dist_path = SCRIPT_DIR / "../../data/parameter_distributions.csv"
    obs_path = SCRIPT_DIR / "../../data/setting"
    result_path = SCRIPT_DIR / "../../results/hmc"

    parameter_dists = pd.read_csv(param_dist_path)
    parameter_dists = parameter_dists.set_index(parameter_dists["parameter"], drop=True)
    parameter_dists = parameter_dists.to_dict(orient="index")

    path = Path(obs_path)
    with open(path/"case_study.json", "r") as f: data = json.load(f)
    true_params = data['12']["true_params"]
    y_obs = np.asarray(data['12']["deformations"])

    num_cores = os.cpu_count()
    print(f"-------------- USING {num_cores} CORES --------------")

    torch_model = MLP(
        input_dim=11,
        hidden_dims=[1024, 512, 256, 128, 64, 32],
        output_dim=15
    )
    state_dict = torch.load(torch_path, map_location=torch.device('cpu'))
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    weights_np = extract_mlp_weights(torch_model)

    input_dim = weights_np[0][0].shape[1]  # Infer from weight shapes
    loglike_op = TorchModelLogLike(torch_model, y_obs)

    with pm.Model() as pymc_model:

        # Klei_soilphi = pm.Normal("Klei_soilphi", mu=parameter_dists["Klei_soilphi"]["mean"], sigma=parameter_dists["Klei_soilphi"]["std"])
        # Klei_soilcohesion = pm.Normal("Klei_soilcohesion", mu=parameter_dists["Klei_soilcohesion"]["mean"], sigma=parameter_dists["Klei_soilcohesion"]["std"])
        # Klei_soilcurkb1 = pm.Normal("Klei_soilcurkb1", mu=parameter_dists["Klei_soilcurkb1"]["mean"], sigma=parameter_dists["Klei_soilcurkb1"]["std"])
        # Zand_soilphi = pm.Normal("Zand_soilphi", mu=parameter_dists["Zand_soilphi"]["mean"], sigma=parameter_dists["Zand_soilphi"]["std"])
        # Zand_soilcurkb1 = pm.Normal("Zand_soilcurkb1", mu=parameter_dists["Zand_soilcurkb1"]["mean"], sigma=parameter_dists["Zand_soilcurkb1"]["std"])
        # Zandvast_soilphi = pm.Normal("Zandvast_soilphi", mu=parameter_dists["Zandvast_soilphi"]["mean"], sigma=parameter_dists["Zandvast_soilphi"]["std"])
        # Zandvast_soilcurkb1 = pm.Normal("Zandvast_soilcurkb1", mu=parameter_dists["Zandvast_soilcurkb1"]["mean"], sigma=parameter_dists["Zandvast_soilcurkb1"]["std"])
        # Zandlos_soilphi = pm.Normal("Zandlos_soilphi", mu=parameter_dists["Zandlos_soilphi"]["mean"], sigma=parameter_dists["Zandlos_soilphi"]["std"])
        # Zandlos_soilcurkb1 = pm.Normal("Zandlos_soilcurkb1", mu=parameter_dists["Zandlos_soilcurkb1"]["mean"], sigma=parameter_dists["Zandlos_soilcurkb1"]["std"])
        # Wall_SheetPilingElementEI = pm.Normal("Wall_SheetPilingElementEI", mu=parameter_dists["Wall_SheetPilingElementEI"]["mean"], sigma=parameter_dists["Wall_SheetPilingElementEI"]["std"])
        #
        # x = pt.stack([
        #     Klei_soilphi,
        #     Klei_soilcohesion,
        #     Klei_soilcurkb1,
        #     Zand_soilphi,
        #     Zand_soilcurkb1,
        #     Zandvast_soilphi,
        #     Zandvast_soilcurkb1,
        #     Zandlos_soilphi,
        #     Zandlos_soilcurkb1,
        #     Wall_SheetPilingElementEI,
        #     true_params["Water_lvl"]
        # ])

        x = pm.Normal("x", mu=0, sigma=1, shape=input_dim)
        y_hat = mlp_forward_pt(x, weights_np).squeeze()
        y = pm.Normal("y", mu=y_hat, sigma=0.1, observed=y_obs)


    # with pm.Model() as pymc_model:
    #     x = pm.Normal("x", mu=0, sigma=1, shape=input_dim)
    #
    #     h = pt.dot(x, weights_np[0][0].T) + weights_np[0][1]
    #     h = pt.switch(h > 0, h, 0)
    #
    #     h = pt.dot(h, weights_np[1][0].T) + weights_np[1][1]
    #     h = pt.switch(h > 0, h, 0)
    #
    #     h = pt.dot(h, weights_np[2][0].T) + weights_np[2][1]
    #     h = pt.switch(h > 0, h, 0)
    #
    #     h = pt.dot(h, weights_np[3][0].T) + weights_np[3][1]
    #     h = pt.switch(h > 0, h, 0)
    #
    #     h = pt.dot(h, weights_np[4][0].T) + weights_np[4][1]
    #     h = pt.switch(h > 0, h, 0)
    #
    #     h = pt.dot(h, weights_np[5][0].T) + weights_np[5][1]
    #     h = pt.switch(h > 0, h, 0)
    #
    #     y_hat = pt.dot(h, weights_np[6][0].T) + weights_np[6][1]
    #
    #     sigma = pm.HalfNormal("sigma", sigma=5)
    #
    #     y = pm.Normal("y", mu=y_hat, sigma=0.1, observed=y_obs)



    # idata = pm.sample(model=pymc_model, draws=1_000, tune=1_000, chains=4, cores=num_cores, progressbar=True, target_accept=0.9)
    idata = pm.sample(model=pymc_model, draws=5, tune=5, chains=4, cores=num_cores, progressbar=True, target_accept=0.9)


    result_path.mkdir(parents=True, exist_ok=True)
    means = [val["mean"] for (key, val) in parameter_dists.items() if key in list(true_params.keys())]
    sigmas = [val["std"] for (key, val) in parameter_dists.items() if key in list(true_params.keys())]
    ref_vals = {f"x[{i}]": (val-mean)/sigma for i, (val, mean, sigma) in enumerate(zip(true_params.values(), means, sigmas))}
    ref_vals["x[11]"] = 0
    az.plot_posterior(idata, var_names=["x"], ref_val=ref_vals, ref_val_color="c")
    plt.savefig(result_path/"posterior_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

