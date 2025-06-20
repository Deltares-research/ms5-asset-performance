import os
import json
import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import pytensor
import pytensor.tensor as pt
from pathlib import Path
import joblib
import typer
from main.case_study_2025.train.hmc.utils import *
from main.case_study_2025.train.srg.mlp_train import MLP, MinMaxScaler
import matplotlib.pyplot as plt
import typer


app_train = typer.Typer()
app_param = typer.Typer()

main = typer.Typer()
main.add_typer(app_train, name="train")
main.add_typer(app_param, name="train_param")

pytensor.config.cxx = "/usr/bin/clang++"  # <-- Fix for MacOS


@app_train.command()
def train(draws: int = 1_000, tune: int = 1_000, targetaccept: float = 0.8, seed: int = 42):

    SCRIPT_DIR = Path(__file__).resolve().parent

    mlp_path = SCRIPT_DIR / "../results/srg/mlp/lr_1.0e-06_epochs_100000"
    torch_path = mlp_path / "torch_weights.pth"
    param_dist_path = SCRIPT_DIR / "../../data/parameter_distributions.csv"
    obs_path = SCRIPT_DIR / "../../data/setting"
    result_path = SCRIPT_DIR / "../../results/hmc/all_params"

    parameter_dists = pd.read_csv(param_dist_path)
    parameter_dists = parameter_dists.set_index(parameter_dists["parameter"], drop=True)
    parameter_dists = parameter_dists.to_dict(orient="index")
    cols_keep = [
        'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1', 'Zand_soilphi', 'Zand_soilcurkb1', 'Zandvast_soilphi',
        'Zandvast_soilcurkb1', 'Zandlos_soilphi', 'Zandlos_soilcurkb1', 'Wall_SheetPilingElementEI'
    ]
    mus = np.array([val["mean"] for (key, val) in parameter_dists.items() if key in cols_keep])
    sigmas = np.array([val["std"] for (key, val) in parameter_dists.items() if key in cols_keep])

    scaler_x = joblib.load(mlp_path / r"scaler_x.joblib")
    scaler_y = joblib.load(mlp_path / r"scaler_y.joblib")

    with open(obs_path / "case_study.json", "r") as f: data = json.load(f)
    true_params = data['12']["true_params"]
    true_params = {key: val for (key, val) in zip(cols_keep + ["Water_lvl"], true_params.values())}
    ref_vals = [val for (key, val) in true_params.items() if key in cols_keep]
    y_obs = np.asarray(data['12']["deformations"])

    num_cores = os.cpu_count()
    print(f"USING {num_cores} CORES")

    torch_model = MLP(
        input_dim=11,
        hidden_dims=[1024, 512, 256, 128, 64, 32],
        output_dim=y_obs.shape[-1]
    )
    state_dict = torch.load(torch_path, map_location=torch.device('cpu'))
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    weights_np = extract_mlp_weights(torch_model)
    input_dim = weights_np[0][0].shape[1]  # Infer from weight shapes

    with pm.Model() as pymc_model:
        x = pm.Normal("x", mu=mus, sigma=sigmas)
        x_full = pt.concatenate([x, pt.constant([true_params["Water_lvl"]])])
        x_scaled = (x_full - scaler_x.data_min_) / scaler_x.scale_
        y_hat_scaled = mlp_forward_pt(x_scaled, weights_np).squeeze()
        y_hat = pm.Deterministic("y_hat", scaler_y.data_min_ + scaler_y.scale_ * y_hat_scaled)
        sigma = pm.HalfNormal("sigma", sigma=5)
        y = pm.Normal("y", mu=y_hat, sigma=sigma, observed=y_obs)

    idata_prior = pm.sample_prior_predictive(
        model=pymc_model,
        random_seed=seed
    )

    idata_posterior = pm.sample(
        model=pymc_model,
        draws=draws,
        tune=tune,
        chains=4,
        cores=num_cores,
        progressbar=True,
        target_accept=targetaccept,
        random_seed=seed
    )

    idata = idata_posterior.copy()
    idata.add_groups({"prior": idata_prior.prior})

    idata.to_netcdf(result_path / "posterior_data.netcdf")

    summarize(idata, result_path, var_names=["x", "sigma"])

    posterior_plot(idata, ref_vals, result_path)


@app_param.command()
def train(param: str = "Klei_soilcurkb1", draws: int = 1_000, tune: int = 1_000, targetaccept: float = 0.8, seed: int = 42):
    SCRIPT_DIR = Path(__file__).resolve().parent

    mlp_path = SCRIPT_DIR / "../results/srg/mlp/lr_1.0e-06_epochs_100000"
    torch_path = mlp_path / "torch_weights.pth"
    param_dist_path = SCRIPT_DIR / "../../data/parameter_distributions.csv"
    obs_path = SCRIPT_DIR / "../../data/setting"
    result_path = SCRIPT_DIR / f"../../results/hmc/{param}"

    parameter_dists = pd.read_csv(param_dist_path)
    parameter_dists = parameter_dists.set_index(parameter_dists["parameter"], drop=True)
    parameter_dists = parameter_dists.to_dict(orient="index")
    cols_keep = [
        'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1', 'Zand_soilphi', 'Zand_soilcurkb1', 'Zandvast_soilphi',
        'Zandvast_soilcurkb1', 'Zandlos_soilphi', 'Zandlos_soilcurkb1', 'Wall_SheetPilingElementEI'
    ]
    mus = np.array([val["mean"] for (key, val) in parameter_dists.items() if key in cols_keep])
    sigmas = np.array([val["std"] for (key, val) in parameter_dists.items() if key in cols_keep])

    scaler_x = joblib.load(mlp_path / r"scaler_x.joblib")
    scaler_y = joblib.load(mlp_path / r"scaler_y.joblib")

    with open(obs_path / "case_study.json", "r") as f: data = json.load(f)
    true_params = data['12']["true_params"]
    true_params = {key: val for (key, val) in zip(cols_keep + ["Water_lvl"], true_params.values())}
    ref_vals = [val for (key, val) in true_params.items() if key in cols_keep]
    y_obs = np.asarray(data['12']["deformations"])

    num_cores = os.cpu_count()
    print(f"USING {num_cores} CORES")

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

    idx = list(true_params.keys()).index(param)
    true_params = np.array([val for (key, val) in true_params.items() if key != param])

    with pm.Model() as pymc_model:
        # x = pm.Normal("x", mu=mus[idx], sigma=sigmas[idx], shape=(1,))
        x = pm.HalfNormal("x", sigma=sigmas[idx], shape=(1,))
        x_full = pt.concatenate([
            pt.constant(true_params[:idx])[None, :],  # shape (1, len(a1))
            x.reshape((1, 1)),  # shape (1, 1)
            pt.constant(true_params[idx:])[None, :]  # shape (1, len(a2))
        ], axis=1)
        x_scaled = (x_full - scaler_x.data_min_) / scaler_x.scale_
        y_hat_scaled = mlp_forward_pt(x_scaled, weights_np).squeeze()
        y_hat = pm.Deterministic("y_hat", scaler_y.data_min_ + scaler_y.scale_ * y_hat_scaled)
        sigma = pm.HalfNormal("sigma", sigma=5)
        y = pm.Normal("y", mu=y_hat, sigma=sigma, observed=y_obs)

    idata_prior = pm.sample_prior_predictive(
        model=pymc_model,
        random_seed=seed
    )

    idata_posterior = pm.sample(
        model=pymc_model,
        draws=draws,
        tune=tune,
        chains=4,
        cores=num_cores,
        progressbar=True,
        target_accept=targetaccept,
        random_seed=seed
    )

    idata = idata_posterior.copy()
    idata.add_groups({"prior": idata_prior.prior})

    idata.to_netcdf(result_path / "posterior_data.netcdf")

    summarize(idata, result_path, var_names=["x", "sigma"])

    posterior_plot(idata, [ref_vals[idx]], result_path)


if __name__  == "__main__":

    main()

