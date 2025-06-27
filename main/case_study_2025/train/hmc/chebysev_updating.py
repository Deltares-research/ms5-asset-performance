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
from main.case_study_2025.train.srg.chebysev_train import Chebysev, MinMaxScaler
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

    chebysev_path = SCRIPT_DIR / "../results/srg/chebysev/lr_1.0e-05_epochs_100000_fullprofile_False"
    torch_path = chebysev_path / "torch_weights.pth"
    param_dist_path = SCRIPT_DIR / "../../data/parameter_distributions.csv"
    obs_path = SCRIPT_DIR / "../../data/setting"
    result_path = SCRIPT_DIR / "../../results/hmc/chebysev/all_params"
    result_path.mkdir(parents=True, exist_ok=True)

    parameter_dists = pd.read_csv(param_dist_path)
    parameter_dists = parameter_dists.set_index(parameter_dists["parameter"], drop=True)
    parameter_dists = parameter_dists.to_dict(orient="index")
    cols_keep = [
        'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1', 'Zand_soilphi', 'Zand_soilcurkb1', 'Zandvast_soilphi',
        'Zandvast_soilcurkb1', 'Zandlos_soilphi', 'Zandlos_soilcurkb1', 'Wall_SheetPilingElementEI'
    ]
    mus = np.array([val["mean"] for (key, val) in parameter_dists.items() if key in cols_keep])
    sigmas = np.array([val["std"] for (key, val) in parameter_dists.items() if key in cols_keep])

    scaler_x = joblib.load(chebysev_path / r"scaler_x.joblib")
    scaler_y = joblib.load(chebysev_path / r"scaler_y.joblib")

    with open(obs_path / "case_study.json", "r") as f: data = json.load(f)
    true_params = data['12']["true_params"]
    true_params = {key: val for (key, val) in zip(cols_keep + ["Water_lvl"], true_params.values())}
    ref_vals = [val for (key, val) in true_params.items() if key in cols_keep]
    y_obs = np.asarray(data['12']["deformations"])

    num_cores = os.cpu_count()
    print(f"USING {num_cores} CORES")

    x = np.linspace(0, 10, y_obs.shape[-1])
    x = np.cumsum(x)

    torch_model = Chebysev(
        input_dim=11,
        hidden_dims=[1024, 512, 256, 128, 64, 32],
        x=x,
        degree=10
    )

    state_dict = torch.load(torch_path, map_location=torch.device('cpu'))
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    weights_np = extract_mlp_weights(torch_model)
    input_dim = weights_np[0][0].shape[1]  # Infer from weight shapes
    basis = torch_model.basis.detach().numpy()

    with pm.Model() as pymc_model:
        x = pm.Normal("x", mu=mus, sigma=sigmas)
        x_full = pt.concatenate([x, pt.constant([true_params["Water_lvl"]])])
        x_scaled = pm.Deterministic("x_scaled", (x_full - scaler_x.data_min_) * scaler_x.scale_ - 1)
        y_hat_scaled = chebysev_forward_pt(x_scaled, weights_np, basis).squeeze()
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
def train(param: str = "Wall_SheetPilingElementEI", draws: int = 10, tune: int = 10, targetaccept: float = 0.8, seed: int = 42):
    SCRIPT_DIR = Path(__file__).resolve().parent

    chebysev_path = SCRIPT_DIR / "../results/srg/chebysev/lr_1.0e-05_epochs_100000_fullprofile_False"
    torch_path = chebysev_path / "torch_weights.pth"
    param_dist_path = SCRIPT_DIR / "../../data/parameter_distributions.csv"
    obs_path = SCRIPT_DIR / "../../data/setting"
    result_path = SCRIPT_DIR / f"../../results/hmc/chebysev/{param}"
    result_path.mkdir(parents=True, exist_ok=True)

    parameter_dists = pd.read_csv(param_dist_path)
    parameter_dists = parameter_dists.set_index(parameter_dists["parameter"], drop=True)
    parameter_dists = parameter_dists.to_dict(orient="index")
    cols_keep = [
        'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1', 'Zand_soilphi', 'Zand_soilcurkb1', 'Zandvast_soilphi',
        'Zandvast_soilcurkb1', 'Zandlos_soilphi', 'Zandlos_soilcurkb1', 'Wall_SheetPilingElementEI'
    ]
    mus = np.array([val["mean"] for (key, val) in parameter_dists.items() if key in cols_keep])
    sigmas = np.array([val["std"] for (key, val) in parameter_dists.items() if key in cols_keep])

    scaler_x = joblib.load(chebysev_path / r"scaler_x.joblib")
    scaler_y = joblib.load(chebysev_path / r"scaler_y.joblib")

    with open(obs_path / "case_study.json", "r") as f: data = json.load(f)
    true_params = data['12']["true_params"]
    true_params = {key: val for (key, val) in zip(cols_keep + ["Water_lvl"], true_params.values())}
    ref_vals = [val for (key, val) in true_params.items() if key in cols_keep]
    y_obs = np.asarray(data['12']["deformations"])

    num_cores = os.cpu_count()
    print(f"USING {num_cores} CORES")

    x = np.linspace(0, 10, y_obs.shape[-1])
    x = np.cumsum(x)

    torch_model = Chebysev(
        input_dim=11,
        hidden_dims=[1024, 512, 256, 128, 64, 32],
        x=x,
        degree=10
    )

    state_dict = torch.load(torch_path, map_location=torch.device('cpu'))
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    weights_np = extract_mlp_weights(torch_model)
    input_dim = weights_np[0][0].shape[1]  # Infer from weight shapes
    basis = torch_model.basis.detach().numpy()

    idx = list(true_params.keys()).index(param)
    # true_params = np.array([val for (key, val) in true_params.items() if key != param])
    # true_params = np.array([val for (key, val) in true_params.items()])
    true_params = np.array([8.21, 39.65, 2300, 27.86, 12726, 37.75, 27084, 29.78, 8852, -1])

    m = (scaler_x.data_min_ + scaler_x.data_max_) / 2

    print(scaler_x.data_min_)
    print(scaler_x.data_max_)

    m = true_params.copy()
    m[0] = 3.1
    m[1] = true_params[0]

    with pm.Model() as pymc_model:
        # x = pm.Normal("x", mu=mus[idx], sigma=sigmas[idx], shape=(1,))
        data_min = pt.constant(scaler_x.data_min_)
        data_max = pt.constant(scaler_x.data_max_)
        scale = pt.constant(scaler_x.scale_)
        x = pm.Uniform("x", lower=data_min[idx], upper=data_max[idx], shape=(1,))
        x_full = pt.concatenate([
            pt.constant(m[:idx])[None, :],  # shape (1, len(a1))
            x.reshape((1, 1)),  # shape (1, 1)
            pt.constant(m[idx:])[None, :]  # shape (1, len(a2))
        ], axis=1)
        # x_full = pt.constant(m)[None, :]
        x_scaled = pm.Deterministic("x_scaled", ((x_full - data_min) * scale - 1).squeeze())
        # x_scaled = pm.Deterministic("x_scaled", 2 * (x_full - data_min) / (data_max-data_min) - 1)
        y_hat_scaled = pm.Deterministic("y_hat_scaled", chebysev_forward_pt(x_scaled, weights_np, basis).squeeze())
        y_hat = pm.Deterministic("y_hat", 0.5 * (y_hat_scaled + 1) * (2 / scaler_y.scale_) + scaler_y.data_min_)
        sigma = pm.HalfNormal("sigma", sigma=5)
        sigma = 5
        y = pm.Normal("y", mu=y_hat, sigma=sigma, observed=y_obs)

    idata_prior = pm.sample_prior_predictive(
        model=pymc_model,
        random_seed=seed
    )

    # idata_posterior = pm.sample(
    #     model=pymc_model,
    #     draws=draws,
    #     tune=tune,
    #     chains=4,
    #     cores=num_cores,
    #     progressbar=True,
    #     target_accept=targetaccept,
    #     random_seed=seed
    # )
    #
    # idata = idata_posterior.copy()
    # idata.add_groups({"prior": idata_prior.prior})
    #
    # idata.to_netcdf(result_path / "posterior_data.netcdf")
    #
    # summarize(idata, result_path, var_names=["x", "sigma"])
    #
    # posterior_plot(idata, [ref_vals[idx]], result_path)
    #
    # print(f"Prior mean={(scaler_x.data_min_[idx]+scaler_x.data_max_[idx])/2:.1f}")
    # print("Prior Min/max:", scaler_x.data_min_[idx], scaler_x.data_max_[idx])
    # print("Prior mean of x:", idata_prior.prior["x"].mean())
    # print("Min/max:", idata_prior.prior["x"].min(), idata_prior.prior["x"].max())
    # print(f"True value={ref_vals[idx]:.0f}")
    # print(f"Prior parameter mean={idata_prior.prior.x.values.mean():.0f}")
    # print(f"Posterior parameter mean={idata.posterior.x.values.mean():.0f}")
    # print(f"Posterior scaled mean={idata.posterior.x_scaled.values[..., idx].mean():.2f}")
    # print(f"Posterior y_hat_scaled mean={idata.posterior.y_hat_scaled.values[..., 5].mean():.2f}")
    # print(f"Posterior y_hat mean={idata.posterior.y_hat.values[..., 5].mean():.2f}")
    # print(f"Observation y={y_obs[..., 5]:.2f}")
    # print(f"Sigma posterior mean={idata.posterior.sigma.values.mean()}")


    from scipy.stats import norm
    import matplotlib.pyplot as plt
    def loglike(x):
        x_full = np.concatenate((m[:idx], np.array([x]), m[idx:]))
        x_scaled = ((x_full - scaler_x.data_min_) * scaler_x.scale_ - 1).squeeze()
        y_hat_scaled = chebysev_forward_np(x_scaled, weights_np, basis).squeeze()
        y_hat = 0.5 * (y_hat_scaled + 1) * (2 / scaler_y.scale_) + scaler_y.data_min_
        sigma = 5
        loglike = norm(loc=y_hat, scale=sigma).logpdf(y_obs).sum()
        return loglike

    grid = np.linspace(scaler_x.data_min_[idx], scaler_x.data_max_[idx], 100)
    loglikes = [loglike(x) for x in grid]

    fig = plt.figure()
    plt.plot(grid, loglikes)
    plt.axvline(ref_vals[idx], c="k")
    plt.close()
    fig.savefig('dummy.png')


    # df = pd.DataFrame(
    #     data=np.c_[scaler_x.data_min_, scaler_x.data_max_, scaler_x.scale_, m],
    #     columns=["Min", "Max", "Scale", "True value"]
    # )
    # df["True value scaled"] = (df["True value"] - df["Min"]) * df["Scale"] - 1
    # df.to_csv("dummy.csv", index=True)


if __name__  == "__main__":

    main()
    # app_param()

