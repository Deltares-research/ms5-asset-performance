import torch
import numpy as np
import json
from main.case_study_2025.train.hmc.utils import *
from main.case_study_2025.train.srg.mlp_train import MLP, MinMaxScaler
from main.case_study_2025.chebysev_moments import FoSCalculator
from main.case_study_2025.train.srg.utils import load_data


if __name__ == "__main__":

    model_path = r"../results/srg/chebysev/lr_1.0e-05_epochs_100000_fullprofile_True/torch_weights.pth"
    posterior_path = r"../../results/hmc/posterior_data.netcdf"
    scaler_x_path = r"../results/srg/chebysev/lr_1.0e-05_epochs_100000_fullprofile_True/scaler_x.joblib"
    scaler_y_path = r"../results/srg/chebysev/lr_1.0e-05_epochs_100000_fullprofile_True/scaler_y.joblib"
    srg_data_path = r"../data/srg_data_20250604_100638.csv"

    X_srg, y_srg = load_data(srg_data_path, full_profile=True)
    n_points = y_srg.shape[-1]

    # with open(obs_path / "case_study.json", "r") as f: data = json.load(f)
    # wall_locs = data["12"]["wall_locs"]
    # monitoring_locs = data["12"]["monitoring_locs"]
    wall_locs = np.linspace(0, 10, n_points)
    monitoring_locs = np.linspace(0, 10, 15)
    wall_props = (1e+4, 15., wall_locs, monitoring_locs)

    fos_calculator = FoSCalculator(n_points, wall_props, model_path, posterior_path, scaler_x_path, scaler_y_path)

    # Prepare posterior samples for MLP inference of displacements
    X = fos_calculator.idata.posterior.x.values
    X = X.reshape(-1, X.shape[-1])

    # Append water level which was not an RV in the PYMC model
    with open(r"../../data/setting/case_study.json", "r") as f: data = json.load(f)
    water_lvl = data['12']["true_params"]["Water_lvl"]
    X = np.column_stack((X, np.ones(X.shape[0])*water_lvl))

    posterior_displacements = fos_calculator.inference(X_srg)

    XX = torch.from_numpy(X_srg).float()
    with torch.no_grad():
        coeffs = fos_calculator.model(XX, return_coeffs=True)
    curvatures = coeffs @ fos_calculator.model.basis_der
    curvatures = curvatures.detach().numpy()
    curvatures /= (1_000)  # Convert [mm/m^2] displacements to [1/m].
    EI = 12_000
    moments = - EI * curvatures  # Minus for proper sign in moment convention

    plot_path = Path("../../results/hmc/chebysev")
    plot_path.mkdir(exist_ok=True, parents=True)
    fos_calculator.plot_moments(posterior_displacements, moments, plot_path)

