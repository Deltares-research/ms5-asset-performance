import pandas as pd
import numpy as np
from src.rvs.state import MvnRV, GaussianState


def parse_parameter_dist(path=r"../../data/parameter_distributions.csv"):

    # NOTE: Stiffness parameters are deterministically controlled by soilcurkb1 by the geomodel.

    parameter_dists = pd.read_csv(path)

    rv_names = parameter_dists["parameter"].tolist()
    means = parameter_dists["mean"].values
    stds = parameter_dists["std"].values

    soil_rv_names = [rv_name.split("_")[0] for rv_name in rv_names if rv_name.split("_")[0] not in ["Wall", "Water", "Anchor"]]
    soil_rv_names_unique = set(soil_rv_names)
    soil_rv_names = list(soil_rv_names_unique)
    structural_rvs = [rv_name for rv_name in rv_names if rv_name.split("_")[0] not in soil_rv_names]

    parameter_dists = parameter_dists.set_index(parameter_dists["parameter"])
    rvs = []
    for soil in soil_rv_names:

        param_names = ["soilphi", "soilcohesion", "soilcurkb1"]
        params = [f"{soil}_{param}" for param in param_names]
        soil_rows = parameter_dists.loc[params]

        if soil_rows.loc[f"{soil}_soilcohesion", "std"] > 0:
            means = soil_rows["mean"].values
            stds = soil_rows["std"].values
            soil_state = MvnRV(mus=means, stds=stds, names=params)
        else:
            params = [f"{soil}_{param}" for param in ["soilphi", "soilcurkb1"]]
            means = soil_rows.loc[params, "mean"]
            stds = soil_rows.loc[params, "std"]
            soil_state = MvnRV(mus=means, stds=stds, names=params)

        rvs.append(soil_state)

    for structural_rv in structural_rvs:
        mean = parameter_dists.loc[structural_rv, "mean"]
        std = parameter_dists.loc[structural_rv, "std"]
        structural_state = MvnRV(mus=[mean], stds=[std], names=[structural_rv])
        rvs.append(structural_state)

    rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.2]), names=["water_lvl"])
    resistance_state = GaussianState(rvs=rvs)
    state = GaussianState(rvs=rvs+[rv_water])

    return state, resistance_state


