import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
from src.geotechnical_models.dsheetpiling.model import DSheetPiling, DSheetPilingResults
from src.reliability_models.dsheetpiling.lsf import *
import collections


def run_model(rvs, model, rv_names):
    params = {name: rv for (name, rv) in zip(rv_names, rvs)}
    soil_data = unpack_soil_params(params, list(model.soils.keys()))
    water_data = unpack_water_params(params, [lvl.name for lvl in model.water.water_lvls])
    wall_data = unpack_wall_data(params, model.wall._asdict())
    model.update_soils(soil_data)
    model.update_water(water_data)
    model.update_wall(wall_data)
    model.execute()
    return model.results.displacement[0]


if __name__ == "__main__":

    times_corrosion = [5, 10, 15, 20]
    corrosions = [0.8, 0.7, 0.6, 0.5]
    corrosions = {
        time: {
            "corrosion": corrosion,
            "deformations": []
        }
        for (time, corrosion) in zip(times_corrosion, corrosions)
    }

    times_deform = [12]
    df = pd.read_csv(r"data/true_params.csv")
    param_names = list(df["parameter"])
    true_params = df["value"].values.tolist()
    geomodel_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    geomodel = DSheetPiling(geomodel_path)
    deformations = run_model(true_params, geomodel, param_names)
    idx_locs = list(range(1, 151, 10))
    deformations = {
        time: {
            "corrosion": [],
            "true_params": {n: t for (n, t) in zip(param_names, true_params)},
            "deformations_all": deformations,
            "deformations": [d for i, d in enumerate(deformations) if i in idx_locs],
            "idx": idx_locs
        }
        for (time, deformations) in zip(times_deform, [deformations])
    }

    data = corrosions
    data.update(deformations)
    data = collections.OrderedDict(sorted(data.items()))

    path = Path(r"data/setting")
    path.mkdir(parents=True, exist_ok=True)
    with open(path/"case_study.json", "w") as f:
        json.dump(data, f, indent=4)

