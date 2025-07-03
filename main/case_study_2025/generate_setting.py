import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
from math import fabs
from src.geotechnical_models.dsheetpiling.model import DSheetPiling, DSheetPilingResults
from src.reliability_models.dsheetpiling.lsf import *
from src.corrosion.corrosion_model import CorrosionModel
import collections
from tqdm import tqdm


def run_model(params, model):
    soil_data = unpack_soil_params(params, list(model.soils.keys()))
    water_data = unpack_water_params(params, [lvl.name for lvl in model.water.water_lvls])
    wall_data = unpack_wall_data(params, model.wall._asdict())
    model.update_soils(soil_data)
    model.update_water(water_data)
    model.update_wall(wall_data)
    model.execute()
    return model.results.displacement[0], model.results.moment[0], model.results.z


if __name__ == "__main__":

    df = pd.read_csv(r"data/true_params.csv", index_col="parameter")
    true_params = df["value"].to_dict()

    times = list(range(0, 31))
    corossion_model = CorrosionModel()
    corrosions = corossion_model.generate_observations(np.array(times), seed=42)

    geomodel_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    geomodel = DSheetPiling(geomodel_path)
    monitoring_locs = list(range(1, 156, 10))

    data = {}
    for i, time in enumerate(tqdm(times)):
        corrosion = corrosions[i]
        corrosion_ratio = corrosion / corossion_model.start_thickness
        EI_corroded = true_params["Wall_SheetPilingElementEI"] * (1 - corrosion_ratio)
        time_params = deepcopy(true_params)
        time_params["Wall_SheetPilingElementEI"] = EI_corroded
        deformations, moments, z = run_model(time_params, geomodel)
        data[time] = {
            "time": time,
            "corrosion": corrosion,
            "corrosion_ratio": corrosion_ratio,
            "EI_corroded": EI_corroded,
            "true_params": true_params,
            "time_params": time_params,
            "z": z,
            "deformations": deformations,
            "deformations_monitoring": [d for i, d in enumerate(deformations) if i in monitoring_locs],
            "moments": moments,
            "max_moment": max([fabs(m) for m in moments])
        }

    path = Path(r"data/setting")
    path.mkdir(parents=True, exist_ok=True)
    with open(path/"case_study.json", "w") as f:
        json.dump(data, f, indent=4)

