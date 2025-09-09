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
from dotenv import load_dotenv
from argparse import ArgumentParser


def run_model(params, model):
    soil_data = unpack_soil_params(params, list(model.soils.keys()))
    water_data = unpack_water_params(params, [lvl.name for lvl in model.water.water_lvls])
    wall_data = unpack_wall_data(params, model.wall._asdict())
    model.update_soils(soil_data)
    model.update_water(water_data)
    model.update_wall(wall_data)
    model.execute()
    return model.results.displacement[0], model.results.moment[0], model.results.z


def main():

    rv_names = [
        'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1','Zand_soilphi', 'Zand_soilcurkb1','Zandvast_soilphi',
        'Zandvast_soilcurkb1','Zandlos_soilphi', 'Zandlos_soilcurkb1','Wall_SheetPilingElementEI', 'water_lvl'
    ]

    srg_samples_path = Path(__file__).parents[1] / "data/surrogate_data.csv"
    env_path = Path(__file__).parents[3] / ".env"

    load_dotenv(env_path)

    df = pd.read_csv(srg_samples_path)
    true_values = df[rv_names].iloc[0]
    true_params = true_values.to_dict()

    monitoring_cols = [col for col in df.columns if col.split("_")[0] == "disp"]
    monitoring_locs = [int(monitoring_col.split("_")[-1]) for monitoring_col in monitoring_cols]

    times = [50 + time for time in range(0, 31, 1)]
    corossion_model = CorrosionModel()
    corrosions = corossion_model.generate_observations(np.array(times), seed=42)

    geomodel_path = os.environ["DSHEET_MODEL_PATH"]  # model_path defined as environment variable
    geomodel = DSheetPiling(geomodel_path)

    data = {}
    moments_survived = []
    for i, time in enumerate(tqdm(times)):
        corrosion = corrosions[i]
        corrosion_ratio = corrosion / corossion_model.start_thickness
        EI_corroded = true_params["Wall_SheetPilingElementEI"] * (1 - corrosion_ratio)
        time_params = deepcopy(true_params)
        time_params["Wall_SheetPilingElementEI"] = EI_corroded
        deformations, moments, z = run_model(time_params, geomodel)
        # Maximum survived moment is 80% of the one met in D-SheetPiling, bound to 600 to fix D-SheetPiling non-convergence.
        moment_survived = min(600, np.abs(moments).max().item() * 0.8)
        moments_survived.append(moment_survived)
        data[float(time)] = {
            "time": float(time),
            "corrosion": corrosion.tolist(),
            "corrosion_ratio": corrosion_ratio.tolist(),
            "EI_corroded": EI_corroded.tolist(),
            "true_params": true_params,
            "time_params": {key: val.tolist() if isinstance(val, np.ndarray) else val for (key, val) in time_params.items()},
            "deformations": deformations,
            "moments": moments,
            "max_moment": max([fabs(m) for m in moments]),
            "moment_survived": max(moments_survived)
        }

    data_path = Path(__file__).parents[1] / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    with open(data_path/"case_study.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    main()

