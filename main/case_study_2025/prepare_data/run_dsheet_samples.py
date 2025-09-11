import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from src.geotechnical_models.dsheetpiling.model import DSheetPiling, DSheetPilingResults
from src.reliability_models.dsheetpiling.lsf import *
from pathlib import Path
import json
from tqdm import tqdm


def run_model(rvs, model, rv_names):
    params = {name: rv for (name, rv) in zip(rv_names, rvs)}
    soil_data = unpack_soil_params(params, list(model.soils.keys()))
    water_data = unpack_water_params(params, [lvl.name for lvl in model.water.water_lvls])
    wall_data = unpack_wall_data(params, model.wall._asdict())
    model.update_soils(soil_data)
    model.update_water(water_data)
    model.update_wall(wall_data)
    model.execute()
    return model.results


def sample_disp(rv_sample, rv_names, model):

    disp_sample = []
    moment_sample = []

    for i_run, rvs in enumerate(tqdm(rv_sample, desc="Calculating sample")):
        results = run_model(rvs, model, rv_names)
        disp_sample.append(results.displacement[0])
        moment_sample.append(results.moment[0])

    return disp_sample, moment_sample


def main(n_samples_to_use=1_000):

    path = Path(__file__).parent

    load_dotenv(path.parents[2] / ".env")

    geomodel_path = os.environ["DSHEET_MODEL_PATH"]
    data_path = path.parent / "data/srg_samples_uniform_100000.npy"
    result_path = path.parent / "data"
    result_path.mkdir(exist_ok=True, parents=True)

    rv_names = [
        'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1', 'Zand_soilphi', 'Zand_soilcurkb1', 'Zandvast_soilphi',
        'Zandvast_soilcurkb1', 'Zandlos_soilphi', 'Zandlos_soilcurkb1', 'Wall_SheetPilingElementEI', 'water_lvl'
    ]

    samples = np.load(data_path)[:n_samples_to_use]
    samples = np.unique(samples, axis=0)

    geomodel = DSheetPiling(geomodel_path)

    disp_sample, moment_sample = sample_disp(samples, rv_names, geomodel)

    results = {
        "sample": samples.tolist(),
        "displacement": disp_sample,
        "moment": moment_sample,
    }

    with open(result_path / "surrogate_data.json", "w") as f:
        json.dump(results, f, indent=4)

    data = np.c_[np.array(samples), np.array(disp_sample), np.array(moment_sample)]
    df_srg = pd.DataFrame(
        data=data,
        columns=rv_names + [f"disp_{i}" for i in range(1, len(disp_sample[0]) + 1)] + [f"moment_{i}" for i in range(1, len(moment_sample[0]) + 1)]
    )

    df_srg.to_csv(result_path / "surrogate_data.csv")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--n_samples_to_use", type=int, default=1_000)
    args = parser.parse_args()

    main(n_samples_to_use=1_000)

