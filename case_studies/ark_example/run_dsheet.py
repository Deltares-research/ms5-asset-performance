from pathlib import Path
from argparse import ArgumentParser
from dotenv import load_dotenv
import os
import json
from typing import Dict, List
from datetime import datetime

from case_studies.ark_example.jpdf import JPDF
from src.reliability_models.dsheetpiling.lsf import *


RV_NAMES = [
    'Klei_soilcohesion',
    'Klei_soilphi',
    'Klei_soilcurkb1',
    'Zand_soilphi',
    'Zand_soilcurkb1',
    'Zandvast_soilphi',
    'Zandvast_soilcurkb1',
    'Zandlos_soilphi',
    'Zandlos_soilcurkb1',
    'Wall_SheetPilingElementEI'
]


def run_model(params: Dict[str, float], model: DSheetPiling) -> DSheetPilingResults:
    soil_data = unpack_soil_params(params, list(model.soils.keys()))
    water_data = unpack_water_params(params, [lvl.name for lvl in model.water.water_lvls])
    model.update_soils(soil_data)
    model.update_water(water_data)
    model.execute()
    return model.results


def main(n_samples: int, moment_capacity: float) -> None:

    # Paths
    load_dotenv(".env")
    os.environ["REMOTE_DATA_PATH"] = os.environ["REMOTE_PATH"] + r"/input"
    specs_path = Path(os.environ["REMOTE_DATA_PATH"]) / "settings.json"

    username = os.environ.get("USER", "unknown").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_folder = Path(os.environ["REMOTE_PATH"]) / "output/mcs_results/"
    output_folder.mkdir(exist_ok=True, parents=True)
    output_file = output_folder / f"{username}_{timestamp}.jsonl"

    with open(specs_path, "r") as f:
        specs = json.load(f)
    config = specs.get("parameters", {})
    jpdf = JPDF(name="dsheet", config=config)
    jpdf.set_prior_from_specs(specs_path)
    jpdf.initiate_samples(n_samples, seed=42)

    geomodel_path = os.environ["DSHEET_MODEL_PATH"]
    console_path = Path(os.environ["CONSOLE_PATH"])
    geomodel = DSheetPiling(geomodel_path)
    geomodel.geomodel.set_meta_property("dsheetpiling_console_path", console_path)

    all_results = []
    for i, X_sample in enumerate(jpdf.X_samples, start=1):
        sample = {rv: x.item() for (rv, x) in zip(RV_NAMES, X_sample)}
        res = run_model(sample, geomodel)
        res = res.to_dict()
        row = {**{"simulation_number": i}, **sample, **res}
        row["moment_fos"] = [moment_capacity / m for m in row["max_moment"]]
        all_results.append(row)

    with open(output_file, "w") as f:
        for row in all_results:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--moment_capacity", type=float, default=500)
    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        moment_capacity=args.moment_capacity,
        )

