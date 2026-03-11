from settlement_engine import *
import numpy as np
from pathlib import Path
import json
from dotenv import load_dotenv
import os
import random


if __name__ == "__main__":

    load_dotenv("settlement_example.env")
    os.environ["REMOTE_DATA_PATH"] = os.environ["REMOTE_PATH"] + r"/input"
    specs_path = Path(os.environ["REMOTE_DATA_PATH"]) / "case_study_specifications.json"
    setting_path = Path(os.environ["REMOTE_DATA_PATH"]) / "case_study_setting.json"

    with open(specs_path, "r") as f:
        specs = json.load(f)

    true_vars = {row["name"]: row["true"] for row in specs["variables"]}

    times = np.linspace(30, specs["parameters"]["preload_removal_time"], 36)

    layer_thicknesses = specs["parameters"]["layer_thickness"]
    if not isinstance(layer_thicknesses, list):
        layer_thicknesses = [layer_thicknesses]

    settlements_per_location = []
    for loc, layer_thickness in enumerate(layer_thicknesses, start=1):

        settlements = get_settlement(
            t=times,
            CR=true_vars["CR"],
            k=true_vars["k"],
            RR=specs["parameters"]["RR"],
            Ca=specs["parameters"]["Ca"],
            h=layer_thickness,
            sigma_0=specs["parameters"]["sigma_0"],
            sigma_v=specs["parameters"]["sigma_0"]+specs["parameters"]["preload"],
            sigma_p=specs["parameters"]["sigma_p"],
            method=specs["parameters"]["doc_method"],
        )

        obs_error = specs["parameters"]["obs_error"]
        random.seed(42 + loc)
        gauss_errors = [random.gauss(mu=0, sigma=1) for _ in range(len(settlements))]
        settlements = [s+gauss_err*obs_error for (s, gauss_err) in zip(settlements, gauss_errors)]

        settlements_per_location.append(settlements)

    setting = []
    for i, time in enumerate(times):
        row = {"time": f"{time:.1f}"}
        for loc, s_loc in enumerate(settlements_per_location, start=1):
            row[f"settlement_{loc}"] = f"{s_loc[i]:.4f}"
        setting.append(row)

    with open(setting_path, "w") as f:
        json.dump(setting, f, indent=2)

