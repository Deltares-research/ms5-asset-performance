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

    times = np.linspace(0., specs["parameters"]["preload_removal_time"], 13)

    settlements = get_settlement(
        t=times,
        CR=true_vars["CR"],
        k=true_vars["k"]*3_600*24,
        RR=specs["parameters"]["RR"],
        Ca=specs["parameters"]["Ca"],
        h=specs["parameters"]["layer_thickness"],
        sigma_0=specs["parameters"]["sigma_0"],
        sigma_v=specs["parameters"]["sigma_0"]+specs["parameters"]["preload"],
        sigma_p=specs["parameters"]["sigma_p"],
        method=specs["parameters"]["doc_method"],
    )

    obs_error = specs["parameters"]["obs_error"]
    random.seed(42)
    gauss_errors = [random.gauss(mu=0, sigma=1) for _ in range(len(settlements))]
    settlements = [s+gauss_err*obs_error for (s, gauss_err) in zip(settlements, gauss_errors)]

    setting = {f"{time:.1f}": f"{settlement.item():.4f}" for (time, settlement) in zip(times, settlements)}

    with open(setting_path, "w") as f:
        json.dump(setting, f, indent=2)

