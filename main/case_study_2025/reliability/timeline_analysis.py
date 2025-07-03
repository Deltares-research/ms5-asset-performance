from pathlib import Path

import numpy as np

from main.case_study_2025.reliability.chebysev_reliability import reliability
from main.case_study_2025.reliability.utils import *
from src.corrosion.corrosion_model import CorrosionModel


if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    setting_path = SCRIPT_DIR / "data/setting/case_study.json"
    z_path = SCRIPT_DIR / "data/setting/z.json"
    mcs_samples_path = SCRIPT_DIR / f"data/mc_samples_normal_100000000.npy"
    chebysev_path = SCRIPT_DIR / "train/results/srg/chebysev/lr_1.0e-05_epochs_100000_fullprofile_True"

    n_mcs = 100_000
    # fos_calculator = load_chebysev_calculator(chebysev_path, z_path)

    corrosion_model = CorrosionModel()

    moment_cap = 40
    start_thickness = corrosion_model.start_thickness

    with open(setting_path, "r") as f:
        setting_data = json.load(f)

    for time, data in setting_data.items():

        time = float(time)

        if time != 0:
            break

        corrosion_dist = corrosion_model.corrosion_distribution(time)
        np.random.seed(42)
        corrosions = corrosion_dist.rvs(n_mcs)
