
#%%

import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
import os

import sys
sys.path.append(r'C:\Users\eijnden\OneDrive - Stichting Deltares\Desktop\MS5_2\ms5-asset-performance')
sys.path.append(r'C:\Users\eijnden\OneDrive - Stichting Deltares\Desktop\MS5_2\ms5-asset-performance\src\geotechnical_models\dsheetpiling')

os.environ['DSHEET_MODEL_PATH'] = r'C:\Users\eijnden\Stichting Deltares\SITO-IS 2025 Moonshot 5 - 02_Asset performance\ARK case study\Geotechnical models\D-Sheet Piling\dummy_higher_load.shi'

from src.geotechnical_models.dsheetpiling.model import DSheetPiling

from main.case_study_2025.prepare_data import generate_samples, generate_setting
from main.case_study_2025.prepare_data import run_dsheet_samples
from main.case_study_2025.train.surrogate import mlp_moment_train
from main.case_study_2025.reliability import timeline_analysis
from main.case_study_2025.reliability import visualize_timeline, visualize_timeline_prior_posterior

#parser = ArgumentParser()
#parser.add_argument("--n_mc_samples", type=int, default=10_000_000)
#parser.add_argument("--n_srg_samples", type=int, default=1_000)
#args = parser.parse_args()

generate_samples.main(1_000_000, 1_000)
run_dsheet_samples.main(n_samples_to_use=100)
generate_setting.main()
mlp_moment_train.main(epochs=5000, lr=0.0001)
timeline_analysis.main()
visualize_timeline_prior_posterior.main()