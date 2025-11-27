
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

os.environ['DSHEET_MODEL_PATH'] = r'C:\Users\eijnden\Stichting Deltares\SITO-IS 2025 Moonshot 5 - 02_Asset performance\ARK case study\Geotechnical models\D-Sheet Piling\dummy_higher_load.shi'

from src.geotechnical_models.dsheetpiling.model import DSheetPiling

from main.case_study_2025.prepare_data.generate_samples import main, calculate


#parser = ArgumentParser()
#parser.add_argument("--n_mc_samples", type=int, default=10_000_000)
#parser.add_argument("--n_srg_samples", type=int, default=1_000)
#args = parser.parse_args()

main(1_000_000, 1_000)

calculate(n_samples_to_use=1_000)