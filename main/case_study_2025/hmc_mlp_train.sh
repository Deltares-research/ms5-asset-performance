#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate ms5

draws=1000
tune=1000
targetccept=0.8
seed=42

#SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python -m main.case_study_2025.train.hmc.updating train train --draws $draws --tune $tune --targetaccept $targetccept 2>&1

