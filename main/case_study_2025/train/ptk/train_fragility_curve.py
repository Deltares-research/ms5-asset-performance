import os
import numpy as np
from pathlib import Path
import pandas as pd
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.rvs.state import MvnRV, GaussianState
from src.reliability_models.dsheetpiling.lsf import package_lsf
from src.reliability_models.dsheetpiling.reliability import ReliabilityFragilityCurve
from utils import parse_parameter_dist


if __name__ == "__main__":


    state, resistance_state = parse_parameter_dist(r"../../data/parameter_distributions.csv")

    geomodel_path = os.environ["MODEL_PATH"]
    form_path = os.environ["FORM_PATH"]
    geomodel = DSheetPiling(geomodel_path, form_path)

    performance_config = ("max_moment", lambda x: 20. / (x[0] + 1e-5))
    form_params = (0.15, 30, 0.02)
    lsf = package_lsf(geomodel, state, performance_config, True)

    fc_savedir = r"../../results/ptk/fragility_curve.json"
    rm = ReliabilityFragilityCurve(lsf, state, "form", form_params, integration_rv_names=["water_lvl"])

    retrain = True

    if retrain:
        rm.build_fragility(n_integration_grid=10, fc_savedir=fc_savedir)
    else:
        rm.load_fragility(fc_savedir)

    pf, beta = rm.integrate_fragility()

