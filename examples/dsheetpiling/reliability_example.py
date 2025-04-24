import os
import numpy as np
from pathlib import Path
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.rvs.state import MvnRV, GaussianState
from src.reliability_models.dsheetpiling.lsf import package_lsf
from src.reliability_models.dsheetpiling.reliability import ReliabilityFragilityCurve


if __name__ == "__main__":

    geomodel_path = os.environ["MODEL_PATH"]
    form_path = os.environ["FORM_PATH"]
    geomodel = DSheetPiling(geomodel_path, form_path)

    soil_layers = {"Klei": ("soilphi", "soilcohesion")}

    rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["Klei_soilphi", "Klei_soilcohesion"])
    rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.08]), names=["water_A"])
    state = GaussianState(rvs=[rv_strength, rv_water])

    performance_config = ("max_moment", lambda x: 150. / (x[0] + 1e-5))
    form_params = (0.15, 30, 0.02)
    lsf = package_lsf(geomodel, state, performance_config, True)

    fc_savedir = r"../../../examples/reliability/fragility_curve_100.json"
    rm = ReliabilityFragilityCurve(lsf, state, "form", form_params, integration_rv_names=["water_A"])
    # rm.build_fragility(n_integration_grid=3, fc_savedir=fc_savedir)
    # rm.generate_integration_mesh(n_grid=3)
    # rm.load_fragility(fc_savedir)
    # pf, beta = rm.integrate_fragility()

    path = Path(Path(r"/examples/dsheetpiling/reliability\bash").as_posix())
    rm.compile_fragility_points(path)