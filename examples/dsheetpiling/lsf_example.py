import os
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.rvs.state import MvnRV, GaussianState
from src.reliability_models.dsheetpiling.lsf import package_lsf


if __name__ == "__main__":

    geomodel_path = os.environ["MODEL_PATH"]
    form_path = os.environ["FORM_PATH"]
    geomodel = DSheetPiling(geomodel_path, form_path)

    state = GaussianState(rvs=[
        MvnRV(mus=[30, 10], stds=[3, 1], names=["Klei_soilphi", "Klei_soilcohesion"]),
        MvnRV(mus=[1], stds=[0.1], names=["water_A"])
    ])
    performance_config = ("max_moment", lambda x: 150. / (x[0] + 1e-5))

    """ The args are "soilphi", "soilcohesion" and "water_A" respectively. """
    lsf = package_lsf(geomodel, state, performance_config, False)
    limit_state = lsf(30, 10, 1)

    lsf_st = package_lsf(geomodel, state, performance_config, True)
    limit_state_st = lsf_st(0, 0, 0)