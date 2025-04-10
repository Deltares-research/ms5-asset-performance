import os
import math
from src.reliability_models.dsheetpiling.lsf import package_lsf
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.rvs.state import GaussianState, MvnRV


def test_lsf():

    geomodel_path = os.environ["MODEL_PATH"]
    form_path = os.environ["FORM_PATH"]
    geomodel = DSheetPiling(geomodel_path, form_path)

    state = GaussianState(rvs=[
        MvnRV(mus=[30, 10], stds=[3, 1], names=["Klei_soilphi", "Klei_soilcohesion"]),
        MvnRV(mus=[1], stds=[0.1], names=["h"])  # Unused, just for testing
    ])

    performance_config = ("max_moment", lambda x: 150. / (x[0] + 1e-5))

    lsf = package_lsf(geomodel, state, performance_config, False)
    limit_state = lsf(30, 10, 1)

    assert math.isclose(limit_state, 4.863, abs_tol=1e-3)


def test_standardized_lsf():

    geomodel_path = os.environ["MODEL_PATH"]
    form_path = os.environ["FORM_PATH"]
    geomodel = DSheetPiling(geomodel_path, form_path)

    state = GaussianState(rvs=[
        MvnRV(mus=[30, 10], stds=[3, 1], names=["Klei_soilphi", "Klei_soilcohesion"]),
        MvnRV(mus=[1], stds=[0.1], names=["h"])  # Unused, just for testing
    ])

    performance_config = ("max_moment", lambda x: 150. / (x[0] + 1e-5))

    lsf_st = package_lsf(geomodel, state, performance_config, True)
    limit_state_st = lsf_st(0, 0, 0)

    assert math.isclose(limit_state_st, 4.863, abs_tol=1e-3)

