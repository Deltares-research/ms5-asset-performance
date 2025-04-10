import os
import math
from src.reliability_models.dsheetpiling.lsf import package_lsf
from src.geotechnical_models.dsheetpiling.model import DSheetPiling


def test_execution():

    geomodel_path = os.environ["MODEL_PATH"]
    form_path = os.environ["FORM_PATH"]
    geomodel = DSheetPiling(geomodel_path, form_path)

    soil_layers = {"Klei": ("soilphi", "soilcohesion")}
    req = (0, "max_moment", lambda x: 150./(x+1e-5))

    lsf = package_lsf(geomodel, soil_layers, req)

    limit_state = lsf(30, 10)  # The first arg is "soilphi" and the second is "soilcohesion" for "Klei", as declared.

    assert math.isclose(limit_state, -4.863, abs_tol=1e-3)

