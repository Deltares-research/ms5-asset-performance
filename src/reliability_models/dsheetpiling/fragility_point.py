from src.reliability_models.dsheetpiling.lsf import package_lsf
from src.geotechnical_models.dsheetpiling.model import *
from src.rvs.state import *
from src.reliability_models.dsheetpiling.reliability import FragilityPoint, ReliabilityFragilityCurve
import json
from pathlib import Path


if __name__ == "__main__":

    POINT = float(os.environ["POINT"])

    form_path = r"C:\Users\mavritsa\Stichting Deltares\Sito-IS 2023 SO Emerging Topics - Moonshot 5 - 02_Asset performance\ARK case study\Geotechnical models\D-Sheet Piling\FORM example"
    geomodel_path = r"C:\Users\mavritsa\Stichting Deltares\Sito-IS 2023 SO Emerging Topics - Moonshot 5 - 02_Asset performance\ARK case study\Geotechnical models\D-Sheet Piling\N60_3_5-060514-red.shi"
    geomodel = DSheetPiling(geomodel_path, form_path)

    soil_layers = {"Klei": ("soilphi", "soilcohesion")}

    rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["Klei_soilphi", "Klei_soilcohesion"])
    rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.08]), names=["water_A"])
    state = GaussianState(rvs=[rv_strength, rv_water])

    performance_config = ("max_moment", lambda x: 150. / (x[0] + 1e-5))
    form_params = (0.15, 30, 0.02)
    lsf = package_lsf(geomodel, state, performance_config, True)

    r = ReliabilityFragilityCurve(lsf, state, "form", form_params, integration_rv_names=["water_A"])
    fp = r.fragility_point(np.asarray([POINT]))
    fp_dict = fp._asdict()

    script_dir = Path(__file__).parent.parent.parent.parent
    output_file = script_dir / "examples" / "reliability" / "bash" / f"fragility_point_{POINT}.json"
    output_file = Path(output_file.as_posix())
    with open(output_file, 'w') as f: json.dump(fp_dict, f, indent=4)
