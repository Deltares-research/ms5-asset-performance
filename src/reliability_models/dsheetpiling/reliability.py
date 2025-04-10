import numpy as np
from scipy import stats
import probabilistic_library as ptk
from src.reliability_models.base import ReliabilityBase
from src.reliability_models.dsheetpiling.lsf import package_lsf, LSFType
from src.geotechnical_models.dsheetpiling.model import *
from src.rvs.state import *
from typing import Optional, Type, Tuple, Dict
from tqdm import tqdm


def init_project(lsf: LSFType, form_params: Tuple[float, int, float] = (0.15, 5, 0.02)) -> ptk.ReliabilityProject:
    project = ptk.ReliabilityProject()
    project.model = lsf
    project.settings.reliability_method = ptk.ReliabilityMethod.form
    relaxation_factor, maximum_iterations, variation_coefficient = form_params
    project.settings.relaxation_factor = relaxation_factor
    project.settings.maximum_iterations = maximum_iterations
    project.settings.variation_coefficient = variation_coefficient
    return project


def set_fragility_rvs(project: ptk.ReliabilityProject, rv_names: List[str]) -> ptk.ReliabilityProject:
    for rv_name in rv_names:
        if state.marginal_pdf_type[rv_name] in ["normal", "multivariate_normal"]:
            project.variables[rv_name].distribution = ptk.DistributionType.normal
            project.variables[rv_name].mean = 0
            project.variables[rv_name].deviation = 1
        else:
            raise NotImplementedError("Non-normal distributions have not been implemented yet.")
    return project


def fragility_point(
        project: ptk.ReliabilityProject,
        point: Annotated[NDArray[float], "integration_dims"],
        integration_rv_names: List[str]
) -> Tuple[ptk.FragilityValue, float]:

    for i, rv_name in enumerate(integration_rv_names):
        project.variables[rv_name].distribution = ptk.DistributionType.deterministic
        project.variables[rv_name].mean = float(point[i])

    project.run()
    dp = project.design_point

    frag_value = ptk.FragilityValue()
    frag_value.x = point
    frag_value.reliability_index = dp.reliability_index
    frag_value.design_point = dp
    fragility_logpf = np.log(dp.probability_failure).item()

    return frag_value, fragility_logpf


def build_fragility(
        lsf: LSFType,
        state: Type[StateBase],
        fragility_rv_names: Optional[Annotated[Tuple[str, ...] | List[str], "fragility_dims"]] = None,
        integration_rv_names: Optional[Annotated[Tuple[str, ...] | List[str], "integration_dims"]] = None,
        form_params: Tuple[float, int, float] = (0.15, 5, 0.02),
        n_integration_grid: int = 10
) -> ptk.FragilityCurve:

    if (fragility_rv_names is None) == (integration_rv_names is None):
        raise ValueError("You must provide exactly one of 'fragility_rv_names' or 'integration_rv_names'")

    if fragility_rv_names is None:
        fragility_rv_names = [rv_name for rv_name in state.names if rv_name not in integration_rv_names]

    if integration_rv_names is None:
        integration_rv_names = [rv_name for rv_name in state.names if rv_name not in fragility_rv_names]

    project = init_project(lsf, form_params)

    project = set_fragility_rvs(project, fragility_rv_names)

    cdf_grid = np.linspace(1e-5, 1-1e-5, n_integration_grid)
    grid = stats.norm(0, 1).ppf(cdf_grid)
    mesh = np.meshgrid([grid]*len(integration_rv_names))
    mesh = np.c_[*mesh]

    fragility_curve = ptk.FragilityCurve()
    fragility_curve.name = "conditional"
    fragility_logpfs = []
    for point in tqdm(mesh, desc="Running FORM for combination of integration variables:"):
        frag_value, fragility_logpf = fragility_point(project, point, integration_rv_names)
        fragility_curve.fragility_values.append(frag_value)
        fragility_logpfs.append(fragility_logpf)

    return fragility_curve


if __name__ == "__main__":

    geomodel_path = os.environ["MODEL_PATH"]
    form_path = os.environ["FORM_PATH"]
    geomodel = DSheetPiling(geomodel_path, form_path)

    soil_layers = {"Klei": ("soilphi", "soilcohesion")}

    rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["Klei_soilphi", "Klei_soilcohesion"])
    rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.08]), names=["water_lvl"])
    state = GaussianState(rvs=[rv_strength, rv_water])

    performance_config = ("max_moment", lambda x: 150. / (x[0] + 1e-5))

    lsf = package_lsf(geomodel, state, performance_config, True)

    build_fragility(lsf, state, integration_rv_names=["water_lvl"], n_integration_grid=3)

