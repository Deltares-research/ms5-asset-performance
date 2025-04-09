import numpy as np
import probabilistic_library as ptk
from src.reliability_models.base import ReliabilityBase
from src.geotechnical_models.dsheetpiling.model import *
from src.rvs.state import *
from typing import Optional, Type, Tuple, Dict


class DsheetReliability:

    def __init__(self, geomodel: DSheetPilingModel) -> None:
        self.geomodel = geomodel


def hunt(t_p, tan_alpha, h_s, h_crest, h):
    g = 9.81
    l_0 = g  * t_p * t_p
    xi = tan_alpha / np.sqrt(2 * np.pi * h_s / l_0)
    r_u = xi * h_s

    return h_crest - (h + r_u)


def build_fragility(
        lsf,
        state: Type[StateBase],
        fragility_rv_names: Optional[Annotated[Tuple[str, ...] | List[str], "fragility_dims"]] = None,
        integration_rv_names: Optional[Annotated[Tuple[str, ...] | List[str], "integration_dims"]] = None,
        n_integration_grid: int = 10
) -> None:

    if (fragility_rv_names is None) == (integration_rv_names is None):
        raise ValueError("You must provide exactly one of 'fragility_rv_names' or 'integration_rv_names'")

    if fragility_rv_names is None:
        fragility_rv_names = [rv_name for rv_name in state.names if rv_name not in integration_rv_names]
        fragility_rv_idx = np.asarray([i for i, name in enumerate(state.names) if name in fragility_rv_names])

    if integration_rv_names is None:
        integration_rv_names = [rv_name for rv_name in state.names if rv_name not in fragility_rv_names]
        integration_rv_idx = np.asarray([i for i, name in enumerate(state.names) if name in integration_rv_names])

    project = ptk.ReliabilityProject()
    project.model = lsf
    project.settings.reliability_method = ptk.ReliabilityMethod.form
    project.settings.relaxation_factor = 0.15
    project.settings.maximum_iterations = 50
    project.settings.variation_coefficient = 0.02

    for rv_name in fragility_rv_names:
        if state.marginal_dist_type[rv_name] in ["normal", "multivariate_normal"]:
            project.variables[rv_name+"_stdandard"].distribution = ptk.DistributionType.normal
            project.variables[rv_name+"_stdandard"].mean = 0
            project.variables[rv_name+"_stdandard"].deviation = 1
        else:
            raise NotImplementedError("Non-normal distributions have not been implemented yet.")

    cdf_grid = np.linspace(1e-5, 1-1e-5, n_integration_grid)
    grids = []
    for rv_name in integration_rv_names:
        grids.append(state.marginal_dist[rv_name].ppf(cdf_grid))
    mesh = np.meshgrid(grids)
    mesh = np.c_[*mesh]

    fragility_curve = ptk.FragilityCurve()
    fragility_curve.name = "conditional"
    fragility_logpf = []
    for val in mesh:
        project.variables["h"].distribution = ptk.DistributionType.deterministic
        project.variables["h"].mean = val
        project.run()
        dp = project.design_point

        value = ptk.FragilityValue()
        value.x = val
        value.reliability_index = dp.reliability_index
        value.design_point = dp

        fragility_curve.fragility_values.append(value)
        fragility_logpf.append(np.log(dp.probability_failure).item())


    pass



if __name__ == "__main__":

    pass

    model_path = os.environ["MODEL_PATH"]
    model = DSheetPiling(model_path)

    rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["phi", "c"])
    rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.08]), names=["water"])
    state = GaussianState([rv_strength, rv_water])

    # r = DsheetReliability(model)
    build_fragility(hunt, state, integration_rv_names=["water"])
