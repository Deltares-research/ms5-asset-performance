import numpy as np
from numpy.typing import NDArray
from scipy import stats
import probabilistic_library as ptk
from src.reliability_models.base import ReliabilityBase
from src.reliability_models.dsheetpiling.lsf import package_lsf, LSFType
from src.geotechnical_models.dsheetpiling.model import *
from src.rvs.state import *
from typing import Optional, Type, Tuple, Dict, NamedTuple
from dataclasses import dataclass, asdict, field
from tqdm import tqdm
import json


class FragilityPoint(NamedTuple):
    point: Annotated[List[float] | Tuple[float, ...] | NDArray[np.float64], "integration_dims"]
    pf: float | Annotated[NDArray[np.float64], "1"]
    beta: float | Annotated[NDArray[np.float64], "1"]
    design_point: Dict[str, float]
    alphas: Dict[str, float]
    logpf: float | Annotated[NDArray[np.float64], "1"]
    convergence: bool


@dataclass
class FragilityCurve:
    fragility_points: Optional[Annotated[List[FragilityPoint] | Tuple[FragilityPoint, ...], "n_points"]] = None
    points: Annotated[NDArray[np.float64], "n_points integration_dims"] = field(init=False)
    design_points: Annotated[NDArray[np.float64], "n_points rv_dims"] = field(init=False)
    pfs: Annotated[NDArray[np.float64], "n_points"] = field(init=False)
    betas: Annotated[NDArray[np.float64], "n_points"] = field(init=False)
    logpfs: Annotated[NDArray[np.float64], "n_points"] = field(init=False)
    alphas: Annotated[NDArray[np.float64], "n_points"] = field(init=False)
    convergences: Annotated[NDArray[np.bool], "n_points"] = field(init=False)

    def __post_init__(self) -> None:
        if self.fragility_points is not None:
            self.parse_fragility_points(self.fragility_points)

    def parse_fragility_points(
            self,
            fragility_points: Annotated[List[FragilityPoint] | Tuple[FragilityPoint, ...], "n_points"]
    ) -> None:
        self.points = np.asarray([fp.point for fp in fragility_points])
        self.design_points = np.asarray([fp.design_point for fp in fragility_points])
        self.pfs = np.asarray([fp.pf for fp in fragility_points])
        self.betas = np.asarray([fp.beta for fp in fragility_points])
        self.logpfs = np.asarray([fp.logpf for fp in fragility_points])
        self.alphas = np.asarray([fp.alphas for fp in fragility_points])
        self.convergences = np.asarray([fp.convergence for fp in fragility_points])

    def save(self, path: str | Path) -> None:
        if not isinstance(path, Path): path = Path(Path(path).as_posix())
        fc_dict = asdict(self)
        fc_dict = {key: val.squeeze().tolist() if isinstance(val, np.ndarray) else val for (key, val) in fc_dict.items()}
        fc_dict["fragility_points"] = [{
            key: val.squeeze().tolist() if isinstance(val, np.ndarray) else val for (key, val) in fp._asdict().items()
        } for fp in fc_dict["fragility_points"]]
        if not path.parent.exists(): path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f: json.dump(fc_dict, f)

    def load(self, path: str | Path) -> "FragilityCurve":
        if not isinstance(path, Path): path = Path(Path(path).as_posix())
        with open(path, 'r') as f: fc_dict = json.load(f)
        fragility_points = [
            FragilityPoint(
                point=fc_dict["points"][i],
                pf=fc_dict["pfs"][i],
                beta=fc_dict["betas"][i],
                design_point=fc_dict["design_points"][i],
                logpf=fc_dict["logpfs"][i],
                alphas = fc_dict["alphas"][i],
                convergence=fc_dict["convergences"][i]
        )
            for i, fragility_point in enumerate(fc_dict["fragility_points"])
        ]
        fc_dict["fragility_points"] = fragility_points
        return FragilityCurve(fragility_points)


class ReliabilityFragilityCurve(ReliabilityBase):

    def __init__(self, lsf: LSFType, reliability_method: str, method_params: Tuple[float | int, ...]) -> None:
        self.lsf = lsf
        self.project = ptk.ReliabilityProject()
        self.project.model = lsf
        if reliability_method.lower() == "form":
            self.adjust_form(method_params)
        else:
            raise NotImplementedError(f"Method '{reliability_method}' has not yet been implemented.")

    def adjust_form(self, form_params: Tuple[float, int, float]) -> None:
        relaxation_factor, maximum_iterations, variation_coefficient = form_params
        self.project.settings.reliability_method = ptk.ReliabilityMethod.form
        self.project.settings.relaxation_factor = relaxation_factor
        self.project.settings.maximum_iterations = maximum_iterations
        self.project.settings.variation_coefficient = variation_coefficient

    def set_fragility_rvs(self, state: Type[StateBase]) -> None:
        for rv_name in state.names:
            if state.marginal_pdf_type[rv_name] in ["normal", "multivariate_normal"]:
                self.project.variables[rv_name].distribution = ptk.DistributionType.normal
                self.project.variables[rv_name].mean = 0
                self.project.variables[rv_name].deviation = 1
            else:
                raise NotImplementedError("Non-normal distributions have not been implemented yet.")

    def generate_integration_mesh(self, n_rvs: int = 1, lims: Tuple[float, float] = (1e-5, 1-1e-5), n_grid: int = 20):
        cdf_grid = np.linspace(min(lims), max(lims), n_grid)
        grid = stats.norm(0, 1).ppf(cdf_grid)
        mesh = np.meshgrid([grid]*n_rvs)
        mesh = np.c_[*mesh]
        return mesh

    def fragility_point(
            self,
            point: Annotated[NDArray[float], "integration_dims"],
            integration_rv_names: List[str]
    ) -> ptk.FragilityValue:

        for i, rv_name in enumerate(integration_rv_names):
            self.project.variables[rv_name].distribution = ptk.DistributionType.deterministic
            self.project.variables[rv_name].mean = float(point[i])

        self.project.run()
        dp = self.project.design_point

        fragility_point = FragilityPoint(
            point=point,
            pf=dp.probability_failure,
            beta=dp.reliability_index,
            design_point={alpha.variable.name: alpha.x for alpha in dp.alphas
                          if alpha.variable.name not in integration_rv_names},
            logpf=np.log(dp.probability_failure),
            alphas = {alpha.variable.name: alpha.alpha for alpha in dp.alphas
                      if alpha.variable.name not in integration_rv_names},
            convergence=dp.is_converged
        )

        return fragility_point

    def build_fragility(
            self,
            state: Type[StateBase],
            fragility_rv_names: Optional[Annotated[Tuple[str, ...] | List[str], "fragility_dims"]] = None,
            integration_rv_names: Optional[Annotated[Tuple[str, ...] | List[str], "integration_dims"]] = None,
            n_integration_grid: int = 20,
            integration_lims: Tuple[float, float] = (1e-5, 1-1e-5),
            fc_savedir: Optional[str | Path] = None
    ) -> None:

        if (fragility_rv_names is None) == (integration_rv_names is None):
            raise ValueError("You must provide exactly one of 'fragility_rv_names' or 'integration_rv_names'")

        if fragility_rv_names is None:
            self.integration_rv_names = integration_rv_names
            self.fragility_rv_names = [rv_name for rv_name in state.names if rv_name not in integration_rv_names]

        if integration_rv_names is None:
            self.fragility_rv_names = fragility_rv_names
            self.integration_rv_names = [rv_name for rv_name in state.names if rv_name not in fragility_rv_names]

        self.set_fragility_rvs(state)

        mesh = self.generate_integration_mesh(len(integration_rv_names), integration_lims, n_integration_grid)

        fragility_points = []
        for point in tqdm(mesh, desc="Running FORM for combination of integration variables:"):
            fragility_point = self.fragility_point(point, integration_rv_names)
            fragility_points.append(fragility_point)

        self.fragility_curve = FragilityCurve(fragility_points)

        if fc_savedir is not None: self.fragility_curve.save(fc_savedir)


if __name__ == "__main__":

    geomodel_path = os.environ["MODEL_PATH"]
    form_path = os.environ["FORM_PATH"]
    geomodel = DSheetPiling(geomodel_path, form_path)

    soil_layers = {"Klei": ("soilphi", "soilcohesion")}

    rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["Klei_soilphi", "Klei_soilcohesion"])
    rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.08]), names=["water_lvl"])
    state = GaussianState(rvs=[rv_strength, rv_water])

    performance_config = ("max_moment", lambda x: 150. / (x[0] + 1e-5))
    form_params = (0.15, 3, 0.02)
    lsf = package_lsf(geomodel, state, performance_config, True)

    r = ReliabilityFragilityCurve(lsf, "form", form_params)
    r.build_fragility(
        state=state,
        integration_rv_names=["water_lvl"],
        n_integration_grid=2,
        fc_savedir=r"../../../examples/reliability/fragility_curve.json"
    )

