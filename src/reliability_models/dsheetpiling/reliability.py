import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator
import math
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
    point: Annotated[List[float] | Tuple[float, ...], "integration_dims"]
    pf: float
    beta: float
    design_point: Dict[str, float]
    alphas: Dict[str, float]
    logpf: float
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
        with open(path, 'w') as f: json.dump(fc_dict, f, indent=4)


class ReliabilityFragilityCurve(ReliabilityBase):

    def __init__(
            self,
            lsf: LSFType,
            state: Type[StateBase],
            reliability_method: str,
            form_params: Tuple[float | int, ...],
            fragility_rv_names: Optional[Annotated[Tuple[str, ...] | List[str], "fragility_dims"]] = None,
            integration_rv_names: Optional[Annotated[Tuple[str, ...] | List[str], "integration_dims"]] = None
    ) -> None:
        self.lsf = lsf
        self.project = ptk.ReliabilityProject()
        self.project.model = lsf
        if reliability_method.lower() == "form":
            self.adjust_form(form_params)
        else:
            raise NotImplementedError(f"Method '{reliability_method}' has not yet been implemented.")

        if (fragility_rv_names is None) == (integration_rv_names is None):
            raise ValueError("You must provide exactly one of 'fragility_rv_names' or 'integration_rv_names'")

        if fragility_rv_names is None:
            self.integration_rv_names = integration_rv_names
            self.fragility_rv_names = [rv_name for rv_name in state.names if rv_name not in integration_rv_names]

        if integration_rv_names is None:
            self.fragility_rv_names = fragility_rv_names
            self.integration_rv_names = [rv_name for rv_name in state.names if rv_name not in fragility_rv_names]

        self.set_fragility_rvs(state)
        self.state = state

    def adjust_form(self, form_params: Tuple[float, int, float]) -> None:
        relaxation_factor, maximum_iterations, variation_coefficient = form_params
        self.project.settings.reliability_method = ptk.ReliabilityMethod.form
        self.project.settings.relaxation_factor = relaxation_factor
        self.project.settings.maximum_iterations = maximum_iterations
        self.project.settings.variation_coefficient = variation_coefficient

    def set_fragility_rvs(self, state: Type[StateBase]) -> None:
        for i, rv_name in enumerate(state.names):
            if state.marginal_pdf_type[rv_name] in ["normal", "multivariate_normal"]:
                self.project.variables[rv_name].distribution = ptk.DistributionType.normal
                self.project.variables[rv_name].mean = 0
                self.project.variables[rv_name].deviation = 1
            else:
                raise NotImplementedError("Non-normal distributions have not been implemented yet.")

    def generate_integration_mesh(self, lims: Tuple[float, float] = (1e-5, 1-1e-5), n_grid: int = 20) -> None:
        n_rvs = len(self.integration_rv_names)
        cdf_grid = np.linspace(min(lims), max(lims), n_grid)
        grid = stats.norm(0, 1).ppf(cdf_grid)
        mesh = np.meshgrid(*[[grid] for _ in range(n_rvs)])
        mesh = np.c_[*[m.flatten() for m in mesh]]
        self.fc_mesh = mesh

    def fragility_point(self, point: Annotated[NDArray[float], "integration_dims"]) -> FragilityPoint:

        for i, rv_name in enumerate(self.integration_rv_names):
            self.project.variables[rv_name].distribution = ptk.DistributionType.deterministic
            self.project.variables[rv_name].mean = float(point[i])

        self.project.run()
        dp = self.project.design_point

        fragility_point = FragilityPoint(
            point=point.tolist(),
            pf=dp.probability_failure,
            beta=dp.reliability_index,
            design_point={alpha.variable.name: alpha.x for alpha in dp.alphas
                          if alpha.variable.name not in self.integration_rv_names},
            logpf=math.log(dp.probability_failure),
            alphas = {alpha.variable.name: alpha.alpha for alpha in dp.alphas
                      if alpha.variable.name not in self.integration_rv_names},
            convergence=dp.is_converged
        )

        return fragility_point

    def build_fragility(
            self,
            n_integration_grid: int = 20,
            integration_lims: Tuple[float, float] = (1e-5, 1-1e-5),
            fc_savedir: Optional[str | Path] = None
    ) -> None:

        if fc_savedir is not None:
            if not isinstance(fc_savedir, Path): fc_savedir = Path(Path(fc_savedir).as_posix())
            fc_savedir.parent.mkdir(parents=True, exist_ok=True)

        self.generate_integration_mesh(integration_lims, n_integration_grid)

        fragility_points = []
        for point in tqdm(self.fc_mesh, desc="Running FORM for combination of integration variables:"):
            fragility_point = self.fragility_point(point)
            fragility_points.append(fragility_point)

        self.fragility_curve = FragilityCurve(fragility_points)

        if fc_savedir is not None: self.fragility_curve.save(fc_savedir)

    def load_fragility(self, path: str | Path) -> None:
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
        self.fragility_curve = FragilityCurve(fragility_points)

    def integrate_fragility(self, n_interp: int  = 1_000) -> Tuple[float, float]:

        n_integration_rvs = len(self.integration_rv_names)

        if n_integration_rvs > 1:

            self.generate_integration_mesh(n_grid=n_interp)

            idx_integration_rvs = np.asarray([self.state.names.index(rv) for rv in self.integration_rv_names])
            mus = self.state.mus[idx_integration_rvs]
            cov = self.state.cov[idx_integration_rvs][:, idx_integration_rvs]

            detransformed_mesh = mus + np.sqrt(np.diag(cov)) * self.fc_mesh
            grids = tuple(np.sort(np.unique(m)) for m in detransformed_mesh.T)
            shapes = tuple(grid.size for grid in grids)

            coords = self.fragility_curve.points
            detransformed_coords = mus + np.sqrt(np.diag(cov)) * coords
            x = tuple([np.unique(coord) for coord in detransformed_coords.T])
            y = self.fragility_curve.logpfs
            n_grid = int(np.sqrt(y.size))
            y = y.reshape(len(x[0]), len(x[1])).T
            interp = RegularGridInterpolator(x, y, bounds_error=False, fill_value=None)
            logpfs_interp = interp(detransformed_mesh)

            log_prob = stats.multivariate_normal(mus, cov).logpdf(detransformed_mesh)
            pf = log_prob + logpfs_interp
            pf = np.exp(pf).reshape(shapes)

            for i_axis, grid in reversed(list(enumerate(grids))):
                pf = trapezoid(pf, grid, axis=i_axis)

        else:

            marginal_pdf = self.state.marginal_pdf[self.integration_rv_names[0]]
            quantiles = marginal_pdf.ppf([0.001, 0.999])
            x = np.linspace(quantiles[0], quantiles[1], n_interp)
            log_prob = marginal_pdf.logpdf(x)

            log_pfs = self.fragility_curve.logpfs
            xs = self.fragility_curve.points

            log_pf = np.interp(x, xs, log_pfs)

            pf = log_prob + log_pf
            pf = np.exp(pf)
            pf = trapezoid(pf, x)

        beta = stats.norm.ppf(1-pf)

        return pf, beta

    def pf_point(self, point):

        idx_integration_rvs = np.asarray([self.state.names.index(rv) for rv in self.integration_rv_names])
        mus = self.state.mus[idx_integration_rvs]
        cov = self.state.cov[idx_integration_rvs][:, idx_integration_rvs]

        coords = self.fragility_curve.points
        detransformed_coords = mus + np.sqrt(np.diag(cov)) * coords
        x = tuple([np.unique(coord) for coord in detransformed_coords.T])
        y = self.fragility_curve.logpfs
        y = y.reshape(len(x[0]), len(x[1])).T
        interp = RegularGridInterpolator(x, y, bounds_error=False, fill_value=None)
        logpfs_interp = interp(point)

        pfs = np.exp(logpfs_interp)

        return pfs

    def compile_fragility_points(self, path: str | Path) -> None:

        json_files = list(path.glob("*.json"))

        fragility_points = []
        for json_file in json_files:
            with open(json_file, 'r') as f: fp_dict = json.load(f)
            fragility_point = FragilityPoint(**fp_dict)
            if np.isnan(fragility_point.pf):
                continue
            fragility_points.append(fragility_point)

        if len(fragility_points) > 0:
            self.fragility_curve = FragilityCurve(fragility_points)
        else:
            raise ValueError("No acceptable entries in the fragility points!")


if __name__ == "__main__":

    pass

