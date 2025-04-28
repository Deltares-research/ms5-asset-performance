import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import trapezoid
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.rvs.state import MvnRV, GaussianState
from generate_data import run_model
from pathlib import Path
from typing import List, Tuple, Dict, Annotated
import json
import os
from tqdm import tqdm


def calculate_mesh(
        rv_names: List[str] | Tuple[str, ...],
        state: GaussianState,
        model: DSheetPiling,
        data: Dict[str, List[float] | float | str],
        n_grid: int = 10
) -> Tuple[
    Dict[str, Annotated[NDArray[np.float64], "n_grid"]],
    Dict[str, Annotated[NDArray[np.float64], "n_grid"]]
]:

    grids = {
        name: np.linspace(state.marginal_pdf[name].ppf(1e-3), state.marginal_pdf[name].ppf(1 - 1e-3), n_grid)
        for name in state.names
    }

    for name in grids.keys():
        if name not in rv_names:
            grids[name] = np.asarray(data[name])

    mesh = np.meshgrid(*list(grids.values()))
    mesh = np.c_[[m.flatten() for m in mesh]].T

    y_hat = []
    for rvs in tqdm(mesh, desc="Calculating mesh point"):
        results = run_model(rvs, model, state)
        y_hat.append(results.displacement)

    y_hat = np.asarray(y_hat)

    return y_hat, grids


def apply_bayes(
        y: Annotated[NDArray[np.float64], "n_points"],
        y_hat: Annotated[NDArray[np.float64], "mesh_size n_points"],
        cov_grid: Annotated[NDArray[np.float64], "n_grid"],
        state: GaussianState,
        grids: Dict[str, Annotated[NDArray[np.float64], "n_grid"]],
        log_likelihood_type: str = "normal"
) -> Annotated[NDArray[np.float64], "mesh_size"]:

    n_rvs = len(list(grids.values())) + 1  # +1 for CoV
    n_grid = list(grids.values())[0].size

    if log_likelihood_type == "normal":

        sigma = y_hat[np.newaxis, ...] * cov_grid[:, *tuple([np.newaxis]*y_hat.ndim)]
        log_likehoods = stats.norm(loc=y_hat[np.newaxis, ...], scale=sigma).logpdf(y)
        log_likehood = np.nansum(log_likehoods, axis=-1).squeeze()
        log_likehood = log_likehood.reshape(cov_grid.size, *tuple([n_grid]*(n_rvs-1)))

    elif log_likelihood_type == "lognormal":

        log_likehood_fn = 0

    else:

        raise NotImplementedError(f"Loglikelihood model '{log_likelihood_type}' not implemented.")

    log_prior_rv = sum(np.meshgrid(*[state.marginal_pdf[name].logpdf(grid) for (name, grid) in grids.items()]))
    log_prior_cov = 0  # Uniform prior for CoV
    log_prior = log_prior_rv + log_prior_cov

    log_posterior = log_prior + log_likehood
    posterior = np.exp(log_posterior)

    grids_all = [cov_grid] + list(grids.values())

    posterior_integral = posterior.copy()
    for i_axis, grid in reversed(list(enumerate(grids_all))):
        if posterior_integral.shape[i_axis] > 1:  # Safeguard aganist nullification if not using CoV grid, but a value
            posterior_integral = trapezoid(posterior_integral, grid, axis=i_axis)
        else:
            continue

    posterior /= posterior_integral

    return posterior


def update(
        rv_names: List[str] | Tuple[str, ...],
        state: GaussianState,
        model: DSheetPiling,
        path: str | Path,
        n_grid: int = 10,
        use_noisy: bool = True,
        log_likelihood_type: str = "normal"
) -> Tuple[
    Annotated[NDArray[np.float64], "n_points**(n_rvs+1)"],
    Dict[str, Annotated[NDArray[np.float64], "n_points"]]
]:

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    with open(path, "r") as f: data = json.load(f)

    if use_noisy:
        y = np.asarray(data["displacement_noisy"]).squeeze()
    else:
        y = np.asarray(data["displacement"]).squeeze()

    y_hat, grids = calculate_mesh(rv_names, state, model, data, n_grid)
    y_hat = y_hat.squeeze()

    # cov_grid = np.logspace(-3, 0, n_grid)
    cov_grid = np.array([0.1])  # TODO: Treat CoV as a variable!

    grids = {key: val for (key, val) in grids.items() if key in rv_names}
    posterior = apply_bayes(y, y_hat, cov_grid, state, grids, log_likelihood_type)

    grids.update({"cov": cov_grid})

    return posterior, grids


if __name__ == "__main__":

    data_path = r"results/sample.json"

    geomodel_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    geomodel = DSheetPiling(geomodel_path)

    soil_layers = {"Klei": ("soilphi", "soilcohesion")}
    rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["Klei_soilphi", "Klei_soilcohesion"])
    rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.08]), names=["water_A"])
    state = GaussianState(rvs=[rv_strength, rv_water])

    rv_names = ["Klei_soilphi", "Klei_soilcohesion"]
    # update(rv_names, state, geomodel, data_path, use_noisy=False, log_likelihood_type="lognormal", n_grid=3)
    posterior, grids = update(rv_names, state, geomodel, data_path, use_noisy=False, log_likelihood_type="normal", n_grid=20)
    posterior = posterior.squeeze()



