import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import trapezoid
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.rvs.state import MvnRV, GaussianState
from generate_data import run_model
from pathlib import Path
from typing import List, Tuple, Dict, Annotated, Optional
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def calculate_mesh(
        rv_names: List[str] | Tuple[str, ...],
        state: GaussianState,
        model: DSheetPiling,
        data: Dict[str, List[float] | float | str],
        n_grid: int = 10,
        grid_lims: Tuple[float, float] = (1e-3, 1-1e-3)
) -> Tuple[
    Dict[str, Annotated[NDArray[np.float64], "n_grid"]],
    Dict[str, Annotated[NDArray[np.float64], "n_grid"]]
]:

    grid_min, grid_max = grid_lims
    grids = {
        name: np.linspace(state.marginal_pdf[name].ppf(grid_min), state.marginal_pdf[name].ppf(grid_max), n_grid)
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
        use_noisy: bool = True,
        log_likelihood_type: str = "normal",
        n_grid: int = 10,
        grid_lims: Tuple[float, float] = (1e-3, 1-1e-3)
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

    y_hat, grids = calculate_mesh(rv_names, state, model, data, n_grid, grid_lims)
    y_hat = y_hat.squeeze()

    # cov_grid = np.logspace(-3, 0, n_grid)
    cov_grid = np.array([0.1])  # TODO: Treat CoV as a variable!

    grids = {key: val for (key, val) in grids.items() if key in rv_names}
    posterior = apply_bayes(y, y_hat, cov_grid, state, grids, log_likelihood_type)

    grids.update({"cov": cov_grid})

    return posterior.squeeze(), grids


def plot_posterior(
        posterior: Annotated[NDArray[np.float64], "n_points**(n_rvs+1)"],
        grids: Dict[str, Annotated[NDArray[np.float64], "n_points"]],
        path: str | Path,
        true_params: Optional[Tuple[float,...]] = None
) -> None:
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.0, wspace=0.0)

    ax_joint = fig.add_subplot(gs[1, 0])
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    ax_marg_x.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_marg_y.tick_params(axis="y", which="both", left=False, labelleft=False)

    extent = (
        grids["Klei_soilphi"].min(), grids["Klei_soilphi"].max(),
        grids["Klei_soilcohesion"].min(), grids["Klei_soilcohesion"].max()
    )
    im = ax_joint.imshow(posterior, cmap="viridis", extent=extent, origin="lower", aspect="auto")
    ax_joint.set_xlabel("Klei phi [deg]", fontsize=12)
    ax_joint.set_ylabel("Klei cohesion [kPa]", fontsize=12)

    marginal_phi = trapezoid(posterior, grids["Klei_soilcohesion"], axis=-1)
    ax_marg_x.fill_between(grids["Klei_soilphi"], marginal_phi, color="b", alpha=0.6)
    ax_marg_x.set_ylabel("Density [-]", fontsize=10)
    ax_marg_x.yaxis.set_visible(False)

    marginal_cohesion = trapezoid(posterior, grids["Klei_soilphi"], axis=0)
    ax_marg_y.fill_betweenx(grids["Klei_soilcohesion"], marginal_cohesion, color="b", alpha=0.6)
    ax_marg_y.set_xlabel("Density [-]", fontsize=10)
    ax_marg_y.xaxis.set_visible(False)

    ax_marg_x.set_xlim(grids["Klei_soilphi"].min(), grids["Klei_soilphi"].max())
    ax_marg_y.set_ylim(grids["Klei_soilcohesion"].min(), grids["Klei_soilcohesion"].max())

    if not true_params is None:
        true_phi, true_cohesion = true_params
        ax_joint.axvline(true_phi, c="r")
        ax_joint.axhline(true_cohesion, c="r")
        ax_joint.scatter(true_phi, true_cohesion, marker="x", c="r")
        ax_marg_x.axvline(true_phi, c="r")
        ax_marg_y.axhline(true_cohesion, c="r")

    for spine in ["top", "right", "left"]:
        ax_marg_x.spines[spine].set_visible(False)
    for spine in ["top", "bottom", "right"]:
        ax_marg_y.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.9, wspace=0.0, hspace=0.0)
    fig.subplots_adjust()

    fig.suptitle("Posterior distribution\nvia grid integration", fontsize=16)

    plt.close()
    fig.savefig(path)


if __name__ == "__main__":

    data_path = r"results/sample.json"

    geomodel_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    geomodel = DSheetPiling(geomodel_path)

    soil_layers = {"Klei": ("soilphi", "soilcohesion")}
    rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["Klei_soilphi", "Klei_soilcohesion"])
    rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.08]), names=["water_A"])
    state = GaussianState(rvs=[rv_strength, rv_water])

    rv_names = ["Klei_soilphi", "Klei_soilcohesion"]
    # posterior, grids = update(
    #     rv_names=rv_names,
    #     state=state,
    #     model=geomodel,
    #     path=data_path,
    #     use_noisy=False,
    #     log_likelihood_type="normal",
    #     n_grid=30,
    #     grid_lims=(1e-1, 1-1e-1)
    # )
    #
    # d = {"posterior": posterior.flatten().tolist(), "grids": {key: val.tolist() for (key, val) in grids.items()}}
    # with open(r"results/posterior.json", "w") as f: json.dump(d, f)

    with open(r"results/posterior.json", "r") as f: d = json.load(f)
    grids = {key: np.asarray(val) for (key, val) in d["grids"].items()}
    posterior = np.asarray(d["posterior"]).reshape(grids["Klei_soilphi"].size, grids["Klei_soilcohesion"].size)

    with open(data_path, "r") as f: data = json.load(f)
    true_params = (data["Klei_soilphi"], data["Klei_soilcohesion"])
    plot_posterior(posterior, grids, r"results/posterior.png", true_params=true_params)

