import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import trapezoid
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.rvs.state import MvnRV, GaussianState
from generate_data import run_model
from pathlib import Path
from typing import List, Tuple, Dict, Annotated, Optional, NamedTuple
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class PosteriorResults(NamedTuple):
    rv_names: List[str] | Tuple[str, ...]
    posterior: List[List[float]]
    grids: Dict[str, List[float]]
    data: Dict[str, List[float]]
    true_params: Tuple[float, ...]
    map: Tuple[float, ...]
    mse: Tuple[float, ...]
    use_noisy: bool
    log_likelihood_type: str
    n_grid: int
    grid_lims: Tuple[float, float]


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
            grids[name] = np.unique(np.asarray(data[name]))

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

    if y.ndim == 1: y = np.expand_dims(y, axis=0)

    if log_likelihood_type == "normal":

        mean = y_hat[np.newaxis, :, np.newaxis, :]
        sigma = np.abs(mean) * cov_grid[:, *tuple([np.newaxis]*y_hat.ndim)]
        sigma = np.clip(sigma, 1e-3, None)
        log_likehoods = stats.norm(loc=mean, scale=sigma).logpdf(y)
        # Sum over axis=-1 (point along wall) and axis=-2 (location)
        log_likehood = np.nansum(log_likehoods, axis=(-2, -1)).squeeze()
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
        grid_lims: Tuple[float, float] = (1e-3, 1-1e-3),
        export_path: Optional[str | Path] = None
) -> PosteriorResults:

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
    posterior = posterior.squeeze()

    grids.update({"cov": cov_grid})

    true_params = tuple([data[name][0] for name in rv_names])

    map_idxs = np.unravel_index(np.argmax(posterior), posterior.shape)
    map_point = (
        grids["Klei_soilphi"][map_idxs[0]].tolist(),
        grids["Klei_soilcohesion"][map_idxs[1]].tolist()
    )

    marginals = (
        trapezoid(posterior, grids["Klei_soilcohesion"], axis=-1),
        trapezoid(posterior, grids["Klei_soilphi"], axis=0)
    )

    mses = (
        np.sum(marginals[0] * (grids["Klei_soilphi"] - true_params[0]) ** 2).item(),
        np.sum(marginals[1] * (grids["Klei_soilphi"] - true_params[1]) ** 2).item()
    )

    posterior_results = PosteriorResults(
        rv_names=rv_names,
        posterior=posterior.tolist(),
        grids={key: val.tolist() for (key, val) in grids.items()},
        data=data,
        true_params=true_params,
        map=map_point,
        mse=mses,
        use_noisy=use_noisy,
        log_likelihood_type=log_likelihood_type,
        n_grid=n_grid,
        grid_lims=grid_lims
    )

    if export_path is not None:
        if not isinstance(export_path, Path): export_path = Path(Path(export_path).as_posix())
        posterior_dict = posterior_results._asdict()
        with open(export_path / "posterior_results.json", "w") as f: json.dump(posterior_dict, f, indent=4)

    return posterior_results


def plot_posterior(
        posterior_results: PosteriorResults,
        path: str | Path,
) -> None:
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    posterior = np.asarray(posterior_results.posterior)
    grids = posterior_results.grids

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.0, wspace=0.0)

    ax_joint = fig.add_subplot(gs[1, 0])
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    ax_marg_x.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_marg_y.tick_params(axis="y", which="both", left=False, labelleft=False)

    extent = (
        min(grids["Klei_soilphi"]), max(grids["Klei_soilphi"]),
        min(grids["Klei_soilcohesion"]), max(grids["Klei_soilcohesion"])
    )
    im = ax_joint.imshow(posterior.T, cmap="viridis", extent=extent, origin="lower", aspect="auto")  # Transpose for imshow
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

    ax_marg_x.set_xlim(min(grids["Klei_soilphi"]), max(grids["Klei_soilphi"]))
    ax_marg_y.set_ylim(min(grids["Klei_soilcohesion"]), max(grids["Klei_soilcohesion"]))

    map_phi, map_cohesion = posterior_results.map
    ax_joint.axvline(map_phi, c="b")
    ax_joint.axhline(map_cohesion, c="b")
    ax_joint.scatter(map_phi, map_cohesion, marker="x", c="r", label="True MAP")
    ax_marg_x.axvline(map_phi, c="b")
    ax_marg_y.axhline(map_cohesion, c="b")

    if not posterior_results.true_params is None:
        true_phi, true_cohesion = posterior_results.true_params
        ax_joint.axvline(true_phi, c="r")
        ax_joint.axhline(true_cohesion, c="r")
        ax_joint.scatter(true_phi, true_cohesion, marker="x", c="r", label="True values")
        ax_marg_x.axvline(true_phi, c="r")
        ax_marg_y.axhline(true_cohesion, c="r")

    for spine in ["top", "right", "left"]:
        ax_marg_x.spines[spine].set_visible(False)
    for spine in ["top", "bottom", "right"]:
        ax_marg_y.spines[spine].set_visible(False)

    # plt.legend(fontsize=10)

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.9, wspace=0.0, hspace=0.0)
    fig.subplots_adjust()

    fig.suptitle("Posterior distribution\nvia grid integration", fontsize=16)

    plt.close()
    fig.savefig(path)


if __name__ == "__main__":

    data_path = r"results/sample.json"
    # data_path = r"results/sample_10_pooled.json"

    geomodel_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    geomodel = DSheetPiling(geomodel_path)

    soil_layers = {"Klei": ("soilphi", "soilcohesion")}
    rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["Klei_soilphi", "Klei_soilcohesion"])
    rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.08]), names=["water_A"])
    state = GaussianState(rvs=[rv_strength, rv_water])

    rv_names = ["Klei_soilphi", "Klei_soilcohesion"]

    posterior_results = update(
        rv_names=rv_names,
        state=state,
        model=geomodel,
        path=data_path,
        use_noisy=False,
        # use_noisy=False,
        log_likelihood_type="normal",
        n_grid=10,
        grid_lims=(1e-1, 1-1e-1),
        export_path=r"results"
    )

    with open(r"results/posterior_results.json", "r") as f: posterior_dict = json.load(f)
    posterior_results = PosteriorResults(**posterior_dict)
    plot_posterior(posterior_results, r"figs/posterior.png")

