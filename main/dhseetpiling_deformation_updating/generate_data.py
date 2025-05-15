import json
import os
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from src.geotechnical_models.dsheetpiling.model import DSheetPiling, DSheetPilingResults
from src.rvs.state import MvnRV, GaussianState
from src.reliability_models.dsheetpiling.lsf import unpack_soil_params, unpack_water_params
from typing import Dict, Optional, Annotated, Tuple
from tqdm import tqdm


def sample_rvs(
        state: GaussianState,
        config: Dict[str, str | float| int],
) -> Annotated[NDArray[np.float64], "n_samples n_rvs"]:

    rv_pooling = config["rv_pooling"]
    n_locs = config["n_locs"]
    seed = config["seed"]

    if rv_pooling not in ["pooled", "unpooled", "partially_pooled"]:
        raise ValueError("Unknown pooling configuration.")

    if rv_pooling == "partially_pooled":
        raise NotImplementedError("Partial pooling not implemented yet.")

    if rv_pooling == "pooled":
        rv_sample = state.sample(1, seed)
        rv_sample = np.repeat(rv_sample[np.newaxis, :], n_locs, axis=0)
    elif rv_pooling == "unpooled":
        rv_sample = state.sample(n_locs, seed)
        if n_locs == 1: rv_sample = rv_sample[np.newaxis, :]

    return rv_sample


def run_model(
        rvs: Annotated[NDArray[np.float64], "n_rvs"],
        model: DSheetPiling,
        state: GaussianState
) -> DSheetPilingResults:
    params = {name: rv for (name, rv) in zip(state.names, rvs)}
    soil_data = unpack_soil_params(params, list(model.soils.keys()))
    water_data = unpack_water_params(params, [lvl.name for lvl in model.water.water_lvls])
    model.update_soils(soil_data)
    model.update_water(water_data)
    model.execute()
    return model.results


def sample_disp(
        rv_sample: Annotated[NDArray[np.float64], "n_samples n_rvs"],
        state: GaussianState,
        model: DSheetPiling,
        config: Dict[str, str | float | int]
) -> Tuple[
    Annotated[NDArray[np.float64], "n_samples n_points"],
    Annotated[NDArray[np.float64], "n_samples n_points"],
    Annotated[NDArray[np.float64], "n_samples n_points"]
]:

    # TODO Use list of geotechnical models. So far, I simulate n_locs locations but only use one model, assuming the
    #  same cross-section everywhere.

    rv_pooling = config["rv_pooling"]
    n_locs = config["n_locs"]
    seed = config["seed"]
    disp_dist_type = config["disp_dist_type"]
    disp_cov = config["disp_cov"]

    if rv_pooling == "pooled": rv_sample = rv_sample[:1]

    disp_sample = []
    moment_sample = []
    for rvs in tqdm(rv_sample, desc="Calculating sample"):
        results = run_model(rvs, model, state)
        disp_sample.append(results.displacement)
        moment_sample.append(results.moment)

    # For some samples, DSheetpiling returns fewer points along the wall. Reject these samples.
    max_n_points = max(len(disp[0]) for disp in disp_sample)
    disp_sample = [disp if len(disp[0]) == max_n_points else [[np.nan]*max_n_points] for disp in disp_sample]
    disp_sample = np.asarray(disp_sample)
    # Drop empty dimension if it exists (only the first one if two empty dimensions exist)
    if disp_sample.ndim > 2: disp_sample = disp_sample[:, 0]

    sample_shape = (n_locs,) + (disp_sample.shape[1:]) if rv_pooling == "pooled" else disp_sample.shape

    np.random.seed(seed)

    if disp_dist_type == "normal":
        disp_noisy = disp_sample * (1 + np.random.randn(*sample_shape) * disp_cov)
    elif disp_dist_type == "lognormal":
        sign_disp_sample = np.sign(disp_sample)
        log_disp_sample = np.log(np.abs(disp_sample))
        log_disp_noise = np.random.randn(*sample_shape) * disp_cov * log_disp_sample
        disp_noisy = sign_disp_sample * np.exp(log_disp_sample+log_disp_noise)
    else:
        raise NotImplementedError(f"Distribution '{disp_dist_type}' not implemented.")


    max_n_points = max(len(moment[0]) for moment in moment_sample)
    moment_sample = [moment if len(moment[0]) == max_n_points else [[np.nan]*max_n_points] for moment in moment_sample]
    moment_sample = np.asarray(moment_sample)
    # Drop empty dimension if it exists (only the first one if two empty dimensions exist)
    if moment_sample.ndim > 2: moment_sample = moment_sample[:, 0]

    return disp_sample, disp_noisy, moment_sample


def draw_sample(
        state: GaussianState,
        model: DSheetPiling,
        config: Dict[str, str | float | int],
        path: str | Path
) -> None:

    rv_sample = sample_rvs(state, config)

    disp_sample, disp_noisy, moment_sample = sample_disp(rv_sample, state, model, config)

    log = {key: val for (key, val) in config.items()}
    log.update({name: rv.tolist() for (name, rv) in zip(state.names, rv_sample.T)})
    log.update({
        "displacement": [[None if np.isnan(x) else x for x in row] for row in disp_sample],
        "displacement_noisy": [[None if np.isnan(x) else x for x in row] for row in disp_noisy],
        "moment":[[None if np.isnan(x) else x for x in row] for row in moment_sample],
    })

    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    if not path.exists(): os.mkdir(path)
    with open(path, "w") as f: json.dump(log, f, indent=4)


if __name__ == "__main__":

    geomodel_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    geomodel = DSheetPiling(geomodel_path)

    soil_layers = {"Klei": ("soilphi", "soilcohesion")}
    rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["Klei_soilphi", "Klei_soilcohesion"])
    # rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.08]), names=["water_A"])
    state = GaussianState(rvs=[rv_strength])

    config = {
        "rv_pooling": "unpooled",
        "n_locs": 1_000,
        "seed": 42,
        # "disp_dist_type": "lognormal",
        "disp_dist_type": "normal",
        "disp_cov": 0.1
    }

    if config["n_locs"] == 1:
        draw_sample(state, geomodel, config, path=r"results/sample.json")
    else:
        draw_sample(state, geomodel, config, path=f"results/sample_{config['n_locs']}_{config['rv_pooling']}.json")

