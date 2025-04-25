import json
import os
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.rvs.state import MvnRV, GaussianState
from src.reliability_models.dsheetpiling.lsf import unpack_soil_params, unpack_water_params
from typing import Dict, Optional, Annotated, Tuple
from tqdm import tqdm


def sample_rvs(
        state: GaussianState,
        config: Dict[str, str | float| int],
        path: Optional[str | Path] = None
) -> Annotated[NDArray, "n_samples n_rvs"]:

    var_pooling = config["var_pooling"]
    n_locs = config["n_locs"]
    seed = config["seed"]

    if var_pooling not in ["pooled", "unpooled", "partially_pooled"]:
        raise ValueError("Unknown pooling configuration.")

    if var_pooling == "partially_pooled":
        raise NotImplementedError("Partial pooling not implemented yet.")

    if var_pooling == "pooled":
        rv_sample = state.sample(1, seed)
        rv_sample = np.repeat(rv_sample[np.newaxis, :], n_locs, axis=0)
    elif var_pooling == "unpooled":
        rv_sample = state.sample(n_locs, seed)

    return rv_sample


def sample_disp(
        rv_sample: Annotated[NDArray, "n_samples n_rvs"],
        state: GaussianState,
        model: DSheetPiling,
        config: Dict[str, str | float | int]
) -> Tuple[
    Annotated[NDArray, "n_samples n_points"],
    Annotated[NDArray, "n_samples n_points"],
    Annotated[NDArray, "n_samples n_points"]
]:

    # TODO Use list of geotechnical models. So far, I simulate n_locs locations but only use one model, assuming the
    #  same cross-section everywhere.

    var_pooling = config["var_pooling"]
    n_locs = config["n_locs"]
    seed = config["seed"]
    disp_dist_type = config["disp_dist_type"]
    disp_cov = config["disp_cov"]

    if var_pooling == "pooled": rv_sample = rv_sample[:1]

    disp_sample = []
    moment_sample = []
    for rvs in tqdm(rv_sample, desc="Calculating sample"):
        params = {name: rv for (name, rv) in zip(state.names, rvs)}
        soil_data = unpack_soil_params(params, list(model.soils.keys()))
        water_data = unpack_water_params(params, [lvl.name for lvl in model.water.water_lvls])
        model.update_soils(soil_data)
        model.update_water(water_data)
        model.execute()
        results = model.results
        disp_sample.append(results.displacement)
        moment_sample.append(results.moment)

    disp_sample = np.asarray(disp_sample)
    # Drop empty dimension if it exists (only the first one if two empty dimensions exist)
    if disp_sample.ndim > 2: disp_sample = disp_sample[0, ...]

    sample_shape = (n_locs,) + (disp_sample.shape[1:]) if var_pooling == "pooled" else disp_sample.shape

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

    moment_sample = np.asarray(moment_sample)
    # Drop empty dimension if it exists (only the first one if two empty dimensions exist)
    if moment_sample.ndim > 2: moment_sample = moment_sample[0, ...]

    return disp_sample, disp_noisy, moment_sample


def draw_sample(
        state: GaussianState,
        model: DSheetPiling,
        config: Dict[str, str | float | int],
        path: str | Path
) -> None:

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    rv_sample = sample_rvs(state, config)

    disp_sample, disp_noisy, moment_sample = sample_disp(rv_sample, state, model, config)

    log = {key: val for (key, val) in config.items()}
    log.update({name: rv.tolist() for (name, rv) in zip(state.names, rv_sample.T)})
    log.update({
        "disp_sample": disp_sample.tolist(),
        "disp_noisy": disp_noisy.tolist(),
        "moment_sample": moment_sample.tolist(),
    })

    with open(path, "w") as f: json.dump(log, f, indent=4)


if __name__ == "__main__":

    geomodel_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    geomodel = DSheetPiling(geomodel_path)

    soil_layers = {"Klei": ("soilphi", "soilcohesion")}
    rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["Klei_soilphi", "Klei_soilcohesion"])
    rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.08]), names=["water_A"])
    state = GaussianState(rvs=[rv_strength, rv_water])

    config = {
        "var_pooling": "unpooled",
        "n_locs": 1,
        "seed": 42,
        "disp_dist_type": "lognormal",
        "disp_cov": 0.1
    }

    draw_sample(state, geomodel, config, path=r"results/sample.json")

