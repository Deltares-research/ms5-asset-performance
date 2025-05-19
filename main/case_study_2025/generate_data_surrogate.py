import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from src.geotechnical_models.dsheetpiling.model import DSheetPiling, DSheetPilingResults
from src.rvs.state import MvnRV, GaussianState
from src.reliability_models.dsheetpiling.lsf import *
from typing import Dict, Optional, Annotated, Tuple
from tqdm import tqdm
import shutil


def run_model(
        rvs: Annotated[NDArray[np.float64], "n_rvs"],
        model: DSheetPiling,
        rv_names
) -> DSheetPilingResults:
    params = {name: rv for (name, rv) in zip(rv_names, rvs)}
    soil_data = unpack_soil_params(params, list(model.soils.keys()))
    water_data = unpack_water_params(params, [lvl.name for lvl in model.water.water_lvls])
    wall_data = unpack_wall_data(params, model.wall._asdict())
    model.update_soils(soil_data)
    model.update_water(water_data)
    model.update_wall(wall_data)
    model.execute()
    return model.results


def log(run_index, disp_sample, moment_sample, path):

    # For some samples, DSheetpiling returns fewer points along the wall. Reject these samples.
    max_n_points = max(len(disp[0]) for disp in disp_sample)
    disp_sample = [disp if len(disp[0]) == max_n_points else [[np.nan] * max_n_points] for disp in disp_sample]
    # disp_sample = np.asarray(disp_sample).squeeze()

    max_n_points = max(len(moment[0]) for moment in moment_sample)
    moment_sample = [moment if len(moment[0]) == max_n_points else [[np.nan] * max_n_points] for moment in
                     moment_sample]
    # moment_sample = np.asarray(moment_sample).squeeze()

    results = {
        "idx": run_index,
        "displacement": disp_sample,
        "moment": moment_sample,
    }

    with open(path, "w") as f: json.dump(results, f, indent=4)


def sample_disp(
        rv_sample: Annotated[NDArray[np.float64], "n_samples n_rvs"],
        rv_names,
        model: DSheetPiling,
) -> Tuple[
    Annotated[NDArray[np.float64], "n_samples n_points"],
    Annotated[NDArray[np.float64], "n_samples n_points"],
    Annotated[NDArray[np.float64], "n_samples n_points"]
]:

    # TODO Use list of geotechnical models. So far, I simulate n_locs locations but only use one model, assuming the
    #  same cross-section everywhere.

    disp_sample = []
    moment_sample = []

    for i_run, rvs in enumerate(tqdm(rv_sample, desc="Calculating sample")):
        results = run_model(rvs, model, rv_names)
        disp_sample.append(results.displacement)
        moment_sample.append(results.moment)

    return disp_sample, moment_sample


def draw_sample(
        model: DSheetPiling,
        rv_names,
        packages,
        result_path: str | Path,
        data_path
) -> None:

    if not isinstance(result_path, Path): result_path = Path(Path(result_path).as_posix())
    if result_path.exists(): shutil.rmtree(result_path)  # Delete files
    result_path.mkdir(parents=True, exist_ok=True)

    if not isinstance(data_path, Path): data_path = Path(Path(data_path).as_posix())

    all_packages = [f for f in data_path.iterdir() if f.is_file()]
    packages = [f for f in all_packages if int(f.name.split("_")[1]) in packages]

    for package in packages:

        path = result_path/package.name
        path = path.with_suffix(".json")

        rv_samples = np.load(package)

        # rv_samples = rv_samples[:5]  # TODO: @Dafydd check this first

        run_index = rv_samples[:, 0].tolist()
        rv_samples = rv_samples[:, 1:]

        disp_sample, moment_sample = sample_disp(rv_samples, rv_names, model)

        log(run_index, disp_sample, moment_sample, path)


def split_packages(path, samples_per_split=1_000):

    path = Path(Path(path).as_posix())

    rv_samples = pd.read_csv(path)
    rv_samples = rv_samples.reset_index()
    rv_samples = rv_samples.values

    sample_packages = np.array_split(rv_samples, rv_samples.shape[0] // samples_per_split)

    path = path.parent / "data_packages"
    if path.exists(): shutil.rmtree(path)  # Delete files
    path.mkdir(parents=True, exist_ok=True)

    for i_package, package in enumerate(sample_packages):
        package_path = path / f"package_{i_package+1}_from_{i_package*samples_per_split}_to_{i_package*samples_per_split+samples_per_split-1}.npy"
        np.save(package_path, package)


def get_samples(n_runs, start, idx_start, path, split_dataset=False):

    if split_dataset: split_packages(path)

    if start == "top":
        rv_samples = rv_samples[idx_start: idx_start+n_runs]
        result_path = f"data/surrogate_sample_{idx_start}_{idx_start+n_runs-1}.json"
    else:
        rv_samples = rv_samples[idx_start-n_runs-1: idx_start-1]
        result_path = f"data/surrogate_sample_{idx_start-n_runs-1}_{idx_start-2}.json"

    return rv_samples, result_path, rv_names


if __name__ == "__main__":

    geomodel_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    geomodel = DSheetPiling(geomodel_path)

    samples_path = r"data/1M_parameter_samples.csv"
    samples_path = Path(Path(samples_path).as_posix())
    df = pd.read_csv(samples_path)
    rv_names = list(df.columns)

    # split_packages(samples_path, samples_per_split=1_000)

    packages = list(range(1, 3))
    data_path = r"data/data_packages"
    result_path = r"data/result_packages"
    draw_sample(geomodel, rv_names, packages=packages, result_path=result_path, data_path=data_path)

