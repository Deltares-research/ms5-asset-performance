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
    # max_n_points = max(len(disp[0]) for disp in disp_sample)
    # disp_sample = [disp if len(disp[0]) == max_n_points else [[np.nan] * max_n_points] for disp in disp_sample]
    # disp_sample = np.asarray(disp_sample).squeeze()

    # max_n_points = max(len(moment[0]) for moment in moment_sample)
    # moment_sample = [moment if len(moment[0]) == max_n_points else [[np.nan] * max_n_points] for moment in moment_sample]
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
        df,
        rv_names,
        result_path: str | Path,
) -> None:

    if not isinstance(result_path, Path): result_path = Path(Path(result_path).as_posix())
    result_path.mkdir(parents=True, exist_ok=True)

    samples = df.values
    samples = samples[:2_000]

    disp_sample, moment_sample = sample_disp(samples, rv_names, model)

    results = {
        "sample": df[cols_keep].values.tolist(),
        "displacement": disp_sample,
        "moment": moment_sample,
    }

    idx = [i for (i, d) in enumerate(disp_sample) if len(d[0]) == 150]
    print(f"NUMBER OF USABLE SAMPLES COLLECTED: {len(idx)}")

    samples = samples[idx]
    disp_sample = [d for (i, d) in enumerate(disp_sample) if i in idx]
    moment_sample = [d for (i, d) in enumerate(moment_sample) if i in idx]

    disp_sample_np = np.array(disp_sample).squeeze()
    moment_sample_np = np.array(moment_sample).squeeze()
    data = np.c_[df[cols_keep].values[:len(samples)], disp_sample_np, moment_sample_np]
    df_srg = pd.DataFrame(
        data=data,
        columns=cols_keep + [f"disp_{i}" for i in range(1, disp_sample_np.shape[1]+1)] + [f"moment_{i}" for i in range(1, moment_sample_np.shape[1]+1)]
    )
    df_srg.to_csv(result_path/"surrogate_data.csv")

    # with open(result_path / "surrogate_data.json", "w") as f:
    #     json.dump(results, f, indent=4)


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

    samples_path = r"../../../data/1M_parameter_samples_uniformly_distributed.csv"
    samples_path = Path(Path(samples_path).as_posix())
    df = pd.read_csv(samples_path)
    df["water_lvl"] = -1.1 + (df["water_lvl"] - df["water_lvl"].min()) / (df["water_lvl"].max() - df["water_lvl"].min()) * (-0.5+1.1)
    rv_names = list(df.columns)

    cols_keep = [
        'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1','Zand_soilphi', 'Zand_soilcurkb1','Zandvast_soilphi',
        'Zandvast_soilcurkb1','Zandlos_soilphi', 'Zandlos_soilcurkb1','Wall_SheetPilingElementEI', 'water_lvl'
    ]
    # df = df.loc[:, cols_keep]

    result_path = Path(__file__).parents[3] / "data"
    draw_sample(geomodel, df, rv_names, result_path=result_path)

