import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import datetime


if __name__ == "__main__":

    path = os.environ["SRG_DATA_PATH"]

    path = Path(Path(path).as_posix())

    df = pd.read_csv(path/"1M_parameter_samples.csv")
    rv_names = list(df.columns)

    rv_path = path / "data_packages"
    data_package_paths = {int(f.name.split("_")[1]): f for f in rv_path.iterdir()}

    result_path_1 = path / "result_packages_Dafydd"
    result_path_2 = path / "result_packages_Antonis"
    result_files = [f for f in result_path_1.iterdir()] + [f for f in result_path_2.iterdir()]
    result_files = {int(f.name.split("_")[1]): f for f in result_files}
    packages = sorted(list(result_files.keys()))

    idxs = []
    Xs = []
    ys = []
    for package in packages:

        rvs = np.load(data_package_paths[package])[:, 1:]

        with open(result_files[package], "r") as f:
            results = json.load(f)

        idx = results["idx"]
        displacements = np.asarray(results["displacement"]).squeeze()
        # moments = np.asarray(results["moment"]).squeeze()

        idxs.append(idx)
        Xs.append(rvs)
        ys.append(displacements)

    idxs = np.hstack(idxs)
    Xs = np.vstack(Xs)
    ys = np.vstack(ys)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    output_name = "srg_data_" + timestamp + ".csv"
    output_path = path / "compiled_data" / output_name
    df_data = pd.DataFrame(
        data=np.c_[idxs, Xs, ys],
        columns=["index"]+rv_names+[f"disp_{i+1}" for i in range(ys.shape[1])]
    )
    df_data.to_csv(output_path, index=False)

    output_name = "srg_data_" + timestamp + ".json"
    output_path = path / "compiled_data" / output_name
    data = {
        "idx": idxs.tolist(),
        "X": Xs.tolist(),
        "y": ys.tolist()
    }
    with open(output_path, "w") as f: json.dump(data, f, indent=4)

