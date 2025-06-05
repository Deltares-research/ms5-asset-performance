import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import datetime


if __name__ == "__main__":

    path = os.environ["SRG_DATA_PATH"]

    path = Path(Path(path).as_posix())

    df = pd.read_csv(path/"1M_parameter_samples_uniformly_distributed.csv")
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

        idx = [idx for (d, idx) in zip(results["displacement"], results["idx"]) if len(d[0]) == 156]
        rvs = [rv for (d, rv) in zip(results["displacement"], rvs) if len(d[0]) == 156]
        displacements = [d[0] for d in results["displacement"] if len(d[0]) == 156]
        displacements = np.asarray(displacements).squeeze()

        idxs.append(idx)
        Xs.append(rvs)
        ys.append(displacements)

    idxs = np.concatenate(idxs)
    Xs = np.vstack(Xs)
    ys = np.vstack(ys)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    output_name = "srg_data_" + timestamp + ".csv"
    output_path = path / "compiled_data" / output_name
    df_data = pd.DataFrame(
        data=np.column_stack([idxs, Xs, ys]),
        columns=["index"]+rv_names+[f"disp_{i+1}" for i in range(ys.shape[1])]
    )
    cols_keep = [
        'Klei_soilcohesion', 'Klei_soilphi', 'Klei_soilcurkb1','Zand_soilphi', 'Zand_soilcurkb1', 'Zandvast_soilphi',
        'Zandvast_soilcurkb1','Zandlos_soilphi', 'Zandlos_soilcurkb1','Wall_SheetPilingElementEI', 'water_lvl'
    ]
    df_data = df_data.loc[:, ["index"]+cols_keep+[f"disp_{i+1}" for i in range(ys.shape[1])]]
    df_data.to_csv(output_path, index=False)
    Xs = df_data[cols_keep].values

    output_name = "srg_data_" + timestamp + ".json"
    output_path = path / "compiled_data" / output_name
    data = {
        "idx": idxs.tolist(),
        "X": Xs.tolist(),
        "y": ys.tolist()
    }
    with open(output_path, "w") as f: json.dump(data, f, indent=4)

