from pathlib import Path
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":

    moment_cap = 15.
    output_dir = r"results"

    output_dir = Path(Path(output_dir).as_posix())

    sample_path = Path(Path(r"results/sample_10000_unpooled.json").as_posix())
    with open(sample_path, "r") as f: sample_data = json.load(f)

    rvs_dict = {
        "phi": np.asarray(sample_data["Klei_soilphi"]),
        "cohesion": np.asarray(sample_data["Klei_soilcohesion"]),
        "water": np.asarray(sample_data["water_A"]),
    }

    moments = np.array([[np.nan if x is None else x for x in row] for row in sample_data["moment"]])
    displacements = np.array([[np.nan if x is None else x for x in row] for row in sample_data["displacement"]])

    idx_keep = np.unique(np.concatenate((
        np.where(np.all(~np.isnan(displacements), axis=1)),
        np.where(np.all(~np.isnan(moments), axis=1))
    )))

    rvs_dict = {key: val[idx_keep] for (key, val) in rvs_dict.items()}
    rvs = np.stack(list(rvs_dict.values()), axis=-1)

    moments = moments[idx_keep]
    displacements = displacements[idx_keep]
    max_moments = moments.max(axis=1)
    fos = moment_cap / max_moments

    df = pd.DataFrame(
        data=np.concatenate((rvs, np.expand_dims(fos, 1)), axis=1),
        columns=list(rvs_dict.keys()) + ["fos"]
    )
    df["fail"] = df["fos"] < 1

    hist = sns.histplot(data=df, x="fos", stat="density")
    plt.savefig(output_dir/"fos_histogram.png")

    plot = sns.pairplot(data=df, vars=["phi", "cohesion", "water"], hue="fail")
    plot.savefig(output_dir/"survival.png")



