from pathlib import Path
import json
import numpy as np
from docutils.nodes import title
from numpy.typing import NDArray
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from typing import Annotated


def load_data(path: str | Path, moment_cap: float = 15., displacement_cap: float = .155) -> pd.DataFrame:

    path = Path(Path(path).as_posix())

    with open(path, "r") as f: sample_data = json.load(f)

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
    
    top_displacement = np.abs(displacements[:, 0])
    meets_displacement = top_displacement / displacement_cap

    df = pd.DataFrame(
        data=np.concatenate((rvs, np.expand_dims(fos, 1)), axis=1),
        columns=list(rvs_dict.keys()) + ["fos"]
    )

    df["req"] = meets_displacement < 1
    df["failure"] = df["fos"] < 1
    df["true_failure"] = df["req"] * df["fos"]
    df["true_failure"] = df["true_failure"]
    df["hue"] = df["req"].astype(str) + " " + df["req"].astype(str)

    return df


def plot_svm(
        X: Annotated[NDArray[np.float64], "data_size features"],
        y: Annotated[NDArray[int], "data_size"],
        clf: svm.SVC,
        path: str | Path
) -> None:

    disp = DecisionBoundaryDisplay.from_estimator(
        clf, X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.5
    )

    fig = plt.figure()
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    disp.ax_.set_xlabel("φ [deg]", fontsize=12)
    disp.ax_.set_ylabel("c [kPa]", fontsize=12)
    disp.figure_.savefig(path/"svm.png")


if __name__ == "__main__":

    moment_cap = 15.
    displacement_cap = 0.16

    sample_path = r"results/sample_10000_unpooled.json"
    output_path = Path(Path(r"results").as_posix())

    df = load_data(sample_path, moment_cap, displacement_cap)

    plot = sns.pairplot(data=df, vars=["phi", "cohesion"], hue="hue")
    plot._legend.set_title(title="Displacement / Failure")
    plot.savefig(output_path/"survival.png")

    X = df[["phi", "cohesion"]].to_numpy()
    y = df["true_failure"].to_numpy().astype(int)
    # clf = svm.SVC(C=1e10).fit(X, y)
    clf = svm.SVC().fit(X, y)

    plot_svm(X, y, clf, output_path)


