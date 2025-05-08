from pathlib import Path
import json
import os
import numpy as np
from docutils.nodes import title
from numpy.typing import NDArray
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Annotated, Optional
from src.rvs.state import MvnRV, GaussianState
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from scipy.integrate import trapezoid


def load_data(path: str | Path, moment_cap: float = 15., displacement_cap: float = .155) -> pd.DataFrame:

    path = Path(Path(path).as_posix())

    with open(path, "r") as f: sample_data = json.load(f)

    rvs_dict = {
        "phi": np.asarray(sample_data["Klei_soilphi"]),
        "cohesion": np.asarray(sample_data["Klei_soilcohesion"]),
        # "water": np.asarray(sample_data["water_A"]),
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
    df["true_failure"] = df["req"] * df["failure"]
    df["true_failure"] = df["true_failure"]
    df["hue"] = df["req"].astype(str) + " " + df["failure"].astype(str)

    return df


def plot_svm(df: pd.DataFrame, target: str, clf: svm.SVC, path: str | Path) -> None:

    X = df[["phi", "cohesion"]].to_numpy()
    y = df[target].to_numpy().astype(int)

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
    disp.figure_.savefig(path)


def fit_svm(df: pd.DataFrame, target: str = "failure", path: Optional[str | Path] = None) -> svm.SVC:

    X = df[["phi", "cohesion"]].to_numpy()
    y = df[target].to_numpy().astype(int)

    pipeline = make_pipeline(
        StandardScaler(),
        svm.SVC()
    )

    param_grid = {
        'svc__C': [1_000, 10_000, 100_000, 1_000_000],
        'svc__gamma': ['scale', 0.01, 0.1, 1],
        'svc__kernel': ['rbf']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5)
    grid.fit(X, y)

    best_model = grid.best_estimator_

    if path is not None:
        if not isinstance(path, Path): path = Path(Path(path).as_posix())
        plot_svm(df, target, best_model, path)

    return best_model


def failure_probability(model: svm.SVC, state: GaussianState) -> float:
    grids = {
        name: np.linspace(state.marginal_pdf[name].ppf(1e-3), state.marginal_pdf[name].ppf(1 - 1e-3), 100)
        for name in state.names
    }
    grids = list(grids.values())
    mesh = np.stack(np.meshgrid(*grids), axis=-1)

    pdf = state.joint_prob(mesh)
    failure = model.predict(mesh.reshape(-1, 2)).reshape(mesh.shape[:-1])

    pf = trapezoid(trapezoid(pdf * failure, grids[1], axis=1), grids[0], axis=0).item()

    return pf


if __name__ == "__main__":

    moment_cap = 15.
    displacement_cap = 0.15

    rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["Klei_soilphi", "Klei_soilcohesion"])
    state = GaussianState(rvs=[rv_strength])

    sample_path = r"results/sample_1000_unpooled.json"
    output_path = Path(Path(r"figures").as_posix())
    if not output_path.exists(): os.mkdir(output_path)

    df = load_data(sample_path, moment_cap, displacement_cap)

    # plot = sns.pairplot(data=df, vars=["phi", "cohesion"], hue="hue")
    plot = sns.scatterplot(data=df, x="phi", y="cohesion", hue="hue")
    plot.legend(title="Displacement / Failure")
    plt.grid()
    plt.savefig(output_path/"survival.png")

    failure_model = fit_svm(df, "failure", path=output_path/"failure_svm.png")
    true_failure_model = fit_svm(df, "true_failure", path=output_path/"true_failure_svm.png")

    print(f"Probability of failure = {failure_probability(failure_model, state) * 100:.1f} %")
    print(f"Probability of failure = {failure_probability(true_failure_model, state) * 100:.1f} %")

