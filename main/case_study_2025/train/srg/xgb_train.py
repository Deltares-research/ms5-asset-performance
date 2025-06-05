import os
import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from main.case_study_2025.train.srg.utils import load_data, plot
from typing import Tuple, Optional
from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib
from tqdm import tqdm
import typer
from datetime import datetime
import joblib


app = typer.Typer()


@app.command()
def train(n_estimators: int = 1_000, max_depth: int = 20, lr: float = 0.05):

    base_dir = Path(__file__).resolve().parent

    data_dir = base_dir.parent / "data"
    data_path = data_dir / "srg_data_20250604_100638.csv"

    output_path = base_dir.parent / f"results/srg/xgb/lr_{lr:.1e}_estimators_{n_estimators:d}_maxdepth_{max_depth:d}"
    output_path.mkdir(parents=True, exist_ok=True)

    X, y = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    print("Training...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        verbosity=1,
    )

    model = MultiOutputRegressor(xgb_model)
    model.fit(X_train, y_train)

    y_hat = inference(model, X_test, scaler_x, scaler_y)
    rmse = mean_squared_error(y_hat, y_test)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"lr={lr:.1e} | {n_estimators:d} estimators | {max_depth:d} max_depth | RMSE: {rmse:.2f}"
    with open(output_path/"training_log.txt", "a") as f:
        f.write(f"{timestamp} | " + message)

    print("Training completed! ✅")

    print("[SUMMARY] "+message)

    joblib.dump(model, output_path / "model.pkl")

    print("Plotting results...")

    plot(inference, model, X_train, X_test, y_train, y_test, scaler_x, scaler_y, output_path, None)

    print("Results plotted! ✅")


def inference(model, x, scaler_x, scaler_y, device=None):
    # x_scaled = scaler_x.transform(x)
    # y_hat_scaled = model.predict(x_scaled)
    # y_hat = scaler_y.inverse_transform(y_hat_scaled)
    y_hat = model.predict(x)
    return y_hat


if __name__ == "__main__":

    app()
