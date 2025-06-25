import os
import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from srg_utils import load_data, plot_without_inference
# from gpr_classes import DependentGPRModels, MultitaskGPModel
from main.case_study_2025.train.srg.srg_utils import load_data, plot_without_inference
# from srg_utils import load_data, plot_without_inference
# import sys
# project_root = Path(__file__).parent.parent.parent.parent.parent
# sys.path.insert(0, str(project_root))
from src.geotechnical_models.gpr.gpr_classes import DependentGPRModels, MultitaskGPModel

import torch
import joblib
import typer
from datetime import datetime



app = typer.Typer()


@app.command()
def train(lr: float = 1e-2, epochs: int = 1_0, rank: int = 1, quiet: bool = False):

    base_dir = Path(__file__).resolve().parent

    data_dir = base_dir.parent / "data"
    # data_path = data_dir / "srg_moments_20250616_104316.csv"
    # data_path = data_dir / "srg_data_20250604_100638.csv"
    # data_path = data_dir / "srg_data_20250520_094244.csv"
    data_name = "srg_moments_20250617_101504"
    data_path = data_dir / f"{data_name}.csv"
    # data_path = data_dir / "srg_data_20250604_100638.csv"

    output_path = base_dir.parent / f"results_moments/{data_name}/gpr/lr_{lr:.1e}_epochs_{epochs:d}_rank_{rank}"
    output_path.mkdir(parents=True, exist_ok=True)

    X, y = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # scaler_x = StandardScaler()
    # scaler_y = StandardScaler()
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
    model_type = "multitask"

    gpr_model = DependentGPRModels()

    epoch_losses, model_state_dict = gpr_model.train(
                x_train=X_train_tensor, 
                y_train=y_train_tensor, 
                scaler_x=scaler_x, 
                scaler_y=scaler_y,
                n_epochs=epochs,
                lr=lr, 
                n_inducing_points=None, 
                model_type=model_type,
                path=output_path,
                rank=rank
            )

    y_hat_test, y_hat_var = gpr_model.predict(X_test_tensor)
    rmse = mean_squared_error(y_hat_test, y_test)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"lr={lr:.1e} | {epochs:d} epochs | RMSE: {rmse:.2f}"
    with open(output_path/"training_log.txt", "a") as f:
        f.write(f"{timestamp} | " + message)

    print("Training completed! ✅")

    # Saving already done in the train function
    torch.save(model_state_dict, output_path/r"model_weights.pth")
    joblib.dump(scaler_x, output_path/r"scaler_x.joblib")
    joblib.dump(scaler_y, output_path/r"scaler_y.joblib")

    print("[SUMMARY] "+message)

    print("Plotting results...")


    # y_hat_train, y_hat_var_train = gpr_model.predict(X_train_tensor)
    plot_without_inference(X_train, X_test, y_train, y_test, y_hat_test, output_path, y_hat_train=None, losses=epoch_losses)
    print("Results plotted! ✅")



if __name__ == "__main__":
    app()
    # train(lr=1e-2, epochs=1000, rank=1)