import os
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.chebyshev import chebvander, chebder
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from main.case_study_2025.train.srg.utils import load_data, plot

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Tuple, Optional
from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib
from tqdm import tqdm
import typer
from datetime import datetime


app = typer.Typer()


def make_chebyshev_basis(x, degree=10):
    x_scaled = 2 * (x - x.min()) / (x.max() - x.min()) - 1
    phi = chebvander(x_scaled, degree)  # shape (n_points, degree+1)
    return phi


def make_chebyshev_basis_second_derivative(x, degree):
    x_scaled = 2 * (x - x.min()) / (x.max() - x.min()) - 1
    T = np.polynomial.chebyshev.chebvander(x_scaled, degree)

    T_dd = np.zeros_like(T)
    for k in range(degree + 1):
        coefs = np.zeros(degree + 1)
        coefs[k] = 1
        d2T_coefs = chebder(coefs, m=2)
        T_dd[:, k] = np.polynomial.chebyshev.chebval(x_scaled, d2T_coefs)

    # Rescale derivatives from d²/d(𝑥̃)² to d²/dx²
    dx_dxtilde_sq = (2 / (x.max() - x.min())) ** 2
    T_dd_rescaled = T_dd * dx_dxtilde_sq

    return T_dd_rescaled


class Chebysev(nn.Module):
    def __init__(self, input_dim, hidden_dims, degree, x):
        super().__init__()
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, degree+1))

        self.net = nn.Sequential(*layers)

        phi = torch.from_numpy(make_chebyshev_basis(x, degree)).float()
        self.register_buffer("basis", phi.T)  # shape (degree+1, n_points)

        phi_der = torch.from_numpy(make_chebyshev_basis_second_derivative(x, degree)).float()
        self.register_buffer("basis_der", phi_der.T)  # shape (degree+1, n_points)

    def forward(self, x, return_coeffs=False):
        coeffs = self.net(x)               # shape (batch, degree+1)
        if return_coeffs: return coeffs
        return coeffs @ self.basis         # shape (batch, n_points)


@app.command()
def train(epochs: int = 1_000, lr: float = 1e-4, full_profile: bool = False, quiet: bool = False):

    base_dir = Path(__file__).resolve().parent

    data_dir = base_dir.parent / "data"
    data_path = data_dir / "srg_data_20250604_100638.csv"
    if full_profile:
        z_path = base_dir.parent.parent / "data/setting/z.json"
    else:
        z_path = base_dir.parent.parent / "data/setting/z_monitoring.json"

    output_path = base_dir.parent / f"results/srg/chebysev/lr_{lr:.1e}_epochs_{epochs:d}_fullprofile_{full_profile}"
    output_path.mkdir(parents=True, exist_ok=True)

    X, y = load_data(data_path, full_profile)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("✅ Using MPS (Metal) backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA backend")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS and CUDA not available — using CPU")

    with open(z_path, "r") as f: z = np.array(json.load(f))

    model = Chebysev(
        input_dim=X.shape[-1],
        hidden_dims=[1024, 512, 256, 128, 64, 32],
        x=z,
        degree=10
    ).to(device)

    torch.manual_seed(42)

   # criterion = nn.MSELoss()
    criterion = torch.nn.HuberLoss(delta=1.0)
   # criterion = torch.nn.SmoothL1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    x_torch = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_torch = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)

    epoch_losses = []

    print("Training...")
    pbar = tqdm(range(1, epochs + 1)) if not quiet else range(1, epochs + 1)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        preds = model(x_torch)
        loss = criterion(preds, y_torch)
        loss.backward()
        optimizer.step()
        epoch_loss = loss.item()
        scheduler.step(epoch_loss)
        epoch_losses.append(epoch_loss)

    epoch_losses = np.asarray(epoch_losses)

    model.eval()

    with torch.no_grad():
        y_hat = inference(model, X_test, scaler_x, scaler_y)
        rmse = mean_squared_error(y_hat, y_test)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"lr={lr:.1e} | {epochs:d} epochs | RMSE: {rmse:.2f}"
    with open(output_path/"training_log.txt", "a") as f:
        f.write(f"{timestamp} | " + message)

    print("Training completed! ✅")

    torch.save(model.state_dict(), output_path/r"torch_weights.pth")
    joblib.dump(scaler_x, output_path/r"scaler_x.joblib")
    joblib.dump(scaler_y, output_path/r"scaler_y.joblib")

    print("[SUMMARY] "+message)

    print("Plotting results...")

    plot(inference, model, X_train, X_test, y_train, y_test, scaler_x, scaler_y, output_path, epoch_losses)

    print("Results plotted! ✅")


def inference(model, x, scaler_x, scaler_y, device=None):
    if device is None:
        device = next(model.parameters()).device
    x_scaled = scaler_x.transform(x)
    x_scaled_torch = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_hat_scaled = model(x_scaled_torch)
    y_hat_scaled = y_hat_scaled.detach().cpu().numpy()
    y_hat = scaler_y.inverse_transform(y_hat_scaled)
    return y_hat


if __name__ == "__main__":

    app()
