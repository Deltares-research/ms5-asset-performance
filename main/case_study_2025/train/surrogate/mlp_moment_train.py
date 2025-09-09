import os
import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from main.case_study_2025.train.surrogate.utils import load_data, plot
import joblib
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from typing import Tuple, Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):

        super().__init__()
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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


def main(epochs: int = 10_000, lr: float = 1e-5, full_profile: bool = True, quiet: bool = False):

    base_dir = Path(__file__).resolve().parent

    data_dir = base_dir.parent / "data"
    data_path = Path(__file__).parents[2] / "data/surrogate_data.csv"

    output_path = base_dir.parent.parent / f"results/surrogate/mlp_moment/lr_{lr:.1e}_epochs_{epochs:d}"
    output_path.mkdir(parents=True, exist_ok=True)

    X, y = load_data(data_path, full_profile=full_profile, target="moment")

    y = y[:, 60: 110]  # Keep only locations with important information
    moment_cutoff = 600

    y = np.abs(y).max(1)
    # Bound maximum moment, reject samples with too high moment (indicates D-SheetPiling non-convergence)
    X = X[y<=moment_cutoff]
    y = y[y<=moment_cutoff]
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("✅ Using MPS backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA backend")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS and CUDA not available — using CPU")

    model = MLP(
        input_dim=X.shape[-1],
        hidden_dims=[1024, 512, 256, 128, 64, 32],
        output_dim=y.shape[-1]
    ).to(device)

    torch.manual_seed(42)

    criterion = torch.nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1.,
        end_factor=.01,
        total_iters=int(epochs*0.8)
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
        scheduler.step()
        epoch_losses.append(epoch_loss)

    epoch_losses = np.asarray(epoch_losses)

    model.eval()

    with torch.no_grad():
        y_hat = inference(model, X_test, scaler_x, scaler_y)
        rmse = np.sqrt(mean_squared_error(y_test.squeeze(), y_hat.squeeze()))
        r2 = r2_score(y_test.squeeze(), y_hat.squeeze())

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"lr={lr:.1e} | {epochs:d} epochs | RMSE: {rmse:.2f} | R2: {r2:.2f}"
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


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10_000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--full_profile', action='store_false')
    parser.add_argument('--quiet', action='store_false')
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        lr=args.lr,
        full_profile=args.full_profile,
        quiet=args.quiet
    )

