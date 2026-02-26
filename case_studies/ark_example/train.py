"""
Train surrogate model (MLP) for D-Sheet piling moment prediction.

Usage:
    python -m case_studies.ark_example.train --epochs 10000 --lr 1e-4
"""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from case_studies.dsheet_example.performance_function import MLP
from case_studies.dsheet_example import io
from case_studies.dsheet_example import plotting


def get_device() -> torch.device:
    """Get best available device."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_training_data(
    moment_cutoff: float = 600.0,
) -> Tuple[NDArray, NDArray]:
    """
    Load and preprocess surrogate training data.

    Args:
        moment_cutoff: Maximum moment threshold (filter outliers).

    Returns:
        Tuple of (X, y) arrays.
    """
    df = io.load_surrogate_data()

    # Input columns (11 features)
    input_cols = [
        "Klei_soilcohesion", "Klei_soilphi", "Klei_soilcurkb1",
        "Zand_soilphi", "Zand_soilcurkb1",
        "Zandvast_soilphi", "Zandvast_soilcurkb1",
        "Zandlos_soilphi", "Zandlos_soilcurkb1",
        "Wall_SheetPilingElementEI", "water_lvl",
    ]
    X = df[input_cols].values

    # Compute max_moment from moment columns (absolute value)
    moment_cols = [c for c in df.columns if c.startswith("moment_")]
    moments = df[moment_cols].values
    y = np.abs(moments).max(axis=1)

    # Filter outliers (non-convergence in D-SheetPiling)
    mask = y <= moment_cutoff
    X = X[mask]
    y = y[mask].reshape(-1, 1)

    return X, y


def train_model(
    X_train: NDArray,
    y_train: NDArray,
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    epochs: int = 10_000,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, NDArray]:
    """
    Train MLP surrogate model.

    Args:
        X_train: Training inputs (scaled).
        y_train: Training targets (scaled).
        input_dim: Input dimension.
        hidden_dims: Hidden layer sizes.
        output_dim: Output dimension.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Torch device.
        verbose: Show progress bar.

    Returns:
        Tuple of (trained model, loss history).
    """
    if device is None:
        device = get_device()

    model = MLP(input_dim, hidden_dims, output_dim).to(device)

    torch.manual_seed(42)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=int(epochs * 0.8),
    )

    X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

    losses = []

    iterator = tqdm(range(1, epochs + 1), desc="Training") if verbose else range(1, epochs + 1)

    for epoch in iterator:
        model.train()
        optimizer.zero_grad()

        preds = model(X_tensor)
        loss = criterion(preds, y_tensor)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

    return model, np.array(losses)


def evaluate_model(
    model: nn.Module,
    X_test: NDArray,
    y_test: NDArray,
    scaler_x: MinMaxScaler,
    scaler_y: MinMaxScaler,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Evaluate trained model on test set.

    Args:
        model: Trained model.
        X_test: Test inputs (unscaled).
        y_test: Test targets (unscaled).
        scaler_x: Input scaler.
        scaler_y: Output scaler.
        device: Torch device.

    Returns:
        Dict with RMSE, R2, predictions.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    X_scaled = scaler_x.transform(X_test)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "rmse": rmse,
        "r2": r2,
        "y_pred": y_pred.squeeze(),
        "y_true": y_test.squeeze(),
    }


def main(
    epochs: int = 10_000,
    lr: float = 1e-4,
    hidden_dims: list = [1024, 512, 256, 128, 64, 32],
    test_size: float = 0.2,
    moment_cutoff: float = 600.0,
    save_plots: bool = True,
    verbose: bool = True,
) -> None:
    """
    Train and save surrogate model.

    Args:
        epochs: Number of training epochs.
        lr: Learning rate.
        hidden_dims: MLP hidden layer sizes.
        test_size: Fraction of data for testing.
        moment_cutoff: Max moment threshold for filtering.
        save_plots: Save diagnostic plots.
        verbose: Print progress.
    """
    device = get_device()
    if verbose:
        print(f"Using device: {device}")

    # Load data
    if verbose:
        print("Loading training data...")

    X, y = load_training_data(moment_cutoff=moment_cutoff)

    if verbose:
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}, Target dim: {y.shape[1]}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Scale data
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Train
    if verbose:
        print(f"\nTraining MLP ({epochs} epochs, lr={lr})...")

    model, losses = train_model(
        X_train_scaled,
        y_train_scaled,
        input_dim=X.shape[1],
        hidden_dims=hidden_dims,
        output_dim=y.shape[1],
        epochs=epochs,
        lr=lr,
        device=device,
        verbose=verbose,
    )

    # Evaluate
    if verbose:
        print("\nEvaluating...")

    results = evaluate_model(model, X_test, y_test, scaler_x, scaler_y, device)

    if verbose:
        print(f"  RMSE: {results['rmse']:.2f}")
        print(f"  R²:   {results['r2']:.4f}")

    # Save model and scalers
    if verbose:
        print("\nSaving model...")

    model_cpu = model.cpu()
    io.save_surrogate_model(model_cpu, scaler_x, scaler_y)

    # Save training log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "epochs": epochs,
        "lr": lr,
        "hidden_dims": hidden_dims,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "rmse": results["rmse"],
        "r2": results["r2"],
    }

    log_path = io.get_remote_path() / "surrogate/model/training_log.json"
    try:
        existing_log = io.load_json("surrogate/model/training_log.json")
        if not isinstance(existing_log, list):
            existing_log = [existing_log]
    except FileNotFoundError:
        existing_log = []

    existing_log.append(log_entry)
    io.save_json(existing_log, "surrogate/model/training_log.json")

    if verbose:
        print(f"  Saved to: {io.get_remote_path() / 'surrogate/model'}")

    # Save plots
    if save_plots:
        if verbose:
            print("\nSaving plots...")

        plot_dir = io.get_remote_path() / "surrogate/plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Predictions plot
        fig = plotting.plot_predictions(
            y_true=results["y_true"],
            y_pred=results["y_pred"],
            title=f"Surrogate predictions (R²={results['r2']:.3f})",
            xlabel="Observed moment [kNm]",
            ylabel="Predicted moment [kNm]",
        )
        plotting.save_figure(fig, plot_dir / "predictions.png")

        # Loss history
        fig = plotting.plot_loss_history(
            losses=losses,
            title="Training loss",
        )
        plotting.save_figure(fig, plot_dir / "loss_history.png")

        if verbose:
            print(f"  Plots saved to: {plot_dir}")

    if verbose:
        print("\nDone!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train MLP surrogate model")
    parser.add_argument("--epochs", type=int, default=10_000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--moment-cutoff", type=float, default=600.0, help="Max moment filter")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    main(
        epochs=args.epochs,
        lr=args.lr,
        test_size=args.test_size,
        moment_cutoff=args.moment_cutoff,
        save_plots=not args.no_plots,
        verbose=not args.quiet,
    )
