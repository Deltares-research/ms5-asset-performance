"""
I/O utilities for the D-Sheet piling case study.

Reads/writes data from a remote folder specified in dsheet_example.env.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from numpy.typing import NDArray


# -----------------------------------------------------------------------------
# Remote path configuration
# -----------------------------------------------------------------------------


env_path = Path(__file__).parent / "dsheet_example.env"
load_dotenv(env_path)


def get_remote_path() -> Path:
    """
    Load remote data folder path from environment file.

    Returns:
        Path to remote data folder.

    Raises:
        ValueError: If REMOTE_DATA_PATH not set in environment.
    """
    remote_path = os.environ.get("REMOTE_DATA_PATH")
    if remote_path is None:
        raise ValueError("REMOTE_DATA_PATH not set in dsheet_example.env")

    return Path(remote_path)


def load_json(filename: str) -> Dict[str, Any]:
    """
    Load JSON file from remote folder.

    Args:
        filename: Name of JSON file (relative to remote folder).

    Returns:
        Parsed JSON content.
    """
    filepath = get_remote_path() / filename
    with open(filepath, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filename: str) -> None:
    """
    Save JSON file to remote folder.

    Args:
        data: Data to save.
        filename: Name of JSON file (relative to remote folder).
    """
    filepath = get_remote_path() / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def load_npy(filename: str) -> NDArray:
    """
    Load numpy array from remote folder.

    Args:
        filename: Name of .npy file (relative to remote folder).

    Returns:
        Numpy array.
    """
    filepath = get_remote_path() / filename
    return np.load(filepath)


def save_npy(data: NDArray, filename: str) -> None:
    """
    Save numpy array to remote folder.

    Args:
        data: Array to save.
        filename: Name of .npy file (relative to remote folder).
    """
    filepath = get_remote_path() / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, data)


# -----------------------------------------------------------------------------
# Parameter distributions
# -----------------------------------------------------------------------------

def load_parameter_distributions() -> pd.DataFrame:
    """
    Load parameter distributions from CSV.

    Expected columns: parameter, mean, std, lower, upper

    Returns:
        DataFrame with parameter distribution definitions.
    """
    filepath = get_remote_path() / "parameter_distributions.csv"
    return pd.read_csv(filepath)


def save_parameter_distributions(df: pd.DataFrame) -> None:
    """
    Save parameter distributions to CSV.

    Args:
        df: DataFrame with columns: parameter, mean, std, lower, upper
    """
    filepath = get_remote_path() / "parameter_distributions.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


# -----------------------------------------------------------------------------
# Monte Carlo and surrogate samples
# -----------------------------------------------------------------------------

def load_mc_samples(fmt: str = "npy") -> NDArray:
    """
    Load Monte Carlo samples (normally distributed).

    Args:
        fmt: File format, "npy" or "csv".

    Returns:
        Sample array.
    """
    if fmt == "csv":
        filepath = get_remote_path() / "samples/mc_samples.csv"
        return pd.read_csv(filepath).values
    return load_npy("samples/mc_samples.npy")


def save_mc_samples(samples: NDArray, fmt: str = "npy") -> None:
    """
    Save Monte Carlo samples.

    Args:
        samples: Sample array.
        fmt: File format, "npy" or "csv".
    """
    if fmt == "csv":
        filepath = get_remote_path() / "samples/mc_samples.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(samples).to_csv(filepath, index=False)
    else:
        save_npy(samples, "samples/mc_samples.npy")


def load_surrogate_samples(fmt: str = "npy") -> NDArray:
    """
    Load surrogate training samples (uniformly distributed).

    Args:
        fmt: File format, "npy" or "csv".

    Returns:
        Sample array.
    """
    if fmt == "csv":
        filepath = get_remote_path() / "samples/surrogate_samples.csv"
        return pd.read_csv(filepath).values
    return load_npy("samples/surrogate_samples.npy")


def save_surrogate_samples(samples: NDArray, fmt: str = "npy") -> None:
    """
    Save surrogate training samples.

    Args:
        samples: Sample array.
        fmt: File format, "npy" or "csv".
    """
    if fmt == "csv":
        filepath = get_remote_path() / "samples/surrogate_samples.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(samples).to_csv(filepath, index=False)
    else:
        save_npy(samples, "samples/surrogate_samples.npy")


# -----------------------------------------------------------------------------
# Surrogate training data (samples + D-SheetPiling outputs)
# -----------------------------------------------------------------------------

def load_surrogate_data() -> pd.DataFrame:
    """
    Load surrogate training data with input samples and model outputs.

    Returns:
        DataFrame with input parameters and displacement/moment columns.
    """
    filepath = get_remote_path() / "surrogate/surrogate_data.csv"
    return pd.read_csv(filepath)


def save_surrogate_data(df: pd.DataFrame) -> None:
    """
    Save surrogate training data.

    Args:
        df: DataFrame with input parameters and displacement/moment columns.
    """
    filepath = get_remote_path() / "surrogate/surrogate_data.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


# -----------------------------------------------------------------------------
# Surrogate model (MLP weights + scalers)
# -----------------------------------------------------------------------------

def load_surrogate_model(
    model_class: type,
    model_kwargs: Dict[str, Any],
    device: torch.device | None = None,
) -> Tuple[torch.nn.Module, Any, Any]:
    """
    Load trained surrogate model with scalers.

    Args:
        model_class: PyTorch model class to instantiate.
        model_kwargs: Keyword arguments for model instantiation.
        device: Torch device. Defaults to CPU.

    Returns:
        Tuple of (model, scaler_x, scaler_y).
    """
    model_dir = get_remote_path() / "surrogate/model"
    device = device or torch.device("cpu")

    model = model_class(**model_kwargs)
    weights_path = model_dir / "torch_weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    scaler_x = joblib.load(model_dir / "scaler_x.joblib")
    scaler_y = joblib.load(model_dir / "scaler_y.joblib")

    return model, scaler_x, scaler_y


def save_surrogate_model(
    model: torch.nn.Module,
    scaler_x: Any,
    scaler_y: Any,
) -> None:
    """
    Save trained surrogate model with scalers.

    Args:
        model: Trained PyTorch model.
        scaler_x: Fitted input scaler.
        scaler_y: Fitted output scaler.
    """
    model_dir = get_remote_path() / "surrogate/model"
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / "torch_weights.pth")
    joblib.dump(scaler_x, model_dir / "scaler_x.joblib")
    joblib.dump(scaler_y, model_dir / "scaler_y.joblib")


# -----------------------------------------------------------------------------
# Case study setting (time-series with corrosion, moments, etc.)
# -----------------------------------------------------------------------------

def load_case_study_setting() -> Dict[str, Any]:
    """
    Load case study setting with time-series data.

    Returns:
        Dictionary keyed by time with corrosion, moments, and other data.
    """
    return load_json("case_study_setting.json")


def save_case_study_setting(setting: Dict[str, Any]) -> None:
    """
    Save case study setting.

    Args:
        setting: Dictionary with time-series case study data.
    """
    save_json(setting, "case_study_setting.json")


# -----------------------------------------------------------------------------
# Fragility curves
# -----------------------------------------------------------------------------

def load_fragility_curve(name: str = "fragility"):
    """
    Load fragility curve from remote folder.

    Args:
        name: Base name of fragility file (without extension).

    Returns:
        FragilityCurve instance.
    """
    # Import here to avoid circular imports
    from case_studies.dsheet_example.performance_function import FragilityCurve

    # Try npz first (has cached moments), then json
    npz_path = get_remote_path() / f"results/{name}.npz"
    json_path = get_remote_path() / f"results/{name}.json"

    if npz_path.exists():
        return FragilityCurve.load(npz_path)
    elif json_path.exists():
        return FragilityCurve.load(json_path)
    else:
        raise FileNotFoundError(f"No fragility curve found: {npz_path} or {json_path}")


def save_fragility_curve(fragility, name: str = "fragility", fmt: str = "npz") -> None:
    """
    Save fragility curve to remote folder.

    Args:
        fragility: FragilityCurve instance.
        name: Base name of fragility file (without extension).
        fmt: Format - "npz" (with cached moments) or "json" (portable summary).
    """
    ext = ".npz" if fmt == "npz" else ".json"
    filepath = get_remote_path() / f"results/{name}{ext}"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fragility.save(filepath, fmt=fmt)


def fragility_exists(name: str = "fragility") -> bool:
    """Check if a fragility curve exists in the remote folder."""
    npz_path = get_remote_path() / f"results/{name}.npz"
    json_path = get_remote_path() / f"results/{name}.json"
    return npz_path.exists() or json_path.exists()


# -----------------------------------------------------------------------------
# Fragility surface index (2D: corrosion_ratio x moment_survived)
# -----------------------------------------------------------------------------

def load_fragility_surface(name: str = "fragility_surface"):
    """
    Load fragility surface index from remote folder.

    Only loads the manifest; individual curves are loaded on-demand.

    Args:
        name: Name of fragility surface directory.

    Returns:
        FragilitySurfaceIndex instance.
    """
    from case_studies.dsheet_example.performance_function import FragilitySurfaceIndex

    surface_dir = get_remote_path() / f"results/{name}"
    if not surface_dir.exists():
        raise FileNotFoundError(f"Fragility surface not found: {surface_dir}")

    return FragilitySurfaceIndex.load(surface_dir)


def save_fragility_surface(surface, name: str = "fragility_surface") -> None:
    """
    Save fragility surface index to remote folder.

    Creates a directory with manifest.json and one .npz file per moment value.

    Args:
        surface: FragilitySurfaceIndex instance.
        name: Name of fragility surface directory.
    """
    surface_dir = get_remote_path() / f"results/{name}"
    surface.save(surface_dir)


def fragility_surface_exists(name: str = "fragility_surface") -> bool:
    """Check if a fragility surface exists in the remote folder."""
    surface_dir = get_remote_path() / f"results/{name}"
    manifest_path = surface_dir / "manifest.json"
    return manifest_path.exists()


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------

def load_reliability_results() -> Dict[str, Any]:
    """Load reliability analysis results."""
    return load_json("results/reliability_results.json")


def save_reliability_results(results: Dict[str, Any]) -> None:
    """Save reliability analysis results."""
    save_json(results, "results/reliability_results.json")
