"""
Generate mock samples for D-Sheet piling case study.

Creates:
- MC samples (normally distributed) for reliability analysis
- Surrogate samples (uniformly distributed) for training
- Mock surrogate data (inputs + simulated outputs)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_mc_samples(
    means: np.ndarray,
    stds: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_samples: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate Monte Carlo samples from truncated normal distributions.

    Args:
        means: Mean values for each parameter.
        stds: Standard deviations for each parameter.
        lower: Lower bounds for each parameter.
        upper: Upper bounds for each parameter.
        n_samples: Number of samples to generate.
        seed: Random seed.

    Returns:
        Array of shape (n_samples, n_parameters).
    """
    np.random.seed(seed)
    n_params = len(means)

    samples = means + stds * np.random.randn(n_samples, n_params)
    samples = np.clip(samples, lower, upper)

    return samples


def generate_surrogate_samples(
    means: np.ndarray,
    stds: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_samples: int = 1_000,
    n_sigma: float = 6.0,
    seed: int = 43,
) -> np.ndarray:
    """
    Generate uniform samples for surrogate training.

    Samples uniformly within ±n_sigma standard deviations, clipped to bounds.

    Args:
        means: Mean values for each parameter.
        stds: Standard deviations for each parameter.
        lower: Lower bounds for each parameter.
        upper: Upper bounds for each parameter.
        n_samples: Number of samples to generate.
        n_sigma: Range in standard deviations.
        seed: Random seed.

    Returns:
        Array of shape (n_samples, n_parameters).
    """
    np.random.seed(seed)
    n_params = len(means)

    uniform = np.random.uniform(size=(n_samples, n_params))
    samples = means - n_sigma * stds + 2 * n_sigma * stds * uniform
    samples = np.clip(samples, lower, upper)

    return samples


def generate_mock_moments(samples: np.ndarray, seed: int = 44) -> np.ndarray:
    """
    Generate mock max moment values (simulating D-Sheet output).

    Uses a simple polynomial model + noise to create realistic-looking data.

    Args:
        samples: Input parameter samples.
        seed: Random seed.

    Returns:
        Array of mock max moment values.
    """
    np.random.seed(seed)
    n_samples = samples.shape[0]

    # Normalize inputs to [0, 1]
    x_norm = (samples - samples.min(axis=0)) / (samples.max(axis=0) - samples.min(axis=0) + 1e-8)

    # Mock model: combination of inputs + nonlinearity + noise
    # EI is last column - higher EI -> lower moment
    ei_effect = -50 * x_norm[:, -1]
    soil_effect = 20 * (x_norm[:, 0] + x_norm[:, 3] + x_norm[:, 5]) / 3
    base_moment = 300

    moments = base_moment + ei_effect + soil_effect + 15 * np.random.randn(n_samples)
    moments = np.clip(moments, 150, 600)

    return moments


def main():
    """Generate all mock sample files."""
    data_dir = Path(__file__).parent / "data"
    samples_dir = data_dir / "samples"
    surrogate_dir = data_dir / "surrogate"

    samples_dir.mkdir(parents=True, exist_ok=True)
    surrogate_dir.mkdir(parents=True, exist_ok=True)

    # Load parameter distributions
    with open(data_dir / "case_study_specifications.json", "r") as f:
        specs = json.load(f)

    variables = specs["variables"]
    means = np.array([v["mean"] for v in variables])
    stds = np.array([v["standard_deviation"] for v in variables])
    lower = np.array([v["lower_bound"] for v in variables])
    upper = np.array([v["upper_bound"] for v in variables])
    var_names = [v["name"] for v in variables]

    # Generate MC samples
    print("Generating MC samples...")
    mc_samples = generate_mc_samples(means, stds, lower, upper, n_samples=100_000)

    # Add water level column (deterministic)
    water_lvl = -1.0 * np.ones((mc_samples.shape[0], 1))
    mc_samples = np.hstack([mc_samples, water_lvl])

    np.save(samples_dir / "mc_samples.npy", mc_samples)
    print(f"  Saved: samples/mc_samples.npy, shape={mc_samples.shape}")

    # Also save as CSV for portability
    mc_df = pd.DataFrame(mc_samples, columns=var_names + ["water_lvl"])
    mc_df.to_csv(samples_dir / "mc_samples.csv", index=False)
    print(f"  Saved: samples/mc_samples.csv")

    # Generate surrogate samples
    print("Generating surrogate samples...")
    srg_samples = generate_surrogate_samples(means, stds, lower, upper, n_samples=1_000)

    # Add water level column
    water_lvl = -1.0 * np.ones((srg_samples.shape[0], 1))
    srg_samples = np.hstack([srg_samples, water_lvl])

    np.save(samples_dir / "surrogate_samples.npy", srg_samples)
    print(f"  Saved: samples/surrogate_samples.npy, shape={srg_samples.shape}")

    # Generate mock surrogate data (with D-Sheet outputs)
    print("Generating mock surrogate data...")
    max_moments = generate_mock_moments(srg_samples[:, :-1])  # Exclude water_lvl for model

    # Create DataFrame
    columns = var_names + ["water_lvl", "max_moment"]
    data = np.hstack([srg_samples, max_moments.reshape(-1, 1)])
    df = pd.DataFrame(data, columns=columns)

    df.to_csv(surrogate_dir / "surrogate_data.csv", index=False)
    print(f"  Saved: surrogate/surrogate_data.csv, shape={df.shape}")

    print("Done!")


if __name__ == "__main__":
    main()
