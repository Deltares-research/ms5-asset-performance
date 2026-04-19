"""
Monte Carlo simulation for a D-SheetPiling LSF.

Samples the stochastic variables from their distributions (as defined in
settings.json), evaluates the selected LSF at a fixed corrosion rate, and
stores per-sample results (g-value, model outputs) with per-sample caching.

Usage:
    python run_mc.py --lsf lsf_wall --n_samples 100 --corrosion_rate 0.3
    python run_mc.py --lsf lsf_wall --n_samples 1000 --use_api
"""

import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from scipy import stats as st

from src.io import get_remote_path
from build_fragility import LSF_REGISTRY, init_model, load_settings, load_api_key


_ENV = Path(__file__).parent / ".env"
_settings = load_settings()
_config = _settings["parameters"]
_remote = get_remote_path(_ENV)


def sample_variables(variables: list, n_samples: int, seed: int = 42) -> np.ndarray:
    """Draw MC samples from the variable distributions.

    Args:
        variables: List of variable defs from settings.json.
        n_samples: Number of samples.
        seed: Random seed.

    Returns:
        Array of shape (n_samples, n_vars).
    """
    rng = np.random.default_rng(seed)
    samples = []

    for v in variables:
        dist_type = v.get("distribution_type", "normal").lower()
        mean = v["mean"]
        std = v["standard_deviation"]
        lo = v.get("lower_bound", -np.inf)
        hi = v.get("upper_bound", np.inf)

        if dist_type in ["normal", "norm", "n", "gaussian"]:
            a = (lo - mean) / std if np.isfinite(lo) else -10
            b = (hi - mean) / std if np.isfinite(hi) else 10
            s = st.truncnorm.rvs(a, b, loc=mean, scale=std, size=n_samples, random_state=rng)
        elif dist_type in ["lognormal", "lognorm"]:
            sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
            mu = np.log(mean) - 0.5 * sigma ** 2
            s = st.lognorm.rvs(sigma, scale=np.exp(mu), size=n_samples, random_state=rng)
        elif dist_type in ["uniform", "unif"]:
            s = st.uniform.rvs(loc=lo, scale=hi - lo, size=n_samples, random_state=rng)
        else:
            s = rng.normal(mean, std, size=n_samples)

        samples.append(s)

    return np.column_stack(samples)


def main(
    lsf_name: str = "lsf_wall",
    n_samples: int = 100,
    corrosion_rate: float = 0.0,
    use_api: bool = False,
    seed: int = 42,
):
    if lsf_name not in LSF_REGISTRY:
        raise ValueError(f"Unknown LSF '{lsf_name}'. Available: {list(LSF_REGISTRY.keys())}")

    init_model(use_api=use_api)
    lsf_fn = LSF_REGISTRY[lsf_name]
    variables = _settings["variables"]
    var_names = [v["name"] for v in variables]

    print("=" * 60)
    print("Monte Carlo Simulation")
    print("=" * 60)
    print(f"LSF: {lsf_name}")
    print(f"Samples: {n_samples}")
    print(f"Corrosion rate: {corrosion_rate:.3f}")
    print(f"Variables: {var_names}")

    # Cache directory
    cache_dir = _remote / "output" / f"mc_{lsf_name}" / f"cr_{corrosion_rate:.4f}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check existing samples
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        completed = set(manifest.get("completed_indices", []))
    else:
        completed = set()

    n_remaining = n_samples - len(completed)
    print(f"Cached: {len(completed)}, remaining: {n_remaining}\n")

    # Draw samples
    X = sample_variables(variables, n_samples, seed=seed)

    # Evaluate LSF per sample with caching
    g_values = []
    for i in range(n_samples):
        if i in completed:
            # Load cached result
            with open(cache_dir / f"sample_{i:06d}.json", "r") as f:
                result = json.load(f)
            g_values.append(result["g"])
            continue

        # Build kwargs for LSF
        kwargs = {name: float(X[i, j]) for j, name in enumerate(var_names)}
        kwargs["corrosion_rate"] = corrosion_rate

        g = lsf_fn(**kwargs)
        g_values.append(g)

        # Cache result
        result = {
            "index": i,
            "g": float(g),
            "corrosion_rate": corrosion_rate,
            "failed": g < 0,
            "params": kwargs,
        }
        with open(cache_dir / f"sample_{i:06d}.json", "w") as f:
            json.dump(result, f, indent=2)

        completed.add(i)

        # Update manifest
        manifest = {
            "lsf_name": lsf_name,
            "corrosion_rate": corrosion_rate,
            "n_samples": n_samples,
            "seed": seed,
            "var_names": var_names,
            "n_completed": len(completed),
            "completed_indices": sorted(completed),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Progress
        n_done = len(completed)
        n_fail = sum(1 for g in g_values if g < 0)
        if n_done % max(1, n_samples // 20) == 0 or n_done == n_samples:
            print(f"  [{n_done:6d}/{n_samples}]  failures: {n_fail}  "
                  f"Pf_hat: {n_fail/n_done:.4e}")

    # Summary
    g_values = np.array(g_values)
    n_fail = np.sum(g_values < 0)
    pf = n_fail / n_samples
    beta = float(st.norm.ppf(1 - pf)) if 0 < pf < 1 else (np.inf if pf == 0 else -np.inf)

    print(f"\n{'='*60}")
    print(f"Results (cr={corrosion_rate:.3f})")
    print(f"{'='*60}")
    print(f"  N samples:   {n_samples}")
    print(f"  N failures:  {n_fail}")
    print(f"  Pf:          {pf:.4e}")
    print(f"  beta:        {beta:.3f}")
    print(f"  g mean:      {g_values.mean():.2f}")
    print(f"  g std:       {g_values.std():.2f}")
    print(f"  g min:       {g_values.min():.2f}")
    print(f"  Cache:       {cache_dir}")

    # Save summary
    summary = {
        "lsf_name": lsf_name,
        "corrosion_rate": corrosion_rate,
        "n_samples": n_samples,
        "n_failures": int(n_fail),
        "pf": pf,
        "beta": beta,
        "g_mean": float(g_values.mean()),
        "g_std": float(g_values.std()),
        "g_min": float(g_values.min()),
        "g_max": float(g_values.max()),
        "seed": seed,
    }
    with open(cache_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:     {cache_dir / 'summary.json'}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf", type=str, default="lsf_wall",
                        help=f"LSF to use. Available: {list(LSF_REGISTRY.keys())}")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--corrosion_rate", type=float, default=0.0)
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        lsf_name=args.lsf,
        n_samples=args.n_samples,
        corrosion_rate=args.corrosion_rate,
        use_api=args.use_api,
        seed=args.seed,
    )
