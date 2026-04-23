"""
One-at-a-time sensitivity analysis for a D-SheetPiling LSF.

For each stochastic variable, evaluates the LSF at its 1st and 99th
percentiles while keeping all other variables at their mean. Reports
the g-value and the change from the baseline (all-at-mean) evaluation.

Usage:
    python run_sensitivity.py --lsf lsf_wall --corrosion_rate 0.0
    python run_sensitivity.py --lsf lsf_anchor --corrosion_rate 0.3 --use_api
"""

import json
import os
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from scipy import stats as st

from src.io import get_remote_path
from build_fragility import LSF_REGISTRY, init_model, load_settings, load_api_key


_ENV = Path(__file__).parent / ".env"
_settings = load_settings()
_config = _settings["parameters"]
_remote = get_remote_path(_ENV)


def get_percentiles(v: dict, percentiles=(0.01, 0.99)):
    """Get percentile values for a variable definition."""
    dist_type = v.get("distribution_type", "normal").lower()
    mean = v["mean"]
    std = v["standard_deviation"]
    lo = v.get("lower_bound", -np.inf)
    hi = v.get("upper_bound", np.inf)

    if dist_type in ["normal", "norm", "n", "gaussian"]:
        a = (lo - mean) / std if np.isfinite(lo) else -10
        b = (hi - mean) / std if np.isfinite(hi) else 10
        dist = st.truncnorm(a, b, loc=mean, scale=std)
    elif dist_type in ["lognormal", "lognorm"]:
        sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
        mu = np.log(mean) - 0.5 * sigma ** 2
        dist = st.lognorm(sigma, scale=np.exp(mu))
    elif dist_type in ["uniform", "unif"]:
        dist = st.uniform(loc=lo, scale=hi - lo)
    else:
        dist = st.norm(loc=mean, scale=std)

    return {p: float(dist.ppf(p)) for p in percentiles}


def main(
    lsf_name: str = "lsf_wall",
    corrosion_rate: float = 0.0,
    use_api: bool = False,
):
    if lsf_name not in LSF_REGISTRY:
        raise ValueError(f"Unknown LSF '{lsf_name}'. Available: {list(LSF_REGISTRY.keys())}")

    init_model(use_api=use_api)
    lsf_fn = LSF_REGISTRY[lsf_name]
    variables = _settings["variables"]
    var_names = [v["name"] for v in variables]

    print("=" * 60)
    print("One-at-a-Time Sensitivity Analysis")
    print("=" * 60)
    print(f"LSF: {lsf_name}")
    print(f"Corrosion rate: {corrosion_rate:.3f}")

    # Baseline: all variables at mean
    baseline_kwargs = {v["name"]: v["mean"] for v in variables}
    baseline_kwargs["Wall_SheetPilingElementEI"] =  _settings["parameters"]["EI_start"]
    baseline_kwargs["corrosion_rate"] = corrosion_rate
    if lsf_name == "lsf_wall_anchor":
        baseline_kwargs["return_separate"] = True
    g_baseline = lsf_fn(**baseline_kwargs)
    if not lsf_name == "lsf_wall_anchor":
        print(f"\nBaseline (all at mean): g = {g_baseline:.2f}\n")
    else:
        print(f"\nBaseline - Wall (all at mean): g = {g_baseline[0]:.2f}\n")
        print(f"\nBaseline - Anchor (all at mean): g = {g_baseline[1]:.2f}\n")

    # Sensitivity per variable
    results = []
    header = f"{'Variable':<30s} {'p01':>12s} {'g(p01)':>10s} {'p99':>12s} {'g(p99)':>10s} {'dg':>10s}"
    print(header)
    print("-" * len(header))

    for v in variables:
        name = v["name"]
        pcts = get_percentiles(v, (0.01, 0.99))

        g_vals = {}
        for p, val in pcts.items():
            kwargs = {vv["name"]: vv["mean"] for vv in variables}
            kwargs["Wall_SheetPilingElementEI"] = _settings["parameters"]["EI_start"]
            kwargs["corrosion_rate"] = corrosion_rate
            if lsf_name == "lsf_wall_anchor":
                kwargs["return_separate"] = True
            kwargs[name] = val
            g = lsf_fn(**kwargs)
            if not isinstance(g, tuple):
                g = (g,)
            g_vals[p] = g

        dg = [p99 - p01 for (p01, p99) in zip(g_vals[0.01], g_vals[0.99])]
        if lsf_name != "lsf_wall_anchor":
            print(f"{name:<30s} {pcts[0.01]:>12.4f} {g_vals[0.01][0]:>10.2f} {pcts[0.99]:>12.4f} {g_vals[0.99][0]:>10.2f} {dg[0]:>10.2f}")
        else:
            print(f"{name:<30s} - Wall {pcts[0.01]:>12.4f} {g_vals[0.01][0]:>10.2f} {pcts[0.99]:>12.4f} {g_vals[0.99][0]:>10.2f} {dg[0]:>10.2f}")
            print(f"{name:<30s} - Anchor {pcts[0.01]:>12.4f} {g_vals[0.01][0]:>10.2f} {pcts[0.99]:>12.4f} {g_vals[0.99][1]:>10.2f} {dg[1]:>10.2f}")

        results.append({
            "variable": name,
            "mean": v["mean"],
            "p01": pcts[0.01],
            "p99": pcts[0.99],
            "g_p01": g_vals[0.01],
            "g_p99": g_vals[0.99],
            "g_baseline": g_baseline,
            "dg": dg,
        })
    # Save
    cache_dir = _remote / "output" / f"sensitivity_{lsf_name}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "lsf_name": lsf_name,
        "corrosion_rate": corrosion_rate,
        "g_baseline": g_baseline,
        "variables": results,
    }
    output_path = cache_dir / f"cr_{corrosion_rate:.4f}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf", type=str, default="lsf_wall_anchor", help=f"LSF to use. Available: {list(LSF_REGISTRY.keys())}")
    parser.add_argument("--corrosion_rate", type=float, default=0.0)
    parser.add_argument("--use_api", action="store_true")
    args = parser.parse_args()
    main(lsf_name=args.lsf, corrosion_rate=args.corrosion_rate, use_api=args.use_api)
