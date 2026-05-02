"""
One-at-a-time sensitivity analysis for a D-SheetPiling LSF.

For each stochastic variable, evaluates the LSF at its 1st and 99th
percentiles while keeping all other variables at their mean. Reports
the g-value and the change from the baseline (all-at-mean) evaluation.

Outputs per run:
  - cr_<rate>.json        full results
  - cr_<rate>.csv         flat per-component table for spreadsheets
  - plots/tornado_<c>.png tornado plot per LSF component

Usage:
    python run_sensitivity.py --lsf lsf_wall --corrosion_rate 0.0
    python run_sensitivity.py --lsf lsf_anchor --corrosion_rate 0.3 --use_api
"""

import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
from scipy import stats as st

from src.io import get_remote_path
from src.plotting import save_figure
from build_fragility import LSF_REGISTRY, init_model, load_settings


_ENV = Path(__file__).parent / ".env"
_settings = load_settings()
_config = _settings["parameters"]
_remote = get_remote_path(_ENV)
_EULER = 0.5772156649


def is_deterministic(v: dict) -> bool:
    return v.get("distribution_type", "normal").lower() in ("deterministic", "constant", "fixed")


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
    elif dist_type in ["gumbel", "gumbel_max", "gumbel_r"]:
        beta = std * np.sqrt(6) / np.pi
        dist = st.gumbel_r(loc=mean - _EULER * beta, scale=beta)
    elif dist_type in ["gumbel_min", "gumbel_l"]:
        beta = std * np.sqrt(6) / np.pi
        dist = st.gumbel_l(loc=mean + _EULER * beta, scale=beta)
    elif dist_type in ["uniform", "unif"]:
        dist = st.uniform(loc=lo, scale=hi - lo)
    else:
        dist = st.norm(loc=mean, scale=std)

    # Truncate to [lo, hi] via inner-CDF remap (idempotent for truncnorm/uniform)
    F_lo = float(dist.cdf(lo)) if np.isfinite(lo) else 0.0
    F_hi = float(dist.cdf(hi)) if np.isfinite(hi) else 1.0
    return {p: float(dist.ppf(F_lo + p * (F_hi - F_lo))) for p in percentiles}


def component_names(lsf_name: str) -> list[str]:
    if lsf_name == "lsf_wall_anchor":
        return ["wall", "anchor"]
    if lsf_name == "lsf_wall":
        return ["wall"]
    if lsf_name == "lsf_anchor":
        return ["anchor"]
    return ["g"]


def export_csv(results: list, comps: list, lsf_name: str, corrosion_rate: float,
               g_baseline: tuple, out_path: Path) -> None:
    """Flat per-component CSV for spreadsheets."""
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "lsf", "corrosion_rate", "component", "variable",
            "mean", "p01", "p99",
            "g_baseline", "g_p01", "g_p99", "dg", "abs_dg",
        ])
        for r in results:
            for c_idx, c in enumerate(comps):
                w.writerow([
                    lsf_name, corrosion_rate, c, r["variable"],
                    r["mean"], r["p01"], r["p99"],
                    g_baseline[c_idx],
                    r["g_p01"][c_idx], r["g_p99"][c_idx],
                    r["dg"][c_idx], abs(r["dg"][c_idx]),
                ])


def plot_tornado(results: list, comps: list, c_idx: int, component: str,
                 g_baseline_c: float, out_path: Path) -> None:
    """Horizontal tornado plot — bars from g_p01 to g_p99 per variable.

    Variables are sorted by |dg| descending. Vertical lines mark the baseline
    g-value (all-at-mean) and the failure threshold (g = 0) when visible.
    """
    rows = [(r["variable"], r["g_p01"][c_idx], r["g_p99"][c_idx], r["dg"][c_idx])
            for r in results]
    rows.sort(key=lambda x: abs(x[3]), reverse=True)

    names = [r[0] for r in rows]
    g_lo = np.array([min(r[1], r[2]) for r in rows])
    g_hi = np.array([max(r[1], r[2]) for r in rows])
    width = g_hi - g_lo
    sign = np.array([np.sign(r[3]) for r in rows])  # which side moves on p99

    n = len(rows)
    fig_h = max(3.5, 0.32 * n + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    y = np.arange(n)
    colors = ["#4c8dde" if s >= 0 else "#d6604d" for s in sign]
    ax.barh(y, width, left=g_lo, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)

    ax.axvline(g_baseline_c, color="k", linestyle="--", linewidth=1,
               label=f"baseline g = {g_baseline_c:.2f}")
    xlim_lo, xlim_hi = ax.get_xlim()
    if xlim_lo <= 0.0 <= xlim_hi:
        ax.axvline(0.0, color="#a00", linestyle=":", linewidth=1.2, label="g = 0")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()  # largest at top
    ax.set_xlabel(f"g_{component}")
    ax.set_title(
        f"Tornado — {component}   (sorted by |g(p99) - g(p01)|)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)


def main(
    lsf_name: str = "lsf_wall_anchor",
    corrosion_rate: float = 0.0,
    use_api: bool = False,
):
    if lsf_name not in LSF_REGISTRY:
        raise ValueError(f"Unknown LSF '{lsf_name}'. Available: {list(LSF_REGISTRY.keys())}")

    init_model(use_api=use_api)
    lsf_fn = LSF_REGISTRY[lsf_name]
    variables = _settings["variables"]
    comps = component_names(lsf_name)

    print("=" * 60)
    print("One-at-a-Time Sensitivity Analysis")
    print("=" * 60)
    print(f"LSF: {lsf_name}")
    print(f"Components: {comps}")
    print(f"Corrosion rate: {corrosion_rate:.3f}")

    # Baseline: all variables at mean
    baseline_kwargs = {v["name"]: v["mean"] for v in variables}
    baseline_kwargs["Wall_SheetPilingElementEI"] = _settings["parameters"]["EI_start"]
    baseline_kwargs["corrosion_rate"] = corrosion_rate
    if lsf_name == "lsf_wall_anchor":
        baseline_kwargs["return_separate"] = True

    g_baseline_raw = lsf_fn(**baseline_kwargs)
    g_baseline = g_baseline_raw if isinstance(g_baseline_raw, tuple) else (g_baseline_raw,)

    print()
    for c, gb in zip(comps, g_baseline):
        print(f"Baseline ({c}, all at mean): g = {gb:.2f}")
    print()
    # Sensitivity per variable
    results = []
    header = (f"{'Variable':<28s} {'comp':<7s} {'p01':>12s} {'g(p01)':>12s}  "
              f"{'p99':>12s} {'g(p99)':>12s} {'dg':>10s}")
    print(header)
    print("-" * len(header))

    for v in variables:
        if is_deterministic(v):
            continue
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
        for c_idx, c in enumerate(comps):
            print(f"{name:<28s} {c:<7s} {pcts[0.01]:>12.4f} {g_vals[0.01][c_idx]:>12.2f}  "
                  f"{pcts[0.99]:>12.4f} {g_vals[0.99][c_idx]:>12.2f} {dg[c_idx]:>10.2f}")

        results.append({
            "variable": name,
            "mean": v["mean"],
            "p01": pcts[0.01],
            "p99": pcts[0.99],
            "g_p01": list(g_vals[0.01]),
            "g_p99": list(g_vals[0.99]),
            "dg": dg,
        })

    # Save
    out_dir = _remote / "output" / f"sensitivity_{lsf_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"cr_{corrosion_rate:.4f}.json"
    with open(json_path, "w") as f:
        json.dump({
            "lsf_name": lsf_name,
            "components": comps,
            "corrosion_rate": corrosion_rate,
            "g_baseline": list(g_baseline),
            "variables": results,
        }, f, indent=2)
    print(f"\nJSON: {json_path}")

    csv_path = out_dir / f"cr_{corrosion_rate:.4f}.csv"
    export_csv(results, comps, lsf_name, corrosion_rate, g_baseline, csv_path)
    print(f"CSV:  {csv_path}")

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for c_idx, c in enumerate(comps):
        plot_path = plots_dir / f"tornado_{c}_cr_{corrosion_rate:.4f}.png"
        plot_tornado(results, comps, c_idx, c, g_baseline[c_idx], plot_path)
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf", type=str, default="lsf_wall_anchor",
                        help=f"LSF to use. Available: {list(LSF_REGISTRY.keys())}")
    parser.add_argument("--corrosion_rate", type=float, default=0.0)
    parser.add_argument("--use_api", action="store_true")
    args = parser.parse_args()
    main(lsf_name=args.lsf, corrosion_rate=args.corrosion_rate, use_api=args.use_api)
