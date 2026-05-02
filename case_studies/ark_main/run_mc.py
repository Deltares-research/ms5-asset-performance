"""
Monte Carlo simulation for a D-SheetPiling LSF (corrosion fixed at 0).

Samples the stochastic variables from their distributions (as defined in
settings.json), evaluates the selected LSF at corrosion_rate = 0, and stores
per-sample results (g-value(s), model outputs) with per-sample caching.

For ``lsf_wall_anchor`` the LSF is evaluated with ``return_separate=True``
so wall and anchor g-values are tracked independently. The system fails if
either component fails (series system).

Usage:
    python run_mc.py --lsf lsf_wall_anchor --n_samples 100
    python run_mc.py --lsf lsf_wall --n_samples 1000 --use_api
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
from scipy import stats as st
from tqdm import tqdm

from src.io import get_remote_path
from src.plotting import save_figure
from build_fragility import LSF_REGISTRY, init_model, load_settings


_ENV = Path(__file__).parent / ".env"
_settings = load_settings()
_config = _settings["parameters"]
_remote = get_remote_path(_ENV)


_EULER = 0.5772156649


def _build_marginal(v: dict):
    """Construct a frozen scipy distribution for a variable definition."""
    dist_type = v.get("distribution_type", "normal").lower()
    mean = v["mean"]
    std = v["standard_deviation"]
    lo = v.get("lower_bound", -np.inf)
    hi = v.get("upper_bound", np.inf)

    if dist_type in ["normal", "norm", "n", "gaussian"]:
        if np.isfinite(lo) and np.isfinite(hi) and not (lo <= mean <= hi):
            raise ValueError(
                f"Variable '{v['name']}': mean={mean} lies outside "
                f"[lower_bound={lo}, upper_bound={hi}] — check settings.json."
            )
        a = (lo - mean) / std if np.isfinite(lo) else -10
        b = (hi - mean) / std if np.isfinite(hi) else 10
        return st.truncnorm(a, b, loc=mean, scale=std)
    if dist_type in ["lognormal", "lognorm"]:
        if mean <= 0:
            raise ValueError(
                f"Variable '{v['name']}': lognormal requires mean > 0, got {mean}."
            )
        sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
        mu = np.log(mean) - 0.5 * sigma ** 2
        return st.lognorm(sigma, scale=np.exp(mu))
    if dist_type in ["gumbel", "gumbel_max", "gumbel_r"]:
        beta = std * np.sqrt(6) / np.pi
        return st.gumbel_r(loc=mean - _EULER * beta, scale=beta)
    if dist_type in ["gumbel_min", "gumbel_l"]:
        beta = std * np.sqrt(6) / np.pi
        return st.gumbel_l(loc=mean + _EULER * beta, scale=beta)
    if dist_type in ["uniform", "unif"]:
        return st.uniform(loc=lo, scale=hi - lo)
    return st.norm(loc=mean, scale=std)


def _truncated_ppf(dist, u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """PPF of `dist` truncated to [lo, hi].

    Maps uniform u in (0, 1) into the inner CDF range [F(lo), F(hi)] before
    inverting. Idempotent for distributions already supported on [lo, hi]
    (e.g. truncnorm / uniform), since F(lo)=0 and F(hi)=1 in that case.
    """
    F_lo = float(dist.cdf(lo)) if np.isfinite(lo) else 0.0
    F_hi = float(dist.cdf(hi)) if np.isfinite(hi) else 1.0
    return dist.ppf(F_lo + u * (F_hi - F_lo))


def _build_correlation_matrix(var_names: list, pairs: list | None) -> np.ndarray:
    """Build an n x n correlation matrix in u-space from a list of pair specs.

    Each pair spec is {"vars": ["A", "B"], "rho": float}.
    """
    n = len(var_names)
    R = np.eye(n)
    if not pairs:
        return R
    name_to_idx = {name: i for i, name in enumerate(var_names)}
    for spec in pairs:
        a, b = spec["vars"]
        rho = float(spec["rho"])
        if a not in name_to_idx or b not in name_to_idx:
            raise ValueError(
                f"correlation_in_u_space references unknown variable: {a} or {b}"
            )
        i, j = name_to_idx[a], name_to_idx[b]
        R[i, j] = R[j, i] = rho
    return R


def is_deterministic(v: dict) -> bool:
    return v.get("distribution_type", "normal").lower() in ("deterministic", "constant", "fixed")


def sample_variables(
    variables: list,
    n_samples: int,
    seed: int = 42,
    correlations: list | None = None,
) -> np.ndarray:
    """Draw MC samples via Nataf transform: correlate in standard normal
    (u-space), then apply each variable's marginal PPF.

    Variables with ``distribution_type == "deterministic"`` are held at their
    mean for every sample and excluded from the correlation matrix.

    Args:
        variables: List of variable defs from settings.json.
        n_samples: Number of samples.
        seed: Random seed.
        correlations: Optional list of {"vars": [a, b], "rho": float} pairs.
            Off-diagonal entries default to 0 (independent). Pairs that
            reference a deterministic variable are silently skipped.

    Returns:
        Array of shape (n_samples, n_vars).
    """
    n_vars = len(variables)
    samples = np.empty((n_samples, n_vars))

    stoch_idx = [i for i, v in enumerate(variables) if not is_deterministic(v)]
    det_idx = [i for i in range(n_vars) if i not in stoch_idx]

    # Deterministic columns: hold at mean
    for i in det_idx:
        samples[:, i] = variables[i]["mean"]

    if not stoch_idx:
        return samples

    # Correlation among stochastic variables only
    stoch_names = [variables[i]["name"] for i in stoch_idx]
    stoch_set = set(stoch_names)
    stoch_pairs = [p for p in (correlations or [])
                   if p["vars"][0] in stoch_set and p["vars"][1] in stoch_set]
    R = _build_correlation_matrix(stoch_names, stoch_pairs)

    # PSD-safe square root via eigendecomposition (handles rho = 1, singular R).
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, 0.0, None)
    L = eigvecs * np.sqrt(eigvals)  # L @ L.T == R

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_samples, len(stoch_idx)))
    U = Z @ L.T
    u_clipped = np.clip(st.norm.cdf(U), 1e-12, 1 - 1e-12)

    for k, i in enumerate(stoch_idx):
        v = variables[i]
        dist = _build_marginal(v)
        lo = v.get("lower_bound", -np.inf)
        hi = v.get("upper_bound", np.inf)
        samples[:, i] = _truncated_ppf(dist, u_clipped[:, k], lo, hi)

    return samples


def component_names(lsf_name: str) -> list[str]:
    """Names of the g-value components returned by the LSF."""
    if lsf_name == "lsf_wall_anchor":
        return ["wall", "anchor"]
    if lsf_name == "lsf_wall":
        return ["wall"]
    if lsf_name == "lsf_anchor":
        return ["anchor"]
    return ["g"]


def plot_component(
    g: np.ndarray,
    component: str,
    summary: dict,
    n_samples: int,
    out_path: Path,
) -> None:
    """Histogram + ECDF for a single g-value component."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Histogram
    ax = axes[0]
    safe = g[g >= 0]
    fail = g[g < 0]
    bins = np.linspace(g.min(), g.max(), 50) if g.size > 1 else 10
    if safe.size:
        ax.hist(safe, bins=bins, color="#4c8dde", alpha=0.85, label=f"safe (n={safe.size})")
    if fail.size:
        ax.hist(fail, bins=bins, color="#d6604d", alpha=0.85, label=f"failed (n={fail.size})")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(f"g_{component}")
    ax.set_ylabel("count")
    ax.set_title(f"Histogram — {component}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # CDF
    ax = axes[1]
    g_sorted = np.sort(g)
    ecdf = np.arange(1, g.size + 1) / g.size
    ax.plot(g_sorted, ecdf, color="#333", linewidth=1.5)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1, label="g = 0")
    pf = summary["pf"]
    ax.axhline(pf, color="#d6604d", linestyle=":", linewidth=1,
               label=f"Pf = {pf:.3e}")
    ax.set_xlabel(f"g_{component}")
    ax.set_ylabel("F(g)")
    ax.set_title(f"ECDF — {component}")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    beta = summary["beta"]
    fig.suptitle(
        f"MC results — {component}   "
        f"N = {n_samples}   Pf = {pf:.3e}   beta = {beta:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, out_path)


def plot_joint(
    g_arrays: dict,
    components: list,
    pf_sys: float,
    beta_sys: float,
    out_path: Path,
) -> None:
    """Scatter of g_a vs g_b for two-component LSFs, with failure quadrants."""
    a, b = components[0], components[1]
    g_a = g_arrays[a]
    g_b = g_arrays[b]
    safe = (g_a >= 0) & (g_b >= 0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(g_a[safe], g_b[safe], s=10, c="#4c8dde", alpha=0.6, label="safe")
    ax.scatter(g_a[~safe], g_b[~safe], s=10, c="#d6604d", alpha=0.7, label="failed")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(f"g_{a}")
    ax.set_ylabel(f"g_{b}")
    ax.set_title(
        f"Joint g — {a} vs {b}\n"
        f"Pf_sys = {pf_sys:.3e}   beta_sys = {beta_sys:.3f}"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)


def evaluate_lsf(lsf_fn, kwargs: dict, lsf_name: str) -> dict[str, float]:
    """Evaluate an LSF and return a dict of component_name -> g_value."""
    call_kwargs = dict(kwargs)
    if lsf_name == "lsf_wall_anchor":
        call_kwargs["return_separate"] = True
    g = lsf_fn(**call_kwargs)
    if not isinstance(g, tuple):
        g = (g,)
    return {name: float(val) for name, val in zip(component_names(lsf_name), g)}


def main(
    lsf_name: str = "lsf_wall_anchor",
    n_samples: int = 100,
    use_api: bool = False,
    seed: int = 42,
):
    if lsf_name not in LSF_REGISTRY:
        raise ValueError(f"Unknown LSF '{lsf_name}'. Available: {list(LSF_REGISTRY.keys())}")

    corrosion_rate = 0.0

    init_model(use_api=use_api)
    lsf_fn = LSF_REGISTRY[lsf_name]
    variables = _settings["variables"]
    var_names = [v["name"] for v in variables]
    comps = component_names(lsf_name)

    print("=" * 60)
    print("Monte Carlo Simulation")
    print("=" * 60)
    print(f"LSF: {lsf_name}")
    print(f"Components: {comps}")
    print(f"Samples: {n_samples}")
    print(f"Corrosion rate: {corrosion_rate:.3f} (fixed)")
    print(f"Variables: {var_names}")

    # Output directory (no per-sample caching — runs always recompute)
    out_dir = _remote / "output" / f"mc_{lsf_name}" / f"cr_{corrosion_rate:.4f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Draw samples (with optional u-space correlation from settings)
    correlations = _settings.get("correlation_in_u_space")
    X = sample_variables(variables, n_samples, seed=seed, correlations=correlations)

    # Evaluate LSF per sample
    g_history: dict[str, list[float]] = {c: [] for c in comps}
    n_fail_sys = 0
    pbar = tqdm(range(n_samples), desc="MC", unit="sample", dynamic_ncols=True)
    for i in pbar:
        # Build kwargs for LSF (mirrors sensitivity: EI is held at EI_start)
        kwargs = {name: float(X[i, j]) for j, name in enumerate(var_names)}
        kwargs["Wall_SheetPilingElementEI"] = _settings["parameters"]["EI_start"]
        kwargs["corrosion_rate"] = corrosion_rate

        g_dict = evaluate_lsf(lsf_fn, kwargs, lsf_name)
        for c in comps:
            g_history[c].append(g_dict[c])

        if any(g_dict[c] < 0 for c in comps):
            n_fail_sys += 1
        pf_running = n_fail_sys / (i + 1)
        beta_running = (
            float(st.norm.ppf(1 - pf_running)) if 0 < pf_running < 1
            else (np.inf if pf_running == 0 else -np.inf)
        )
        pbar.set_postfix(
            fail=n_fail_sys,
            Pf=f"{pf_running:.2e}",
            beta=f"{beta_running:.2f}",
        )

    # Summary
    g_arrays = {c: np.array(g_history[c]) for c in comps}
    sys_failed = np.zeros(n_samples, dtype=bool)
    for c in comps:
        sys_failed |= g_arrays[c] < 0

    n_fail_sys = int(sys_failed.sum())
    pf_sys = n_fail_sys / n_samples
    beta_sys = (
        float(st.norm.ppf(1 - pf_sys)) if 0 < pf_sys < 1
        else (np.inf if pf_sys == 0 else -np.inf)
    )

    per_component = {}
    for c in comps:
        n_fail_c = int(np.sum(g_arrays[c] < 0))
        pf_c = n_fail_c / n_samples
        beta_c = (
            float(st.norm.ppf(1 - pf_c)) if 0 < pf_c < 1
            else (np.inf if pf_c == 0 else -np.inf)
        )
        per_component[c] = {
            "n_failures": n_fail_c,
            "pf": pf_c,
            "beta": beta_c,
            "g_mean": float(g_arrays[c].mean()),
            "g_std": float(g_arrays[c].std()),
            "g_min": float(g_arrays[c].min()),
            "g_max": float(g_arrays[c].max()),
        }

    print(f"\n{'='*60}")
    print(f"Results (cr={corrosion_rate:.3f})")
    print(f"{'='*60}")
    print(f"  N samples:     {n_samples}")
    for c in comps:
        s = per_component[c]
        print(f"  [{c}]")
        print(f"    N failures: {s['n_failures']}")
        print(f"    Pf:         {s['pf']:.4e}")
        print(f"    beta:       {s['beta']:.3f}")
        print(f"    g mean:     {s['g_mean']:.2f}")
        print(f"    g std:      {s['g_std']:.2f}")
        print(f"    g min:      {s['g_min']:.2f}")
    if len(comps) > 1:
        print(f"  [system (any component fails)]")
        print(f"    N failures: {n_fail_sys}")
        print(f"    Pf:         {pf_sys:.4e}")
        print(f"    beta:       {beta_sys:.3f}")
    print(f"  Output:        {out_dir}")

    # Save summary
    summary = {
        "lsf_name": lsf_name,
        "components": comps,
        "corrosion_rate": corrosion_rate,
        "n_samples": n_samples,
        "seed": seed,
        "per_component": per_component,
        "system": {
            "n_failures": n_fail_sys,
            "pf": pf_sys,
            "beta": beta_sys,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:       {out_dir / 'summary.json'}")

    # Plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for c in comps:
        plot_component(
            g_arrays[c],
            component=c,
            summary=per_component[c],
            n_samples=n_samples,
            out_path=plots_dir / f"mc_{c}.png",
        )
    if len(comps) >= 2:
        plot_joint(
            g_arrays,
            components=comps[:2],
            pf_sys=pf_sys,
            beta_sys=beta_sys,
            out_path=plots_dir / f"mc_joint_{comps[0]}_{comps[1]}.png",
        )
    print(f"  Plots:         {plots_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf", type=str, default="lsf_wall", help=f"LSF to use. Available: {list(LSF_REGISTRY.keys())}")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        lsf_name=args.lsf,
        n_samples=args.n_samples,
        use_api=args.use_api,
        seed=args.seed,
    )
