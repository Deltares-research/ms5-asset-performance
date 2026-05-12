"""
Monte Carlo simulation for a D-SheetPiling LSF at a user-selected corrosion
ratio that is held fixed across the entire run.

Samples the stochastic variables from their distributions (as defined in
settings.json), evaluates the selected LSF at the chosen ``corrosion_rate``,
and stores per-sample results (g-value(s)) with periodic checkpointing
(every ``max(10, n_samples // 50)`` iterations, plus a final flush on the
last sample). An interrupted run can be resumed by re-invoking with the
same ``--lsf``, ``--n_samples``, ``--seed`` and ``--cr``.

For ``lsf_wall_anchor`` the LSF is evaluated with ``return_separate=True``
so wall and anchor g-values are tracked independently. The system fails if
either component fails (series system).

Usage:
    python run_mc.py --lsf lsf_wall_anchor --n_samples 100 --cr 0.0
    python run_mc.py --lsf lsf_wall --n_samples 1000 --cr 0.2 --use_api
"""

import sys
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
from scipy import stats as st
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Geolib reads ``geolib.env`` relative to cwd when its ``MetaData`` BaseSettings
# is first instantiated (during ``import geolib`` from src.geotechnical_models).
# When this script is launched from the repo root there is no geolib.env there,
# so we explicitly hydrate the env from the case-study copy before any import
# that pulls geolib in.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / "geolib.env")

# Silence geolib chatter that would interleave with the tqdm progress bar.
# ``geolib.utils`` emits cosmetic ``run_identification`` newline warnings; the
# base_model logger emits a one-line error before raising ``CalculationError``
# whenever DSheetPiling.exe can't be located.
import logging
logging.getLogger("geolib.utils").setLevel(logging.ERROR)
logging.getLogger("geolib.models.base_model").setLevel(logging.CRITICAL)

from src.io import get_remote_path
from src.plotting import save_figure
from reliability.build_fragility import LSF_REGISTRY, init_model, load_settings


_ENV = Path(__file__).resolve().parents[1] / ".env"
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


def _proportion_ci(
    k: np.ndarray, n: np.ndarray, z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray]:
    """Normal-approximation CI for a binomial proportion (vectorised over k, n).

    Half-width is ``z * sqrt(p_hat * (1 - p_hat) / n)``. Default z=1.96 gives
    a 95% interval. Bounds are clipped to [0, 1]; at k = 0 or k = n the band
    has zero width.
    """
    p = k / n
    half = z * np.sqrt(p * (1.0 - p) / n)
    lo = np.clip(p - half, 0.0, 1.0)
    hi = np.clip(p + half, 0.0, 1.0)
    return lo, hi


def plot_convergence(
    g_arrays: dict,
    comps: list[str],
    out_path: Path,
) -> None:
    """Running Pf and beta vs iteration for system + each component.

    Useful for visually judging whether the MC estimator has stabilised: a
    flat tail on both panels indicates convergence; a still-drifting curve
    (or a still-wide CI) indicates too few samples for the (rare-event) Pf
    being estimated. Shaded bands are 95% CIs on the binomial proportion;
    beta bounds are the bounds of the Pf interval pushed through
    ``-Phi^{-1}`` (i.e. upper Pf bound -> lower beta bound).
    """
    n = len(next(iter(g_arrays.values())))
    iters = np.arange(1, n + 1)

    sys_fail = np.zeros(n, dtype=bool)
    for c in comps:
        sys_fail |= g_arrays[c] < 0
    cum_sys = np.cumsum(sys_fail)

    fig, (ax_pf, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

    x_label = n * 1.01  # x position for endpoint labels (in data coords)

    def _add(cum: np.ndarray, label: str, color: str, lw: float):
        pf = cum / iters
        pf_lo, pf_hi = _proportion_ci(cum, iters)

        # Pf panel: mean line + shaded CI band. Mask the leading zero-failure
        # prefix so the log axis doesn't have to deal with pf = 0.
        m_pf = pf > 0
        if m_pf.any():
            ax_pf.fill_between(
                iters[m_pf], pf_lo[m_pf], pf_hi[m_pf],
                color=color, alpha=0.18, linewidth=0,
            )
            ax_pf.plot(iters[m_pf], pf[m_pf],
                       color=color, linewidth=lw, label=label)
            ax_pf.text(
                x_label, pf[m_pf][-1],
                f"Pf={pf[m_pf][-1]:.2e}",
                va="center", ha="left",
                color=color, fontsize=9, clip_on=False,
            )

        # beta panel: invert the Pf band via -Phi^{-1}. Clamp the Pf
        # bounds away from {0, 1} so the inverse-CDF is finite.
        m_b = (pf > 0) & (pf < 1)
        if m_b.any():
            eps = 1e-300
            pf_lo_c = np.clip(pf_lo[m_b], eps, 1.0 - 1e-15)
            pf_hi_c = np.clip(pf_hi[m_b], eps, 1.0 - 1e-15)
            beta_lo = st.norm.ppf(1.0 - pf_hi_c)  # higher Pf -> lower beta
            beta_hi = st.norm.ppf(1.0 - pf_lo_c)  # lower  Pf -> higher beta
            beta = st.norm.ppf(1.0 - pf[m_b])
            ax_b.fill_between(
                iters[m_b], beta_lo, beta_hi,
                color=color, alpha=0.18, linewidth=0,
            )
            ax_b.plot(iters[m_b], beta,
                      color=color, linewidth=lw, label=label)
            ax_b.text(
                x_label, float(beta[-1]),
                f"beta={float(beta[-1]):.2f}",
                va="center", ha="left",
                color=color, fontsize=9, clip_on=False,
            )

    component_colors = ["#4c8dde", "#f0a000", "#2ca02c", "#9467bd"]
    if len(comps) > 1:
        for c, color in zip(comps, component_colors):
            _add(np.cumsum(g_arrays[c] < 0), label=c, color=color, lw=1.0)
        _add(cum_sys, label="system", color="#222", lw=1.8)
    else:
        _add(cum_sys, label=comps[0], color="#222", lw=1.8)

    # Headroom on the right so end-of-curve annotations don't clip.
    x_right = n * 1.18
    ax_pf.set_xlim(right=x_right)
    ax_b.set_xlim(right=x_right)

    ax_pf.set_xlabel("iteration")
    ax_pf.set_ylabel("running Pf")
    ax_pf.set_yscale("log")
    ax_pf.set_title("Running Pf (95% CI shaded)")
    ax_pf.grid(alpha=0.3, which="both")

    ax_b.set_xlabel("iteration")
    ax_b.set_ylabel("running beta")
    ax_b.set_title("Running beta (95% CI shaded)")
    ax_b.grid(alpha=0.3)

    fig.suptitle(f"MC convergence — N = {n}", fontsize=11)
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


def _checkpoint_path(out_dir: Path, corrosion_rate: float) -> Path:
    return out_dir / f"checkpoint_cr_{corrosion_rate:.4f}.csv"


def _load_checkpoint(
    out_dir: Path,
    lsf_name: str,
    n_samples: int,
    seed: int,
    corrosion_rate: float,
    comps: list[str],
    var_names: list[str],
) -> tuple[int, dict[str, list[float]]]:
    """Resume from CSV checkpoint if its run config matches; otherwise start fresh.

    File layout::

        # lsf_name = ...
        # n_samples = ...
        # seed = ...
        # corrosion_rate = ...
        i,<var1>,<var2>,...,<comp1>,<comp2>,...
        0,x0_v1,x0_v2,...,g0_c1,g0_c2,...
        1,x1_v1,x1_v2,...,g1_c1,g1_c2,...

    Sample inputs (X) are written for inspection / downstream analysis but are
    not consumed on resume — X is regenerated deterministically from ``seed``,
    so we only validate that the column names line up and load g-values.
    """
    ckpt = _checkpoint_path(out_dir, corrosion_rate)
    if not ckpt.exists():
        return 0, {c: [] for c in comps}

    meta: dict[str, str] = {}
    with open(ckpt, newline="") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith("#"):
                f.seek(pos)
                break
            kv = line.lstrip("#").strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                meta[k.strip()] = v.strip()
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    try:
        meta_lsf = meta["lsf_name"]
        meta_n = int(meta["n_samples"])
        meta_seed = int(meta["seed"])
        meta_cr = float(meta["corrosion_rate"])
    except (KeyError, ValueError):
        print(f"  Checkpoint {ckpt} metadata unreadable — ignoring.")
        return 0, {c: [] for c in comps}

    expected_header = ["i"] + list(var_names) + list(comps)
    matches = (
        meta_lsf == lsf_name
        and meta_n == n_samples
        and meta_seed == seed
        and abs(meta_cr - corrosion_rate) < 1e-12
        and header == expected_header
    )
    if not matches:
        print(f"  Checkpoint {ckpt} has a different run config — ignoring.")
        return 0, {c: [] for c in comps}

    g_offset = 1 + len(var_names)
    g_history = {c: [] for c in comps}
    for row in rows:
        for k, c in enumerate(comps):
            g_history[c].append(float(row[g_offset + k]))
    i_next = len(rows)
    print(f"  Resuming from {ckpt.name}: {i_next}/{n_samples} samples already computed.")
    return i_next, g_history


def _save_checkpoint(
    out_dir: Path,
    i_next: int,
    g_history: dict[str, list[float]],
    X: np.ndarray,
    var_names: list[str],
    lsf_name: str,
    n_samples: int,
    seed: int,
    corrosion_rate: float,
) -> None:
    """Atomically write the CSV checkpoint (samples + g-values per row)."""
    ckpt = _checkpoint_path(out_dir, corrosion_rate)
    tmp = ckpt.with_suffix(".csv.tmp")
    comps = list(g_history.keys())
    with open(tmp, "w", newline="") as f:
        f.write(f"# lsf_name = {lsf_name}\n")
        f.write(f"# n_samples = {n_samples}\n")
        f.write(f"# seed = {seed}\n")
        f.write(f"# corrosion_rate = {corrosion_rate:.10g}\n")
        writer = csv.writer(f)
        writer.writerow(["i"] + list(var_names) + comps)
        for k in range(i_next):
            writer.writerow(
                [k]
                + [repr(float(X[k, j])) for j in range(len(var_names))]
                + [repr(g_history[c][k]) for c in comps]
            )
    tmp.replace(ckpt)


def postprocess(
    g_history: dict[str, list[float]],
    comps: list[str],
    lsf_name: str,
    corrosion_rate: float,
    seed: int,
    out_dir: Path,
) -> None:
    """Compute summary stats, write summary.json, and emit all plots.

    Postprocessing operates on whatever samples are in ``g_history`` — for
    partial runs the reported N is the number of completed iterations, not
    the originally requested ``n_samples``.
    """
    g_arrays = {c: np.array(g_history[c]) for c in comps}
    n_loaded = len(g_arrays[comps[0]])
    if n_loaded == 0:
        raise RuntimeError(f"No samples to postprocess in {out_dir}.")

    sys_failed = np.zeros(n_loaded, dtype=bool)
    for c in comps:
        sys_failed |= g_arrays[c] < 0

    n_fail_sys = int(sys_failed.sum())
    pf_sys = n_fail_sys / n_loaded
    beta_sys = (
        float(st.norm.ppf(1 - pf_sys)) if 0 < pf_sys < 1
        else (np.inf if pf_sys == 0 else -np.inf)
    )

    per_component = {}
    for c in comps:
        n_fail_c = int(np.sum(g_arrays[c] < 0))
        pf_c = n_fail_c / n_loaded
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
    print(f"  N samples:     {n_loaded}")
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

    summary = {
        "lsf_name": lsf_name,
        "components": comps,
        "corrosion_rate": corrosion_rate,
        "n_samples": n_loaded,
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

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for c in comps:
        plot_component(
            g_arrays[c],
            component=c,
            summary=per_component[c],
            n_samples=n_loaded,
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
    plot_convergence(
        g_arrays,
        comps=comps,
        out_path=plots_dir / "mc_convergence.png",
    )
    print(f"  Plots:         {plots_dir}")


def main(
    lsf_name: str = "lsf_wall_anchor",
    n_samples: int = 100,
    cr: float = 0.0,
    use_api: bool = False,
    seed: int = 42,
    postprocess_only: bool = False,
):
    if lsf_name not in LSF_REGISTRY:
        raise ValueError(f"Unknown LSF '{lsf_name}'. Available: {list(LSF_REGISTRY.keys())}")
    if not 0.0 <= cr < 1.0:
        raise ValueError(f"corrosion_rate must be in [0, 1), got {cr}")

    corrosion_rate = float(cr)
    variables = _settings["variables"]
    var_names = [v["name"] for v in variables]
    comps = component_names(lsf_name)
    out_dir = _remote / "output" / f"mc_{lsf_name}" / f"cr_{corrosion_rate:.4f}"

    if postprocess_only:
        print("=" * 60)
        print("Monte Carlo Postprocessing (no MC runs)")
        print("=" * 60)
        print(f"LSF: {lsf_name}")
        print(f"Components: {comps}")
        print(f"Corrosion rate: {corrosion_rate:.4f}")
        print(f"Expecting: n_samples={n_samples}, seed={seed}")

        ckpt = _checkpoint_path(out_dir, corrosion_rate)
        if not ckpt.exists():
            raise RuntimeError(
                f"Checkpoint file does not exist: {ckpt}\n"
                f"  Double-check --lsf, --cr, and the remote path. "
                f"Note that the directory is selected from --lsf "
                f"(here: 'mc_{lsf_name}/')."
            )
        i_loaded, g_history = _load_checkpoint(
            out_dir, lsf_name, n_samples, seed, corrosion_rate, comps, var_names,
        )
        if i_loaded == 0:
            raise RuntimeError(
                f"Checkpoint {ckpt} exists but its metadata or column layout "
                f"does not match (lsf={lsf_name}, n_samples={n_samples}, "
                f"seed={seed}, cr={corrosion_rate:.4f}). Check the leading "
                f"'#' header lines in the CSV to see what it was written with."
            )
        postprocess(g_history, comps, lsf_name, corrosion_rate, seed, out_dir)
        return

    init_model(use_api=use_api)
    lsf_fn = LSF_REGISTRY[lsf_name]

    ckpt_interval = max(10, n_samples // 50)

    print("=" * 60)
    print("Monte Carlo Simulation")
    print("=" * 60)
    print(f"LSF: {lsf_name}")
    print(f"Components: {comps}")
    print(f"Samples: {n_samples}")
    print(f"Corrosion rate: {corrosion_rate:.4f} (fixed for entire run)")
    print(f"Variables: {var_names}")
    print(f"Checkpoint every: {ckpt_interval} samples")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Draw samples (with optional u-space correlation from settings).
    # The full sample array is deterministic in `seed`, so resuming from a
    # checkpoint just skips the rows that have already been evaluated.
    correlations = _settings.get("correlation_in_u_space")
    X = sample_variables(variables, n_samples, seed=seed, correlations=correlations)

    i_start, g_history = _load_checkpoint(
        out_dir, lsf_name, n_samples, seed, corrosion_rate, comps, var_names,
    )
    n_fail_sys = sum(
        1 for k in range(i_start)
        if any(g_history[c][k] < 0 for c in comps)
    )

    pbar = tqdm(
        range(i_start, n_samples),
        initial=i_start,
        total=n_samples,
        desc="MC",
        unit="sample",
        dynamic_ncols=True,
    )
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

        if (i + 1) % ckpt_interval == 0 or (i + 1) == n_samples:
            _save_checkpoint(
                out_dir, i + 1, g_history, X, var_names,
                lsf_name, n_samples, seed, corrosion_rate,
            )

    postprocess(g_history, comps, lsf_name, corrosion_rate, seed, out_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lsf", type=str, default="lsf_wall", help=f"LSF to use. Available: {list(LSF_REGISTRY.keys())}")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--cr", type=float, default=0.0,
                        help="Corrosion ratio (held fixed for the entire run).")
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--postprocess-only", action="store_true",
        help="Skip the MC loop. Load the checkpoint matching "
             "(--lsf, --n-samples, --seed, --cr) and re-emit the summary and "
             "plots (including a Pf/beta convergence plot).",
    )
    args = parser.parse_args()
    main(
        lsf_name=args.lsf,
        n_samples=args.n_samples,
        cr=args.cr,
        use_api=args.use_api,
        seed=args.seed,
        postprocess_only=args.postprocess_only,
    )
