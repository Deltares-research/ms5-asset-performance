"""Spatial-MCS plots: Pf(cr) parametric curve, per-section beta along the
wall at time t, and a handful of example ``g(x)`` realisations at cr = 0.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

from src.plotting import save_figure


# ----------------------------------------------------------------------
# Wald 95% CI helper (clipped to [0, 1])
# ----------------------------------------------------------------------

def _proportion_ci(
    k: np.ndarray, n: int | np.ndarray, z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray]:
    n_arr = np.asarray(n, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    p = k_arr / n_arr
    half = z * np.sqrt(p * (1.0 - p) / n_arr)
    return np.clip(p - half, 0.0, 1.0), np.clip(p + half, 0.0, 1.0)


# ----------------------------------------------------------------------
# Pf(cr) parametric curve
# ----------------------------------------------------------------------

def pf_vs_cr(
    cr_values: np.ndarray,
    n_fail_section: np.ndarray,   # (n_cr, n_sections)
    n_fail_system: np.ndarray,    # (n_cr,)
    betas_form: np.ndarray,       # (n_cr,) — single-section FORM beta per cr
    n_samples: int,
    out_path: Path,
) -> None:
    """Two-panel plot of Pf(cr) and beta(cr) for the system and per-section avg.

    The single-section FORM curve (from the cached fragility) is overlaid as
    a reference — the spatial system Pf sits above it, the per-section avg
    Pf is its MCS estimate at the section level.
    """
    pf_system = n_fail_system / n_samples
    pf_sys_lo, pf_sys_hi = _proportion_ci(n_fail_system, n_samples)

    # Per-section avg: pool across sections (since they share the same fragility)
    n_fail_sec_pooled = n_fail_section.sum(axis=1)            # (n_cr,)
    n_trials_sec_pooled = n_samples * n_fail_section.shape[1]
    pf_section_avg = n_fail_sec_pooled / n_trials_sec_pooled
    pf_sec_lo, pf_sec_hi = _proportion_ci(n_fail_sec_pooled, n_trials_sec_pooled)

    pf_form_single = st.norm.cdf(-betas_form)

    fig, (ax_pf, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- Pf panel
    ax_pf.fill_between(cr_values, pf_sys_lo, pf_sys_hi,
                       color="#d6604d", alpha=0.20, linewidth=0)
    ax_pf.plot(cr_values, pf_system, "o-", color="#d6604d", linewidth=1.6,
               label="system Pf (MCS)")
    ax_pf.fill_between(cr_values, pf_sec_lo, pf_sec_hi,
                       color="#4c8dde", alpha=0.20, linewidth=0)
    ax_pf.plot(cr_values, pf_section_avg, "s-", color="#4c8dde", linewidth=1.2,
               markersize=4, label="per-section avg Pf (MCS)")
    ax_pf.plot(cr_values, pf_form_single, "--", color="#2ca02c", linewidth=1.2,
               label="single-section Pf (FORM)")
    ax_pf.set_xlabel("cr")
    ax_pf.set_ylabel("Pf")
    ax_pf.set_yscale("log")
    ax_pf.set_title(f"Pf vs cr — N = {n_samples} (95% CI shaded)")
    ax_pf.grid(alpha=0.3, which="both")
    ax_pf.legend(loc="best", fontsize=9)

    # ---- beta panel
    def _b(p):
        out = np.full_like(p, np.nan, dtype=float)
        m = (p > 0) & (p < 1)
        out[m] = st.norm.ppf(1.0 - p[m])
        return out

    ax_b.plot(cr_values, _b(pf_system), "o-", color="#d6604d", linewidth=1.6,
              label="system beta (MCS)")
    ax_b.plot(cr_values, _b(pf_section_avg), "s-", color="#4c8dde", linewidth=1.2,
              markersize=4, label="per-section avg beta (MCS)")
    ax_b.plot(cr_values, betas_form, "--", color="#2ca02c", linewidth=1.2,
              label="single-section beta (FORM)")
    ax_b.set_xlabel("cr")
    ax_b.set_ylabel(r"$\beta$")
    ax_b.set_title("beta vs cr")
    ax_b.grid(alpha=0.3)
    ax_b.legend(loc="best", fontsize=9)

    fig.tight_layout()
    save_figure(fig, out_path)


# ----------------------------------------------------------------------
# Pf and beta vs time, prior-cr integration
# ----------------------------------------------------------------------

def pf_vs_time(
    forecast_times: np.ndarray,
    pf_section_t: np.ndarray,     # (n_t, n_sections)
    pf_system_t: np.ndarray,      # (n_t,)
    out_path: Path,
) -> None:
    """Pf and beta vs forecast time.

    Shows the system curve as a heavy line and the per-section spread
    (min-to-max across sections at each t) as a band, with the per-section
    mean as a thin line. With a single fragility curve shared across
    sections the spread is just MCS noise and the band tightens as
    ``n_samples`` grows.
    """
    sec_min = pf_section_t.min(axis=1)
    sec_max = pf_section_t.max(axis=1)
    sec_mean = pf_section_t.mean(axis=1)

    def to_beta(pf):
        out = np.full_like(pf, np.nan, dtype=float)
        m = (pf > 0) & (pf < 1)
        out[m] = st.norm.ppf(1.0 - np.clip(pf[m], 1e-300, 1.0 - 1e-15))
        return out

    fig, (ax_pf, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_pf.fill_between(forecast_times, sec_min, sec_max,
                       color="#4c8dde", alpha=0.20, linewidth=0,
                       label="per-section range (MCS)")
    ax_pf.plot(forecast_times, sec_mean, "-", color="#4c8dde", linewidth=1.0,
               label="per-section mean")
    ax_pf.plot(forecast_times, pf_system_t, "-", color="#d6604d", linewidth=1.8,
               label="system")
    ax_pf.set_xlabel("t [years]")
    ax_pf.set_ylabel("Pf")
    ax_pf.set_yscale("log")
    ax_pf.set_title("Pf vs time (prior cr integration)")
    ax_pf.grid(alpha=0.3, which="both")
    ax_pf.legend(loc="best", fontsize=9)

    ax_b.fill_between(forecast_times, to_beta(sec_max), to_beta(sec_min),
                      color="#4c8dde", alpha=0.20, linewidth=0,
                      label="per-section range (MCS)")
    ax_b.plot(forecast_times, to_beta(sec_mean), "-", color="#4c8dde", linewidth=1.0,
              label="per-section mean")
    ax_b.plot(forecast_times, to_beta(pf_system_t), "-", color="#d6604d", linewidth=1.8,
              label="system")
    ax_b.set_xlabel("t [years]")
    ax_b.set_ylabel(r"$\beta$")
    ax_b.set_title("beta vs time")
    ax_b.grid(alpha=0.3)
    ax_b.legend(loc="best", fontsize=9)

    fig.tight_layout()
    save_figure(fig, out_path)


# ----------------------------------------------------------------------
# Per-section beta along the wall (at a single time t)
# ----------------------------------------------------------------------

def section_beta(
    pf_section_t: np.ndarray,     # (n_sections,)
    x: np.ndarray,                # (n_sections,)
    beta_single: float,           # single-section reference at this t
    pf_system_t: float,           # system Pf at this t
    out_path: Path,
    t_label: str = "t = 0",
) -> None:
    """Per-section beta along the wall, with single-section and system refs."""
    beta_per_sec = np.array([
        float(st.norm.ppf(1 - p)) if 0 < p < 1
        else (np.inf if p == 0 else -np.inf)
        for p in pf_section_t
    ])
    beta_system_t = (
        float(st.norm.ppf(1 - pf_system_t)) if 0 < pf_system_t < 1
        else (np.inf if pf_system_t == 0 else -np.inf)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, beta_per_sec, "o-", color="#4c8dde",
            label="per-section beta (MCS)")
    ax.axhline(beta_single, color="#2ca02c", linestyle=":",
               label=f"single-section FORM beta = {beta_single:.3f}")
    ax.axhline(beta_system_t, color="#d6604d", linestyle="--",
               label=f"system beta (MCS) = {beta_system_t:.3f}")
    ax.set_xlabel("position along wall [m]")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(f"Beta along the wall — {t_label}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_figure(fig, out_path)


# ----------------------------------------------------------------------
# Example field realisations (sanity-check view)
# ----------------------------------------------------------------------

def realizations(
    x: np.ndarray,
    L_eff: np.ndarray,
    beta_single: float,
    n_show: int,
    seed: int,
    out_path: Path,
) -> None:
    """Draw a handful of fresh ``g(x)`` curves for visual sanity-checking."""
    rng = np.random.default_rng(seed + 9999)
    fig, ax = plt.subplots(figsize=(10, 4))
    for k in range(n_show):
        g = beta_single - L_eff @ rng.standard_normal(len(x))
        ax.plot(x, g, linewidth=0.8, alpha=0.7,
                label=f"sample {k+1}" if k < 4 else None)
        ax.scatter(x[g < 0], g[g < 0], s=20, color="#d6604d", zorder=5)

    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("position along wall [m]")
    ax.set_ylabel(r"$g_i(x)$ in u-space")
    ax.set_title(f"Example g(x) realisations (red = failed section);  "
                 f"single-section beta = {beta_single:.3f}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_path)
