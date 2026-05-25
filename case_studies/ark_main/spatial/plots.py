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
# Pf and beta vs time, posterior leg (one curve per obs time)
# ----------------------------------------------------------------------

def pf_vs_time_posterior(
    forecast_times_prior: np.ndarray,
    pf_system_prior: np.ndarray,
    posterior_results: dict[str, dict],
    out_path: Path,
) -> None:
    """Posterior system-Pf vs t, one curve per obs time.

    Each obs scenario gets a thin viridis-coloured curve on ``[t_obs,
    t_end]``. A ``+`` marker sits at the curve's leftmost point (the obs
    time itself). The prior leg's system Pf is drawn as a bold reference
    line so the obs-conditional shift is readable at a glance.

    ``posterior_results[t_obs_str]`` is expected to expose
    ``forecast_times`` (n_t,) and ``pf_system`` (n_t,) — convertable from
    the cached ``n_fail_system / n_samples``.
    """
    obs_keys = sorted(posterior_results.keys(), key=float)
    n_obs = len(obs_keys)
    cmap = plt.get_cmap("viridis")

    def to_beta(pf):
        out = np.full_like(pf, np.nan, dtype=float)
        m = (pf > 0) & (pf < 1)
        out[m] = st.norm.ppf(1.0 - np.clip(pf[m], 1e-300, 1.0 - 1e-15))
        return out

    fig, (ax_pf, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_pf.plot(forecast_times_prior, pf_system_prior,
               color="#222", linewidth=2.0, label="prior")
    ax_b.plot(forecast_times_prior, to_beta(pf_system_prior),
              color="#222", linewidth=2.0, label="prior")

    for k, key in enumerate(obs_keys):
        block = posterior_results[key]
        t_arr = np.asarray(block["forecast_times"], dtype=float)
        pf_sys = np.asarray(block["pf_system"], dtype=float)
        color = cmap((k + 0.5) / max(n_obs, 1))

        ax_pf.plot(t_arr, pf_sys, color=color, linewidth=1.0, alpha=0.9)
        ax_pf.scatter([t_arr[0]], [pf_sys[0]], marker="+",
                      color=color, s=40, zorder=5)
        ax_pf.text(t_arr[-1] * 1.005, pf_sys[-1],
                   f"t_obs={float(key):.1f}", color=color, fontsize=8,
                   va="center", ha="left", clip_on=False)

        beta_sys = to_beta(pf_sys)
        ax_b.plot(t_arr, beta_sys, color=color, linewidth=1.0, alpha=0.9)
        ax_b.scatter([t_arr[0]], [beta_sys[0]], marker="+",
                     color=color, s=40, zorder=5)
        ax_b.text(t_arr[-1] * 1.005, float(beta_sys[-1]),
                  f"t_obs={float(key):.1f}", color=color, fontsize=8,
                  va="center", ha="left", clip_on=False)

    t_right = forecast_times_prior[-1] * 1.18
    ax_pf.set_xlim(right=t_right)
    ax_b.set_xlim(right=t_right)

    ax_pf.set_xlabel("t [years]")
    ax_pf.set_ylabel("Pf system")
    ax_pf.set_yscale("log")
    ax_pf.set_title("Posterior Pf vs t (one curve per obs time)")
    ax_pf.grid(alpha=0.3, which="both")

    ax_b.set_xlabel("t [years]")
    ax_b.set_ylabel(r"$\beta$ system")
    ax_b.set_title(r"Posterior $\beta$ vs t")
    ax_b.grid(alpha=0.3)

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
# Spatial corrosion forecast at one obs time
# ----------------------------------------------------------------------

def cr_along_wall(
    x: np.ndarray,
    prior_q: np.ndarray,      # (n_q,) — typically (q05, q50, q95) in cr units
    cond_q: np.ndarray,       # (n_sections, n_q)
    t_obs: float,
    t_at: float,
    wall_thickness: float,
    out_path: Path,
    obs_value_mm: float | None = None,
    obs_error_std_mm: float | None = None,
    ylim_mm: tuple[float, float] | None = None,
) -> None:
    """Corrosion vs position at time ``t_at`` given obs through ``t_obs``.

    Layout:

    * Prior 90% band shaded blue, prior median as a solid blue horizontal
      line — both stationary along ``x``.
    * Posterior conditional 90% band shaded red, posterior median as a
      solid red curve — narrows at ``x = x[0]`` (where the obs is taken)
      and fans out to the prior at sections beyond the cr correlation
      length.
    * Observation marker at ``x = 0`` with measurement-noise error bars
      (95% CI ≈ ±1.96·sigma) when ``obs_value_mm`` is provided.

    Units: ``prior_q`` and ``cond_q`` are dimensionless corrosion ratios;
    we multiply by ``wall_thickness`` for the y-axis so the values read in
    millimetres alongside the observation.
    """
    pq05, pq50, pq95 = prior_q[0], prior_q[1], prior_q[2]
    cq05 = cond_q[:, 0] * wall_thickness
    cq50 = cond_q[:, 1] * wall_thickness
    cq95 = cond_q[:, 2] * wall_thickness

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Prior — constant horizontal band/line across the whole wall.
    ax.axhspan(pq05 * wall_thickness, pq95 * wall_thickness,
               color="#4c8dde", alpha=0.18, linewidth=0)
    ax.axhline(pq50 * wall_thickness, color="#4c8dde", linewidth=1.5,
               label="prior median (constant)")
    # Two fake patches just to get the legend rendering on the band:
    ax.plot([], [], color="#4c8dde", linewidth=8, alpha=0.18,
            label="prior 90% CI")

    # Posterior — varies along x.
    ax.fill_between(x, cq05, cq95, color="#d6604d", alpha=0.20,
                    linewidth=0, label="posterior 90% CI")
    ax.plot(x, cq50, color="#d6604d", linewidth=1.8,
            label="posterior median")

    # Observation marker at x = 0.
    if obs_value_mm is not None:
        if obs_error_std_mm and obs_error_std_mm > 0:
            ax.errorbar(
                [float(x[0])], [obs_value_mm],
                yerr=[1.96 * obs_error_std_mm],
                fmt="o", color="k", capsize=4, zorder=5,
                label="observation (95% CI)",
            )
        else:
            ax.scatter([float(x[0])], [obs_value_mm], color="k",
                       s=40, zorder=5, label="observation")

    ax.set_xlabel("position along wall [m]")
    ax.set_ylabel("corrosion [mm]")
    ax.set_title(
        f"Spatial corrosion at t = {t_at:.1f} yr  |  "
        f"conditioned on obs through t = {t_obs:.1f} yr"
    )
    if ylim_mm is not None:
        ax.set_ylim(ylim_mm)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, out_path)


# ----------------------------------------------------------------------
# Split-violin spatial corrosion plot (prior vs posterior per section)
# ----------------------------------------------------------------------

def cr_along_wall_violin(
    x: np.ndarray,
    cr_prior_samples_mm: np.ndarray,    # (n_samples,) — stationary prior
    cr_post_samples_mm: np.ndarray,     # (n_samples, n_sections)
    t_obs: float,
    t_at: float,
    wall_thickness: float,
    out_path: Path,
    obs_value_mm: float | None = None,
    obs_error_std_mm: float | None = None,
    ylim_mm: tuple[float, float] | None = None,
) -> None:
    """Per-section split violin: prior (left, blue) vs posterior (right, red).

    Both half-violins share a single density normalisation so the visual
    scale is consistent: the widest violin in the figure (prior or
    posterior, any section) hits the half-width budget; narrower violins
    are proportionally thinner. This makes the posterior collapse at the
    obs location visually obvious.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    n_sections = len(x)
    spacing = (x[-1] - x[0]) / max(n_sections - 1, 1)
    violin_width = 0.7 * spacing            # half ≤ ~0.35 of section spacing

    y_lo = min(cr_prior_samples_mm.min(), cr_post_samples_mm.min())
    y_hi = max(cr_prior_samples_mm.max(), cr_post_samples_mm.max())
    y_grid = np.linspace(y_lo, y_hi, 100)

    kde_prior = st.gaussian_kde(cr_prior_samples_mm)
    density_prior = kde_prior(y_grid)
    max_density_prior = float(density_prior.max())

    densities_post = []
    for i in range(n_sections):
        col = cr_post_samples_mm[:, i]
        # gaussian_kde rejects (near-)constant samples — fall back to a
        # narrow spike at the median when that happens.
        if col.std() < 1e-9:
            d = np.zeros_like(y_grid)
            d[np.argmin(np.abs(y_grid - col[0]))] = 1.0
        else:
            d = st.gaussian_kde(col)(y_grid)
        densities_post.append(d)
    max_density_post = max(float(d.max()) for d in densities_post)
    norm_factor = (violin_width / 2.0) / max(
        max_density_prior, max_density_post, 1e-12
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))

    for i, xi in enumerate(x):
        d_left = density_prior * norm_factor
        ax.fill_betweenx(
            y_grid, xi - d_left, np.full_like(d_left, float(xi)),
            color="#4c8dde", alpha=0.75, edgecolor="k", linewidth=0.3,
        )
        d_right = densities_post[i] * norm_factor
        ax.fill_betweenx(
            y_grid, np.full_like(d_right, float(xi)), xi + d_right,
            color="#d6604d", alpha=0.75, edgecolor="k", linewidth=0.3,
        )

    if obs_value_mm is not None:
        if obs_error_std_mm and obs_error_std_mm > 0:
            ax.errorbar(
                [float(x[0])], [obs_value_mm],
                yerr=[1.96 * obs_error_std_mm],
                fmt="o", color="k", capsize=4, zorder=5,
            )
        else:
            ax.scatter([float(x[0])], [obs_value_mm],
                       color="k", s=40, zorder=5)

    legend_handles = [
        Patch(color="#4c8dde", alpha=0.75, label="prior"),
        Patch(color="#d6604d", alpha=0.75, label="posterior"),
    ]
    if obs_value_mm is not None:
        legend_handles.append(
            Line2D([], [], color="k", marker="o", linestyle="",
                   label="obs (95% CI)")
        )

    ax.set_xlabel("position along wall [m]")
    ax.set_ylabel("corrosion [mm]")
    ax.set_title(
        f"cr distribution per section at t = {t_at:.1f} yr  |  "
        f"left=prior, right=posterior (obs through t = {t_obs:.1f} yr)"
    )
    if ylim_mm is not None:
        ax.set_ylim(ylim_mm)
    ax.set_xlim(float(x[0]) - violin_width, float(x[-1]) + violin_width)
    ax.grid(alpha=0.3)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
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


# ----------------------------------------------------------------------
# Nested-FORM alpha visualisations
# ----------------------------------------------------------------------

def _alpha_grid_layout(n_panels: int) -> tuple[int, int]:
    """Pick a (rows, cols) layout that's roughly square for n_panels."""
    cols = int(np.ceil(np.sqrt(n_panels)))
    rows = int(np.ceil(n_panels / cols))
    return rows, cols


def alpha_heatmap(
    *,
    forecast_times: np.ndarray,
    x: np.ndarray,
    alpha_cr: np.ndarray,            # (n_t, n_sections)
    alpha_basic: np.ndarray,         # (n_t, n_sections, n_active)
    active_vars: list[str],
    t_obs: float,
    out_path: Path,
) -> None:
    """One heatmap per variable: alpha_v(t, section), plus alpha_cr.

    Color is diverging (RdBu) and symmetric around 0 with a shared vmax across
    all panels in the figure so per-variable contributions are comparable.
    Hot red = positive alpha (load-like), cold blue = negative alpha
    (resistance-like). White = ~0 (variable doesn't matter at that (t, section)).
    """
    # Order panels: alpha_cr first, then basic vars in input order.
    panels = [("cr", alpha_cr)] + [
        (active_vars[j], alpha_basic[:, :, j]) for j in range(len(active_vars))
    ]
    n_panels = len(panels)
    rows, cols = _alpha_grid_layout(n_panels)

    # Shared symmetric color scale across all panels.
    vmax = max(0.05, float(np.max(np.abs(np.stack([p[1] for p in panels])))))

    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows),
                             squeeze=False)
    extent = [forecast_times[0], forecast_times[-1], x[0], x[-1]]
    for k, (name, arr) in enumerate(panels):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        im = ax.imshow(
            arr.T,                                 # (n_sections, n_t)
            origin="lower", aspect="auto", extent=extent,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest",
        )
        ax.axvline(t_obs, color="k", linestyle=":", linewidth=0.8)
        ax.set_title(name, fontsize=10)
        if r == rows - 1:
            ax.set_xlabel("t")
        if c == 0:
            ax.set_ylabel("section x [m]")

    # Blank any spare panels in the last row.
    for k in range(n_panels, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02,
                        location="right", aspect=30)
    cbar.set_label(r"$\alpha_v$")
    fig.suptitle(
        rf"Nested-FORM $\alpha_v(t, x)$  —  $t_{{\rm obs}}={t_obs:.1f}$",
        fontsize=12,
    )
    save_figure(fig, out_path)


def alpha_lines_at_sections(
    *,
    forecast_times: np.ndarray,
    x: np.ndarray,
    alpha_cr: np.ndarray,            # (n_t, n_sections)
    alpha_basic: np.ndarray,         # (n_t, n_sections, n_active)
    active_vars: list[str],
    section_indices: list[int],
    t_obs: float,
    out_path: Path,
) -> None:
    """alpha_v(t) at a handful of sections — one subplot per section.

    cr is drawn as a thick dashed black line; basic variables share a viridis
    colour ramp keyed by variable index so the same variable has the same
    colour across all subplots and across obs scenarios. A horizontal y = 0
    line marks the resistance/load sign flip.
    """
    n_sec = len(section_indices)
    fig, axes = plt.subplots(
        1, n_sec, figsize=(4.2 * n_sec, 3.6), sharey=True, squeeze=False,
    )
    axes = axes[0]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(j / max(1, len(active_vars) - 1)) for j in range(len(active_vars))]

    for ax_i, sec_idx in enumerate(section_indices):
        ax = axes[ax_i]
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        ax.axvline(t_obs, color="k", linestyle=":", linewidth=0.8)
        # cr line first so it's on top of legend reading order.
        ax.plot(forecast_times, alpha_cr[:, sec_idx],
                color="black", linestyle="--", linewidth=2.2, label="cr")
        for j, v in enumerate(active_vars):
            ax.plot(forecast_times, alpha_basic[:, sec_idx, j],
                    color=colors[j], linewidth=1.4, label=v)
        ax.set_title(f"x = {x[sec_idx]:.0f} m", fontsize=10)
        ax.set_xlabel("t")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\alpha_v$")
    # One legend for the whole figure on the right.
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=8, frameon=False)
    fig.suptitle(
        rf"Nested-FORM $\alpha_v(t)$ at selected sections  —  $t_{{\rm obs}}={t_obs:.1f}$",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.85, 0.94))
    save_figure(fig, out_path)


def alpha_lines_prior(
    *,
    forecast_times: np.ndarray,
    alpha_cr: np.ndarray,            # (n_t,)
    alpha_basic: np.ndarray,         # (n_t, n_active)
    active_vars: list[str],
    out_path: Path,
) -> None:
    """alpha_v(t) for the unconditional (prior) nested-FORM leg.

    Single panel — no section dimension because the prior is stationary along
    the wall, so ``(alpha_cr_t, alpha_basic_t)`` is the same at every section.
    Same colour palette as :func:`alpha_lines_at_sections`: cr as thick dashed
    black, basic vars on a viridis ramp keyed by variable index.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(j / max(1, len(active_vars) - 1)) for j in range(len(active_vars))]
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.plot(forecast_times, alpha_cr,
            color="black", linestyle="--", linewidth=2.2, label="cr")
    for j, v in enumerate(active_vars):
        ax.plot(forecast_times, alpha_basic[:, j],
                color=colors[j], linewidth=1.4, label=v)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\alpha_v$")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False)
    ax.set_title(r"Nested-FORM $\alpha_v(t)$ — unconditional (prior) leg")
    fig.tight_layout(rect=(0, 0, 0.80, 1.0))
    save_figure(fig, out_path)
