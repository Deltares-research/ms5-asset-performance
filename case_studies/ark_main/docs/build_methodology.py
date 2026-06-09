"""
Build the ARK-main methodology documentation as a Word document.

Produces:
- ``figures/*.png`` — illustrative figures for each section. Synthetic
  curves built from the same equations the pipeline uses, so they
  visually represent the method without depending on the live remote
  share or specific output runs.
- ``methodology.docx`` — the assembled document, written to this ``docs/``
  folder and (if reachable) copied to the synced Deltares Documentation
  folder.

Run::

    python case_studies/ark_main/docs/build_methodology.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Inches, Pt, RGBColor
from scipy import stats as st
from scipy.optimize import minimize_scalar

DOCS = Path(__file__).resolve().parent
FIG = DOCS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DOCX_OUT = DOCS / "methodology.docx"

# Synced Deltares Documentation copy. Best-effort target — if the share is
# offline or the file is open in Word the copy is skipped with a warning.
DELTARES_OUT = Path(
    r"C:\Users\mavritsa\Stichting Deltares\SITO-IS 2025 Moonshot 5 - "
    r"MS5.2 Smart & resilient infra\02_Asset performance\03 Products and "
    r"workshops\Documentation\methodology.docx"
)


# ======================================================================
# Figure generators (synthetic / illustrative)
# ======================================================================

def fig_wall_schematic(path: Path) -> None:
    """Schematic of the sheet-pile wall + anchor + soil layers + canal."""
    fig, ax = plt.subplots(figsize=(8, 5))
    # ground
    ax.fill_between([0, 1], -3.5, 0, color="#d8c5a0", alpha=0.4, linewidth=0)
    ax.axhline(0, color="#7a6a4d", linewidth=1)
    # canal (left side, lower)
    ax.fill_between([-0.5, 0.45], -2.5, -0.4, color="#4c8dde", alpha=0.5,
                    linewidth=0)
    # phreatic level (right side)
    ax.axhline(-0.8, xmin=0.55, xmax=1.0, color="#4c8dde",
               linestyle="--", linewidth=1.0)
    # wall
    wall_x = 0.5
    ax.plot([wall_x, wall_x], [0.4, -3.0], color="#444",
            linewidth=4, solid_capstyle="butt")
    # anchor
    ax.plot([wall_x, wall_x + 0.35], [-0.5, -0.5], color="#a83232",
            linewidth=2)
    ax.scatter([wall_x + 0.35], [-0.5], s=80, color="#a83232", zorder=5)
    # soil layers (right of wall)
    ax.fill_between([wall_x, 1.0], -1.0, 0, color="#e6c98a", alpha=0.6,
                    linewidth=0)
    ax.fill_between([wall_x, 1.0], -2.0, -1.0, color="#c9a96e", alpha=0.7,
                    linewidth=0)
    ax.fill_between([wall_x, 1.0], -3.5, -2.0, color="#9a7d4a", alpha=0.7,
                    linewidth=0)
    # labels
    ax.text(0.0, -0.18, "canal", fontsize=10, color="#1a3d6d")
    ax.text(0.78, -0.5, "Klei", fontsize=10, ha="center")
    ax.text(0.78, -1.5, "Zand", fontsize=10, ha="center")
    ax.text(0.78, -2.7, "Zandvast", fontsize=10, ha="center")
    ax.text(wall_x + 0.36, -0.4, "anchor", fontsize=9, color="#a83232")
    ax.text(wall_x - 0.02, 0.55, "sheet pile wall",
            fontsize=10, ha="right", color="#444")
    ax.annotate("", xy=(0.45, 0.10), xytext=(0.25, 0.40),
                arrowprops=dict(arrowstyle="->", color="#1a3d6d"))
    ax.text(0.18, 0.45, "passive load\n(water + soil)",
            fontsize=8, color="#1a3d6d")
    ax.annotate("", xy=(0.52, -0.25), xytext=(0.85, -0.10),
                arrowprops=dict(arrowstyle="->", color="#7a3a3a"))
    ax.text(0.86, 0.05, "active load\n(soil + surcharge q)",
            fontsize=8, color="#7a3a3a")
    ax.set_xlim(-0.5, 1.05)
    ax.set_ylim(-3.5, 0.9)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("ARK sheet-pile wall cross section (single section)")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_fragility(path: Path) -> None:
    """Synthetic β(cr) + Pf(cr) curve resembling the cached fragility."""
    cr = np.linspace(0, 1, 50)
    beta = 4.0 - 6.0 * cr ** 0.9 + 0.2 * np.sin(4 * cr)
    pf = 0.5 * (1 - np.tanh(2.5 * beta))   # placeholder for Φ(-β)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    ax1.plot(cr, beta, color="#1a3d6d", linewidth=2)
    ax1.scatter(cr[::5], beta[::5], color="#a83232", s=20,
                label="cached design points")
    ax1.set_xlabel("corrosion ratio cr")
    ax1.set_ylabel(r"$\beta$  (FORM)")
    ax1.set_title(r"Fragility: $\beta$ vs cr")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.plot(cr, pf, color="#1a3d6d", linewidth=2)
    ax2.set_xlabel("corrosion ratio cr")
    ax2.set_ylabel(r"$P_f = \Phi(-\beta)$")
    ax2.set_yscale("log")
    ax2.set_title(r"Fragility: $P_f$ vs cr")
    ax2.grid(alpha=0.3, which="both")
    fig.suptitle("Fragility curve (cached one-shot FORM per cr)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_corrosion_model(path: Path) -> None:
    """Power-law corrosion model — prior bands + posterior tightening."""
    t = np.linspace(0, 60, 200)
    rng = np.random.default_rng(0)
    n = 1000
    A = rng.lognormal(np.log(0.15), 0.35, n)
    B = rng.normal(0.70, 0.07, n)
    cr_t = A[:, None] * t[None, :] ** B[:, None]
    cr_t = np.clip(cr_t, 0, 5)
    q05 = np.quantile(cr_t, 0.05, axis=0)
    q50 = np.quantile(cr_t, 0.50, axis=0)
    q95 = np.quantile(cr_t, 0.95, axis=0)

    # mock posterior given an "observation" at t=20 of 1.4 mm with sigma=0.2
    t_obs, y_obs, sig = 20.0, 1.4, 0.2
    pred = A[:, None] * t_obs ** B[:, None]
    log_w = -0.5 * ((pred.squeeze() - y_obs) / sig) ** 2
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    q05p = np.array([np.interp(0.05, np.cumsum(w[np.argsort(cr_t[:, i])]),
                               np.sort(cr_t[:, i])) for i in range(len(t))])
    q50p = np.array([np.interp(0.50, np.cumsum(w[np.argsort(cr_t[:, i])]),
                               np.sort(cr_t[:, i])) for i in range(len(t))])
    q95p = np.array([np.interp(0.95, np.cumsum(w[np.argsort(cr_t[:, i])]),
                               np.sort(cr_t[:, i])) for i in range(len(t))])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(t, q05, q95, color="#4c8dde", alpha=0.18,
                    label="prior 90% band")
    ax.plot(t, q50, color="#1a3d6d", linewidth=1.6, label="prior median")
    ax.fill_between(t, q05p, q95p, color="#d6604d", alpha=0.20,
                    label="posterior 90% band")
    ax.plot(t, q50p, color="#7a3a3a", linewidth=1.6, label="posterior median")
    ax.errorbar([t_obs], [y_obs], yerr=[1.96 * sig], fmt="o", color="k",
                capsize=4, zorder=5, label="observation")
    ax.set_xlabel("time t [yr]")
    ax.set_ylabel("corrosion [mm]")
    ax.set_title(r"Corrosion model $cr(t)=A\cdot t^B$  —  prior vs posterior")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_spatial_kernel(path: Path) -> None:
    """Spatial covariance kernel family C(d) = rho_0 + (1-rho_0)*exp(-(d/theta)^2)."""
    d = np.linspace(0, 1200, 400)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    combos = [
        (50.0,  0.0,  "#1a3d6d", "θ=50, ρ₀=0"),
        (200.0, 0.0,  "#4c8dde", "θ=200, ρ₀=0"),
        (800.0, 0.0,  "#88c4f0", "θ=800, ρ₀=0"),
        (200.0, 0.3,  "#a83232", "θ=200, ρ₀=0.3"),
        (200.0, 0.6,  "#d6604d", "θ=200, ρ₀=0.6"),
    ]
    for theta, rho_0, color, label in combos:
        C = rho_0 + (1.0 - rho_0) * np.exp(-(d / theta) ** 2)
        ax.plot(d, C, color=color, linewidth=1.7, label=label)
    ax.set_xlabel("distance d [m]")
    ax.set_ylabel("correlation C(d)")
    ax.set_title("Squared-exponential kernel family")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_cr_field_along_wall(path: Path) -> None:
    """cr prior horizontal band vs posterior conditioned on x=0 obs."""
    x = np.linspace(0, 1000, 21)
    theta = 200.0
    rho = np.exp(-(x / theta) ** 2)
    # made-up prior + posterior z-moments at the obs location.
    m_post, v_post = -0.8, 0.35
    # closed-form 5/50/95 along x in z, then nominally interpret as cr scale.
    z_mean = rho * m_post
    z_var = rho * rho * v_post + (1.0 - rho * rho)
    z_lo = z_mean - 1.645 * np.sqrt(z_var)
    z_hi = z_mean + 1.645 * np.sqrt(z_var)

    # Reuse the prior q05/q50/q95 from the corrosion model as the horizontal
    # band for visual consistency. Pin them to t = 30 yr.
    rng = np.random.default_rng(0)
    n = 2000
    A = rng.lognormal(np.log(0.15), 0.35, n)
    B = rng.normal(0.70, 0.07, n)
    cr30 = A * 30.0 ** B
    cr30 = np.clip(cr30, 0, 6)
    pq05, pq50, pq95 = np.quantile(cr30, [0.05, 0.5, 0.95])

    # Map z back to a heuristic cr scale anchored on the prior band.
    cr_med = pq50 + (pq50 - pq05) * (z_mean / -1.645)
    cr_lo = pq50 + (pq50 - pq05) * (z_lo / -1.645)
    cr_hi = pq50 + (pq95 - pq50) * (z_hi / 1.645)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhspan(pq05, pq95, color="#4c8dde", alpha=0.18, linewidth=0)
    ax.axhline(pq50, color="#1a3d6d", linewidth=1.5, label="prior median")
    ax.plot([], [], color="#4c8dde", linewidth=10, alpha=0.18,
            label="prior 90% CI")
    ax.fill_between(x, cr_lo, cr_hi, color="#d6604d", alpha=0.20,
                    label="posterior 90% CI")
    ax.plot(x, cr_med, color="#7a3a3a", linewidth=1.8,
            label="posterior median")
    ax.scatter([0], [pq05 * 0.4], color="k", s=40, zorder=5,
               label="obs at x=0")
    ax.set_xlabel("position along wall x [m]")
    ax.set_ylabel("corrosion [mm]")
    ax.set_title("cr field along the wall — prior + Kriging-conditional posterior")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# Shared synthetic fragility + corrosion helpers so the nested-FORM figures
# below use exactly the equations the pipeline uses.

def _synthetic_beta_of_cr(cr: np.ndarray) -> np.ndarray:
    """A monotone-decreasing β(cr) resembling the cached fragility."""
    return 4.0 - 6.0 * np.asarray(cr, dtype=float) ** 0.9


def _prior_cr_cdf_at_t(t: float, cr_grid: np.ndarray):
    """Prior CDF of cr at time t from the synthetic power-law corrosion model."""
    rng = np.random.default_rng(0)
    n = 4000
    A = rng.lognormal(np.log(0.15), 0.35, n)
    B = rng.normal(0.70, 0.07, n)
    cr_mm = A * max(float(t), 1e-6) ** B
    cr_ratio = np.clip(cr_mm / 9.5, 0.0, 1.0)
    # empirical CDF on cr_grid
    cr_sorted = np.sort(cr_ratio)
    cdf = np.searchsorted(cr_sorted, cr_grid, side="right") / len(cr_sorted)
    return np.clip(cdf, 1e-9, 1 - 1e-9)


def _nested_form_1d(mu_z, sigma_z, cr_grid, F_prior, cr_vals, betas):
    """Mirror of spatial.nested_mcs._nested_form_1d for the figures."""
    def cr_from_xi(xi):
        z = mu_z + sigma_z * float(xi)
        u = float(st.norm.cdf(z))
        return float(np.interp(u, F_prior, cr_grid))

    def loss(xi):
        cr = cr_from_xi(xi)
        beta = float(np.interp(cr, cr_vals, betas))
        return float(xi) ** 2 + beta ** 2

    res = minimize_scalar(loss, bounds=(-6.0, 6.0), method="bounded",
                          options={"xatol": 1e-5})
    xi_star = float(res.x)
    cr_star = cr_from_xi(xi_star)
    beta_star = float(np.interp(cr_star, cr_vals, betas))
    return xi_star, cr_star, beta_star


def fig_nested_form(path: Path) -> None:
    """Nested-FORM 1-D search for the cr design point at one forecast t."""
    xi = np.linspace(-4, 4, 400)
    # mock beta(cr(xi)) via a smooth function
    mu_z = -0.5
    sigma_z = 1.0
    cr_of_xi = 0.5 * (1 + np.tanh((mu_z + sigma_z * xi) * 0.7))
    beta_of_cr = 4.0 - 6.0 * cr_of_xi ** 0.9 + 0.2 * np.sin(4 * cr_of_xi)
    obj = xi ** 2 + beta_of_cr ** 2
    i_star = int(np.argmin(obj))
    xi_star = xi[i_star]
    beta_T = np.sqrt(obj[i_star])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(xi, beta_of_cr, color="#1a3d6d", linewidth=1.6)
    ax1.axvline(xi_star, color="#a83232", linestyle="--", linewidth=1)
    ax1.scatter([xi_star], [beta_of_cr[i_star]], color="#a83232", s=40)
    ax1.set_xlabel(r"$\xi$  (z-space)")
    ax1.set_ylabel(r"$\beta(cr(\xi))$ from fragility")
    ax1.set_title(r"Inner: $\beta$ along cr-axis")
    ax1.grid(alpha=0.3)

    ax2.plot(xi, np.sqrt(obj), color="#4c8dde", linewidth=1.6,
             label=r"$\sqrt{\xi^2 + \beta(cr(\xi))^2}$")
    ax2.axvline(xi_star, color="#a83232", linestyle="--", linewidth=1)
    ax2.scatter([xi_star], [beta_T], color="#a83232", s=50,
                label=fr"$\beta_T={beta_T:.2f}$")
    ax2.set_xlabel(r"$\xi$  (z-space)")
    ax2.set_ylabel(r"$\beta_T$")
    ax2.set_title(r"Outer: minimise total reliability index")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.suptitle("Nested-FORM design-point search in cr (alphas method)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_alpha_corrosion(path: Path) -> None:
    """α of corrosion over time: actually run the nested-FORM decomposition.

    Uses the synthetic β(cr) + power-law prior CDF so the curves reflect the
    same equations as ``spatial/nested_mcs.py``. Two panels:
      (1) β_form(cr*) vs β_T(t) — corrosion lifts the total reliability index.
      (2) stacked α² shares: α_cr² (corrosion) + Σ α_basic,v² (= 1).
    """
    cr_grid = np.linspace(0, 1, 600)
    cr_vals = np.linspace(0, 1, 60)
    betas = _synthetic_beta_of_cr(cr_vals)
    times = np.linspace(2, 50, 40)

    beta_form = np.zeros_like(times)
    beta_T = np.zeros_like(times)
    alpha_cr = np.zeros_like(times)
    for k, t in enumerate(times):
        F_prior = _prior_cr_cdf_at_t(t, cr_grid)
        xs, crs, bstar = _nested_form_1d(0.0, 1.0, cr_grid, F_prior,
                                         cr_vals, betas)
        beta_form[k] = bstar
        bT = np.sqrt(xs * xs + bstar * bstar)
        beta_T[k] = bT
        alpha_cr[k] = -xs / bT if bT > 0 else 0.0

    a_cr_sq = alpha_cr ** 2
    a_basic_sq = 1.0 - a_cr_sq

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(times, beta_form, color="#1a3d6d", linewidth=1.8,
             label=r"$\beta(cr^*)$ — fragility at design pt")
    ax1.plot(times, beta_T, color="#a83232", linewidth=1.8,
             label=r"$\beta_T=\sqrt{\xi^{*2}+\beta(cr^*)^2}$")
    ax1.set_xlabel("time t [yr]")
    ax1.set_ylabel(r"$\beta$")
    ax1.set_title("Total reliability index vs fragility β")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.fill_between(times, 0, a_cr_sq, color="#d6604d", alpha=0.7,
                     label=r"$\alpha_{cr}^2$ (corrosion)")
    ax2.fill_between(times, a_cr_sq, 1.0, color="#4c8dde", alpha=0.6,
                     label=r"$\sum_v \alpha_{basic,v}^2$ (structural)")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("time t [yr]")
    ax2.set_ylabel(r"importance share $\alpha^2$")
    ax2.set_title(r"Variance decomposition ($\alpha_{cr}^2+\sum\alpha_v^2=1$)")
    ax2.legend(loc="center right", fontsize=9)
    fig.suptitle("Estimating the importance factor of corrosion (nested FORM)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_pipeline_arch(path: Path) -> None:
    """Pipeline class hierarchy as a simple boxes-and-arrows diagram."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, color="#cfd8e3"):
        rect = plt.Rectangle((x, y), w, h, facecolor=color,
                             edgecolor="#1a3d6d", linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10)

    box(4.0, 4.0, 2.0, 0.7, "BasePipeline", color="#a8c0d8")
    box(0.5, 2.5, 2.2, 0.7, "FragilityPipeline")
    box(3.0, 2.5, 2.2, 0.7, "GridModelPipeline")
    box(5.5, 2.5, 2.2, 0.7, "MCSPipeline")
    box(2.0, 0.8, 2.5, 0.7, "Sampler protocol", color="#e8d8b0")
    box(5.0, 0.8, 2.5, 0.7, "LSFEvaluator protocol", color="#e8d8b0")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#1a3d6d"))

    arrow(5.0, 4.0, 1.6, 3.2)
    arrow(5.0, 4.0, 4.1, 3.2)
    arrow(5.0, 4.0, 6.6, 3.2)
    arrow(6.0, 2.5, 4.0, 1.5)
    arrow(6.6, 2.5, 6.0, 1.5)
    ax.text(5.0, 4.85, "src/pipeline/", fontsize=9,
            color="#7a6a4d", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_method_comparison(path: Path) -> None:
    """Side-by-side cartoon of interpolate vs alphas method on the LSF."""
    cr = np.linspace(0, 1, 100)
    beta = 4.0 - 6.0 * cr ** 0.9
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # interpolate
    ax = axes[0]
    ax.plot(cr, beta, color="#1a3d6d", linewidth=1.8, label="fragility β(cr)")
    cr_samples = np.array([0.1, 0.25, 0.4, 0.55, 0.7])
    beta_samples = np.interp(cr_samples, cr, beta)
    ax.scatter(cr_samples, beta_samples, color="#d6604d", s=40,
               label="per-sample interp")
    for c, b in zip(cr_samples, beta_samples):
        ax.plot([c, c], [0, b], color="#d6604d", linestyle=":",
                linewidth=0.8)
    ax.set_xlabel("cr per sample")
    ax.set_ylabel(r"$\beta$")
    ax.set_title("interpolate — per-sample fragility lookup")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    # alphas
    ax = axes[1]
    ax.plot(cr, beta, color="#1a3d6d", linewidth=1.8, label="fragility β(cr)")
    cr_star = 0.35
    beta_star = float(np.interp(cr_star, cr, beta))
    # tangent line at cr_star
    slope = (np.interp(cr_star + 0.01, cr, beta) -
             np.interp(cr_star - 0.01, cr, beta)) / 0.02
    line = beta_star + slope * (cr - cr_star)
    ax.plot(cr, line, color="#d6604d", linewidth=1.4, linestyle="--",
            label="tangent at design point")
    ax.scatter([cr_star], [beta_star], color="#a83232", s=80, zorder=5,
               label=fr"design point cr*={cr_star:.2f}")
    ax.set_xlabel("cr")
    ax.set_ylabel(r"$\beta$")
    ax.set_title("alphas — tangent-hyperplane LSF")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(-1, 5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_mcs_convergence_cartoon(path: Path) -> None:
    """Cartoon MCS running-Pf + 95% CI for the system + components."""
    rng = np.random.default_rng(1)
    n = 500
    pf_true_wall = 0.06
    pf_true_anc  = 0.005
    fail_wall = rng.random(n) < pf_true_wall
    fail_anc  = rng.random(n) < pf_true_anc
    fail_sys  = fail_wall | fail_anc
    iters = np.arange(1, n + 1)

    def run_ci(fail):
        cum = np.cumsum(fail)
        pf = cum / iters
        half = 1.96 * np.sqrt(np.clip(pf * (1 - pf), 1e-15, None) / iters)
        return pf, np.clip(pf - half, 1e-6, 1), np.clip(pf + half, 1e-6, 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for fail, color, lab, lw in [
        (fail_sys,  "#222",    "system", 1.8),
        (fail_wall, "#4c8dde", "wall",   1.2),
        (fail_anc,  "#f0a000", "anchor", 1.2),
    ]:
        pf, lo, hi = run_ci(fail)
        ax.fill_between(iters, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(iters, pf, color=color, linewidth=lw, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("MC iteration")
    ax.set_ylabel("running Pf")
    ax.set_title("MCS convergence — system + per-component (95% CI)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_series_system(path: Path) -> None:
    """System (series) Pf vs per-section Pf along the wall."""
    x = np.linspace(0, 1000, 11)
    # per-section Pf roughly constant under the prior (homogeneous wall)
    pf_section = np.full_like(x, 1.8e-2)
    # series-system Pf if sections were independent: 1 - prod(1 - pf_i)
    pf_indep = 1.0 - np.prod(1.0 - pf_section)
    # fully correlated lower bound = max(pf_i)
    pf_corr = pf_section.max()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, pf_section, width=60, color="#4c8dde", alpha=0.7,
           label="per-section Pf")
    ax.axhline(pf_indep, color="#a83232", linewidth=1.8,
               label=f"series-system Pf (independent) = {pf_indep:.3f}")
    ax.axhline(pf_corr, color="#1a3d6d", linewidth=1.4, linestyle="--",
               label=f"fully-correlated bound = {pf_corr:.3f}")
    ax.set_xlabel("position along wall x [m]")
    ax.set_ylabel("Pf at fixed t")
    ax.set_title("Series-system inflation: any-section-fails vs single-section")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ======================================================================
# DOCX assembly helpers
# ======================================================================

def _set_doc_defaults(doc: Document) -> None:
    """Set base font + margins for the whole document."""
    for section in doc.sections:
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)


def para(doc: Document, text: str) -> None:
    """Add a justified body paragraph."""
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def bullet(doc: Document, text: str) -> None:
    doc.add_paragraph(style="List Bullet").add_run(text)


def add_code(doc: Document, text: str) -> None:
    """Add a monospaced code block (one paragraph, courier, 9pt)."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)


def add_figure(doc: Document, png: Path, caption: str,
               width: float = 6.2) -> None:
    """Insert a figure with a caption paragraph below it."""
    doc.add_picture(str(png), width=Inches(width))
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(10)


def add_table(doc: Document, rows: list[list[str]],
              header: bool = True) -> None:
    """Add a simple grid table from a list of rows (first row is header)."""
    tbl = doc.add_table(rows=len(rows), cols=len(rows[0]))
    tbl.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.rows[i].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(val)
            if header and i == 0:
                run.bold = True
            run.font.size = Pt(10)


# ======================================================================
# Build
# ======================================================================

def main() -> None:
    print(">> generating figures")
    fig_wall_schematic(FIG / "01_wall_schematic.png")
    fig_fragility(FIG / "02_fragility.png")
    fig_corrosion_model(FIG / "03_corrosion_model.png")
    fig_spatial_kernel(FIG / "04_spatial_kernel.png")
    fig_cr_field_along_wall(FIG / "05_cr_field.png")
    fig_nested_form(FIG / "06_nested_form.png")
    fig_pipeline_arch(FIG / "07_pipeline_arch.png")
    fig_method_comparison(FIG / "08_method_comparison.png")
    fig_mcs_convergence_cartoon(FIG / "09_mcs_convergence.png")
    fig_alpha_corrosion(FIG / "10_alpha_corrosion.png")
    fig_series_system(FIG / "11_series_system.png")

    print(">> assembling docx")
    doc = Document()
    _set_doc_defaults(doc)

    # ------------------------------------------------------------------
    # Title block
    # ------------------------------------------------------------------
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("ARK Main — Methodology Documentation")
    run.bold = True
    run.font.size = Pt(20)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run(
        "Time-dependent reliability of a 1 km anchored sheet-pile wall under "
        "corrosion, with Bayesian updating and spatial variability"
    ).italic = True
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(
        "Moonshot 5 — MS5.2 Smart & Resilient Infrastructure / Asset "
        "Performance.  Companion to the code under case_studies/ark_main/."
    ).font.size = Pt(9)
    doc.add_paragraph()

    # ==================================================================
    # 0. Reading guide
    # ==================================================================
    doc.add_heading("0. Reading guide", level=1)
    para(doc,
        "This document describes the methodology behind the two analysis "
        "entry points in the ARK case study and the modelling decisions that "
        "sit underneath them. It is organised in four parts. Part I "
        "(Sections 1–5) lays out the shared foundations: the structural "
        "problem, the limit-state functions, the random variables, the "
        "cached fragility curve, and the time-dependent corrosion model with "
        "its Bayesian updating. Part II (Section 6) documents run.py, the "
        "per-cross-section pipeline, including how the importance factor of "
        "corrosion is estimated for a single section. Part III "
        "(Sections 7–10) documents run_spatial.py, the whole-wall "
        "series-system pipeline, including the spatial covariance model, the "
        "Nataf+Kriging propagation of a single-location observation, and the "
        "two interchangeable Monte-Carlo strategies. Part IV (Sections 11–14) "
        "covers the supporting machinery — pipeline architecture, "
        "configuration, caching, and the catalogue of approximations.")
    para(doc,
        "Throughout, the corrosion ratio cr ∈ [0, 1] is the fraction of the "
        "original wall thickness that has been lost to corrosion. It is the "
        "single scalar that couples the time domain (how much corrosion has "
        "accumulated by time t) to the structural domain (how the section "
        "responds at that level of corrosion). Almost every modelling choice "
        "in this document is, at bottom, a choice about how to represent the "
        "uncertainty in cr and how it propagates.")

    # ==================================================================
    # PART I — FOUNDATIONS
    # ==================================================================
    doc.add_heading("Part I — Foundations", level=1)

    # ------------------------------------------------------------------
    doc.add_heading("1. Problem statement", level=1)
    para(doc,
        "The ARK case study analyses a 1 km anchored sheet-pile wall along a "
        "canal. The wall is discretised into N cross sections at uniform "
        "spacing (typically 11 sections at 100 m or 21 at 50 m). Each section "
        "is mechanically identical and shares one cached fragility curve — "
        "the wall is treated as spatially homogeneous in its mechanical "
        "properties, so spatial variability enters only through the random "
        "fields placed over the input variables, not through section-to-"
        "section differences in geometry.")
    add_figure(doc, FIG / "01_wall_schematic.png",
        "Figure 1.1 — Schematic of a single cross section: anchored sheet "
        "pile, soil layers (Klei / Zand / Zandvast / Zandlos), canal level, "
        "phreatic level, and the surcharge q on the active side.")
    para(doc,
        "Two component limit states are considered — the bending capacity of "
        "the sheet-pile wall and the anchor force — together with their "
        "series-system combination (the section is deemed to have failed if "
        "either component fails). Corrosion erodes the wall over time, "
        "reducing both its moment capacity and its flexural rigidity, so the "
        "reliability of every section decreases monotonically as cr grows.")
    para(doc, "Every run produces two analysis legs:")
    bullet(doc,
        "Prior leg — no observations. The failure probability is propagated "
        "forward in time using the prior corrosion-ratio PDF derived from the "
        "corrosion model alone.")
    bullet(doc,
        "Posterior leg — corrosion measurements have been taken. The "
        "corrosion-ratio PDF is updated by Bayesian inference; in the spatial "
        "pipeline the measurement is taken at a single location (x = 0) and "
        "Kriged to the rest of the wall through a spatial covariance kernel.")
    para(doc,
        "Keeping the prior and posterior legs methodologically identical — "
        "same integrator, same Monte-Carlo engine, same fragility — is a "
        "deliberate design constraint. It guarantees that any change in β "
        "between the two legs is attributable to the observations and not to "
        "an artefact of switching method.")

    # ------------------------------------------------------------------
    doc.add_heading("2. Limit-state functions", level=1)
    para(doc,
        "Three LSFs share the same FORM-variable signature and differ only in "
        "the failure quantity returned by D-SheetPiling. All are written in "
        "normalised form so that g > 0 is safe and g < 0 is failure:")
    add_code(doc,
        "g_wall   = M_capacity(cr) / (theta_M . |M_max|)     - 1\n"
        "g_anchor = F_yield(cr)     / (theta_F . |F_anchor|)  - 1\n"
        "g_system = min(g_wall, g_anchor)\n")
    para(doc,
        "The corrosion ratio cr ∈ [0, 1] enters mechanically by thinning the "
        "wall uniformly. Both the moment capacity and the flexural rigidity "
        "scale linearly with the remaining thickness:")
    add_code(doc,
        "factor      = 1 - cr\n"
        "M_capacity  = M_capacity_nominal . factor\n"
        "EI          = EI_nominal          . factor\n")
    para(doc,
        "theta_M and theta_F are model-uncertainty factors (lognormal, mean "
        "1) on the predicted demand. The anchor capacity is treated as "
        "independent of cr in the current model — corrosion is assumed to act "
        "on the wall section, not on the anchor rod itself. This is a "
        "modelling choice that could be revisited if anchor corrosion data "
        "warranted it.")
    para(doc,
        "Convergence sentinel. When D-SheetPiling fails to converge for a "
        "given draw — typically at high cr, where the geometric stiffness of "
        "the thinned wall collapses — the LSF returns a sentinel value of "
        "−99999 and the sample is counted as a failure. This is the "
        "conservative default; sentinels are not filtered out. In the MCS "
        "diagnostics they appear as a pile-up at the (−99999, −99999) corner "
        "of the g_wall vs g_anchor scatter, visually separable from the "
        "genuine near-zero failure cluster.")

    # ------------------------------------------------------------------
    doc.add_heading("3. Random variables and the joint PDF", level=1)
    para(doc,
        "The FORM variables, their marginal distributions, and their "
        "pairwise correlations in Nataf u-space are read from "
        "<remote>/input/settings.json. The joint PDF object "
        "(models/jpdf.py, inheriting from src/jpdf/jpdf.py) holds the "
        "marginals on a per-variable grid and provides an importance-sampling "
        "backend for sample-based work. The same JPDF instance is reused by "
        "every downstream module, so the distributional assumptions are "
        "defined in exactly one place:")
    bullet(doc, "run.py — per-section reliability via fragility integration.")
    bullet(doc, "analysis/run_mc.py — per-section MCS with D-SheetPiling in the loop.")
    bullet(doc,
        "io/export_cr_pdfs.py — prior and posterior corrosion-ratio PDFs as "
        "a function of time, persisted to cr_pdfs_<lsf>.json for the spatial "
        "pipeline to consume.")
    para(doc,
        "Soil vs non-soil classification. Variable names encode their type: "
        "soil variables follow the pattern <Layer>_soil<property> (e.g. "
        "Klei_soilphi, Zand_soilgamwet). Everything else — applied loads "
        "(uniform_load_left), water levels (phreatic_level, canal_level), "
        "model factors (model_factor_M/F) and wall stiffness "
        "(Wall_SheetPilingElementEI) — is non-soil. The distinction matters "
        "in the spatial pipeline: soil variables are given genuine random "
        "fields along the wall, whereas non-soil variables are treated as "
        "uniform — one scalar draw per Monte-Carlo sample, broadcast to every "
        "section (Section 8).")

    # ------------------------------------------------------------------
    doc.add_heading("4. Fragility curve (cached FORM)", level=1)
    para(doc,
        "The fragility curve is the bridge between the structural model and "
        "the reliability analysis. For each value of cr on a coarse grid, the "
        "FragilityCurveBuilder runs a single FORM analysis with cr held fixed "
        "as a deterministic parameter. The converged result for each cr-point "
        "is persisted to <remote>/output/fragility_curve_<lsf>/"
        "point_<idx>.json and contains:")
    bullet(doc, "the corrosion ratio cr at that point;")
    bullet(doc, "the failure probability Pf and reliability index β = −Φ⁻¹(Pf);")
    bullet(doc,
        "the unit-norm FORM direction cosines α_v for every basic FORM "
        "variable (the importance factors at that cr);")
    bullet(doc, "the design-point coordinates in physical space; and")
    bullet(doc, "a convergence flag.")
    para(doc,
        "This cache is the canonical β(cr) and α(cr) summary consumed by "
        "every reliability leg in the case study. Because each point is a "
        "one-shot FORM run at fixed cr, the cache embodies the assumption "
        "that, conditional on cr, the design-point direction in the basic "
        "variables is well-described by FORM. Section 6.3 and Section 10 "
        "explain how cr is subsequently promoted from a deterministic "
        "parameter back to a random variable.")
    add_figure(doc, FIG / "02_fragility.png",
        "Figure 4.1 — Synthetic β(cr) and Pf(cr) shapes typical of the ARK "
        "wall fragility cache. β is roughly affine in cr in the low-to-"
        "moderate range and saturates as the wall approaches yield. Black "
        "dots mark the cached design points actually computed.")
    para(doc,
        "Nuance — non-contiguous cache indices. The cache is hybrid: some "
        "cr-points come from full Monte-Carlo (cr = 0, 0.3, 0.5, 0.7), some "
        "from FORM (cr = 0.1, 0.2), and some from manual saturation values "
        "(cr = 0.9, 1.0). The on-disk file indices are therefore not "
        "contiguous, and the manifest's completed_indices list must reference "
        "the real file indices rather than positional order. A mismatch here "
        "silently drops cr-points (e.g. cr = 0.9) from the loaded curve — a "
        "subtle failure mode worth flagging when curating the cache by hand.")

    # ------------------------------------------------------------------
    doc.add_heading("5. Corrosion model and Bayesian updating", level=1)
    para(doc,
        "The corrosion model (models/corrosion.py) maps time to a "
        "corrosion-thickness distribution, which is then converted to the "
        "dimensionless corrosion ratio cr = C(t) / d0, where d0 is the "
        "original wall thickness. Two model forms are supported; the power "
        "law is the default for the 0–50 yr horizon:")
    add_code(doc,
        "power  (default, 0-50 yr):  C(t) = A . t^B\n"
        "linear (legacy, 50-75 yr):  C(t) = C50 . (1 + r/C50_mu . (t - t0))\n")
    para(doc,
        "In the power-law form A is a calibrated constant (typically fitted "
        "to NEN 6766:2023 tabulated values) and the exponent B is the single "
        "random parameter, given a truncated-normal prior on [B_min, B_max]. "
        "In the linear form the random parameter is C50. Both play the same "
        "role inside the pipeline: a one-dimensional quantity carried on a "
        "grid in the JPDF and updated by observations. This common param_* "
        "interface is what lets the two corrosion models share all of the "
        "downstream machinery.")
    para(doc,
        "Forward propagation. The corrosion-ratio PDF at any time t is the "
        "pushforward of the random parameter through the forward model. "
        "Conditional on the parameter, C(t) is given a truncated-normal "
        "scatter (mean from the forward model, standard deviation a fixed "
        "fraction of the mean, truncated to [0, d0]); marginalising over the "
        "parameter's current PDF and applying the change of variables "
        "c → c/d0 yields f_cr(cr; t). This is the distribution every "
        "reliability leg integrates the fragility against.")
    para(doc,
        "Bayesian updating. Observations are corrosion measurements at fixed "
        "times, stored in <remote>/input/data.json. The likelihood is built "
        "on the first observation plus the successive differences of the "
        "standardised residuals. This is the correct sequential weighting for "
        "measurements taken along a single timeline: because every "
        "measurement reflects the same realisation of the random parameter "
        "and differs only by i.i.d. observation noise, treating the raw "
        "residuals as independent would double-count the shared parameter "
        "uncertainty. Differencing removes that shared component. The "
        "posterior parameter PDF is then pushed forward exactly as the prior "
        "is, giving f_cr(cr; t | obs).")
    add_figure(doc, FIG / "03_corrosion_model.png",
        "Figure 5.1 — Prior median and 90% band of cr(t) (blue), and the "
        "posterior conditioned on a single observation at t = 20 yr (red). "
        "The posterior narrows around the observation time and propagates "
        "forward with reduced uncertainty.")
    para(doc,
        "Keeping this logic in the corrosion model + JPDF (rather than in "
        "any one pipeline) is what allows io/export_cr_pdfs.py to precompute "
        "the entire prior/posterior cr-PDF table once and hand it to the "
        "spatial pipeline as a file (Section 7). The per-section and spatial "
        "analyses therefore share a single source of truth for the corrosion "
        "distributions.")

    # ==================================================================
    # PART II — run.py
    # ==================================================================
    doc.add_heading("Part II — Per-section reliability (run.py)", level=1)

    doc.add_heading("6. Methodology of run.py", level=1)
    para(doc,
        "run.py analyses one cross section at a time using FragilityPipeline "
        "(src/pipeline/fragility.py). It produces β(t) for both the prior and "
        "the posterior leg by combining the cached fragility curve with the "
        "time-varying corrosion-ratio PDF. No structural solver runs inside "
        "run.py — all of the mechanics has already been condensed into the "
        "fragility cache — so the pipeline is fast and is the workhorse for "
        "the temporal analysis.")

    doc.add_heading("6.1 Fragility integration", level=2)
    para(doc,
        "The core computation is a one-dimensional integral over the "
        "corrosion ratio. At each forecast time t, the section failure "
        "probability is the fragility-weighted average of the failure "
        "probability over the corrosion-ratio distribution:")
    add_code(doc, "P_f(t) = integral  P_f(cr) . f_cr(cr | t)  dcr")
    para(doc,
        "P_f(cr) = Φ(−β(cr)) is interpolated from the cached fragility "
        "points; f_cr(·|t) comes from the corrosion model. The integral is "
        "evaluated by the trapezoidal rule on a common fine cr-grid that "
        "spans the fragility cache. Because Φ(−β(cr)) varies over many orders "
        "of magnitude across the cr range, the grid is deliberately fine "
        "(of order 10³ points) so the tail of the fragility is resolved "
        "where the corrosion PDF places most of its mass early in life.")

    doc.add_heading("6.2 The sequential-Bayesian time loop", level=2)
    para(doc,
        "FragilityPipeline inherits the generic time loop from BasePipeline "
        "(Section 11). The loop walks the observation dates in order; at each "
        "date it folds in every observation up to and including that date, "
        "re-derives the posterior corrosion parameter, and then forecasts the "
        "failure probability forward to every future grid time under both the "
        "prior and the accumulated-posterior corrosion PDFs:")
    add_code(doc,
        "for t_obs in obs_times:\n"
        "    obs_so_far = observations with date <= t_obs\n"
        "    do_update(jpdf, obs_so_far)          # posterior parameter PDF\n"
        "    for ft in forecast_times >= t_obs:\n"
        "        Pf_prior[ft]     = integral Pf(cr) f_prior(cr|ft) dcr\n"
        "        Pf_posterior[ft] = integral Pf(cr) f_post (cr|ft) dcr\n")
    para(doc,
        "The accumulating-posterior structure (condition on all observations "
        "≤ t_obs, not just the latest) means each successive obs scenario is "
        "strictly more informed than the previous one, which is the physically "
        "meaningful way to present a monitoring campaign.")

    doc.add_heading("6.3 Estimating the importance factor of corrosion", level=2)
    para(doc,
        "A recurring question for a single section is: how much of the "
        "section's unreliability is driven by corrosion uncertainty versus "
        "the structural and soil variables? The fragility cache cannot answer "
        "this directly, because in the cache cr is a deterministic parameter "
        "— the cached importance factors α_v(cr) describe only the basic "
        "variables, conditional on a fixed cr. To obtain an importance factor "
        "for corrosion as a random variable, cr must be promoted back to a "
        "random dimension and folded into the FORM geometry. This is the "
        "single most important nuance in the per-section methodology, and the "
        "same construction reappears in the spatial pipeline (Section 10.2).")
    para(doc,
        "Nataf transform of cr. At time t the corrosion ratio has CDF "
        "F_prior(cr; t). Define the standard-normal image")
    add_code(doc, "z = Phi^{-1}( F_prior(cr; t) )")
    para(doc,
        "so that z is N(0, 1) under the prior. The fragility supplies, for "
        "any cr, both the conditional reliability index β(cr) and the "
        "conditional unit-norm basic-variable directions α_v(cr). In the "
        "augmented standard-normal space spanned by the cr-axis coordinate ξ "
        "and the basic-variable vector u, the limit state is")
    add_code(doc, "g(xi, u) = beta(cr(xi)) - sum_v alpha_v(cr) . u_v")
    para(doc,
        "where cr(ξ) = F_prior⁻¹(Φ(ξ)). At a fixed ξ the FORM distance from "
        "the origin to the failure surface in the u-subspace is exactly "
        "β(cr(ξ)) (the cached conditional index), and the distance travelled "
        "along the cr-axis is ξ itself. Because the two subspaces are "
        "orthogonal, the squared distance to the joint design point is "
        "ξ² + β(cr(ξ))². The nested-FORM design point is the ξ that minimises "
        "it:")
    add_code(doc,
        "xi*    = argmin_xi  [ xi^2 + beta(cr(xi))^2 ]\n"
        "beta_T = sqrt( xi*^2 + beta(cr*)^2 )            # total index\n"
        "alpha_cr   = - xi* / beta_T                     # corrosion importance\n"
        "alpha_v    = ( beta(cr*) / beta_T ) . alpha_v(cr*)\n")
    para(doc,
        "The scaling factor β(cr*)/β_T applied to the cached basic-variable "
        "directions is exactly what is needed to keep the augmented direction "
        "vector unit-norm, so by construction")
    add_code(doc, "alpha_cr^2 + sum_v alpha_v^2 = 1")
    para(doc,
        "α_cr² is therefore the share of the total reliability budget "
        "attributable to corrosion, and the (scaled) α_v² are the shares "
        "attributable to each structural/soil variable. The 1-D minimisation "
        "is solved with a bounded scalar optimiser over ξ ∈ [−6, 6], which "
        "covers tail probabilities below 10⁻⁹ on either side and is more than "
        "adequate for this very smooth objective.")
    add_figure(doc, FIG / "06_nested_form.png",
        "Figure 6.1 — Nested-FORM search. Left: the fragility-derived β along "
        "the cr axis at one forecast time. Right: the total reliability index "
        "√(ξ² + β²) as a function of the cr-axis coordinate; its minimum "
        "locates the corrosion design point cr*.")
    add_figure(doc, FIG / "10_alpha_corrosion.png",
        "Figure 6.2 — Corrosion importance over the forecast horizon, "
        "computed by running the nested-FORM decomposition at each time. "
        "Left: the fragility index β(cr*) and the total index β_T. Right: the "
        "variance decomposition α_cr² (corrosion) plus the structural share, "
        "summing to one at every time.")
    para(doc,
        "Reading the decomposition. Early in life the corrosion PDF is tight "
        "and sits at low cr where the fragility is benign, so the design "
        "point barely moves along the cr-axis (ξ* small) and corrosion "
        "carries little of the importance. As the horizon lengthens the "
        "corrosion PDF both shifts to higher cr and widens, the design point "
        "is pulled along the cr-axis, and α_cr² grows — corrosion becomes the "
        "dominant contributor. Conditioning on observations (the posterior "
        "leg) tightens F(cr; t), which generally pulls α_cr² down relative to "
        "the prior at the same time, because the measurement has removed part "
        "of the corrosion uncertainty. Producing these prior-vs-posterior "
        "α-decompositions over time, both as per-variable line plots and as "
        "corrosion-vs-structural pie charts, is exactly what the analysis "
        "scripts alpha_lines_cross_section.py and alpha_pie_cross_section.py "
        "do, driven by the cr-PDF table that run.py exports.")
    para(doc,
        "Why this is the only consistent way to get α_cr. The fragility "
        "integration of Section 6.1 collapses the corrosion and structural "
        "uncertainty into a single β(t); it cannot separate the two by "
        "construction. The nested-FORM decomposition is the mechanism that "
        "reintroduces the cr dimension and recovers a defensible split, at "
        "the cost of one linearisation of the fragility surface at cr* (the "
        "tangent approximation). For the smooth, near-affine ARK fragility "
        "this linearisation is mild.")

    # ==================================================================
    # PART III — run_spatial.py
    # ==================================================================
    doc.add_heading("Part III — Whole-wall spatial reliability (run_spatial.py)",
                    level=1)

    doc.add_heading("7. Overview and inputs", level=1)
    para(doc,
        "run_spatial.py treats the 1 km wall as a series system of N sections "
        "and asks for the probability that any section fails. Spatial "
        "variability is the whole point: soil properties and the corrosion "
        "ratio vary from section to section, so the system failure event is "
        "no longer a function of a single scalar cr. The pipeline runs the "
        "same two legs (prior, posterior) but now both legs are Monte-Carlo "
        "over random fields along the wall.")
    para(doc,
        "Rather than re-deriving the corrosion distributions from the model, "
        "the spatial pipeline reads them from the precomputed file "
        "<remote>/output/cr_pdfs_<lsf>.json (produced by "
        "io/export_cr_pdfs.py). That file holds the common cr-grid, the "
        "forecast times, the prior PDF per time, and the posterior PDF per "
        "(observation time, forecast time) block. The path to this file is "
        "resolvable explicitly via the --cr-pdfs flag or a cr_pdfs_path "
        "setting, falling back to the conventional location next to the "
        "fragility cache. Using the exported file guarantees the temporal and "
        "spatial analyses share identical corrosion distributions.")

    doc.add_heading("8. Spatial covariance kernels", level=1)
    para(doc,
        "Each soil variable and the corrosion field are equipped with a "
        "squared-exponential covariance kernel over the section coordinate:")
    add_code(doc, "C(d) = rho_0 + (1 - rho_0) . exp[ - (d / theta)^2 ]")
    para(doc,
        "θ is the correlation length in metres; ρ₀ is a correlation floor "
        "that bounds the kernel from below at large separation. ρ₀ = 0 gives "
        "pure squared-exponential decay, so distant sections decorrelate "
        "completely; ρ₀ = 1 forces perfect correlation along the whole wall "
        "(the kernel matrix becomes all-ones and the variable behaves as a "
        "single scalar draw broadcast to every section).")
    add_figure(doc, FIG / "04_spatial_kernel.png",
        "Figure 8.1 — Squared-exponential kernel family for several (θ, ρ₀) "
        "combinations. θ sets the decay rate; ρ₀ sets the asymptotic floor.")
    para(doc,
        "The soil/non-soil classification of Section 3 maps directly onto the "
        "kernels. Soil variables receive genuine fields with their configured "
        "(θ, ρ₀). Non-soil variables (loads, water levels, model factors, "
        "wall stiffness) default to ρ₀ = 1, i.e. one scalar per Monte-Carlo "
        "sample broadcast to all sections — a deliberate simplification, on "
        "the grounds that those quantities act on the wall as a whole rather "
        "than varying section by section. The default can be overridden "
        "per variable if a non-soil quantity should in fact vary spatially.")
    para(doc,
        "Effective collapse for the basic field. For a FORM point with "
        "unit-norm directions α_v, the projected field Y(x) = Σ_v α_v U_v(x) "
        "is itself a zero-mean, unit-variance Gaussian field whose covariance "
        "is the α²-weighted sum of the per-variable kernels, C_eff = Σ_v α_v² "
        "C_v. A single Cholesky factor of C_eff then lets the inner loop draw "
        "Y directly, collapsing N_v per-variable matrix multiplies down to "
        "one — an important efficiency in the prior-leg engine.")
    add_table(doc, [
        ["Settings block", "Applies to", "Typical value"],
        ["wall.{theta, rho_0}", "soil variables without an explicit override",
         "θ = 200 m, ρ₀ = 0"],
        ["cr.{theta, rho_0}", "the corrosion-ratio field",
         "θ = 50 m, ρ₀ = 0"],
        ["per_variable.{var}", "explicit per-FORM-variable override",
         "e.g. model factors θ = 10 m, ρ₀ = 0.9"],
    ])

    doc.add_heading("9. Posterior corrosion field — Nataf + Kriging", level=1)
    para(doc,
        "In the posterior leg the observation is taken at the first section, "
        "x = 0. Propagating that single-location measurement to the rest of "
        "the wall is done by a Nataf transform of the corrosion field into a "
        "standard-normal z-field, Kriging in z-space, and a back-transform to "
        "cr. The reference marginal for the Nataf transform is the prior "
        "F_prior(cr; t): this keeps the z-field stationary with N(0, 1) "
        "marginals, which is what the squared-exponential covariance form "
        "assumes. The posterior enters only through the moments of z at the "
        "observation location, not as a re-anchoring of the marginal.")
    add_code(doc,
        "z(x) := Phi^{-1}( F_prior( cr(x) ; t ) )\n"
        "\n"
        "Cov_z(d) = rho_0_cr + (1 - rho_0_cr) . exp[ - (d/theta_cr)^2 ]\n"
        "\n"
        "(m_post, v_post) = mean, var of z(x=0) under the posterior cr-PDF\n"
        "      m_post = integral z(cr) f_post(cr;t) dcr\n"
        "      v_post = integral z(cr)^2 f_post(cr;t) dcr  -  m_post^2\n"
        "\n"
        "Kriged moments at section i  (rho_i = Cov_z(|x_i - x_0|)):\n"
        "  E[z_i | obs]      = rho_i . m_post\n"
        "  Cov[z_i,z_j| obs] = rho_i rho_j . v_post + (rho_ij - rho_i rho_j)\n"
        "\n"
        "Per MC sample:\n"
        "  z_field = E[z|obs] + L_z . Z      (L_z = chol of the cov above)\n"
        "  cr_i    = F_prior^{-1}( Phi(z_i) ; t )\n")
    para(doc,
        "The conditional-variance expression is worth reading carefully: the "
        "rho_i rho_j v_post term is the information transmitted from the "
        "observation (it shrinks as ρ_i → 0 far from x = 0), while the "
        "rho_ij − rho_i rho_j term is the residual prior covariance that "
        "survives conditioning. At a section far beyond the correlation "
        "length, ρ_i → 0, the conditional moments revert to (0, 1) and the "
        "Kriged field reproduces the prior exactly — the observation simply "
        "has no reach there. v_post is clipped to [1e-12, 1] because the "
        "posterior z-variance cannot exceed the prior marginal variance of 1.")
    add_figure(doc, FIG / "05_cr_field.png",
        "Figure 9.1 — Corrosion along the wall at one forecast time. The "
        "prior 90% band is constant in x (blue); the posterior band (red) "
        "collapses around the observation at x = 0 and fans back out to the "
        "prior at sections beyond the corrosion correlation length.")

    doc.add_heading("10. Spatial reliability — two Monte-Carlo methods", level=1)
    para(doc,
        "The spatial MCS engine offers two interchangeable strategies, "
        "selected by mcs_method in spatial_settings.json. Both are applied "
        "identically to the prior and posterior legs, so the only difference "
        "between the legs is whether the corrosion field is conditioned on "
        "observations. The two methods are kept side by side (they write to "
        "separate cache files) so the field-exact result and the linearised "
        "result can be cross-checked.")
    add_figure(doc, FIG / "08_method_comparison.png",
        "Figure 10.1 — Left: the interpolate method evaluates the cached "
        "fragility at each realised section cr_i. Right: the alphas method "
        "linearises the fragility at the nested-FORM design point cr* and "
        "evaluates a tangent-hyperplane limit state.")

    doc.add_heading("10.1 Method: interpolate (field-exact)", level=2)
    para(doc,
        "Per Monte-Carlo sample and per section: draw the local corrosion "
        "ratio cr_i from the (Kriged-conditional, or unconditional for the "
        "prior) corrosion field; interpolate the conditional reliability "
        "index β(cr_i) and the basic-variable directions α_v(cr_i) from the "
        "cached fragility; draw the basic-variable spatial fields once; and "
        "evaluate")
    add_code(doc, "g_i = beta(cr_i) - sum_v alpha_v(cr_i) . U_v(x_i)")
    para(doc,
        "Failure indicators are tallied per section (P(g_i < 0)) and as a "
        "system (P(any g_i < 0)). This method uses the fragility pointwise "
        "and is the natural reference. Its one approximation is that linearly "
        "interpolating each α-component in cr does not preserve unit norm, so "
        "|α(cr_i)| drifts slightly from 1 and the marginal per-section "
        "probability deviates from Φ(−β(cr_i)) by the same small factor "
        "(Section 13). An optional renormalisation is available but off by "
        "default.")

    doc.add_heading("10.2 Method: alphas (nested-FORM tangent hyperplane)", level=2)
    para(doc,
        "The alphas method applies the nested-FORM decomposition of "
        "Section 6.3 at every (section, time), promoting cr to a standard-"
        "normal variable and precomputing one linearised limit state per "
        "section. The 1-D search yields β_T, α_cr and the scaled α_v exactly "
        "as before, but now with section-dependent z-moments:")
    add_code(doc,
        "Prior leg     : mu_z = 0,            sigma_z = 1   (stationary)\n"
        "Posterior leg : mu_z = rho_i.m_post, sigma_z^2 = 1 + rho_i^2(v_post-1)\n")
    para(doc,
        "The standardised local corrosion coordinate is "
        "ξ_i = (z_i − μ_z,i)/σ_z,i, which is N(0, 1) at every section. The "
        "Monte-Carlo step then evaluates the tangent-hyperplane limit state "
        "with no per-sample fragility interpolation at all:")
    add_code(doc,
        "g_i = beta_T_i - alpha_cr_i . xi_i - sum_v alpha_v_i . U_v(x_i)")
    para(doc,
        "Because the design-point coefficients are precomputed per (section, "
        "time), the inner loop is a single linear combination per section — "
        "appreciably faster than the per-sample interpolation of the "
        "field-exact method. The trade-off is the extra linearisation of the "
        "fragility surface at cr*. A second, more subtle consequence: in the "
        "prior leg the alphas method samples a genuinely spatially-varying "
        "corrosion field, whereas the interpolate prior leg historically used "
        "a uniform-cr-per-sample assumption at each cr-point of the grid. The "
        "alphas-prior system Pf is therefore generally higher than the "
        "interpolate-prior Pf, because an uncorrelated corrosion field gives "
        "more independent opportunities for some section to fail. This is a "
        "real modelling difference, not numerical noise, and is the reason "
        "the prior leg was also moved onto the alphas machinery — so that "
        "prior and posterior are directly comparable.")
    para(doc,
        "A welcome by-product of the alphas method is that it exposes an "
        "explicit importance factor per (section, time) for every variable, "
        "corrosion included. These drive the alpha-heatmap and alpha-line "
        "plots: near x = 0 the corrosion α stays small (the observation pins "
        "cr), and it grows with distance as the Kriged field reverts to the "
        "prior and corrosion reclaims its share of the unit-norm budget.")
    add_figure(doc, FIG / "11_series_system.png",
        "Figure 10.2 — Series-system inflation. Each section carries a "
        "similar per-section Pf (homogeneous wall), but the probability that "
        "at least one of N sections fails is larger; how much larger depends "
        "on the spatial correlation of the fields. Independent sections give "
        "the upper estimate; perfect correlation collapses the system Pf back "
        "to the single-section value.")
    para(doc,
        "System reliability and correlation. The series-system probability "
        "sits between two bounds: if sections were perfectly correlated the "
        "system Pf equals the single-section Pf, and if independent it "
        "approaches 1 − Π(1 − Pf_i). The correlation lengths θ and the floors "
        "ρ₀ therefore directly control the system result — short correlation "
        "lengths inflate the system Pf, long ones suppress it. This coupling "
        "is precisely why the θ_wall × θ_cr sweep (run_spatial_sweep.py) "
        "exists.")

    # ==================================================================
    # PART IV — SUPPORTING MACHINERY
    # ==================================================================
    doc.add_heading("Part IV — Supporting machinery", level=1)

    doc.add_heading("11. Pipeline architecture", level=1)
    para(doc,
        "All reliability calculations are organised around BasePipeline "
        "(src/pipeline/base.py), an abstract class that owns the JPDF, the "
        "time arrays, and the generic sequential-Bayesian time loop shown in "
        "Section 6.2. Concrete subclasses supply only the Pf computation:")
    add_figure(doc, FIG / "07_pipeline_arch.png",
        "Figure 11.1 — Pipeline class hierarchy. BasePipeline supplies the "
        "time loop; subclasses plug in the Pf computation. MCSPipeline adds a "
        "Sampler + LSFEvaluator strategy pair so spatial and per-section MCS "
        "can share the same shell.")
    add_table(doc, [
        ["Pipeline", "Pf computation", "Used by"],
        ["FragilityPipeline",
         "1-D integration of cached Pf(cr) against the cr-PDF(t)", "run.py"],
        ["GridModelPipeline",
         "pre-evaluated model on a grid / IS samples",
         "legacy / sensitivity studies"],
        ["MCSPipeline (new)",
         "Sampler draws → LSFEvaluator → tally",
         "run_spatial.py and run_mc.py (migration target)"],
    ])
    para(doc,
        "MCSPipeline (src/pipeline/mcs.py) is a drop-in sibling under active "
        "development. Its design delegates the spatial-Bayesian propagation — "
        "observe at x = 0, Krige to all other sections — to the JPDF via an "
        "update_at_location interface, so the observation-propagation logic "
        "lives in one place independent of which sampler/evaluator pair is in "
        "use.")

    doc.add_heading("12. Configuration and workflows", level=1)
    para(doc, "All runs read JSON inputs from <remote>/input/:")
    add_table(doc, [
        ["File", "Contents"],
        ["settings.json",
         "FORM variable marginals, u-space correlations, corrosion-model "
         "parameters."],
        ["spatial_settings.json",
         "LSF name, sample count, seed, wall length L, n_sections, kernel "
         "blocks (wall, cr, per_variable), mcs_method, optional cr_pdfs_path."],
        ["data.json", "Corrosion observations {time, corrosion}."],
    ])
    para(doc, "Top-level entry points:")
    add_table(doc, [
        ["Script", "What it does"],
        ["run.py",
         "Per-section FragilityPipeline; β(t) per obs scenario plus the "
         "exported cr-PDF table and α-decomposition plots."],
        ["analysis/run_mc.py",
         "Per-section MCS at a chosen cr with D-SheetPiling in the loop; "
         "checkpointed; --postprocess-only re-renders plots and design "
         "points."],
        ["run_spatial.py",
         "Spatial pipeline; prior + posterior legs with the chosen MCS "
         "method; writes caches, forecasts, and the full plot bundle."],
        ["run_spatial_sweep.py",
         "Sweeps the wall and corrosion correlation lengths."],
        ["io/export_cr_pdfs.py",
         "Builds and persists the prior + posterior cr-PDF table consumed by "
         "the spatial pipeline."],
    ])
    para(doc,
        "Command-line flags override the corresponding settings-file field "
        "for the duration of a run; unspecified flags inherit from the file. "
        "run_spatial.py exposes --lsf-name, --n-samples, --seed, --L, "
        "--n-sections, --mcs-method, --wall-theta/--wall-rho0, "
        "--cr-theta/--cr-rho0, --cr-pdfs and --settings. Programmatic "
        "overrides (passing a dict to analyze) follow the same precedence, "
        "which is how the sweep driver injects each θ-combination.")

    doc.add_heading("13. Reproducibility, caching, approximations", level=1)
    para(doc,
        "Spatial outputs are keyed by a setup signature encoding the LSF, "
        "sample count, seed, L, n_sections, both kernels, and a hash of any "
        "per-variable overrides. Identical configurations land in the same "
        "signature folder, so reruns hit the caches and overwrite the results "
        "folder in place — there is no timestamping, by design, so that "
        "re-running an experiment reproduces rather than accumulates.")
    add_table(doc, [
        ["Cache file", "Method / leg", "Invalidates on"],
        ["pf_grid.json", "interpolate / prior",
         "LSF, N, seed, n_sections, L, wall kernel, fragility fingerprint"],
        ["posterior_grid.json", "interpolate / posterior",
         "+ cr kernel, obs_times"],
        ["pf_grid_alphas.json", "alphas / prior",
         "+ cr kernel (samples the cr-field directly), forecast_times"],
        ["posterior_grid_alphas.json", "alphas / posterior",
         "+ cr kernel, obs_times"],
    ])
    para(doc,
        "On Windows-mounted SMB shares the atomic-rename used to publish "
        "cache files occasionally fails with PermissionError (WinError 5); "
        "the checkpoint layer wraps the rename in an exponential-backoff "
        "retry. This is benign on POSIX, where the rename succeeds first try.")
    doc.add_heading("Catalogue of known approximations", level=2)
    bullet(doc,
        "α-interpolation (interpolate method) does not preserve unit norm: "
        "|α(cr_i)| drifts from 1, so the per-section P(g_i < 0) deviates "
        "slightly from Φ(−β(cr_i)). Optional renormalisation "
        "α ← α/|α|, β ← β/|α| is available but disabled by default.")
    bullet(doc,
        "Tangent linearisation (alphas method) replaces the fragility surface "
        "by its tangent at the nested-FORM design point cr*. Accurate for the "
        "smooth, near-affine ARK fragility; would degrade for a strongly "
        "curved fragility.")
    bullet(doc,
        "Nataf reference marginal is the prior, so the z-field stays "
        "stationary; the posterior enters only via (m_post, v_post) at the "
        "observation location, not as a re-anchored marginal.")
    bullet(doc,
        "The corrosion field and the basic-variable u-fields are sampled "
        "independently and couple only at the per-section fragility lookup, "
        "consistent with FORM's design-point construction (the cached α is "
        "already the design-point direction conditional on cr).")
    bullet(doc,
        "No cross-variable correlation within a section beyond what the "
        "fragility α already encodes; the settings' u-space correlation block "
        "is not yet propagated into the spatial fields.")
    bullet(doc,
        "A single fragility curve is shared by all sections (homogeneous "
        "wall). Per-section geometry would require per-section fragilities and "
        "a wider interpolation in the spatial engine.")
    bullet(doc,
        "The D-SheetPiling convergence sentinel (−99999) is counted as a "
        "failure (conservative) and is not filtered.")

    doc.add_heading("14. Verification and sanity checks", level=1)
    para(doc,
        "Several limiting cases give cheap regression checks on the spatial "
        "machinery:")
    bullet(doc,
        "θ_cr → ∞, ρ₀_cr = 1 makes every section perfectly correlated with "
        "x = 0; the posterior system Pf should converge to integrating the "
        "fragility against the global-observation posterior (the "
        "no-spatial-dimension baseline).")
    bullet(doc,
        "ρ₀_cr = 0, θ_cr → 0 makes distant sections independent of x = 0; the "
        "posterior Pf at far sections should match the prior leg at the same "
        "time, since the observation cannot reach them.")
    bullet(doc,
        "Sample-count scaling: moving from 10⁴ to 10⁶ samples should change "
        "results by O(1/√n); the observed ~3% shift at 10× matches the "
        "expected √10 noise reduction, confirming the estimator is unbiased "
        "and the seed handling is sound.")
    bullet(doc,
        "Prior-vs-posterior monotonicity: as more observations are folded in, "
        "the posterior system Pf at the end of the horizon should move "
        "monotonically (down for benign measurements, up for adverse ones), "
        "never jump erratically.")

    # ------------------------------------------------------------------
    doc.add_heading("Appendix A — Module map", level=1)
    add_table(doc, [
        ["Path", "Role"],
        ["run.py", "Per-section reliability driver (FragilityPipeline)"],
        ["run_spatial.py", "Spatial pipeline driver"],
        ["run_spatial_sweep.py", "Correlation-length sweep over the fields"],
        ["analysis/run_mc.py", "Direct per-section MCS with D-SheetPiling"],
        ["analysis/alpha_lines_cross_section.py",
         "Per-variable α(t) decomposition for run.py (prior + posterior)"],
        ["analysis/alpha_pie_cross_section.py",
         "Corrosion-vs-structural α² pie charts for run.py"],
        ["reliability/build_fragility.py",
         "LSF definitions and one-shot FORM fragility builder"],
        ["models/jpdf.py", "Joint PDF + corrosion-variable handling"],
        ["models/corrosion.py", "Power-law / linear corrosion model"],
        ["spatial/engine.py", "Interpolate prior MCS (effective Cholesky)"],
        ["spatial/posterior_mcs.py", "Interpolate posterior MCS (field-exact)"],
        ["spatial/nested_mcs.py", "Alphas prior + posterior MCS (nested FORM)"],
        ["spatial/cr_field.py", "Nataf + Kriging utilities for the cr field"],
        ["spatial/covariance.py",
         "Spatial kernel, Cholesky, config resolver"],
        ["spatial/checkpoint.py", "Cache load/save + SMB rename retry"],
        ["spatial/analyze_sweep.py", "Sweep aggregator + plots"],
        ["io/export_cr_pdfs.py", "Build + persist prior/posterior cr-PDF table"],
        ["src/pipeline/base.py", "BasePipeline (abstract time loop)"],
        ["src/pipeline/fragility.py", "FragilityPipeline"],
        ["src/pipeline/mcs.py", "MCSPipeline (new, partly wired)"],
        ["src/jpdf/jpdf.py", "Base JPDF (grid + IS backends)"],
    ])

    doc.add_heading("Appendix B — Symbol glossary", level=1)
    add_table(doc, [
        ["Symbol", "Meaning"],
        ["cr", "corrosion ratio = lost thickness / original thickness ∈ [0,1]"],
        ["β(cr)", "FORM reliability index from the fragility at fixed cr"],
        ["α_v(cr)", "FORM direction cosine (importance) of basic variable v"],
        ["z", "Nataf image of cr: z = Φ⁻¹(F_prior(cr; t)), N(0,1) under prior"],
        ["ξ", "standardised cr-axis coordinate in the nested-FORM search"],
        ["β_T", "total nested-FORM index, √(ξ*² + β(cr*)²)"],
        ["α_cr", "importance factor of corrosion, −ξ*/β_T"],
        ["θ, ρ₀", "correlation length and floor of a squared-exponential kernel"],
        ["m_post, v_post",
         "mean/variance of z at the obs location under the posterior"],
        ["ρ_i", "Cov_z(|x_i − x_0|), the cr correlation between section i and obs"],
    ])

    # ------------------------------------------------------------------
    print(f">> writing {DOCX_OUT}")
    doc.save(str(DOCX_OUT))

    # Best-effort copy to the synced Deltares folder.
    try:
        if DELTARES_OUT.parent.exists():
            shutil.copyfile(DOCX_OUT, DELTARES_OUT)
            print(f">> copied to {DELTARES_OUT}")
        else:
            print(f"!! Deltares folder not found, skipped: {DELTARES_OUT.parent}")
    except PermissionError:
        print("!! Deltares copy failed (file open in Word?). Local copy is "
              f"up to date at {DOCX_OUT}")
    print(">> done.")


if __name__ == "__main__":
    main()
