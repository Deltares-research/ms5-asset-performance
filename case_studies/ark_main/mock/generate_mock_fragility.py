"""Generate a realistic-looking mock fragility cache + cr_pdfs + data.json.

These fake inputs let the spatial pipeline run end-to-end on a machine that
doesn't have the D-SheetPiling install required to rebuild the real fragility
cache. Everything lands under ``mock/`` so the case-study ``.env`` (which
points ``REMOTE_PATH`` at this directory) picks it up automatically:

    mock/output/fragility_curve_lsf_wall/point_*.json   (fragility cache)
    mock/output/fragility_curve_lsf_wall/manifest.json  (cache manifest)
    mock/output/cr_pdfs_lsf_wall.json                   (prior + posterior cr PDFs)
    mock/input/data.json                                (synthetic obs)

Run:
    python -m case_studies.ark_main.mock.generate_mock_fragility
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as st


HERE = Path(__file__).resolve().parent
LSF_NAME = "lsf_wall"
WALL_THICKNESS_MM = 9.5

# ---------------------------------------------------------------------------
# 1) Fragility cache
# ---------------------------------------------------------------------------
# A smooth monotone-decreasing beta(cr): beta(0) ~ 4 (Pf ~ 3e-5), beta(1) ~ 0.3.
# alphas vary mildly with cr to mimic the design-point drift the real FORM
# search produces (wall-capacity factor dominates more as the wall corrodes).

with open(HERE / "input" / "settings.json") as f:
    _settings = json.load(f)
VAR_NAMES = [v["name"] for v in _settings["variables"]]

# Active-variable alpha reference (everything else stays at 0).
ALPHA_REF = {
    "Klei_soilphi":         -0.30,
    "Zand_soilphi":         -0.40,
    "Zand_soilgamwet":       0.10,
    "Zandvast_soilphi":     -0.20,
    "model_factor_M":        0.60,
    "model_factor_F":        0.10,
    "phreatic_level":        0.30,
    "canal_level":          -0.20,
    "uniform_load_left":     0.40,
}


def beta_of_cr(cr: float) -> float:
    """Smooth monotone-decreasing reliability index."""
    return float(4.0 - 4.0 * cr + 0.3 * cr ** 2)


def alphas_at_cr(cr: float) -> dict[str, float]:
    """Per-variable alpha, drifted slightly with cr and renormalised to unit length."""
    a = dict(ALPHA_REF)
    a["model_factor_M"] *= 1.0 + 0.2 * cr
    for k in ("Klei_soilphi", "Zand_soilphi", "Zandvast_soilphi"):
        a[k] *= 1.0 - 0.1 * cr
    full = {v: 0.0 for v in VAR_NAMES}
    full.update(a)
    arr = np.array([full[v] for v in VAR_NAMES], dtype=float)
    arr = arr / np.linalg.norm(arr)
    return {v: float(arr[i]) for i, v in enumerate(VAR_NAMES)}


def build_fragility_cache() -> None:
    cr_grid_frag = np.array([
        0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
    ])
    out_dir = HERE / "output" / f"fragility_curve_{LSF_NAME}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, cr in enumerate(cr_grid_frag):
        beta = beta_of_cr(float(cr))
        alphas = alphas_at_cr(float(cr))
        # Design point in u-space: u* = -beta * alpha (FORM convention).
        design_point = {v: float(-beta * alphas[v]) for v in VAR_NAMES}
        record = {
            "index": i,
            "point": {"corrosion_rate": float(cr)},
            "beta": beta,
            "pf": float(st.norm.cdf(-beta)),
            "alphas": alphas,
            "design_point": design_point,
            "method": "form",
            "convergence": True,
        }
        with open(out_dir / f"point_{i:04d}.json", "w") as f:
            json.dump(record, f, indent=2)

    # Manifest expected by export_cr_pdfs.py.
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(
            {
                "lsf_name": LSF_NAME,
                "n_points": int(len(cr_grid_frag)),
                "cr_values": cr_grid_frag.tolist(),
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(cr_grid_frag)} fragility points -> {out_dir}")


# ---------------------------------------------------------------------------
# 2) Synthetic observations (data.json)
# ---------------------------------------------------------------------------

OBS = {
    "obs_01": {"time": 10.0, "corrosion": 0.5},
    "obs_02": {"time": 20.0, "corrosion": 1.2},
    "obs_03": {"time": 30.0, "corrosion": 2.0},
}


def build_data_json() -> None:
    out = HERE / "input" / "data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(OBS, f, indent=2)
    print(f"Wrote {len(OBS)} observations -> {out}")


# ---------------------------------------------------------------------------
# 3) cr_pdfs_<lsf>.json
# ---------------------------------------------------------------------------
# Importance-sampling-style construction:
#   * Sample (A, B) from priors. Compute cr_mm(t) = A * t^B per sample.
#   * Prior cr-PDF at time t: KDE of cr_mm(t)/wall_thickness samples.
#   * Posterior given obs:  reweight samples by N(cr_mm(t_obs); c_obs, sigma_obs)
#     and KDE the resulting weighted samples.

OBS_ERROR_STD_MM = 0.2     # matches settings.json parameters.obs_error_std
N_PARAM_SAMP = 5_000
KDE_BW_REL = 0.06          # relative bandwidth (fraction of sample std)
N_CR_GRID = 1000
T_START, T_END = 0, 50


def _kde_on_grid(cr_samples: np.ndarray, weights: np.ndarray,
                 cr_grid: np.ndarray) -> np.ndarray:
    """Weighted Gaussian KDE evaluated on a fine cr_grid, normalised to 1."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return np.zeros_like(cr_grid)
    bw = max(0.005, KDE_BW_REL * cr_samples.std())
    diff = (cr_grid[:, None] - cr_samples[None, :]) / bw
    kernel = np.exp(-0.5 * diff ** 2) / (bw * np.sqrt(2.0 * np.pi))
    pdf = (kernel * w[None, :]).sum(axis=1) / w.sum()
    integral = float(np.trapezoid(pdf, cr_grid))
    if integral > 0:
        pdf /= integral
    return pdf


def build_cr_pdfs() -> None:
    cr_grid = np.linspace(0.0, 1.0, N_CR_GRID)
    obs_times = sorted(float(d["time"]) for d in OBS.values())
    obs_vals = [float(d["corrosion"]) for d in
                sorted(OBS.values(), key=lambda r: r["time"])]
    forecast_times = sorted(set(
        [float(t) for t in range(T_START, T_END + 1)] + obs_times
    ))

    # Prior (A, B) — power-law corrosion parameters.
    rng = np.random.default_rng(0)
    A_samp = rng.lognormal(mean=np.log(0.15), sigma=0.35, size=N_PARAM_SAMP)
    B_samp = np.clip(rng.normal(loc=0.70, scale=0.07, size=N_PARAM_SAMP),
                     0.3, 1.1)

    def cr_ratio_samples_at_t(t: float) -> np.ndarray:
        cr_mm = A_samp * (max(t, 1e-9) ** B_samp)
        return np.clip(cr_mm / WALL_THICKNESS_MM, 0.0, 1.0)

    # Prior cr-PDF per t (delta at t=0 gets a tight spike just left of 0).
    prior_pdf_per_t: dict[str, list[float]] = {}
    w_prior = np.ones(N_PARAM_SAMP)
    for t in forecast_times:
        if t <= 0.0:
            pdf = np.zeros_like(cr_grid)
            pdf[0] = 1.0
            pdf = pdf / np.trapezoid(pdf, cr_grid)
        else:
            pdf = _kde_on_grid(cr_ratio_samples_at_t(t), w_prior, cr_grid)
        prior_pdf_per_t[f"{t:.4f}"] = pdf.tolist()

    # Posterior cr-PDF per (obs scenario, t >= t_obs).
    posterior_pdf_per_obs: dict[str, dict[str, list[float]]] = {}
    log_lik = np.zeros(N_PARAM_SAMP)
    for t_obs, c_obs in zip(obs_times, obs_vals):
        cr_mm_at_obs = A_samp * (t_obs ** B_samp)
        log_lik += -0.5 * ((cr_mm_at_obs - c_obs) / OBS_ERROR_STD_MM) ** 2
        w = np.exp(log_lik - log_lik.max())
        block: dict[str, list[float]] = {}
        for t in forecast_times:
            if t < t_obs:
                continue
            pdf = _kde_on_grid(cr_ratio_samples_at_t(t), w, cr_grid)
            block[f"{t:.4f}"] = pdf.tolist()
        posterior_pdf_per_obs[f"{t_obs:.4f}"] = block

    payload = {
        "lsf_name": LSF_NAME,
        "model_type": "power",
        "wall_thickness": WALL_THICKNESS_MM,
        "cr_grid": cr_grid.tolist(),
        "forecast_times": forecast_times,
        "obs_times": obs_times,
        "prior_pdf_per_t": prior_pdf_per_t,
        "posterior_pdf_per_obs": posterior_pdf_per_obs,
    }
    out = HERE / "output" / f"cr_pdfs_{LSF_NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------

def main() -> None:
    build_fragility_cache()
    build_data_json()
    build_cr_pdfs()


if __name__ == "__main__":
    main()
