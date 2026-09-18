"""
Finalising figure for the theta_cr sweep: system reliability beta vs the
cr-field correlation length theta_cr.

The sweep (``run_spatial_sweep.py``) runs the spatial MCS once per theta_cr,
each writing its own ``results/<signature>/forecasts/prior.json``. This script
reads the **prior-leg system beta at the final forecast time** from each combo
and draws a single theta-beta curve with the two limiting cases as horizontal
reference lines:

  * ``theta_cr -> 0`` (here 0.01 m, << section spacing): the cr field is
    independent section-to-section -> maximum series-system effect -> lowest
    system beta. Drawn as a horizontal line ("no spatial dependency").
  * ``theta_cr -> inf`` (here 1e6 m, >> wall length): cr is perfectly
    correlated along the wall, so the 1 km wall behaves like a single
    cross-section -> highest system beta. Drawn as a horizontal line
    ("full spatial dependency").
  * the intermediate theta values form the connecting curve (log-x).

Run after the sweep finishes::

    python -m case_studies.ark_main.analysis.plot_theta_beta
    python -m case_studies.ark_main.analysis.plot_theta_beta --n-samples 5000000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ARK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ARK))

import numpy as np
import matplotlib.pyplot as plt

from src.io import get_remote_path
from src.plotting import collect_pngs_to_pdf, make_gifs

_ENV = _ARK / ".env"
_remote = get_remote_path(_ENV)

# theta_cr (m) -> role in the figure.
THETA_INDEP = 0.01          # ~0 limit: independent sections (no dependency)
THETA_FULL = 1_000_000.0    # ~inf limit: fully correlated (full dependency)
THETA_CURVE = [10.0, 100.0, 500.0, 1000.0]   # the connecting line


def _find_forecast_json(n_samples: int, theta_cr: float, name: str) -> Path | None:
    """Locate ``forecasts/<name>.json`` for one theta_cr combo.

    The signature encodes ``crth<int(theta_cr)>``; everything else (geometry,
    wall kernel, per_variable / grid hashes) is shared across the sweep, so we
    glob on the ``crth`` token and the sample count and take the unique hit.
    ``name`` is ``"prior"`` or ``"posterior"``.
    """
    base = _remote / "output" / "spatial_analysis" / "results"
    crth = int(theta_cr)
    pat = f"lsf-wall_N{n_samples}_*_crth{crth}_crrh0_*"
    hits = sorted(base.glob(pat))
    # Guard: int(10000)==int(1000) etc. never collide here, but int(0.01)==0
    # and a literal theta_cr=0 would share ``crth0`` — fine, we only sweep one.
    for d in hits:
        p = d / "forecasts" / f"{name}.json"
        if p.exists():
            return p
    return None


def _find_prior_json(n_samples: int, theta_cr: float) -> Path | None:
    return _find_forecast_json(n_samples, theta_cr, "prior")


def _load_beta_series(n_samples: int) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Read every swept theta's full prior system-beta time series.

    Returns ``(forecast_times, {theta_cr: beta_system_t})`` where each
    ``beta_system_t`` is a float array (``None`` -> NaN) over the shared
    ``forecast_times`` grid. Raises if any combo is missing.
    """
    wanted = [THETA_INDEP, *THETA_CURVE, THETA_FULL]
    series: dict[float, np.ndarray] = {}
    forecast_times = None
    missing = []
    for th in wanted:
        p = _find_prior_json(n_samples, th)
        if p is None:
            missing.append(th)
            continue
        d = json.load(open(p))
        ft = np.asarray(d["forecast_times"], dtype=float)
        if forecast_times is None:
            forecast_times = ft
        beta = np.array(
            [np.nan if b is None else float(b) for b in d["beta_system_t"]],
            dtype=float,
        )
        series[th] = beta
    if missing:
        raise FileNotFoundError(
            f"No prior.json for theta_cr = {missing} at N={n_samples}. "
            f"Has the sweep finished those combos?"
        )
    return forecast_times, series


def _draw_frame(
    ax, *, betas_at_t: dict[float, float], t: float, n_samples: int,
    ylim: tuple[float, float] | None = None,
    t_obs: float | None = None,
) -> None:
    """Draw one theta-beta frame on ``ax`` for forecast time ``t``.

    ``betas_at_t`` maps each swept theta_cr to its system beta at this ``t``
    (NaN allowed -> the point/line is simply dropped). ``t_obs=None`` labels
    the frame as the prior leg; a value labels it as the posterior leg
    conditioned on observations up to that obs time.
    """
    b_indep = betas_at_t[THETA_INDEP]
    b_full = betas_at_t[THETA_FULL]
    th_curve = sorted(THETA_CURVE)
    b_curve = [betas_at_t[th] for th in th_curve]

    ax.plot(
        th_curve, b_curve,
        marker="o", color="#1f77b4", lw=2.0, ms=7, zorder=3,
        label=r"$\beta_{\mathrm{sys}}(\theta_{cr})$",
    )
    if np.isfinite(b_indep):
        ax.axhline(
            b_indep, color="#d62728", ls="--", lw=1.8, zorder=2,
            label=rf"$\theta_{{cr}}\to 0$ (independent, no dependency): "
                  rf"$\beta={b_indep:.3f}$",
        )
    if np.isfinite(b_full):
        ax.axhline(
            b_full, color="#2ca02c", ls="-.", lw=1.8, zorder=2,
            label=rf"$\theta_{{cr}}\to\infty$ (fully correlated): "
                  rf"$\beta={b_full:.3f}$",
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"cr-field correlation length  $\theta_{cr}$  [m]")
    ax.set_ylabel(r"system reliability index  $\beta_{\mathrm{sys}}$")
    if t_obs is None:
        leg_label = "prior"
    else:
        leg_label = rf"posterior, $t_{{obs}}={t_obs:.1f}$"
    ax.set_title(
        rf"Wall system reliability vs cr correlation length "
        rf"({leg_label}, $t={t:.1f}$, N={n_samples:,})"
    )
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", framealpha=0.95)
    if ylim is not None:
        ax.set_ylim(*ylim)


def main(n_samples: int, out_path: Path) -> None:
    """Single PNG at the final forecast time."""
    forecast_times, series = _load_beta_series(n_samples)
    i_end = len(forecast_times) - 1
    betas_at_t = {th: float(b[i_end]) for th, b in series.items()}

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    all_b = [v for v in betas_at_t.values() if np.isfinite(v)]
    lo, hi = min(all_b), max(all_b)
    pad = 0.10 * max(hi - lo, 0.05)
    _draw_frame(ax, betas_at_t=betas_at_t, t=float(forecast_times[i_end]),
                n_samples=n_samples, ylim=(lo - pad, hi + pad))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    th_curve = sorted(THETA_CURVE)
    print(f"  theta_cr -> 0   (indep):   beta = {betas_at_t[THETA_INDEP]:.4f}")
    for th in th_curve:
        print(f"  theta_cr = {th:>8.0f} m:       beta = {betas_at_t[th]:.4f}")
    print(f"  theta_cr -> inf (full):    beta = {betas_at_t[THETA_FULL]:.4f}")


def main_over_time(n_samples: int, out_dir: Path) -> None:
    """One PNG per forecast time, then compile into a PDF + GIF.

    Frames share a single y-axis (computed from all finite betas across every
    theta and time) so the GIF reads as a smooth evolution rather than a
    rescaling jitter. ``out_dir`` holds the PNG frames; the PDF/GIF land
    alongside it (so ``make_gifs`` picks the frame dir up as a subdirectory).
    """
    forecast_times, series = _load_beta_series(n_samples)

    # Shared y-range across every frame.
    all_finite = np.concatenate([b[np.isfinite(b)] for b in series.values()])
    lo, hi = float(all_finite.min()), float(all_finite.max())
    pad = 0.10 * max(hi - lo, 0.05)
    ylim = (lo - pad, hi + pad)

    frames_dir = out_dir / "theta_beta_over_time"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(forecast_times):
        betas_at_t = {th: float(b[i]) for th, b in series.items()}
        # Skip degenerate frames where the curve has no finite point.
        if not any(np.isfinite(betas_at_t[th]) for th in THETA_CURVE):
            continue
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        _draw_frame(ax, betas_at_t=betas_at_t, t=float(t),
                    n_samples=n_samples, ylim=ylim)
        fig.tight_layout()
        fig.savefig(frames_dir / f"theta_beta_t{float(t):06.2f}.png", dpi=150)
        plt.close(fig)

    n_frames = len(list(frames_dir.glob("*.png")))
    print(f"Saved {n_frames} frames to {frames_dir}")

    collect_pngs_to_pdf(frames_dir, out_dir / "theta_beta_over_time.pdf")
    print(f"Saved {out_dir / 'theta_beta_over_time.pdf'}")
    # make_gifs scans subdirectories of out_dir and emits <subdir>.gif.
    make_gifs(out_dir)


def _load_posterior_beta_series(
    n_samples: int,
) -> tuple[list[float], dict[float, tuple[np.ndarray, dict[float, np.ndarray]]]]:
    """Read every swept theta's posterior system-beta series, per obs time.

    Returns ``(obs_times, {t_obs: (forecast_times, {theta_cr: beta_system_t})})``.
    Each block's ``forecast_times`` is shared across thetas (set by the cr-PDF
    file, not by theta_cr); ``beta_system_t`` has ``None`` -> NaN. Raises if any
    combo is missing.
    """
    wanted = [THETA_INDEP, *THETA_CURVE, THETA_FULL]
    # per_obs[t_obs] = {"ft": array, "betas": {theta: array}}
    per_obs: dict[float, dict] = {}
    obs_times: list[float] = []
    missing = []
    for th in wanted:
        p = _find_forecast_json(n_samples, th, "posterior")
        if p is None:
            missing.append(th)
            continue
        d = json.load(open(p))
        if not obs_times:
            obs_times = sorted(float(k) for k in d["per_obs"].keys())
        for t_obs_str, block in d["per_obs"].items():
            t_obs = float(t_obs_str)
            ft = np.asarray(block["forecast_times"], dtype=float)
            beta = np.array(
                [np.nan if b is None else float(b)
                 for b in block["beta_system_t"]],
                dtype=float,
            )
            slot = per_obs.setdefault(t_obs, {"ft": ft, "betas": {}})
            slot["betas"][th] = beta
    if missing:
        raise FileNotFoundError(
            f"No posterior.json for theta_cr = {missing} at N={n_samples}. "
            f"Has the sweep finished those combos?"
        )
    out = {t_obs: (slot["ft"], slot["betas"]) for t_obs, slot in per_obs.items()}
    return obs_times, out


def main_posterior_over_time(n_samples: int, out_dir: Path) -> None:
    """Posterior leg: a frame-folder + PDF + GIF per observation time.

    Layout under ``out_dir/theta_beta_posterior/``::

        tobs_006.25/                 # PNG per forecast time t >= t_obs
        tobs_006.25.pdf
        tobs_006.25.gif
        ... one triple per obs scenario

    Every frame across every obs scenario shares ONE y-axis (global finite-beta
    range) so the obs scenarios are directly comparable side by side.
    """
    obs_times, series = _load_posterior_beta_series(n_samples)

    # Shared y-range across every frame of every obs scenario.
    pool = []
    for _ft, betas in series.values():
        for b in betas.values():
            pool.append(b[np.isfinite(b)])
    all_finite = np.concatenate([p for p in pool if p.size])
    lo, hi = float(all_finite.min()), float(all_finite.max())
    pad = 0.10 * max(hi - lo, 0.05)
    ylim = (lo - pad, hi + pad)

    parent = out_dir / "theta_beta_posterior"
    parent.mkdir(parents=True, exist_ok=True)
    for t_obs in obs_times:
        ft, betas = series[t_obs]
        frames_dir = parent / f"tobs_{t_obs:06.2f}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(ft):
            betas_at_t = {th: float(b[i]) for th, b in betas.items()}
            if not any(np.isfinite(betas_at_t[th]) for th in THETA_CURVE):
                continue
            fig, ax = plt.subplots(figsize=(8.0, 5.2))
            _draw_frame(ax, betas_at_t=betas_at_t, t=float(t),
                        n_samples=n_samples, ylim=ylim, t_obs=t_obs)
            fig.tight_layout()
            # Frame name leads with t so collect/make_gifs sort within the
            # folder by forecast time (one t_obs per folder).
            fig.savefig(frames_dir / f"theta_beta_t{float(t):06.2f}.png", dpi=150)
            plt.close(fig)
        n_frames = len(list(frames_dir.glob("*.png")))
        collect_pngs_to_pdf(frames_dir, parent / f"tobs_{t_obs:06.2f}.pdf")
        print(f"  t_obs = {t_obs:>5.1f}: {n_frames} frames -> "
              f"{frames_dir.name}/ + .pdf")
    # One GIF per tobs subdirectory (single-frame scenarios are skipped).
    make_gifs(parent)
    print(f"Saved posterior theta-beta frames/PDFs/GIFs under {parent}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=5_000_000,
                   help="Sample count of the sweep to read (default 5e6)")
    p.add_argument("--over-time", action="store_true",
                   help="Prior leg: one frame per forecast time + PDF + GIF "
                        "(instead of a single t_end PNG)")
    p.add_argument("--posterior", action="store_true",
                   help="Posterior leg: a frame-folder + PDF + GIF per obs time")
    p.add_argument("--out", type=Path, default=None,
                   help="Single-frame mode: output PNG. Over-time/posterior "
                        "modes: output directory. Defaults under <remote>/output/"
                        "spatial_analysis/sweep/")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    sweep_dir = _remote / "output" / "spatial_analysis" / "sweep"
    if args.posterior:
        main_posterior_over_time(args.n_samples, args.out or sweep_dir)
    elif args.over_time:
        main_over_time(args.n_samples, args.out or sweep_dir)
    else:
        main(args.n_samples, args.out or (sweep_dir / "theta_beta.png"))
