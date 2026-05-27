"""
Aggregate a theta_cr x rho_0_cr sweep into two summary plots.

Reads each combo's ``forecasts/posterior.json`` and ``forecasts/prior.json``
from ``<remote>/output/spatial_analysis/results/<signature>/`` (written
earlier by ``run_spatial.py``), then renders:

* ``beta_forecast_grid.png`` — small multiples, one panel per ``theta_cr``.
  Each panel overlays the prior beta(t) curve (grey) with one posterior
  curve per ``rho_0_cr`` value (viridis gradient), same y-axis across
  panels.
* ``beta_contour_tend.png`` — bilinear-interpolated contour of system
  beta at the final forecast time (latest obs scenario) over the
  ``(theta_cr, rho_0_cr)`` plane. theta_cr axis log-scaled; rho_0_cr
  axis linear; sample points marked with white dots.

No MCS is invoked — purely read-only analysis on disk artefacts. If a
combo's folder is missing, the panel/cell shows ``"missing"`` and the
contour interpolates around the gap.

Usage::

    python -m case_studies.ark_main.spatial.analyze_sweep
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

_ARK = Path(__file__).resolve().parents[1]
if str(_ARK) not in sys.path:
    sys.path.insert(0, str(_ARK))

# python-dotenv hydrates DSHEETPILING_CONSOLE_PATH for downstream imports;
# even though this script doesn't touch geolib, ``src.io`` may.
from dotenv import load_dotenv
load_dotenv(_ARK / "geolib.env")

from src.io import get_remote_path
from src.plotting import save_figure

from spatial import results_dir as _results_dir


_ENV = _ARK / ".env"
_remote = get_remote_path(_ENV)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _load_base() -> dict:
    return json.load(open(_remote / "input" / "spatial_settings.json"))


def _override(base: dict, theta_cr: float, rho_0_cr: float) -> dict:
    cfg = json.loads(json.dumps(base))           # deep copy
    cr = cfg.setdefault("cr", {})
    cr["theta"] = float(theta_cr)
    cr["rho_0"] = float(rho_0_cr)
    return cfg


def _to_arr(beta_list) -> np.ndarray:
    """Convert a list[float | None] (None = inf from summary) to float array
    with ``nan`` in the inf slots so plotting can skip them.
    """
    return np.array([
        np.nan if b is None else float(b) for b in beta_list
    ], dtype=float)


def _load_combo_forecasts(rdir: Path) -> dict | None:
    """Read prior.json + posterior.json from a single combo's results dir."""
    prior_path = rdir / "forecasts" / "prior.json"
    post_path = rdir / "forecasts" / "posterior.json"
    if not prior_path.exists() or not post_path.exists():
        return None

    prior = json.load(open(prior_path))
    post = json.load(open(post_path))

    return {
        "prior_t":    np.array(prior["forecast_times"]),
        "prior_beta": _to_arr(prior["beta_system_t"]),
        "post_per_obs": {
            float(k): {
                "t":    np.array(v["forecast_times"]),
                "beta": _to_arr(v["beta_system_t"]),
                "pf":   np.array(v["pf_system_t"]),
            }
            for k, v in post["per_obs"].items()
        },
    }


# ----------------------------------------------------------------------
# Plot 1 — grid of beta-forecast curves
# ----------------------------------------------------------------------

def plot_beta_forecast_grid(
    theta_cr_values: Sequence[float],
    rho_0_cr_values: Sequence[float],
    base: dict,
    out_path: Path,
) -> None:
    """One subplot per ``theta_cr``, one curve per ``rho_0_cr``.

    Each curve is the posterior beta forecast from the **earliest** obs
    scenario (the obs with the longest forecast horizon — the latest obs
    in the current data has ``t_obs = t_end``, a single-point curve). The
    prior beta(t) is overlaid as a grey reference.
    """
    n_t = len(theta_cr_values)
    n_r = len(rho_0_cr_values)
    fig, axes = plt.subplots(
        1, n_t,
        figsize=(5.0 * n_t, 4.0),
        sharey=True,
        squeeze=False,
    )

    cmap_rho = plt.get_cmap("viridis")
    all_betas: list[float] = []

    for i, tc in enumerate(theta_cr_values):
        ax = axes[0, i]
        prior_drawn = False

        for j, rc in enumerate(rho_0_cr_values):
            cfg = _override(base, tc, rc)
            rdir = _results_dir(_remote, cfg)
            data = _load_combo_forecasts(rdir)
            if data is None:
                continue

            # Prior reference (once per subplot — it's identical across
            # rho_0_cr values; only the cr-field kernel changes it).
            if not prior_drawn:
                ax.plot(
                    data["prior_t"], data["prior_beta"],
                    color="#888", linewidth=1.4, label="prior",
                )
                mask = ~np.isnan(data["prior_beta"])
                all_betas.extend(data["prior_beta"][mask].tolist())
                prior_drawn = True

            # Posterior curve from the EARLIEST obs scenario (longest
            # non-degenerate forecast horizon).
            obs_keys = sorted(data["post_per_obs"].keys())
            if not obs_keys:
                continue
            first_block = data["post_per_obs"][obs_keys[0]]
            color = cmap_rho((j + 0.5) / max(n_r, 1))
            ax.plot(
                first_block["t"], first_block["beta"],
                "-", color=color, linewidth=1.6,
                label=fr"$\rho_{{0,cr}}$={rc:g}",
            )
            m = ~np.isnan(first_block["beta"])
            all_betas.extend(first_block["beta"][m].tolist())

        ax.set_xlabel("forecast time [yr]")
        ax.set_title(fr"$\theta_{{cr}}$ = {tc:.0f} m", fontsize=11)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel(r"$\beta$ system")
        ax.legend(loc="best", fontsize=9)

    if all_betas:
        ymin = min(all_betas) - 0.1
        ymax = max(all_betas) + 0.1
        for ax in axes.flat:
            ax.set_ylim(ymin, ymax)

    fig.suptitle(
        "Posterior beta forecast from the earliest obs scenario "
        "(one subplot per theta_cr, one colour per rho_0_cr)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, out_path)


# ----------------------------------------------------------------------
# Plot 2 — contour of beta at t_end vs (theta_cr, rho_0_cr)
# ----------------------------------------------------------------------

def plot_beta_contour(
    theta_cr_values: Sequence[float],
    rho_0_cr_values: Sequence[float],
    base: dict,
    out_path: Path,
) -> None:
    grid = np.full((len(theta_cr_values), len(rho_0_cr_values)), np.nan)
    for i, tc in enumerate(theta_cr_values):
        for j, rc in enumerate(rho_0_cr_values):
            cfg = _override(base, tc, rc)
            rdir = _results_dir(_remote, cfg)
            data = _load_combo_forecasts(rdir)
            if data is None:
                continue
            obs_keys = sorted(data["post_per_obs"].keys())
            if not obs_keys:
                continue
            last_block = data["post_per_obs"][obs_keys[-1]]
            beta_end = last_block["beta"][-1]
            if not np.isnan(beta_end):
                grid[i, j] = beta_end

    tc_arr = np.array(theta_cr_values, dtype=float)
    rc_arr = np.array(rho_0_cr_values, dtype=float)

    # meshgrid with indexing='xy': X varies along columns (theta_cr),
    # Y varies along rows (rho_0_cr). Z must be shape (n_rho, n_theta).
    X, Y = np.meshgrid(tc_arr, rc_arr, indexing="xy")
    Z = grid.T

    fig, ax = plt.subplots(figsize=(9, 6))

    levels = 12
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis_r")
    cs = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    # Mark the actual sample points.
    for tc in theta_cr_values:
        for rc in rho_0_cr_values:
            ax.scatter([tc], [rc], c="white", edgecolor="k",
                       s=40, zorder=5)

    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(r"posterior $\beta$ system at $t_{end}$", fontsize=11)

    ax.set_xscale("log")
    ax.set_xticks(tc_arr)
    ax.set_yticks(rc_arr)
    ax.set_xticklabels([f"{v:g}" for v in tc_arr])
    ax.set_yticklabels([f"{v:g}" for v in rc_arr])
    ax.set_xlabel(r"$\theta_{cr}$ [m]")
    ax.set_ylabel(r"$\rho_{0,cr}$")
    ax.set_title(
        r"Posterior $\beta$ at $t_{end}$ (latest obs scenario) — "
        r"bilinear interp between sample points (white)"
    )
    fig.tight_layout()
    save_figure(fig, out_path)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main(
    theta_cr_values: Sequence[float] = (50.0, 200.0, 800.0),
    rho_0_cr_values: Sequence[float] = (0.0, 0.3, 0.6),
) -> None:
    base = _load_base()
    out_dir = _remote / "output" / "spatial_analysis" / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_combos = len(theta_cr_values) * len(rho_0_cr_values)
    print(f"Reading {n_combos} combo result folders...")

    plot_beta_forecast_grid(
        theta_cr_values, rho_0_cr_values, base,
        out_path=out_dir / "beta_forecast_grid.png",
    )
    print(f"  wrote {out_dir / 'beta_forecast_grid.png'}")

    plot_beta_contour(
        theta_cr_values, rho_0_cr_values, base,
        out_path=out_dir / "beta_contour_tend.png",
    )
    print(f"  wrote {out_dir / 'beta_contour_tend.png'}")

    print(f"\nDone -- sweep analysis in {out_dir}")


if __name__ == "__main__":
    main()
