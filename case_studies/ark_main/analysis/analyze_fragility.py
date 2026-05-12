"""
Per-converged-fragility-point structural diagnostics.

For each fragility point with a converged FORM design point, re-runs the
D-SheetPiling model at that design point (with deterministic variables held
at their means) and extracts:

  - z, moment(z) along the wall (stage 0)
  - anchor force
  - max |moment| and the elevation at which it occurs

Generates three plot families:

  1. <fragility_internals_{lsf}>/moment_profiles/point_NNNN.png
        moment(z) with vertical lines at +/- M_capacity * (1 - cr)
  2. <fragility_internals_{lsf}>/demand_vs_cr.png
        two stacked subplots:
          a) |M|_max vs cr (left y) and |F_anchor| vs cr (right y)
          b) z at |M|_max vs cr

Internals are cached at <fragility_curve>/internals/point_NNNN.json so
re-plotting is fast (no D-Sheet runs needed on a second invocation).

Usage:
    python analyze_fragility.py
    python analyze_fragility.py --lsf-name lsf_wall_anchor --use-api
    python analyze_fragility.py --force-recompute
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.io import get_remote_path
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.reliability_models.dsheetpiling import build_payload, apply_payload


_ENV = Path(__file__).resolve().parents[1] / ".env"
_SECRETS_ENV = Path(__file__).resolve().parents[3] / "secrets" / ".env"
_DETERMINISTIC_TYPES = ("deterministic", "constant", "fixed")
# Pure LSF multipliers — not D-Sheet inputs.
_LSF_ONLY = ("model_factor_M", "model_factor_F")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_api_key():
    if _SECRETS_ENV.exists():
        load_dotenv(_SECRETS_ENV)
    return os.environ.get("DSHEET_API_KEY")


def load_settings():
    remote = get_remote_path(_ENV)
    with open(remote / "input" / "settings.json", "r") as f:
        return json.load(f)


def load_points(cache_dir: Path) -> list[dict]:
    pts = []
    for f in sorted(cache_dir.glob("point_*.json")):
        with open(f) as fh:
            pts.append(json.load(fh))
    pts.sort(key=lambda r: r["index"])
    return pts


def is_deterministic(v: dict) -> bool:
    return v.get("distribution_type", "normal").lower() in _DETERMINISTIC_TYPES


# ---------------------------------------------------------------------------
# D-Sheet at the FORM design point
# ---------------------------------------------------------------------------

def design_kwargs(point: dict, variables: list) -> dict:
    """FORM design values for stochastic vars + means for deterministic vars.

    The cached design_point has 0.0 for deterministic vars (PTK doesn't track
    them as stochastic dimensions), so we always overwrite those with their
    means from settings.json.
    """
    dp = point["design_point"]
    kwargs = {}
    for v in variables:
        name = v["name"]
        if is_deterministic(v):
            kwargs[name] = float(v["mean"])
        elif name in dp:
            kwargs[name] = float(dp[name])
        else:
            kwargs[name] = float(v["mean"])
    return kwargs


def compute_internals(base_model: DSheetPiling, point: dict, variables: list,
                      config: dict) -> dict:
    """Run D-Sheet at the design point and return its internal forces."""
    cr = float(point["point"]["corrosion_rate"])
    factor = 1.0 - cr
    EI_start = float(config.get(
        "EI_start",
        next((v["mean"] for v in variables if v["name"] == "Wall_SheetPilingElementEI"), 0.0),
    ))

    kwargs = design_kwargs(point, variables)
    # Apply EI degradation (mirrors lsf_wall).
    kwargs["Wall_SheetPilingElementEI"] = EI_start * factor
    # Drop pure-LSF multipliers — they don't enter D-Sheet.
    for k in _LSF_ONLY:
        kwargs.pop(k, None)

    model = deepcopy(base_model)
    payload = build_payload(kwargs, model)
    apply_payload(model, payload)
    model.execute()

    z = list(model.results.z)
    moment = model.results.moment[0]
    if isinstance(moment, np.ndarray):
        moment = moment.tolist()
    anchor_force = model.results.anchor_force[0]
    if isinstance(anchor_force, (list, np.ndarray)):
        anchor_force = float(np.asarray(anchor_force).item())

    moment_arr = np.asarray(moment, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    abs_max_idx = int(np.argmax(np.abs(moment_arr)))

    return {
        "index": int(point["index"]),
        "cr": cr,
        "z": z,
        "moment": [float(m) for m in moment_arr],
        "anchor_force": float(anchor_force),
        "max_moment": float(moment_arr[abs_max_idx]),
        "z_max_moment": float(z_arr[abs_max_idx]),
        "m_capacity": float(config["wall_moment_capacity"]) * factor,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_moment_along_height(internals: dict, out_path: Path, lsf_name: str) -> None:
    z = np.asarray(internals["z"])
    m = np.asarray(internals["moment"])
    cr = internals["cr"]
    cap = internals["m_capacity"]
    mfM = float(internals.get("model_factor_M", 1.0))
    m_eff = m * mfM

    fig, ax = plt.subplots(figsize=(5.5, 8))
    ax.plot(m, z, color="#2E86AB", linewidth=2,
            label=r"$M(z)$ (D-Sheet at design point)")
    ax.plot(m_eff, z, color="#1F4E79", linewidth=1.8, linestyle="--",
            label=fr"$\theta_M \cdot M(z)$ (eff. demand, $\theta_M$={mfM:.3f})")
    ax.axvline(cap, color="r", linestyle="--", linewidth=1.4,
               label=fr"$\pm M_{{cap}}(1-cr)$ = $\pm${cap:.0f} kNm/m")
    ax.axvline(-cap, color="r", linestyle="--", linewidth=1.4)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.6)
    ax.scatter(internals["max_moment"], internals["z_max_moment"],
               s=70, color="k", zorder=5,
               label=fr"$|M|_{{max}}$={abs(internals['max_moment']):.0f} @ z={internals['z_max_moment']:.2f} m")

    ax.set_xlabel("Moment [kNm/m]")
    ax.set_ylabel("Elevation z [m]")
    ax.set_title(f"{lsf_name} — point {internals['index']:04d}, cr = {cr:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_demand_vs_cr(internals_list: list[dict], out_path: Path, lsf_name: str,
                      anchor_capacity: float) -> None:
    """Stacked subplots: (a) |M|_max & F_anchor vs cr, (b) z(|M|_max) vs cr."""
    cr = np.array([d["cr"] for d in internals_list])
    max_m = np.array([abs(d["max_moment"]) for d in internals_list])
    mfM = np.array([d.get("model_factor_M", 1.0) for d in internals_list])
    max_m_eff = max_m * mfM
    cap_m = np.array([d["m_capacity"] for d in internals_list])
    f_a = np.array([d["anchor_force"] for d in internals_list])  # signed
    z_max = np.array([d["z_max_moment"] for d in internals_list])

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    # a) max |M| (left y) and F_anchor (right y) vs cr
    color_m = "#2E86AB"
    color_m_eff = "#1F4E79"
    color_f = "#A6342B"
    ax_top.plot(cr, max_m, "o-", color=color_m, linewidth=1.8, label=r"$|M|_{max}$ (raw demand)")
    ax_top.plot(cr, max_m_eff, "^--", color=color_m_eff, linewidth=1.8,
                label=r"$\theta_M \cdot |M|_{max}$ (eff. demand)")
    ax_top.plot(cr, cap_m, "--", color=color_m, alpha=0.55,
                label=r"$M_{cap}(1-cr)$ (capacity)")
    ax_top.set_ylabel("Max |moment| [kNm/m]", color=color_m)
    ax_top.tick_params(axis="y", labelcolor=color_m)
    ax_top.grid(True, alpha=0.3)

    ax_top2 = ax_top.twinx()
    ax_top2.plot(cr, f_a, "s-", color=color_f, linewidth=1.8, label=r"$F_{anchor}$ (demand)")
    ax_top2.axhline(anchor_capacity, color=color_f, linestyle=":", alpha=0.7,
                    label=fr"$F_{{yield}}$ = {anchor_capacity:.0f}")
    ax_top2.set_ylabel("Anchor force [kN/m]", color=color_f)
    ax_top2.tick_params(axis="y", labelcolor=color_f)

    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_top2.get_legend_handles_labels()
    ax_top.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    ax_top.set_title(f"Demand at FORM design point — {lsf_name}")

    # b) z at max |M| vs cr
    ax_bot.plot(cr, z_max, "o-", color=color_m, linewidth=1.8)
    ax_bot.set_xlabel("Corrosion ratio  $cr$  [-]")
    ax_bot.set_ylabel(r"z at $|M|_{max}$  [m]")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.set_title(r"Location of $|M|_{max}$ along the wall")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(lsf_name: str = "lsf_wall", use_api: bool = False,
         force_recompute: bool = False) -> None:
    settings = load_settings()
    config = settings["parameters"]
    variables = settings["variables"]
    remote = get_remote_path(_ENV)

    cache_dir = remote / "output" / f"fragility_curve_{lsf_name}"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"No cached fragility curve at {cache_dir}. Run build_fragility.py first."
        )

    points = load_points(cache_dir)
    converged = [p for p in points
                 if p.get("convergence") and p.get("method") == "form"
                 and p.get("design_point")]
    print(f"Converged FORM points: {len(converged)} / {len(points)} total")
    if not converged:
        raise RuntimeError("No converged FORM points to analyse.")

    api_key = load_api_key() if use_api else None
    base_model = DSheetPiling(str(remote / "input" / "model.shi"), api_key=api_key)
    print(f"Execution: {'API' if api_key else 'local'}")

    internals_dir = cache_dir / "internals"
    internals_dir.mkdir(parents=True, exist_ok=True)

    internals_list = []
    for p in converged:
        idx = p["index"]
        cache_path = internals_dir / f"point_{idx:04d}.json"
        if cache_path.exists() and not force_recompute:
            with open(cache_path) as fh:
                internals = json.load(fh)
            print(f"  [{idx:04d}] cr={internals['cr']:.3f}  cached")
        else:
            print(f"  [{idx:04d}] cr={p['point']['corrosion_rate']:.3f}  ...", end=" ", flush=True)
            internals = compute_internals(base_model, p, variables, config)
            with open(cache_path, "w") as fh:
                json.dump(internals, fh, indent=2)
            print(f"|M|_max={abs(internals['max_moment']):.1f}  "
                  f"|F|={abs(internals['anchor_force']):.1f}  "
                  f"z_M={internals['z_max_moment']:.2f}")
        # Pull design-point model factors from the source point (works for
        # both fresh and cached internals).
        dp = p.get("design_point", {})
        internals["model_factor_M"] = float(dp.get("model_factor_M", 1.0))
        internals["model_factor_F"] = float(dp.get("model_factor_F", 1.0))
        internals_list.append(internals)

    # Plots
    plots_dir = remote / "output" / f"fragility_internals_{lsf_name}"
    plots_dir.mkdir(parents=True, exist_ok=True)

    moment_dir = plots_dir / "moment_profiles"
    moment_dir.mkdir(parents=True, exist_ok=True)
    for internals in internals_list:
        out = moment_dir / f"moment_profile_{internals['index']:04d}.png"
        plot_moment_along_height(internals, out, lsf_name)
    print(f"\nMoment profiles -> {moment_dir}")

    summary_path = plots_dir / "demand_vs_cr.png"
    plot_demand_vs_cr(internals_list, summary_path, lsf_name,
                      float(config["anchor_capacity"]))
    print(f"Summary       -> {summary_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lsf-name", type=str, default="lsf_wall",
                        help="LSF name (selects fragility_curve_{lsf}/). Default: lsf_wall.")
    parser.add_argument("--use-api", action="store_true",
                        help="Use the D-SheetPiling compute API.")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-run D-Sheet even if internals are cached.")
    args = parser.parse_args()
    main(lsf_name=args.lsf_name, use_api=args.use_api,
         force_recompute=args.force_recompute)
