"""Plot wall-failure fragility: FORM curve vs MCS points.

FORM curve: <remote>/output/fragility_curve_lsf_wall/point_*.json
MCS points: per_component "wall" block of
            <remote>/output/mc_lsf_wall_anchor/cr_*/summary.json

Writes a two-panel PNG (beta vs cr, Pf vs cr) to
<remote>/output/fragility_plots_lsf_wall/form_vs_mcs.png

Usage:
    python -m case_studies.ark_main.analysis.plot_fragility_form_vs_mcs
"""

import json
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.io import get_remote_path

_ENV = Path(__file__).resolve().parents[1] / ".env"
_REMOTE = get_remote_path(_ENV)

FORM_DIR = _REMOTE / "output" / "fragility_curve_lsf_wall"
# (parent MCS dir, cr folder) — cr=0 was run separately under mc_lsf_wall.
MC_SOURCES = [
    (_REMOTE / "output" / "mc_lsf_wall", "cr_0.0000"),
    (_REMOTE / "output" / "mc_lsf_wall_anchor", "cr_0.3000"),
    (_REMOTE / "output" / "mc_lsf_wall_anchor", "cr_0.5000"),
    (_REMOTE / "output" / "mc_lsf_wall_anchor", "cr_0.7000"),
]
OUT = _REMOTE / "output" / "fragility_plots_lsf_wall" / "form_vs_mcs.png"

# Clip beta for plotting (manual cr=0.9/1.0 points sit at beta=-6).
BETA_CLIP = 6.0


def load_form():
    pts = []
    for f in sorted(glob.glob(str(FORM_DIR / "point_*.json"))):
        d = json.load(open(f))
        pts.append((d["point"]["corrosion_rate"], d["beta"], d["pf"],
                    d.get("method", "form")))
    pts.sort()
    cr = np.array([p[0] for p in pts])
    beta = np.array([p[1] for p in pts])
    pf = np.array([p[2] for p in pts])
    method = [p[3] for p in pts]
    return cr, beta, pf, method


def load_mcs_wall():
    rows = []
    for parent, folder in MC_SOURCES:
        d = json.load(open(parent / folder / "summary.json"))
        w = d["per_component"]["wall"]
        rows.append((d["corrosion_rate"], w["beta"], w["pf"],
                     w["n_failures"], d["n_samples"]))
    rows.sort()
    cr = np.array([r[0] for r in rows])
    beta = np.array([r[1] for r in rows])
    pf = np.array([r[2] for r in rows])
    nfail = np.array([r[3] for r in rows])
    nsamp = np.array([r[4] for r in rows])
    return cr, beta, pf, nfail, nsamp


def main():
    cr_f, beta_f, pf_f, method_f = load_form()
    cr_m, beta_m, pf_m, nfail_m, nsamp_m = load_mcs_wall()

    beta_f_plot = np.clip(beta_f, -BETA_CLIP, BETA_CLIP)

    fig, (ax_b, ax_p) = plt.subplots(1, 2, figsize=(12, 5))

    # --- beta vs cr ---
    ax_b.plot(cr_f, beta_f_plot, "-o", color="tab:blue", lw=1.8, ms=5,
              label="FORM", zorder=2)
    ax_b.plot(cr_m, beta_m, "-x", color="tab:red", ms=10, mew=2.2, lw=1.8,
              label="MCS", zorder=3)
    ax_b.axhline(0.0, color="k", lw=0.8, ls=":")
    ax_b.set_xlabel("corrosion rate  cr")
    ax_b.set_ylabel(r"reliability index  $\beta$")
    ax_b.set_title("Wall fragility: reliability index")
    ax_b.legend()
    ax_b.grid(alpha=0.3)

    # --- Pf vs cr (log) ---
    ax_p.semilogy(cr_f, pf_f, "-o", color="tab:blue", lw=1.8, ms=5,
                  label="FORM", zorder=2)
    ax_p.semilogy(cr_m, pf_m, "-x", color="tab:red", ms=10, mew=2.2, lw=1.8,
                  label="MCS", zorder=3)
    ax_p.set_xlabel("corrosion rate  cr")
    ax_p.set_ylabel(r"failure probability  $P_f$")
    ax_p.set_title("Wall fragility: failure probability")
    ax_p.legend()
    ax_p.grid(alpha=0.3, which="both")

    fig.suptitle("Wall-failure fragility curve — FORM vs MCS",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Wrote {OUT}")

    # Console comparison
    print(f"\n{'cr':>5} {'FORM beta':>10} {'MCS beta':>10} "
          f"{'FORM Pf':>12} {'MCS Pf':>12}")
    fmap = {round(c, 3): (b, p) for c, b, p in zip(cr_f, beta_f, pf_f)}
    for c, bm, pm in zip(cr_m, beta_m, pf_m):
        bf, pf_ = fmap.get(round(c, 3), (float("nan"), float("nan")))
        print(f"{c:>5.2f} {bf:>10.3f} {bm:>10.3f} {pf_:>12.3e} {pm:>12.3e}")


if __name__ == "__main__":
    main()
