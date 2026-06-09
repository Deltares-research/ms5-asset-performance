"""Archive the FORM wall fragility curve and replace MCS-covered points.

The wall fragility curve used by the analysis lives at
    <remote>/output/fragility_curve_lsf_wall/
It is a FORM curve that breaks down for cr >= 0.3 (FORM converges to a bad
design point once corrosion bites). We have trustworthy MCS runs at
cr = 0.0, 0.3, 0.5, 0.7, so we splice those in.

Steps:
  1. Copy the whole curve folder to a sibling FORM archive (abort if the
     archive already exists -- never clobber a prior archive).
  2. For each point whose cr has an MCS wall run, overwrite pf / beta /
     logpf / method with the MCS values. For cr = 0.3 / 0.5 / 0.7 also swap
     in the MCS design point + alphas. For cr = 0.0 keep the FORM design
     point + alphas (the MCS direction is degenerate: only 1 failure ->
     all-zero alphas), replacing pf / beta only.
  3. The original FORM pf / beta are stashed in a ``_form_original`` block;
     points already carrying that block are skipped (safe to re-run).

Only READS the MCS outputs; WRITES the archive copy and the in-place curve.

Usage:
    python -m case_studies.ark_main.analysis.replace_wall_fragility_with_mcs
"""

import json
import shutil
from pathlib import Path

from src.io import get_remote_path

_ENV = Path(__file__).resolve().parents[1] / ".env"
_REMOTE = get_remote_path(_ENV)
_OUT = _REMOTE / "output"

CURVE_DIR = _OUT / "fragility_curve_lsf_wall"
ARCHIVE_DIR = _OUT / "fragility_curve_lsf_wall_FORM_archive"

# cr -> (design-point file, keep_form_direction)
#   keep_form_direction=True  -> replace pf/beta only, retain FORM dp+alphas.
MCS_SOURCES = {
    0.0: (_OUT / "mc_lsf_wall" / "cr_0.0000" / "design_point.json", True),
    0.3: (_OUT / "mc_lsf_wall_anchor" / "cr_0.3000" / "design_point_lsf_wall.json", False),
    0.5: (_OUT / "mc_lsf_wall_anchor" / "cr_0.5000" / "design_point_lsf_wall.json", False),
    0.7: (_OUT / "mc_lsf_wall_anchor" / "cr_0.7000" / "design_point_lsf_wall.json", False),
}
# cr=0.3 fallback: older run only wrote the combined design_point.json
# (system == wall there, both 128 failures).
FALLBACKS = {
    0.3: _OUT / "mc_lsf_wall_anchor" / "cr_0.3000" / "design_point.json",
}


def _resolve(cr):
    src, keep_form = MCS_SOURCES[cr]
    if not src.exists() and cr in FALLBACKS:
        src = FALLBACKS[cr]
    if not src.exists():
        raise FileNotFoundError(f"No MCS design point for cr={cr}: {src}")
    return src, keep_form


def main():
    # --- 1. archive ------------------------------------------------------
    if ARCHIVE_DIR.exists():
        raise SystemExit(
            f"Archive already exists, refusing to overwrite:\n  {ARCHIVE_DIR}\n"
            "Delete it manually if you really want to re-archive."
        )
    shutil.copytree(CURVE_DIR, ARCHIVE_DIR)
    print(f"Archived FORM curve ->\n  {ARCHIVE_DIR}\n")

    # --- 2. replace points ----------------------------------------------
    replaced = []
    for pf_path in sorted(CURVE_DIR.glob("point_*.json")):
        pt = json.load(open(pf_path))
        cr = round(float(pt["point"]["corrosion_rate"]), 2)
        if cr not in MCS_SOURCES:
            continue
        if "_form_original" in pt:
            print(f"  cr={cr:.2f}: already MCS-replaced, skipping ({pf_path.name})")
            continue

        src, keep_form = _resolve(cr)
        mcs = json.load(open(src))

        pt["_form_original"] = {
            "pf": pt["pf"], "beta": pt["beta"], "method": pt.get("method"),
        }
        pt["pf"] = mcs["pf"]
        pt["beta"] = mcs["beta"]
        pt["logpf"] = mcs.get("logpf")
        pt["convergence"] = True
        pt["method"] = "mcs"
        if not keep_form:
            pt["design_point"] = mcs.get("design_point", pt.get("design_point", {}))
            pt["alphas"] = mcs.get("alphas", pt.get("alphas", {}))
        pt["mcs_provenance"] = {
            "source": str(src),
            "n_samples": mcs.get("n_samples"),
            "n_failures": mcs.get("n_failures"),
            "seed": mcs.get("seed"),
            "direction": "form_retained" if keep_form else "mcs",
        }

        json.dump(pt, open(pf_path, "w"), indent=2)
        replaced.append((cr, pt["_form_original"]["beta"], pt["beta"],
                         "form_dir" if keep_form else "mcs_dir"))
        print(f"  cr={cr:.2f}: beta {pt['_form_original']['beta']:+.3f} -> "
              f"{pt['beta']:+.3f}  ({pf_path.name}, {replaced[-1][3]})")

    # --- 3. annotate manifest -------------------------------------------
    man_path = CURVE_DIR / "manifest.json"
    man = json.load(open(man_path))
    man["mcs_replaced_cr"] = sorted({c for c, *_ in replaced})
    man["note"] = ("Hybrid FORM/MCS curve: points at the listed cr were "
                   "replaced with direct-MCS pf/beta (see point '_form_original' "
                   "and 'mcs_provenance'). FORM archive: "
                   f"{ARCHIVE_DIR.name}")
    json.dump(man, open(man_path, "w"), indent=2)

    print(f"\nReplaced {len(replaced)} points. Manifest annotated.")
    print(f"  New hybrid curve : {CURVE_DIR}")
    print(f"  FORM archive     : {ARCHIVE_DIR}")


if __name__ == "__main__":
    main()
