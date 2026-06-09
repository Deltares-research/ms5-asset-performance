"""Drop the broken FORM points from the hybrid wall fragility curve.

After splicing MCS points into <remote>/output/fragility_curve_lsf_wall/,
the FORM points that survive *between* the MCS points are still wrong
(FORM collapsed once corrosion bites), making the curve non-monotone:

    cr=0.30 beta=+1.61 (MCS)
    cr=0.40 beta=-0.75 (FORM, broken)
    cr=0.50 beta=+0.01 (MCS)
    cr=0.60 beta=-0.96 (FORM, broken)
    cr=0.70 beta=-2.21 (MCS)
    cr=0.80 beta=-1.64 (FORM, broken -- above the cr=0.7 point)

This removes the cr=0.4/0.6/0.8 point files and syncs the manifest grid.
They are already preserved in fragility_curve_lsf_wall_FORM_archive/.

``load_points`` reads whatever point_*.json exist and sorts by cr, so gaps
in the index sequence are fine.

Usage:
    python -m case_studies.ark_main.analysis.remove_broken_form_points
"""

import json
from pathlib import Path

from src.io import get_remote_path

_ENV = Path(__file__).resolve().parents[1] / ".env"
_REMOTE = get_remote_path(_ENV)
CURVE_DIR = _REMOTE / "output" / "fragility_curve_lsf_wall"

REMOVE_CR = {0.4, 0.6, 0.8}


def main():
    removed = []
    kept = []
    for pf_path in sorted(CURVE_DIR.glob("point_*.json")):
        pt = json.load(open(pf_path))
        cr = round(float(pt["point"]["corrosion_rate"]), 2)
        if cr in REMOVE_CR:
            if pt.get("method") == "mcs":
                raise SystemExit(f"Refusing to remove an MCS point: {pf_path.name}")
            pf_path.unlink()
            removed.append((cr, pt["beta"], pf_path.name))
            print(f"  removed cr={cr:.2f} beta={pt['beta']:+.3f} ({pf_path.name})")
        else:
            kept.append(cr)

    # --- sync manifest --------------------------------------------------
    man_path = CURVE_DIR / "manifest.json"
    man = json.load(open(man_path))
    kept_sorted = sorted(kept)
    man["grid"]["corrosion_rate"] = kept_sorted
    man["n_total"] = len(kept_sorted)
    man["n_completed"] = len(kept_sorted)
    man["completed_indices"] = list(range(len(kept_sorted)))
    man["removed_broken_form_cr"] = sorted(REMOVE_CR)
    json.dump(man, open(man_path, "w"), indent=2)

    print(f"\nRemoved {len(removed)} broken FORM points. "
          f"Curve now has {len(kept_sorted)} points:")
    print("  cr grid:", ", ".join(f"{c:g}" for c in kept_sorted))


if __name__ == "__main__":
    main()
