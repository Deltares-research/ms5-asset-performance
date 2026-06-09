"""Assemble an MCS-derived fragility curve from existing run_mc outputs.

Reads the per-cr ``design_point.json`` written by ``analysis/run_mc.py``
(postprocess step) for ``lsf_wall_anchor`` and writes them out as a standard
fragility-curve cache (``point_*.json`` + ``manifest.json``) so the rest of
the pipeline can consume them exactly like a FORM curve.

The pf/beta in each ``design_point.json`` is the **system** value
(min(g_wall, g_anchor) < 0), i.e. the (wall OR anchor) failure event.

This script only READS from ``mc_lsf_wall_anchor`` and WRITES to a brand-new
folder; it never touches existing fragility curves.

Usage:
    python -m case_studies.ark_main.analysis.build_mcs_fragility
"""

import json
from pathlib import Path
from src.io import get_remote_path

_ENV = Path(__file__).resolve().parents[1] / ".env"
_REMOTE = get_remote_path(_ENV)

LSF_NAME = "lsf_wall_anchor"
SRC_DIR = _REMOTE / "output" / f"mc_{LSF_NAME}"
OUT_DIR = _REMOTE / "output" / f"fragility_curve_{LSF_NAME}_mcs"

# cr folders produced by run_mc, in ascending cr order.
CR_FOLDERS = ["cr_0.3000", "cr_0.5000", "cr_0.7000"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = []
    completed = []
    points = []

    for index, folder in enumerate(CR_FOLDERS):
        # Older runs wrote a single design_point.json (= system); newer
        # postprocess runs split per component, where the system point is
        # design_point_<lsf>.json.
        candidates = [
            SRC_DIR / folder / "design_point.json",
            SRC_DIR / folder / f"design_point_{LSF_NAME}.json",
        ]
        src = next((c for c in candidates if c.exists()), None)
        if src is None:
            raise FileNotFoundError(
                f"No system design point in {SRC_DIR / folder} "
                f"(looked for {[c.name for c in candidates]})"
            )
        with open(src) as f:
            dp = json.load(f)

        cr = float(dp["point"]["corrosion_rate"])
        point = {
            "index": index,
            "point": {"corrosion_rate": cr},
            "pf": dp["pf"],
            "beta": dp["beta"],
            "logpf": dp.get("logpf"),
            "convergence": True,
            "method": "mcs",
            "design_point": dp.get("design_point", {}),
            "alphas": dp.get("alphas", {}),
            # MCS provenance (extra fields, ignored by FORM consumers)
            "n_samples": dp.get("n_samples"),
            "n_failures": dp.get("n_failures"),
            "seed": dp.get("seed"),
            "source": str(src),
        }
        out = OUT_DIR / f"point_{index:04d}.json"
        with open(out, "w") as f:
            json.dump(point, f, indent=2)

        grid.append(cr)
        completed.append(index)
        points.append(point)
        print(f"  point_{index:04d}: cr={cr:.3f}  pf={dp['pf']:.4e}  "
              f"beta={dp['beta']:.4f}  (n={dp.get('n_samples')}, "
              f"nfail={dp.get('n_failures')})")

    manifest = {
        "lsf_name": LSF_NAME,
        "method": "mcs",
        "deterministic_vars": ["corrosion_rate"],
        "grid": {"corrosion_rate": grid},
        "source_dir": str(SRC_DIR),
        "n_total": len(points),
        "n_completed": len(completed),
        "completed_indices": completed,
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(points)} points + manifest to:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
