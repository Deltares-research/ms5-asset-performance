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

# Low-cr end: there is no MCS run at cr = 0 (an MCS there sees ~zero
# failures, so beta is not estimable by sampling). Borrow the converged
# cr = 0 point from the existing FORM wall_anchor curve to anchor the low end.
FORM_SRC_DIR = _REMOTE / "output" / f"fragility_curve_{LSF_NAME}"
FORM_CR_VALUES = [0.0]

# High-cr end: there is no MCS run at cr = 1 either. Borrow the cr = 1
# saturation point from the lsf_wall curve (a manual Pf~1 / beta~-6 cap):
# at full corrosion the wall — and therefore the wall-OR-anchor system —
# fails with certainty, so the wall point is a valid system upper bound.
# This cr = 1 point is NOT required to be 'converged' (it is a manual cap).
# Set WALL_CR_VALUES = [] to skip the high-cr anchor.
WALL_SRC_DIR = _REMOTE / "output" / "fragility_curve_lsf_wall"
WALL_CR_VALUES = [1.0]


def _borrow_points(src_dir: Path, cr_values: list[float], method_tag: str,
                   require_convergence: bool) -> list[dict]:
    """Pull requested cr points out of another fragility cache.

    Matches by corrosion_rate (tight tolerance) across ALL point_*.json
    files in ``src_dir`` (so 5-digit names like point_00010.json that some
    manifests omit from completed_indices are still found). Returns dicts in
    the standard fragility-point schema, re-tagged with ``method_tag`` and a
    ``source`` provenance field.
    """
    if not cr_values or not src_dir.exists():
        return []
    by_cr: dict[float, dict] = {}
    for f in src_dir.glob("point_*.json"):
        p = json.load(open(f))
        if require_convergence and not p.get("convergence"):
            continue
        by_cr[round(float(p["point"]["corrosion_rate"]), 6)] = p
    out = []
    for cr in cr_values:
        match = by_cr.get(round(float(cr), 6))
        if match is None:
            raise FileNotFoundError(
                f"{src_dir} has no cr={cr} point "
                f"(require_convergence={require_convergence}; "
                f"available: {sorted(by_cr)})"
            )
        out.append({
            "point": {"corrosion_rate": float(match["point"]["corrosion_rate"])},
            "pf": match["pf"],
            "beta": match["beta"],
            "logpf": match.get("logpf"),
            "convergence": True,
            "method": method_tag,
            "design_point": match.get("design_point", {}),
            "alphas": match.get("alphas", {}),
            "source": str(src_dir),
        })
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect (cr, point-dict) entries from both sources, then sort by cr so
    # indices are monotone in corrosion ratio regardless of input order.
    entries: list[tuple[float, dict]] = []

    # --- borrowed end anchors: FORM cr=0 (low) + lsf_wall cr=1 (high) ---
    for bp in (_borrow_points(FORM_SRC_DIR, FORM_CR_VALUES, "form", True)
               + _borrow_points(WALL_SRC_DIR, WALL_CR_VALUES, "wall_proxy", False)):
        entries.append((float(bp["point"]["corrosion_rate"]), bp))

    # --- MCS points ---
    for folder in CR_FOLDERS:
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
        entries.append((cr, {
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
        }))

    entries.sort(key=lambda e: e[0])

    grid = []
    completed = []
    points = []
    for index, (cr, point) in enumerate(entries):
        point["index"] = index
        out = OUT_DIR / f"point_{index:04d}.json"
        with open(out, "w") as f:
            json.dump(point, f, indent=2)
        grid.append(cr)
        completed.append(index)
        points.append(point)
        extra = (f"  (n={point.get('n_samples')}, nfail={point.get('n_failures')})"
                 if point["method"] == "mcs" else "  (FORM)")
        print(f"  point_{index:04d}: cr={cr:.3f}  pf={point['pf']:.4e}  "
              f"beta={point['beta']:.4f}  [{point['method']}]{extra}")

    methods = sorted({p["method"] for p in points})
    manifest = {
        "lsf_name": LSF_NAME,
        "method": "+".join(methods),
        "deterministic_vars": ["corrosion_rate"],
        "grid": {"corrosion_rate": grid},
        "source_dir": str(SRC_DIR),
        "form_source_dir": str(FORM_SRC_DIR) if FORM_CR_VALUES else None,
        "wall_source_dir": str(WALL_SRC_DIR) if WALL_CR_VALUES else None,
        "n_total": len(points),
        "n_completed": len(completed),
        "completed_indices": completed,
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(points)} points + manifest to:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
