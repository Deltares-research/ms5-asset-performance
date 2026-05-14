"""Fragility-cache loading helpers for the spatial MCS.

Reads ``point_NNNN.json`` files produced by ``reliability/build_fragility.py``.
Each point carries ``beta``, ``pf``, the FORM ``design_point``, and a unit-norm
``alphas`` dict over the basic random variables — that is everything the
spatial MCS needs to evaluate ``g_i = beta - alpha . U(x_i)`` per section.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_points(remote: Path, lsf_name: str) -> list[dict]:
    """Read every converged fragility point for an LSF, sorted by cr."""
    cache_dir = remote / "output" / f"fragility_curve_{lsf_name}"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Fragility cache not found: {cache_dir}. Run build_fragility first."
        )
    pts: list[dict] = []
    for p in sorted(cache_dir.glob("point_*.json")):
        pt = json.load(open(p))
        if pt.get("convergence") and "alphas" in pt:
            pts.append(pt)
    pts.sort(key=lambda d: d["point"]["corrosion_rate"])
    return pts


def pick_cr_point(points: list[dict], cr_target: float) -> dict:
    """Return the cached fragility point whose cr is closest to ``cr_target``."""
    cr_vals = np.array([p["point"]["corrosion_rate"] for p in points])
    idx = int(np.argmin(np.abs(cr_vals - cr_target)))
    return points[idx]
