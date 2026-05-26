"""
Persistence for the spatial-MCS Pf grid.

One JSON file per spatial-MCS run, holding the per-cr per-section failure
counts and enough metadata to detect a stale cache::

    {
      "lsf_name":         "lsf_wall",
      "n_samples":        100000,
      "seed":             42,
      "n_sections":       21,
      "L":                1000.0,
      "wall_theta":    200.0,
      "wall_rho_0":    0.3,
      "cr_values":        [0.0, 0.025, ...],
      "betas":            [3.42, 3.31, ...],
      "n_fail_section":   [[..21 ints..], ..n_cr rows..],
      "n_fail_system":    [..n_cr ints..]
    }

The cache is keyed by every field except ``n_fail_*`` — if any of them
differs, the cached file is treated as stale and recomputed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def path(out_dir: Path, method: str = "interpolate") -> Path:
    """File name for the prior-leg cache.

    ``method="interpolate"`` -> the classic ``pf_grid.json`` (one Pf per
    cr-point from :func:`spatial.engine.compute_or_load_pf_grid`).
    ``method="alphas"`` -> ``pf_grid_alphas.json`` (per-t fail counts from
    the nested-FORM unconditional MCS). The two coexist in the same
    signature folder.
    """
    if method == "interpolate":
        return out_dir / "pf_grid.json"
    return out_dir / f"pf_grid_{method}.json"


def _arrays_close(a: list, b: list, tol: float = 1e-9) -> bool:
    if len(a) != len(b):
        return False
    return all(math.isclose(float(x), float(y), abs_tol=tol) for x, y in zip(a, b))


def try_load(
    out_dir: Path,
    *,
    lsf_name: str,
    n_samples: int,
    seed: int,
    n_sections: int,
    L: float,
    wall_theta: float,
    wall_rho_0: float,
    cr_values: np.ndarray,
    betas: np.ndarray,
) -> dict | None:
    """Return cached data if every key matches, otherwise ``None``.

    A mismatch on the fragility-cache fingerprint (``cr_values`` or ``betas``)
    typically means the fragility curve has been rebuilt since this spatial
    run was cached — we play it safe and force a recompute.
    """
    p = path(out_dir)
    if not p.exists():
        return None
    try:
        data = json.load(open(p))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  Cached Pf grid {p} unreadable ({exc!r}) — recomputing.")
        return None

    same = (
        data.get("lsf_name") == lsf_name
        and int(data.get("n_samples", -1)) == int(n_samples)
        and int(data.get("seed", -1)) == int(seed)
        and int(data.get("n_sections", -1)) == int(n_sections)
        and math.isclose(float(data.get("L", -1)), float(L), abs_tol=1e-9)
        and math.isclose(float(data.get("wall_theta", -1)), float(wall_theta), abs_tol=1e-9)
        and math.isclose(float(data.get("wall_rho_0", -1)), float(wall_rho_0), abs_tol=1e-9)
        and _arrays_close(data.get("cr_values", []), list(cr_values))
        and _arrays_close(data.get("betas", []), list(betas))
    )
    if not same:
        print(f"  Cached Pf grid {p} has a different run config — recomputing.")
        return None
    print(f"  Loaded cached Pf grid from {p}.")
    return data


def save(out_dir: Path, data: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = path(out_dir)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(p)


# ----------------------------------------------------------------------
# Prior leg cache for the nested-FORM unconditional MCS (mcs_method="alphas")
# ----------------------------------------------------------------------

def try_load_prior_alphas(
    out_dir: Path,
    *,
    lsf_name: str,
    n_samples: int,
    seed: int,
    n_sections: int,
    L: float,
    wall_theta: float,
    wall_rho_0: float,
    cr_theta: float,
    cr_rho_0: float,
    cr_values: np.ndarray,
    betas: np.ndarray,
    forecast_times: list[float],
) -> dict | None:
    """Return the cached alphas-prior MCS result if every key matches.

    Unlike the interpolate-method prior (``pf_grid.json``), this cache
    depends on the cr kernel parameters because the unconditional cr field
    is sampled with the cr spatial covariance, and on the forecast time
    grid because ``(beta_T, alpha_cr, alpha_basic)`` is recomputed per t.
    """
    p = path(out_dir, method="alphas")
    if not p.exists():
        return None
    try:
        data = json.load(open(p))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  Cached alphas-prior grid {p} unreadable ({exc!r}) — recomputing.")
        return None

    same = (
        data.get("lsf_name") == lsf_name
        and int(data.get("n_samples", -1)) == int(n_samples)
        and int(data.get("seed", -1)) == int(seed)
        and int(data.get("n_sections", -1)) == int(n_sections)
        and math.isclose(float(data.get("L", -1)), float(L), abs_tol=1e-9)
        and math.isclose(float(data.get("wall_theta", -1)), float(wall_theta), abs_tol=1e-9)
        and math.isclose(float(data.get("wall_rho_0", -1)), float(wall_rho_0), abs_tol=1e-9)
        and math.isclose(float(data.get("cr_theta", -1)), float(cr_theta), abs_tol=1e-9)
        and math.isclose(float(data.get("cr_rho_0", -1)), float(cr_rho_0), abs_tol=1e-9)
        and _arrays_close(data.get("cr_values", []), list(cr_values))
        and _arrays_close(data.get("betas", []), list(betas))
        and _arrays_close(data.get("forecast_times", []), list(forecast_times))
    )
    if not same:
        print(f"  Cached alphas-prior grid {p} has a different run config — recomputing.")
        return None
    print(f"  Loaded cached alphas-prior grid from {p}.")
    return data


def save_prior_alphas(out_dir: Path, data: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = path(out_dir, method="alphas")
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(p)


# ----------------------------------------------------------------------
# Posterior leg cache (cr-field MCS, per obs scenario)
# ----------------------------------------------------------------------

def posterior_path(out_dir: Path, method: str = "interpolate") -> Path:
    """File name for the posterior-leg cache.

    Two methods write to different files so they coexist in the same cache
    folder: the interpolate-method posterior at ``posterior_grid.json`` and
    the nested-FORM (alphas) MCS at ``posterior_grid_alphas.json``.
    """
    if method == "interpolate":
        return out_dir / "posterior_grid.json"
    return out_dir / f"posterior_grid_{method}.json"


def try_load_posterior(
    out_dir: Path,
    *,
    lsf_name: str,
    n_samples: int,
    seed: int,
    n_sections: int,
    L: float,
    wall_theta: float,
    wall_rho_0: float,
    cr_theta: float,
    cr_rho_0: float,
    cr_values: np.ndarray,
    betas: np.ndarray,
    obs_times: list[float],
    method: str = "interpolate",
) -> dict | None:
    """Return the cached posterior-leg result if every key matches, else ``None``.

    Cache invalidates on any metadata change OR if the fragility fingerprint
    (``cr_values``, ``betas``) differs from when the cache was written —
    typically the curve was rebuilt — OR if the obs-time set changed (e.g.
    ``data.json`` got a new entry).
    """
    p = posterior_path(out_dir, method=method)
    if not p.exists():
        return None
    try:
        data = json.load(open(p))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  Cached posterior grid {p} unreadable ({exc!r}) — recomputing.")
        return None

    same = (
        data.get("lsf_name") == lsf_name
        and int(data.get("n_samples", -1)) == int(n_samples)
        and int(data.get("seed", -1)) == int(seed)
        and int(data.get("n_sections", -1)) == int(n_sections)
        and math.isclose(float(data.get("L", -1)), float(L), abs_tol=1e-9)
        and math.isclose(float(data.get("wall_theta", -1)), float(wall_theta), abs_tol=1e-9)
        and math.isclose(float(data.get("wall_rho_0", -1)), float(wall_rho_0), abs_tol=1e-9)
        and math.isclose(float(data.get("cr_theta", -1)), float(cr_theta), abs_tol=1e-9)
        and math.isclose(float(data.get("cr_rho_0", -1)), float(cr_rho_0), abs_tol=1e-9)
        and _arrays_close(data.get("cr_values", []), list(cr_values))
        and _arrays_close(data.get("betas", []), list(betas))
        and _arrays_close(data.get("obs_times", []), list(obs_times))
    )
    if not same:
        print(f"  Cached posterior grid {p} has a different run config — recomputing.")
        return None
    print(f"  Loaded cached posterior grid from {p}.")
    return data


def save_posterior(out_dir: Path, data: dict, method: str = "interpolate") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = posterior_path(out_dir, method=method)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(p)
