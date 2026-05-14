"""
Spatial-variability MCS for a 1 km sheet-pile wall — library code.

This module exposes:

* :func:`compute_or_load_pf_grid` — public entry. Returns the cached or
  freshly-computed per-fragility-point Pf grid (steps 1-5 of the
  spatial-MCS pipeline).
* :func:`_run_mcs_over_cr_grid` — the inner-loop MCS, internal.

The orchestration that wraps these into the t=0 integration, summary, and
plots lives in ``case_studies/ark_main/run_spatial.py``.

What the MCS does
-----------------

For each cached fragility cr-point we read ``beta(cr)`` and the unit-norm
``alpha(cr)`` over the basic random variables. The FORM linearisation gives
the per-section LSF in u-space::

    g_i = beta(cr) - alpha(cr) . U(x_i)

Per-variable spatial Gaussian fields collapse into one effective covariance
``C_eff(cr) = sum_v alpha_v(cr)^2 * C_v`` (see ``covariance.py``), so each
MC sample is one Cholesky-multiply ``Y = L_eff @ Z``. Section ``i`` fails
iff ``g_i = beta - Y_i < 0``. Common random numbers (the same ``Z``
stream) are shared across cr-points so the resulting ``Pf(cr)`` curve is
smooth.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ARK = Path(__file__).resolve().parents[1]
if str(_ARK) not in sys.path:
    sys.path.insert(0, str(_ARK))

# Geolib reads ``geolib.env`` relative to cwd when its ``MetaData`` BaseSettings
# is first instantiated. Hydrate it from the case-study copy before any
# transitive geolib import (``reliability.build_fragility`` pulls it in).
from dotenv import load_dotenv
load_dotenv(_ARK / "geolib.env")

import numpy as np
from tqdm import tqdm

from src.io import get_remote_path
from reliability.build_fragility import load_settings

from spatial import checkpoint, covariance, fragility


_ENV = _ARK / ".env"
_settings = load_settings()
_remote = get_remote_path(_ENV)


# ----------------------------------------------------------------------
# Step 1-4: MCS over the cr grid
# ----------------------------------------------------------------------

def _run_mcs_over_cr_grid(
    points: list[dict],
    x: np.ndarray,
    spec: dict[str, tuple[float, float]],
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the spatial MCS at every cached fragility cr-point.

    Uses a single shared ``Z`` stream across cr (common random numbers) so
    that ``Pf(cr)`` is monotone-and-smooth-ish instead of ratty MC noise.
    Returns ``(n_fail_section[n_cr, n_sections], n_fail_system[n_cr])``.

    The inner loop collapses the cr axis into the matmul: with
    ``L_stack`` of shape ``(K, N, N)`` and a per-batch ``Z`` of shape
    ``(T, N)``, the broadcast ``Z @ L_stack.swapaxes(-1, -2)`` yields all
    ``K * T * N`` projected values in one call.
    """
    K = len(points)
    N = len(x)

    L_stack = np.stack([
        covariance.effective_cholesky(
            x, {k: float(v) for k, v in pt["alphas"].items()}, spec,
        ) for pt in points
    ], axis=0)                                       # (K, N, N)
    L_stack_T = L_stack.swapaxes(-1, -2)             # (K, N, N), L^T per cr
    betas = np.array([float(pt["beta"]) for pt in points])  # (K,)

    n_fail_section = np.zeros((K, N), dtype=np.int64)
    n_fail_system = np.zeros(K, dtype=np.int64)

    batch_size = max(200, n_samples // 50)
    rng = np.random.default_rng(seed)
    pbar = tqdm(total=n_samples, desc=f"Spatial MC ({K} cr-points)",
                unit="sample", dynamic_ncols=True)
    done = 0
    while done < n_samples:
        take = min(batch_size, n_samples - done)
        Z = rng.standard_normal((take, N))                          # (T, N)
        Y = Z @ L_stack_T                                            # (K, T, N)
        G = betas[:, None, None] - Y                                 # (K, T, N)
        sec_fail = G < 0                                             # (K, T, N) bool
        n_fail_section += sec_fail.sum(axis=1).astype(np.int64)      # (K, N)
        n_fail_system += sec_fail.any(axis=2).sum(axis=1).astype(np.int64)  # (K,)
        done += take
        pbar.update(take)
    pbar.close()

    return n_fail_section, n_fail_system


# ----------------------------------------------------------------------
# Step 5: cache gateway — public API
# ----------------------------------------------------------------------

def compute_or_load_pf_grid(
    lsf_name: str = "lsf_wall",
    n_samples: int = 10000,
    seed: int = 42,
    theta: float | None = None,
    rho_0: float | None = None,
) -> dict:
    """Return the per-fragility-point Pf grid for ``lsf_name``.

    Loads from ``<remote>/output/spatial_mc_<lsf_name>/pf_grid.json`` if a
    cached file compatible with the requested configuration exists; otherwise
    runs the spatial MCS at every cached fragility cr-point, writes the
    result to that path, and returns it. Subsequent calls with the same
    arguments hit the cache.

    The returned dict mirrors the on-disk JSON: ``lsf_name``, ``n_samples``,
    ``seed``, ``n_sections``, ``L``, ``default_theta``, ``default_rho_0``,
    ``cr_values`` (n_cr), ``betas`` (n_cr), ``n_fail_section`` (n_cr x
    n_sections), ``n_fail_system`` (n_cr). Convert the last two to Pf by
    dividing by ``n_samples``.

    Cache invalidates if any of ``(lsf_name, n_samples, seed, n_sections, L,
    default_theta, default_rho_0)`` or the fragility fingerprint
    ``(cr_values, betas)`` differs from what's on disk — most commonly when
    the fragility curve has been rebuilt or the spatial defaults moved.
    """
    points = fragility.load_points(_remote, lsf_name)
    var_names = list(points[0]["alphas"].keys())
    L, n_sections, default_theta, default_rho_0, spec = covariance.resolve_config(
        var_names, _settings.get("spatial"), theta, rho_0,
    )
    x = np.linspace(0.0, L, n_sections)
    out_dir = _remote / "output" / f"spatial_mc_{lsf_name}"

    cr_values = np.array([pt["point"]["corrosion_rate"] for pt in points])
    betas = np.array([float(pt["beta"]) for pt in points])

    cached = checkpoint.try_load(
        out_dir,
        lsf_name=lsf_name, n_samples=n_samples, seed=seed,
        n_sections=n_sections, L=L,
        default_theta=default_theta, default_rho_0=default_rho_0,
        cr_values=cr_values, betas=betas,
    )
    if cached is not None:
        return cached

    n_fail_section, n_fail_system = _run_mcs_over_cr_grid(
        points, x, spec, n_samples, seed,
    )
    data = {
        "lsf_name": lsf_name,
        "n_samples": int(n_samples),
        "seed": int(seed),
        "n_sections": int(n_sections),
        "L": float(L),
        "default_theta": float(default_theta),
        "default_rho_0": float(default_rho_0),
        "cr_values": cr_values.tolist(),
        "betas": betas.tolist(),
        "n_fail_section": n_fail_section.tolist(),
        "n_fail_system": n_fail_system.tolist(),
    }
    checkpoint.save(out_dir, data)
    print(f"  Saved Pf grid to {checkpoint.path(out_dir)}.")
    return data
