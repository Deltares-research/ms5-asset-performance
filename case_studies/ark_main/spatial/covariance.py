"""Spatial covariance for the per-variable Gaussian fields, plus the
collapsed (effective) Cholesky used by the inner MCS loop.

With independent unit-variance fields per FORM variable and unit-norm alphas,
the FORM projection

    Y(x) = sum_v alpha_v * U_v(x)

is itself a zero-mean unit-variance Gaussian field whose covariance is

    C_eff(x, x') = sum_v alpha_v^2 * C_v(|x - x'|)

So a single Cholesky of ``C_eff`` lets us sample ``Y`` directly, collapsing the
N_v per-variable Cholesky multiplies in the inner loop down to one.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import distance_matrix


def spatial_covariance(x: np.ndarray, theta: float, rho_0: float) -> np.ndarray:
    """``rho_0 + (1 - rho_0) * exp(-(d/theta)^2)`` evaluated on the section grid."""
    D = distance_matrix(x.reshape(-1, 1), x.reshape(-1, 1))
    return rho_0 + (1.0 - rho_0) * np.exp(-(D / theta) ** 2)


def is_soil_variable(name: str) -> bool:
    """Variable name encodes a per-layer soil property.

    The ARK FORM variables follow ``<Layer>_soil<property>`` (e.g.
    ``Klei_soilphi``, ``Zand_soilcurkb1``). Everything else — applied
    loads (``uniform_load_left``), water levels (``phreatic_level``,
    ``canal_level``), model factors (``model_factor_M/F``), wall
    stiffness (``Wall_SheetPilingElementEI``) — applies to the whole
    wall at once and is treated as **uniform along the wall**: a single
    scalar draw per MC sample, broadcast to all sections. The spec for
    those variables uses ``rho_0 = 1`` (perfect spatial correlation),
    which makes the kernel collapse to an all-ones matrix.

    The default classification can be overridden per-variable by setting
    ``rho_0 < 1`` in ``spatial_settings.json``'s ``per_variable`` block.
    """
    return "soil" in name.lower()


def effective_cholesky(
    x: np.ndarray,
    alpha_dict: dict[str, float],
    spec: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Cholesky factor of ``Cov[alpha . U(x)]`` over the section grid.

    Variables with zero alpha are skipped (they cannot influence the
    projection). A tiny diagonal nugget regularises the decomposition for
    parameter combinations that sit on the PSD boundary.
    """
    n = len(x)
    C_eff = np.zeros((n, n))
    for v, (theta, rho_0) in spec.items():
        a = alpha_dict[v]
        if a == 0.0:
            continue
        C_eff += (a * a) * spatial_covariance(x, theta, rho_0)
    return np.linalg.cholesky(C_eff + 1e-8 * np.eye(n))


def resolve_config(
    var_names: list[str],
    spatial_settings: dict | None,
) -> tuple[float, int, float, float, dict[str, tuple[float, float]]]:
    """Resolve the spatial-correlation parameters per variable.

    Returns ``(L, n_sections, wall_theta, wall_rho_0, spec)`` where ``spec``
    is ``{var_name -> (theta, rho_0)}``.

    Source of truth is ``spatial_settings`` (the ``spatial_settings.json``
    payload). The basic FORM variables share one kernel by default — its
    ``(theta, rho_0)`` lives under the ``wall`` block (the same kernel the
    whole 1 km sheet pile uses). Per-variable overrides in
    ``spatial_settings["per_variable"]`` take precedence over ``wall``;
    both fall back to built-in defaults (``theta=200 m``, ``rho_0=0.3``)
    if missing.
    """
    s = spatial_settings or {}
    L = float(s.get("L", 1000.0))
    n_sections = int(s.get("n_sections", 21))

    wall = s.get("wall", {}) or {}
    wall_theta = float(wall.get("theta", 200.0))
    wall_rho_0 = float(wall.get("rho_0", 0.0))

    per_variable = s.get("per_variable", {}) or {}
    spec: dict[str, tuple[float, float]] = {}
    for v in var_names:
        entry = per_variable.get(v, {}) or {}
        if is_soil_variable(v):
            # Soil parameters get spatial fields with the wall kernel.
            # Default rho_0 = 0 means pure exp-squared decay — at distances
            # >> theta the field becomes uncorrelated with the obs side.
            spec[v] = (
                float(entry.get("theta", wall_theta)),
                float(entry.get("rho_0", wall_rho_0)),
            )
        else:
            # Non-soil variables (loads, water levels, model factors, wall
            # stiffness) are applied wall-wide. Encoded as rho_0 = 1 so the
            # kernel matrix is all-ones — each MC sample draws a single
            # scalar that gets broadcast to every section.
            spec[v] = (
                float(entry.get("theta", wall_theta)),
                float(entry.get("rho_0", 1.0)),
            )
    return L, n_sections, wall_theta, wall_rho_0, spec


def resolve_cr_config(spatial_settings: dict | None) -> tuple[float, float]:
    """Resolve ``(theta_cr, rho_0_cr)`` for the corrosion-rate spatial kernel.

    Read from ``spatial_settings["cr"].{theta, rho_0}`` with built-in
    fallback ``(200.0, 0.3)``. The cr field uses the same squared-exponential
    covariance form as the basic FORM variables but with dedicated
    parameters, since cr usually varies on a different length-scale than
    soil properties.
    """
    s = spatial_settings or {}
    cr = s.get("cr", {}) or {}
    return float(cr.get("theta", 200.0)), float(cr.get("rho_0", 0.3))
