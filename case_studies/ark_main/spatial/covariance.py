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
    spatial_block: dict | None,
    cli_theta: float | None,
    cli_rho_0: float | None,
) -> tuple[float, int, float, float, dict[str, tuple[float, float]]]:
    """Resolve the spatial-correlation parameters per variable.

    Returns ``(L, n_sections, default_theta, default_rho_0, spec)`` where
    ``spec`` is ``{var_name -> (theta, rho_0)}``.

    Precedence per variable: ``per_variable[name]`` > CLI override > settings
    ``default`` > built-in default (``theta=200 m``, ``rho_0=0.3``). The two
    resolved scalar defaults are surfaced separately so callers (e.g. the
    cache fingerprint) don't have to reverse-engineer them from ``spec``.
    """
    block = spatial_block or {}
    L = float(block.get("L", 1000.0))
    n_sections = int(block.get("n_sections", 21))

    default = block.get("default", {}) or {}
    default_theta = float(cli_theta if cli_theta is not None
                          else default.get("theta", 200.0))
    default_rho_0 = float(cli_rho_0 if cli_rho_0 is not None
                          else default.get("rho_0", 0.3))

    per_variable = block.get("per_variable", {}) or {}
    spec: dict[str, tuple[float, float]] = {}
    for v in var_names:
        entry = per_variable.get(v, {}) or {}
        spec[v] = (
            float(entry.get("theta", default_theta)),
            float(entry.get("rho_0", default_rho_0)),
        )
    return L, n_sections, default_theta, default_rho_0, spec
