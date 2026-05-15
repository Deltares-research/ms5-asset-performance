"""
Nataf + Kriging utilities for sampling the cr field across the wall.

Pipeline (per forecast time ``t``, conditional on obs up to ``t_obs``):

1. Treat ``z(x) := Phi^{-1}(F_prior(cr(x); t))`` as a unit-variance Gaussian
   field over the section grid, with squared-exponential spatial covariance
   ``C_cr(d) = rho_0_cr + (1 - rho_0_cr) * exp(-(d/theta_cr)^2)``.
2. Extract the posterior mean ``m_post(t)`` and variance ``v_post(t)`` of
   ``z(x = 0)`` from the file's ``posterior_pdf_per_obs[t_obs][t]`` —
   numerical integrals against the posterior PDF on the file's ``cr_grid``.
3. Kriging conditional moments on the section grid (single observation
   point ``x_0 = x[0]``):

        E[z_i | obs]  = rho_cr_i * m_post
        Cov[z_i, z_j | obs]  =  rho_cr_i * rho_cr_j * v_post
                                + (rho_cr_ij - rho_cr_i * rho_cr_j)

4. Cholesky once per ``t``. Sample ``z_field = mean + L_z @ Z_cr`` per MCS
   iteration. Back-transform ``cr_i = F_prior^{-1}(Phi(z_i); t)`` via
   numerical inverse-CDF on the file's ``cr_grid``.

Notes
-----
The reference marginal for the Nataf transform is the **prior**
``F_prior(cr; t)``. This keeps the spatial field stationary (each ``z`` is
N(0, 1) marginal under the prior), which is what the squared-exponential
covariance form assumes. The posterior shows up only via the moments of
``z(x = 0)``, not as a re-anchoring of the reference distribution.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st
from scipy.spatial import distance_matrix


# ----------------------------------------------------------------------
# Numerical CDF / inverse-CDF on a grid
# ----------------------------------------------------------------------

def cdf_on_grid(pdf: np.ndarray, cr_grid: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of a PDF on the ``cr_grid``.

    Returned CDF is clipped to [eps, 1 - eps] so the subsequent
    ``Phi^{-1}`` evaluation is finite for every grid point.
    """
    cdf = np.zeros_like(cr_grid, dtype=float)
    if len(cr_grid) > 1:
        # cumulative trapezoid: cdf[k] = sum_{j<k} 0.5*(pdf[j] + pdf[j+1])*dx[j]
        dx = np.diff(cr_grid)
        seg = 0.5 * (pdf[:-1] + pdf[1:]) * dx
        cdf[1:] = np.cumsum(seg)
    # Normalise (in case the input pdf isn't quite normalised, e.g. boundary
    # truncation in the corrosion model).
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    eps = 1e-12
    return np.clip(cdf, eps, 1.0 - eps)


def inv_cdf_at(u_vals: np.ndarray, cr_grid: np.ndarray,
               cdf: np.ndarray) -> np.ndarray:
    """Evaluate ``F^{-1}(u)`` by interpolating the inverse of ``cdf`` over ``cr_grid``.

    ``u_vals`` may be any shape; we flatten, interpolate, then reshape.
    """
    out = np.interp(np.asarray(u_vals).ravel(), cdf, cr_grid)
    return out.reshape(np.asarray(u_vals).shape)


# ----------------------------------------------------------------------
# Z-moments under the posterior at the obs location
# ----------------------------------------------------------------------

def z_moments_under_posterior(
    prior_pdf: np.ndarray,
    posterior_pdf: np.ndarray,
    cr_grid: np.ndarray,
) -> tuple[float, float]:
    """Mean & variance of ``z = Phi^{-1}(F_prior(cr))`` under the posterior.

    The transform ``z = Phi^{-1}(F_prior(cr))`` is computed once on the
    common ``cr_grid``. We then integrate against ``posterior_pdf`` to get

        m_post = E_post[z]
        v_post = E_post[z^2] - m_post^2

    Both integrals are trapezoidal on ``cr_grid``. The returned variance is
    clipped to ``[1e-12, 1.0]`` so the subsequent ``(1 - rho^2) + rho^2 * v``
    Kriging variance stays positive even with degenerate posteriors.
    """
    F_prior = cdf_on_grid(np.asarray(prior_pdf, dtype=float), cr_grid)
    z_of_cr = st.norm.ppf(F_prior)

    p_post = np.asarray(posterior_pdf, dtype=float)
    norm = np.trapezoid(p_post, cr_grid)
    if norm <= 0:
        # Fully degenerate posterior — fall back to "no information"
        # (z(x_0) ~ N(0, 1)). The Kriged field then equals the prior.
        return 0.0, 1.0
    p_post = p_post / norm

    m = float(np.trapezoid(z_of_cr * p_post, cr_grid))
    e2 = float(np.trapezoid(z_of_cr ** 2 * p_post, cr_grid))
    v = max(e2 - m * m, 1e-12)
    v = min(v, 1.0)  # cannot exceed the prior marginal variance
    return m, v


# ----------------------------------------------------------------------
# Kriged conditional moments on the section grid
# ----------------------------------------------------------------------

def kriged_chol(
    x: np.ndarray,
    x0_idx: int,
    theta_cr: float,
    rho_0_cr: float,
    v_post: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean_factor, L_z)`` for sampling the Kriged z-field.

    ``mean_factor[i] = rho_cr_i`` is the multiplier on ``m_post`` for the
    conditional mean at section ``i``. ``L_z`` is the Cholesky factor of
    the conditional covariance

        Cov[z_i, z_j | obs] = rho_cr_i * rho_cr_j * v_post
                              + (rho_cr_ij - rho_cr_i * rho_cr_j)

    so that ``z = mean_factor * m_post + L_z @ Z`` with ``Z ~ N(0, I)``
    has the correct moments under the Kriging conditioning.

    A tiny diagonal nugget regularises ``L_z`` when ``v_post`` makes the
    matrix near-singular (e.g. at very early forecast times where the
    prior and posterior nearly coincide).
    """
    D = distance_matrix(x.reshape(-1, 1), x.reshape(-1, 1))
    R = rho_0_cr + (1.0 - rho_0_cr) * np.exp(-(D / theta_cr) ** 2)
    rho_i = R[:, x0_idx]                                 # (n_sections,)
    # Conditional covariance: outer product term + spatial residual.
    cov_cond = np.outer(rho_i, rho_i) * v_post + (R - np.outer(rho_i, rho_i))
    L_z = np.linalg.cholesky(cov_cond + 1e-8 * np.eye(len(x)))
    return rho_i, L_z


# ----------------------------------------------------------------------
# Per-sample cr-field sampler
# ----------------------------------------------------------------------

def conditional_cr_quantiles(
    x: np.ndarray,
    x0_idx: int,
    theta_cr: float,
    rho_0_cr: float,
    prior_pdf: np.ndarray,
    posterior_pdf: np.ndarray,
    cr_grid: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form quantiles of the obs-conditioned cr field at each section.

    The cr field in u-space is Gaussian with N(0, 1) marginals under the
    prior. The Kriging conditional on the obs at ``x_0`` yields, per section,

        E[z_i | obs]    = rho_cr_i * m_post
        Var[z_i | obs]  = rho_cr_i^2 * v_post + (1 - rho_cr_i^2)

    so ``z_i | obs ~ N(m_i, s_i^2)`` is again Gaussian. Quantiles in cr-space
    follow by pushing the z-quantiles through ``F_prior^{-1}(Phi(z))``.

    Returns ``(prior_q, cond_q)``:

    * ``prior_q`` shape ``(len(quantiles),)`` — stationary along the wall.
    * ``cond_q`` shape ``(len(sections), len(quantiles))`` — varies across
      sections; converges to ``prior_q`` at sections where ``rho_cr_i`` is
      small.
    """
    F_prior = cdf_on_grid(np.asarray(prior_pdf, dtype=float), cr_grid)

    # Prior quantiles (stationary).
    q_arr = np.asarray(quantiles, dtype=float)
    prior_q = inv_cdf_at(q_arr, cr_grid, F_prior)

    # Posterior z-moments at the obs location.
    m_post, v_post = z_moments_under_posterior(prior_pdf, posterior_pdf, cr_grid)

    # Kriged moments along the section grid.
    D0 = np.abs(np.asarray(x, dtype=float) - float(x[x0_idx]))
    rho_i = rho_0_cr + (1.0 - rho_0_cr) * np.exp(-(D0 / theta_cr) ** 2)
    mean_i = rho_i * m_post
    var_i = rho_i ** 2 * v_post + (1.0 - rho_i ** 2)
    sd_i = np.sqrt(np.maximum(var_i, 0.0))

    # Quantile z-values per section.
    z_q = st.norm.ppf(q_arr)                          # (n_q,)
    z_cond = mean_i[:, None] + sd_i[:, None] * z_q[None, :]   # (n_sections, n_q)

    # Map back to cr space via F_prior^{-1}(Phi(z)).
    u_cond = st.norm.cdf(z_cond)
    cond_q = inv_cdf_at(u_cond, cr_grid, F_prior)
    return prior_q, cond_q


def sample_cr_field(
    rng: np.random.Generator,
    n_samples: int,
    mean_factor: np.ndarray,
    L_z: np.ndarray,
    m_post: float,
    prior_pdf: np.ndarray,
    cr_grid: np.ndarray,
) -> np.ndarray:
    """Draw ``n_samples`` realisations of the Kriged cr field at all sections.

    Returns ``cr_samples`` of shape ``(n_samples, n_sections)``.
    """
    n_sections = len(mean_factor)
    Z_cr = rng.standard_normal((n_samples, n_sections))
    z_field = mean_factor * m_post + Z_cr @ L_z.T            # (n_samples, n_sections)
    u_field = st.norm.cdf(z_field)                            # (n_samples, n_sections)
    # Numerical inverse-CDF: F_prior^{-1}(u) on cr_grid.
    F_prior = cdf_on_grid(np.asarray(prior_pdf, dtype=float), cr_grid)
    return inv_cdf_at(u_field, cr_grid, F_prior)
