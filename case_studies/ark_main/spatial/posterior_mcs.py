"""
Field-sampling MCS for one obs scenario.

For each forecast time ``t >= t_obs``:

* Draw the cr field per realisation from the Kriged conditional in
  ``cr_field.py`` — anchored on the posterior at ``x = 0`` and decaying to
  the prior at far ``x``.
* Per section, look up the FORM linearisation at the realised local cr by
  linearly interpolating ``(beta, alpha_v)`` from the cached fragility
  points.
* Sample the per-variable basic-variable u-fields with their own spatial
  covariances (independent of the cr field).
* Compute the per-section LSF ``g_i = beta_i - sum_v alpha_i_v * U_v(x_i)``,
  tally failures.

Unlike ``engine.py``'s parametric pf-grid MCS this one **does not collapse**
the per-variable Cholesky multiplies: the section-dependent alphas mean the
effective covariance of ``Y(x) = alpha(cr(x)) . U(x)`` is not a clean
sum-of-alpha-squared form. We sample each variable's u-field independently
and take the inner product per section at MC time.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st
from tqdm import tqdm

from spatial import cr_field
from spatial.covariance import spatial_covariance


def _per_variable_chol(
    x: np.ndarray,
    spec_basic: dict[str, tuple[float, float]],
    active_vars: list[str],
) -> dict[str, np.ndarray | None]:
    """Cholesky of ``C_v`` per active basic variable, or ``None`` if uniform.

    A variable with ``rho_0 >= 1`` is treated as uniform along the wall —
    we mark it with ``None`` here and the sampler in ``run_posterior_mcs``
    draws one scalar per MC sample and broadcasts it across sections,
    bypassing the rank-deficient Cholesky of an all-ones matrix.
    """
    n = len(x)
    L_by_var: dict[str, np.ndarray | None] = {}
    for v in active_vars:
        theta_v, rho_0_v = spec_basic[v]
        if rho_0_v >= 1.0 - 1e-12:
            L_by_var[v] = None
            continue
        Cv = spatial_covariance(x, theta_v, rho_0_v)
        L_by_var[v] = np.linalg.cholesky(Cv + 1e-8 * np.eye(n))
    return L_by_var


def _interp_form_per_section(
    cr_samples: np.ndarray,           # (n_samples, n_sections)
    cr_values: np.ndarray,             # (n_cr,)
    betas: np.ndarray,                 # (n_cr,)
    alpha_table: np.ndarray,           # (n_cr, n_vars_active)
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interp (beta, alpha) per (sample, section) from the cached grid.

    ``cr_values`` is the sorted cr grid from the fragility cache. ``betas``
    and each column of ``alpha_table`` are interpolated against ``cr_values``
    with flat extrapolation at the endpoints (``np.interp`` default).

    Returns ``(beta_samples, alpha_samples)`` of shapes
    ``(n_samples, n_sections)`` and ``(n_samples, n_sections, n_vars_active)``.
    """
    flat = cr_samples.ravel()
    beta_flat = np.interp(flat, cr_values, betas)
    beta_samples = beta_flat.reshape(cr_samples.shape)

    n_active = alpha_table.shape[1]
    alpha_flat = np.empty((flat.size, n_active))
    for j in range(n_active):
        alpha_flat[:, j] = np.interp(flat, cr_values, alpha_table[:, j])
    alpha_samples = alpha_flat.reshape(*cr_samples.shape, n_active)
    return beta_samples, alpha_samples


def run_posterior_mcs(
    *,
    points: list[dict],
    x: np.ndarray,
    spec_basic: dict[str, tuple[float, float]],
    theta_cr: float,
    rho_0_cr: float,
    cr_grid_export: np.ndarray,
    prior_pdf_per_t: dict[str, list[float]],
    posterior_pdf_per_t: dict[str, list[float]],
    forecast_times: np.ndarray,
    n_samples: int,
    seed: int,
    desc: str = "posterior MC",
) -> tuple[np.ndarray, np.ndarray]:
    """Run the field-sampling MCS for one obs scenario.

    Returns ``(n_fail_section, n_fail_system)`` with shapes
    ``(n_t, n_sections)`` and ``(n_t,)`` respectively. The counts are
    accumulated over ``n_samples`` MC realisations.

    Implementation notes
    --------------------
    * The set of "active" FORM variables is the union over the fragility
      cache of variables with any non-zero alpha. Variables with alpha = 0
      everywhere contribute zero to ``Y`` for every cr and are dropped.
    * Per-variable u-field Cholesky factors are precomputed once. The cr
      Cholesky is rebuilt per ``t`` because the Kriging variance depends on
      ``v_post(t)``.
    * Batching: each iteration of the outer ``for t`` loop draws all
      ``n_samples`` realisations of the cr field and the basic u-fields in
      one numpy call, then computes per-section failures vectorised.
    """
    n_sections = len(x)
    cr_values = np.array([float(pt["point"]["corrosion_rate"]) for pt in points])
    order = np.argsort(cr_values)
    cr_values = cr_values[order]
    betas = np.array([float(points[i]["beta"]) for i in order])

    # Active variable set: any var with non-zero alpha at any cr.
    var_names = list(points[0]["alphas"].keys())
    alpha_full = np.array([
        [float(points[i]["alphas"][v]) for v in var_names] for i in order
    ])                                                       # (n_cr, n_vars_all)
    active_mask = np.any(np.abs(alpha_full) > 1e-6, axis=0)  # (n_vars_all,)
    active_vars = [v for v, m in zip(var_names, active_mask) if m]
    alpha_table = alpha_full[:, active_mask]                  # (n_cr, n_vars_active)
    n_active = len(active_vars)

    L_by_var = _per_variable_chol(x, spec_basic, active_vars)
    # Partition variables into spatial (Cholesky) and uniform (scalar draw).
    spatial_idx = [j for j, v in enumerate(active_vars) if L_by_var[v] is not None]
    uniform_idx = [j for j, v in enumerate(active_vars) if L_by_var[v] is None]
    n_spatial = len(spatial_idx)
    n_uniform = len(uniform_idx)
    if n_spatial:
        L_basic_stack = np.stack(
            [L_by_var[active_vars[j]] for j in spatial_idx], axis=0,
        )
        L_basic_T = L_basic_stack.swapaxes(-1, -2)           # (n_spatial, N, N)
    else:
        L_basic_T = None

    rng = np.random.default_rng(seed)
    n_t = len(forecast_times)
    n_fail_section = np.zeros((n_t, n_sections), dtype=np.int64)
    n_fail_system = np.zeros(n_t, dtype=np.int64)

    pbar = tqdm(forecast_times, desc=desc, unit="t", dynamic_ncols=True)
    for ti, t in enumerate(pbar):
        key = f"{float(t):.4f}"
        prior_pdf = np.asarray(prior_pdf_per_t[key], dtype=float)
        post_pdf = np.asarray(posterior_pdf_per_t[key], dtype=float)

        m_post, v_post = cr_field.z_moments_under_posterior(
            prior_pdf, post_pdf, cr_grid_export,
        )
        mean_factor, L_z = cr_field.kriged_chol(
            x, x0_idx=0, theta_cr=theta_cr, rho_0_cr=rho_0_cr, v_post=v_post,
        )

        # 1) Sample cr field over all sections: (n_samples, n_sections).
        cr_samples = cr_field.sample_cr_field(
            rng, n_samples, mean_factor, L_z, m_post, prior_pdf, cr_grid_export,
        )

        # 2) Interp FORM coefficients per (sample, section).
        beta_samples, alpha_samples = _interp_form_per_section(
            cr_samples, cr_values, betas, alpha_table,
        )                                                    # (n_s, N), (n_s, N, n_a)

        # 3) Sample basic u-fields:
        #    * Spatial vars (rho_0 < 1): draw a 21-vec via Cholesky, one per
        #      variable, one per MC sample. Stacked shape (n_spatial, n_s, N).
        #    * Uniform vars (rho_0 = 1, the loads/water/model factors/wall
        #      stiffness): one scalar per MC sample, broadcast across the N
        #      sections. Stacked shape (n_uniform, n_s, N).
        # Then rebuild a (n_active, n_s, N) tensor in original variable order
        # so the einsum below can use the alpha_samples shape unchanged.
        U_basic = np.empty((n_active, n_samples, n_sections))
        if n_spatial:
            Z_spatial = rng.standard_normal((n_spatial, n_samples, n_sections))
            U_spatial = Z_spatial @ L_basic_T                # (n_spatial, n_s, N)
            U_basic[spatial_idx] = U_spatial
        if n_uniform:
            Z_uniform = rng.standard_normal((n_uniform, n_samples))
            U_basic[uniform_idx] = Z_uniform[..., None]      # broadcast over N

        # 4) Y_i = sum_v alpha_i_v * U_v(x_i)
        #    alpha_samples: (n_s, N, n_a); U_basic.transpose: (n_s, N, n_a)
        Y = np.einsum("snv,vsn->sn", alpha_samples, U_basic) # (n_samples, N)

        # 5) Failure
        G = beta_samples - Y                                 # (n_samples, N)
        sec_fail = G < 0
        n_fail_section[ti] += sec_fail.sum(axis=0).astype(np.int64)
        n_fail_system[ti] += int(sec_fail.any(axis=1).sum())

        pf_sys = n_fail_system[ti] / n_samples
        pbar.set_postfix(t=f"{float(t):.1f}", Pf=f"{pf_sys:.2e}")
    pbar.close()

    return n_fail_section, n_fail_system
