"""
Nested-FORM MCS for prior and posterior legs.

Companion to ``posterior_mcs.py`` (per-sample fragility interpolation) and to
``engine.py`` (cr-grid-then-integrate prior leg). Both legs of the spatial
pipeline can use this module via ``spatial_settings.mcs_method =
"alphas"``: posterior obs scenarios go through :func:`run_nested_mcs`, and
the unconditional prior leg goes through :func:`run_nested_mcs_prior`. The
two share the same 1-D nested-FORM search; the prior variant just sets
``mu_z = 0``, ``sigma_z = 1`` (no obs conditioning) and collapses the
section dimension because the prior is stationary along the wall.

Idea
----
Instead of looking up ``(beta(cr_i), alpha(cr_i))`` per (MC sample, section)
by interpolating the cached fragility curve, this method **precomputes one
linearised LSF per (section, t)** by treating cr as one more basic random
variable via a nested-FORM 1-D search:

For each section ``i`` and forecast time ``t``:

1.  Marginal of ``z_i = Phi^{-1}(F_prior(cr_i; t))`` under the posterior is
    Gaussian with moments ``(mu_z_i, sigma_z_i)`` from the Kriging
    conditioning at the obs location. The standardised local cr variable is
    ``xi_i := (z_i - mu_z_i) / sigma_z_i`` (N(0, 1) marginal at section i).

2.  1-D search ``xi_i* = argmin xi^2 + beta(cr_i(xi))^2`` over
    ``cr_i(xi) = F_prior^{-1}(Phi(mu_z_i + sigma_z_i * xi))``.

3.  Assemble the joint design point ``(u*, xi*)`` and read off

        beta_T_i      = sqrt(xi_i*^2  +  beta(cr_i*)^2)
        alpha_cr_i    = -xi_i* / beta_T_i
        alpha_basic_i = (beta(cr_i*) / beta_T_i) * alpha(cr_i*)   (unit-norm preserved)

4.  Per MCS realisation we then evaluate the **tangent-hyperplane** LSF::

        g_i = beta_T_i  -  alpha_cr_i * xi_i  -  sum_v alpha_basic_iv * U_v(x_i)

    No per-sample fragility interpolation — one matmul-style evaluation per
    section per sample.

Compared to ``posterior_mcs.run_posterior_mcs`` this is faster in the inner
loop (no interp; one linear combination per section) but is an extra
linearisation of the fragility surface at the design point. Useful as a
second method to A/B against the field-MCS reference.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from spatial import cr_field
from spatial.covariance import spatial_covariance


# ----------------------------------------------------------------------
# 1-D nested-FORM search
# ----------------------------------------------------------------------

def _nested_form_1d(
    *,
    mu_z: float,
    sigma_z: float,
    cr_grid_export: np.ndarray,
    F_prior: np.ndarray,             # standardised CDF on cr_grid_export
    cr_values_frag: np.ndarray,      # sorted cr grid from the fragility cache
    betas_frag: np.ndarray,          # beta values aligned with cr_values_frag
    xi_bound: float = 6.0,
) -> tuple[float, float, float]:
    """Solve ``xi* = argmin xi^2 + beta(cr(xi))^2`` for one section/time.

    ``cr(xi) = F_prior^{-1}(Phi(mu_z + sigma_z * xi))`` via the file's
    cr_grid_export and the prior CDF computed on it. Returns
    ``(xi_star, cr_star, beta_at_cr_star)``.

    ``minimize_scalar(method='bounded')`` is plenty for this very smooth 1-D
    problem; ``xi_bound = 6.0`` covers >1e-9 tail probability either side.
    """
    def cr_from_xi(xi: float) -> float:
        z = mu_z + sigma_z * float(xi)
        u = float(st.norm.cdf(z))
        return float(np.interp(u, F_prior, cr_grid_export))

    def loss(xi: float) -> float:
        cr = cr_from_xi(xi)
        beta = float(np.interp(cr, cr_values_frag, betas_frag))
        return float(xi) ** 2 + beta ** 2

    res = minimize_scalar(
        loss, bounds=(-xi_bound, xi_bound), method="bounded",
        options={"xatol": 1e-5},
    )
    xi_star = float(res.x)
    cr_star = cr_from_xi(xi_star)
    beta_at_cr_star = float(np.interp(cr_star, cr_values_frag, betas_frag))
    return xi_star, cr_star, beta_at_cr_star


# ----------------------------------------------------------------------
# Per-(section, time) precomputation
# ----------------------------------------------------------------------

def compute_nested_form_for_scenario(
    *,
    x: np.ndarray,
    cr_grid_export: np.ndarray,
    prior_pdf_per_t: dict[str, list[float]],
    posterior_pdf_per_t: dict[str, list[float]],
    forecast_times: np.ndarray,
    theta_cr: float,
    rho_0_cr: float,
    cr_values_frag: np.ndarray,
    betas_frag: np.ndarray,
    alpha_table: np.ndarray,              # (n_cr, n_active)
    active_vars: list[str] | None = None,
    x0_idx: int = 0,
) -> dict[str, np.ndarray]:
    """Per (section, t) nested-FORM precompute for one obs scenario.

    Returns a dict with arrays shaped ``(n_t, n_sections)`` (or
    ``(n_t, n_sections, n_active)`` for the basic alphas)::

        mu_z:        (n_t, n_sections)  marginal mean of z(x_i) under posterior
        sigma_z:     (n_t, n_sections)  marginal std of z(x_i) under posterior
        xi_star:     (n_t, n_sections)  argmin of the 1-D search
        cr_star:     (n_t, n_sections)  cr at the design point
        beta_form:   (n_t, n_sections)  beta(cr_star) (the fragility beta)
        beta_T:      (n_t, n_sections)  total nested-FORM beta
        alpha_cr:    (n_t, n_sections)  direction cosine for the standardised cr
        alpha_basic: (n_t, n_sections, n_active)  direction cosines for u-vars

    ``rho_i = R_cr(|x_i - x_0|)`` is computed once. Posterior z-moments
    ``(m_post, v_post)`` at the obs location come from
    :func:`cr_field.z_moments_under_posterior` per t.
    """
    n_sections = len(x)
    n_active = alpha_table.shape[1]
    n_t = len(forecast_times)

    # rho_i is independent of t (only the cr kernel + geometry).
    D0 = np.abs(np.asarray(x, dtype=float) - float(x[x0_idx]))
    rho_i = rho_0_cr + (1.0 - rho_0_cr) * np.exp(-(D0 / theta_cr) ** 2)

    mu_z = np.zeros((n_t, n_sections))
    sigma_z = np.zeros((n_t, n_sections))
    xi_star = np.zeros((n_t, n_sections))
    cr_star = np.zeros((n_t, n_sections))
    beta_form = np.zeros((n_t, n_sections))
    beta_T = np.zeros((n_t, n_sections))
    alpha_cr_out = np.zeros((n_t, n_sections))
    alpha_basic_out = np.zeros((n_t, n_sections, n_active))

    for ti, t in enumerate(forecast_times):
        key = f"{float(t):.4f}"
        prior_pdf = np.asarray(prior_pdf_per_t[key], dtype=float)
        post_pdf = np.asarray(posterior_pdf_per_t[key], dtype=float)
        F_prior = cr_field.cdf_on_grid(prior_pdf, cr_grid_export)

        m_post, v_post = cr_field.z_moments_under_posterior(
            prior_pdf, post_pdf, cr_grid_export,
        )

        # Marginal of z_i under posterior:
        #   E[z_i | obs]    = rho_i * m_post
        #   Var[z_i | obs]  = rho_i^2 * v_post + (1 - rho_i^2)
        mu_z_t = rho_i * m_post
        var_z_t = rho_i ** 2 * v_post + (1.0 - rho_i ** 2)
        sigma_z_t = np.sqrt(np.maximum(var_z_t, 1e-12))
        mu_z[ti] = mu_z_t
        sigma_z[ti] = sigma_z_t

        for i in range(n_sections):
            xs, crs, b_at_crs = _nested_form_1d(
                mu_z=float(mu_z_t[i]),
                sigma_z=float(sigma_z_t[i]),
                cr_grid_export=cr_grid_export,
                F_prior=F_prior,
                cr_values_frag=cr_values_frag,
                betas_frag=betas_frag,
            )
            xi_star[ti, i] = xs
            cr_star[ti, i] = crs
            beta_form[ti, i] = b_at_crs
            bT = float(np.sqrt(xs * xs + b_at_crs * b_at_crs))
            beta_T[ti, i] = bT
            alpha_cr_out[ti, i] = -xs / bT if bT > 0 else 0.0
            # Interp the alpha-vector at cr_star and scale by (beta_at_cr_star/beta_T).
            alpha_at_crs = np.array([
                float(np.interp(crs, cr_values_frag, alpha_table[:, j]))
                for j in range(n_active)
            ])
            alpha_basic_out[ti, i] = (b_at_crs / bT) * alpha_at_crs if bT > 0 else 0.0

    return {
        "mu_z": mu_z,
        "sigma_z": sigma_z,
        "xi_star": xi_star,
        "cr_star": cr_star,
        "beta_form": beta_form,
        "beta_T": beta_T,
        "alpha_cr": alpha_cr_out,
        "alpha_basic": alpha_basic_out,
        "active_vars": list(active_vars) if active_vars is not None else [],
    }


# ----------------------------------------------------------------------
# Field MCS on the linearised LSF
# ----------------------------------------------------------------------

def _per_variable_chol(
    x: np.ndarray,
    spec_basic: dict[str, tuple[float, float]],
    active_vars: list[str],
) -> dict[str, np.ndarray | None]:
    """Cholesky of ``C_v`` per active basic variable, or ``None`` if uniform.

    Mirrors ``posterior_mcs._per_variable_chol`` — same uniform-broadcast
    treatment for non-soil variables (``rho_0 >= 1``).
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


def run_nested_mcs(
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
    desc: str = "nested MC",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run the nested-FORM MCS for one obs scenario.

    Returns ``(n_fail_section, n_fail_system, precompute)`` where
    ``n_fail_section`` is shape ``(n_t, n_sections)``, ``n_fail_system`` is
    ``(n_t,)``, and ``precompute`` is the dict returned by
    :func:`compute_nested_form_for_scenario` (so the caller can persist the
    per-(section, t) ``beta_T`` / ``alpha_cr`` / ``alpha_basic`` for plots
    and post-processing).

    Implementation
    --------------
    * Active vars and the alpha-table are derived the same way as
      ``posterior_mcs`` (any non-zero alpha at any cr in the fragility cache).
    * Per t we sample (a) the cr-field z(x) via Kriging conditional moments,
      (b) the spatial basic-variable u-fields independently per variable.
      ``xi_i = (z_i - mu_z_i) / sigma_z_i`` is the standardised local cr at
      section i (N(0, 1) marginal).
    * ``g_i = beta_T_i - alpha_cr_i * xi_i - sum_v alpha_basic_iv * U_v(x_i)``
      is the linearised LSF. Failure is ``g_i < 0`` per section, system
      failure is ``any(g_i < 0)`` over sections.
    """
    n_sections = len(x)

    # Sort fragility cache by cr and assemble active alpha table.
    cr_values = np.array([float(p["point"]["corrosion_rate"]) for p in points])
    order = np.argsort(cr_values)
    cr_values = cr_values[order]
    betas = np.array([float(points[i]["beta"]) for i in order])
    var_names = list(points[0]["alphas"].keys())
    alpha_full = np.array([
        [float(points[i]["alphas"][v]) for v in var_names] for i in order
    ])
    active_mask = np.any(np.abs(alpha_full) > 1e-6, axis=0)
    active_vars = [v for v, m in zip(var_names, active_mask) if m]
    alpha_table = alpha_full[:, active_mask]
    n_active = len(active_vars)

    # Precompute per-(section, t) nested-FORM design points.
    precompute = compute_nested_form_for_scenario(
        x=x,
        cr_grid_export=cr_grid_export,
        prior_pdf_per_t=prior_pdf_per_t,
        posterior_pdf_per_t=posterior_pdf_per_t,
        forecast_times=forecast_times,
        theta_cr=theta_cr,
        rho_0_cr=rho_0_cr,
        cr_values_frag=cr_values,
        betas_frag=betas,
        alpha_table=alpha_table,
        active_vars=active_vars,
    )
    beta_T_all = precompute["beta_T"]          # (n_t, N)
    alpha_cr_all = precompute["alpha_cr"]      # (n_t, N)
    alpha_basic_all = precompute["alpha_basic"]  # (n_t, N, n_active)
    mu_z_all = precompute["mu_z"]              # (n_t, N)
    sigma_z_all = precompute["sigma_z"]        # (n_t, N)

    # Cholesky of basic-variable spatial fields (shared across t).
    L_by_var = _per_variable_chol(x, spec_basic, active_vars)
    spatial_idx = [j for j, v in enumerate(active_vars) if L_by_var[v] is not None]
    uniform_idx = [j for j, v in enumerate(active_vars) if L_by_var[v] is None]
    n_spatial = len(spatial_idx)
    n_uniform = len(uniform_idx)
    if n_spatial:
        L_basic_stack = np.stack(
            [L_by_var[active_vars[j]] for j in spatial_idx], axis=0,
        )
        L_basic_T = L_basic_stack.swapaxes(-1, -2)  # (n_spatial, N, N)
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

        # Kriged cr-field sampler — same machinery as posterior_mcs.
        m_post, v_post = cr_field.z_moments_under_posterior(
            prior_pdf, post_pdf, cr_grid_export,
        )
        mean_factor, L_z = cr_field.kriged_chol(
            x, x0_idx=0, theta_cr=theta_cr, rho_0_cr=rho_0_cr, v_post=v_post,
        )

        # 1) Sample z field, then standardise per section.
        Z_cr = rng.standard_normal((n_samples, n_sections))
        z_field = mean_factor * m_post + Z_cr @ L_z.T           # (n_s, N)
        mu_z_t = mu_z_all[ti][None, :]                          # (1, N)
        sigma_z_t = sigma_z_all[ti][None, :]                    # (1, N)
        xi_field = (z_field - mu_z_t) / sigma_z_t               # (n_s, N)

        # 2) Sample basic u-fields.
        U_basic = np.empty((n_active, n_samples, n_sections))
        if n_spatial:
            Z_spatial = rng.standard_normal((n_spatial, n_samples, n_sections))
            U_basic[spatial_idx] = Z_spatial @ L_basic_T        # (n_spatial, n_s, N)
        if n_uniform:
            Z_uniform = rng.standard_normal((n_uniform, n_samples))
            U_basic[uniform_idx] = Z_uniform[..., None]         # broadcast over N

        # 3) Evaluate linearised LSF per section per sample.
        #    Per-section beta_T, alpha_cr, alpha_basic (constant across samples).
        beta_T_t = beta_T_all[ti][None, :]                      # (1, N)
        alpha_cr_t = alpha_cr_all[ti][None, :]                  # (1, N)
        # alpha_basic: (N, n_active); U_basic: (n_active, n_s, N)
        Y_basic = np.einsum("nv,vsn->sn", alpha_basic_all[ti], U_basic)  # (n_s, N)
        G = beta_T_t - alpha_cr_t * xi_field - Y_basic          # (n_s, N)

        sec_fail = G < 0
        n_fail_section[ti] += sec_fail.sum(axis=0).astype(np.int64)
        n_fail_system[ti] += int(sec_fail.any(axis=1).sum())

        pf_sys = n_fail_system[ti] / n_samples
        pbar.set_postfix(t=f"{float(t):.1f}", Pf=f"{pf_sys:.2e}")
    pbar.close()

    return n_fail_section, n_fail_system, precompute


# ----------------------------------------------------------------------
# Prior leg — nested-FORM with no observations
# ----------------------------------------------------------------------

def compute_nested_form_prior(
    *,
    cr_grid_export: np.ndarray,
    prior_pdf_per_t: dict[str, list[float]],
    forecast_times: np.ndarray,
    cr_values_frag: np.ndarray,
    betas_frag: np.ndarray,
    alpha_table: np.ndarray,              # (n_cr, n_active)
    active_vars: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Per-t nested-FORM precompute for the unconditional (prior) leg.

    No observation conditioning -> ``mu_z = 0``, ``sigma_z = 1`` everywhere
    and the prior is stationary along the wall, so the design point only
    depends on ``t`` (one (beta_T, alpha_cr, alpha_basic) triple per t).

    Returns 1-D arrays shaped ``(n_t,)`` or ``(n_t, n_active)``::

        xi_star:     (n_t,)             argmin of the 1-D search
        cr_star:     (n_t,)             cr at the design point
        beta_form:   (n_t,)             beta(cr_star)
        beta_T:      (n_t,)             total nested-FORM beta
        alpha_cr:    (n_t,)             direction cosine for standardised cr
        alpha_basic: (n_t, n_active)    direction cosines for u-vars
    """
    n_active = alpha_table.shape[1]
    n_t = len(forecast_times)

    xi_star = np.zeros(n_t)
    cr_star = np.zeros(n_t)
    beta_form = np.zeros(n_t)
    beta_T = np.zeros(n_t)
    alpha_cr_out = np.zeros(n_t)
    alpha_basic_out = np.zeros((n_t, n_active))

    for ti, t in enumerate(forecast_times):
        key = f"{float(t):.4f}"
        prior_pdf = np.asarray(prior_pdf_per_t[key], dtype=float)
        F_prior = cr_field.cdf_on_grid(prior_pdf, cr_grid_export)

        xs, crs, b_at_crs = _nested_form_1d(
            mu_z=0.0,
            sigma_z=1.0,
            cr_grid_export=cr_grid_export,
            F_prior=F_prior,
            cr_values_frag=cr_values_frag,
            betas_frag=betas_frag,
        )
        xi_star[ti] = xs
        cr_star[ti] = crs
        beta_form[ti] = b_at_crs
        bT = float(np.sqrt(xs * xs + b_at_crs * b_at_crs))
        beta_T[ti] = bT
        alpha_cr_out[ti] = -xs / bT if bT > 0 else 0.0
        alpha_at_crs = np.array([
            float(np.interp(crs, cr_values_frag, alpha_table[:, j]))
            for j in range(n_active)
        ])
        alpha_basic_out[ti] = (b_at_crs / bT) * alpha_at_crs if bT > 0 else 0.0

    return {
        "xi_star": xi_star,
        "cr_star": cr_star,
        "beta_form": beta_form,
        "beta_T": beta_T,
        "alpha_cr": alpha_cr_out,
        "alpha_basic": alpha_basic_out,
        "active_vars": list(active_vars) if active_vars is not None else [],
    }


def run_nested_mcs_prior(
    *,
    points: list[dict],
    x: np.ndarray,
    spec_basic: dict[str, tuple[float, float]],
    theta_cr: float,
    rho_0_cr: float,
    cr_grid_export: np.ndarray,
    prior_pdf_per_t: dict[str, list[float]],
    forecast_times: np.ndarray,
    n_samples: int,
    seed: int,
    desc: str = "nested prior MC",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run the nested-FORM MCS on the **unconditional** spatial cr-field.

    Mirrors :func:`run_nested_mcs` but without obs conditioning. The cr field
    is drawn via the prior Cholesky (kernel ``rho_0_cr + (1-rho_0_cr) *
    exp(-(d/theta_cr)**2)``); each ``z_i`` is N(0, 1) marginal so
    ``xi_i == z_i`` directly. The design point ``(beta_T_t, alpha_cr_t,
    alpha_basic_t)`` is the **same at every section** for a given t (prior
    stationarity), broadcast along the section axis at evaluation time.

    Returns ``(n_fail_section, n_fail_system, precompute)`` with the same
    shapes as :func:`run_nested_mcs`.
    """
    n_sections = len(x)

    cr_values = np.array([float(p["point"]["corrosion_rate"]) for p in points])
    order = np.argsort(cr_values)
    cr_values = cr_values[order]
    betas = np.array([float(points[i]["beta"]) for i in order])
    var_names = list(points[0]["alphas"].keys())
    alpha_full = np.array([
        [float(points[i]["alphas"][v]) for v in var_names] for i in order
    ])
    active_mask = np.any(np.abs(alpha_full) > 1e-6, axis=0)
    active_vars = [v for v, m in zip(var_names, active_mask) if m]
    alpha_table = alpha_full[:, active_mask]
    n_active = len(active_vars)

    precompute = compute_nested_form_prior(
        cr_grid_export=cr_grid_export,
        prior_pdf_per_t=prior_pdf_per_t,
        forecast_times=forecast_times,
        cr_values_frag=cr_values,
        betas_frag=betas,
        alpha_table=alpha_table,
        active_vars=active_vars,
    )
    beta_T_all = precompute["beta_T"]              # (n_t,)
    alpha_cr_all = precompute["alpha_cr"]          # (n_t,)
    alpha_basic_all = precompute["alpha_basic"]    # (n_t, n_active)

    # Cholesky of the unconditional cr field along the wall (one factor,
    # shared across t — kernel doesn't depend on t).
    C_cr = spatial_covariance(x, theta_cr, rho_0_cr)
    L_cr = np.linalg.cholesky(C_cr + 1e-8 * np.eye(n_sections))

    # Cholesky of each basic-variable field (same machinery as posterior).
    L_by_var = _per_variable_chol(x, spec_basic, active_vars)
    spatial_idx = [j for j, v in enumerate(active_vars) if L_by_var[v] is not None]
    uniform_idx = [j for j, v in enumerate(active_vars) if L_by_var[v] is None]
    n_spatial = len(spatial_idx)
    n_uniform = len(uniform_idx)
    if n_spatial:
        L_basic_stack = np.stack(
            [L_by_var[active_vars[j]] for j in spatial_idx], axis=0,
        )
        L_basic_T = L_basic_stack.swapaxes(-1, -2)  # (n_spatial, N, N)
    else:
        L_basic_T = None

    rng = np.random.default_rng(seed)
    n_t = len(forecast_times)
    n_fail_section = np.zeros((n_t, n_sections), dtype=np.int64)
    n_fail_system = np.zeros(n_t, dtype=np.int64)

    pbar = tqdm(forecast_times, desc=desc, unit="t", dynamic_ncols=True)
    for ti, t in enumerate(pbar):
        # 1) Unconditional cr field. Each xi_i ~ N(0, 1), correlated by L_cr.
        Z_cr = rng.standard_normal((n_samples, n_sections))
        xi_field = Z_cr @ L_cr.T                                # (n_s, N)

        # 2) Basic u-fields (same as posterior).
        U_basic = np.empty((n_active, n_samples, n_sections))
        if n_spatial:
            Z_spatial = rng.standard_normal((n_spatial, n_samples, n_sections))
            U_basic[spatial_idx] = Z_spatial @ L_basic_T
        if n_uniform:
            Z_uniform = rng.standard_normal((n_uniform, n_samples))
            U_basic[uniform_idx] = Z_uniform[..., None]

        # 3) Tangent-hyperplane LSF. (beta_T_t, alpha_cr_t, alpha_basic_t)
        #    are scalars/1D vectors — broadcast across sections.
        beta_T_t = float(beta_T_all[ti])
        alpha_cr_t = float(alpha_cr_all[ti])
        alpha_basic_t = alpha_basic_all[ti]                     # (n_active,)
        # einsum: 'v,vsn->sn' (broadcast alpha_v across samples and sections)
        Y_basic = np.einsum("v,vsn->sn", alpha_basic_t, U_basic)
        G = beta_T_t - alpha_cr_t * xi_field - Y_basic          # (n_s, N)

        sec_fail = G < 0
        n_fail_section[ti] += sec_fail.sum(axis=0).astype(np.int64)
        n_fail_system[ti] += int(sec_fail.any(axis=1).sum())

        pf_sys = n_fail_system[ti] / n_samples
        pbar.set_postfix(t=f"{float(t):.1f}", Pf=f"{pf_sys:.2e}")
    pbar.close()

    return n_fail_section, n_fail_system, precompute
