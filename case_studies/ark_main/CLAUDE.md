# ark_main — context for resuming work

Notes for continuing the ARK sheet-pile-wall case study from a fresh session
(e.g. a different machine). Focus is the **spatial-variability pipeline** —
that's where the recent work has been. The per-section pipeline in `run.py`
is treated as upstream and stable.

---

## Big picture

The case study is a 1 km sheet-pile wall, modelled as 11 (or 21) cross
sections at 50 m / 100 m spacing. Every section shares the **same** cached
FORM fragility curve (`<remote>/output/fragility_curve_<lsf>/point_*.json`),
parameterised by a single scalar `corrosion_rate` (cr ∈ [0, 1]).

Two reliability legs (each runnable in either the **field** or **nested**
method — see "Posterior leg methods" below; both legs use the same method):

1. **Prior leg** — no observations.
   - `field`: integrate a cached `Pf(cr, section)` grid against the time-
     varying prior cr-PDF (uniform-cr-per-sample assumption).
   - `nested`: nested-FORM tangent-hyperplane MCS on the **unconditional**
     cr-field — cr varies along the wall under the prior covariance.
2. **Posterior leg** — observations at `x = 0`.
   - `field`: per-sample fragility interpolation on the Kriged-conditional
     cr-field.
   - `nested`: nested-FORM tangent-hyperplane MCS on the Kriging-conditional
     cr-field.

The fragility points and corrosion observations are read from
`<remote>/output/fragility_curve_<lsf>/` and `<remote>/input/data.json`
respectively. The wall is **homogeneous**: there is only one fragility curve
shared by all sections.

---

## Path conventions (Windows ↔ Mac)

- Remote share root comes from `case_studies/ark_main/.env` →
  `get_remote_path(_ENV)`. Same key on both OSes; just point it at wherever
  the SMB / mounted share lives on the Mac.
- Outputs are written under `<remote>/output/...`; inputs under
  `<remote>/input/...`. Nothing in the code is Windows-specific except the
  retry-on-SMB-PermissionError logic in `spatial/checkpoint.py`
  (`_atomic_replace` with exponential backoff — harmless on POSIX where the
  rename just succeeds first try).
- Module entry points are run from the **repo root**, e.g.
  `python -m case_studies.ark_main.run_spatial` (not from inside `ark_main/`).
- Unicode in printed output: Windows `cp1252` chokes on `→`, `∈`, `φ`. The
  code now uses ASCII (`->`, `..`) in prints; CSVs use `utf-8-sig`. On the
  Mac you can use UTF-8 freely, but **don't regress the existing files** —
  keep the ASCII / utf-8-sig choices in place so Windows runs stay clean.

---

## Configuration files

### `<remote>/input/spatial_settings.json` — single source of truth for the spatial pipeline

```json
{
  "lsf_name": "lsf_wall",
  "n_samples": 100000,
  "seed": 42,
  "L": 1000.0,
  "n_sections": 11,
  "mcs_method": "field",
  "wall": { "theta": 200.0, "rho_0": 0.0 },
  "cr":   { "theta":  50.0, "rho_0": 0.0 },
  "per_variable": {}
}
```

- `mcs_method` — `"field"` (default) or `"nested"`. Selects the MCS
  method for **both** legs (prior + posterior). The two methods write to
  different cache files in the same signature folder (`pf_grid.json` +
  `posterior_grid.json` for field; `pf_grid_nested.json` +
  `posterior_grid_nested.json` for nested) so they coexist. See "Method
  flag — what gets swapped" below for the per-leg breakdown.
  - Legacy key `posterior_method` is still accepted (emits a
    `DeprecationWarning`) for back-compat with older settings files; remove
    it once you've renamed the keys.

- `wall.{theta, rho_0}` — basic-variable squared-exp kernel
  `C(d) = rho_0 + (1 − rho_0) · exp(−(d/theta)²)`. Applied to **soil**
  variables only (see classification below).
- `cr.{theta, rho_0}` — same kernel form, separate parameters, for the
  corrosion-rate field. **Currently `rho_0 = 0`** so the cr field decays
  cleanly to the prior at distances ≫ θ_cr (no floor).
- `per_variable: { var_name: { "theta": ..., "rho_0": ... } }` — overrides
  for individual variables. Empty in current setup. If you populate it, the
  setup signature gains a `pv-<8-hex>` suffix.
- **No CLI flags**. Programmatic overrides pass a dict to `analyze(cfg)`
  (used by `run_spatial_sweep.py`).

### Soil vs non-soil classification

`spatial/covariance.py::is_soil_variable(name)` → `"soil" in name.lower()`.

- **Soil variables** (`Klei_soilphi`, `Zand_soilgamwet`, …): get a spatial
  field with the wall kernel. Decorrelate along the wall as θ_wall shrinks.
- **Non-soil variables** (`uniform_load_left`, `phreatic_level`,
  `canal_level`, `model_factor_M/F`, `Wall_SheetPilingElementEI`): forced to
  `rho_0 = 1` (perfect spatial correlation) — drawn as one scalar per MC
  sample and broadcast to every section. Encoded by `resolve_config` in
  `covariance.py`.

### `<remote>/input/settings.json` / `settings_full.json`

The full set of FORM variables for the LSFs. `settings.json` is the
sensitivity-pruned set actually used by `build_fragility`. `settings_full.json`
is the un-pruned set used by `io/export_wall_params.py` to dump distribution
tables.

### `<remote>/input/data.json`

Corrosion observations `{ time, corrosion }`. Read by `run_spatial.py` for
the obs marker on `cr_along_wall*.png` plots.

### `<remote>/output/cr_pdfs_<lsf>.json`

Output of `io/export_cr_pdfs.py`. Holds:

- `cr_grid` — common fine grid.
- `forecast_times` — integer grid + obs times.
- `prior_pdf_per_t` — `{ "t.4f": [pdf on cr_grid] }`.
- `posterior_pdf_per_obs` — `{ "t_obs": { "t": [pdf] } }`, one block per
  obs time, only `t >= t_obs`.

The spatial pipeline **reads this file** rather than re-deriving the cr PDFs
from `models/corrosion.py` — one source of truth. Regenerate after any
change to the corrosion model or `data.json`:

```
python -m case_studies.ark_main.io.export_cr_pdfs --lsf-name lsf_wall
```

---

## Output layout

```
<remote>/output/spatial_analysis/
├── cached/
│   └── <setup_signature>/
│       ├── pf_grid.json                  # prior leg cache,     method=field  (Pf at every cr-point)
│       ├── pf_grid_nested.json           # prior leg cache,     method=nested (per-t fail counts)
│       ├── posterior_grid.json           # posterior leg cache, method=field
│       └── posterior_grid_nested.json    # posterior leg cache, method=nested
├── results/
│   └── <setup_signature>/
│       ├── summary.json
│       ├── forecasts/
│       │   ├── prior.json
│       │   └── posterior.json     # per t_obs block
│       └── plots/
│           ├── pf_vs_cr.png             # method=field only (no Pf(cr) table in nested)
│           ├── pf_vs_time.png
│           ├── pf_vs_time_posterior.png
│           ├── beta_along_wall.png
│           ├── realizations.png         # method=field only (g(x) at cr=0)
│           ├── beta_forecast_system/   # PNG per t_obs
│           ├── beta_forecast_system.pdf
│           ├── beta_forecast_system.gif
│           ├── cr_along_wall/          # PNG per t_obs
│           ├── cr_along_wall.pdf
│           ├── cr_along_wall.gif
│           ├── cr_violin/              # PNG per t_obs
│           ├── cr_violin.pdf
│           ├── cr_violin.gif
│           ├── alpha_heatmap/          # PNG per t_obs    (method=nested only)
│           ├── alpha_heatmap.pdf       #                   (method=nested only)
│           ├── alpha_heatmap.gif       #                   (method=nested only)
│           ├── alpha_lines/            # PNG per t_obs    (method=nested only)
│           ├── alpha_lines.pdf         #                   (method=nested only)
│           ├── alpha_lines.gif         #                   (method=nested only)
│           └── alpha_lines_prior.png   # single panel     (method=nested only, prior leg)
└── sweep/                         # outputs of analyze_sweep.py
    ├── beta_forecast_grid.png
    └── beta_contour_tend.png
```

### Setup signature

`spatial/__init__.py::setup_signature(s)` produces:

```
lsf-<tag>_N<n_samples>_seed<seed>_L<int(L)>_n<n_sections>_wth<int>_wrh<g>_crth<int>_crrh<g>[_pv-<sha>]
```

- Strips `lsf_` prefix so it reads `lsf-wall_…`, not `lsf-lsf_wall_…`.
- Floats use `:g` (drops trailing zeros), so `rho_0 = 0.0 → "0"`.
- Identical configs land in the same folder → caches hit and the results
  folder is overwritten in place. Different θ_wall / θ_cr → different folder.

### Cache invalidation

All four caches (`pf_grid.json`, `pf_grid_nested.json`,
`posterior_grid.json`, `posterior_grid_nested.json`) are validated against:

- LSF name, `n_samples`, `seed`, `n_sections`, `L`.
- `wall_theta`, `wall_rho_0`.
- `cr_theta`, `cr_rho_0` (posterior caches always; the nested prior cache
  also depends on the cr kernel because it samples the cr-field directly;
  the field-prior cache `pf_grid.json` does **not** depend on `cr_*`).
- Fragility fingerprint (`cr_values`, `betas`).
- `obs_times` (posterior caches), `forecast_times` (nested-prior cache).
- `method` is encoded in the file name, not in the validator —
  `"field"` writes `pf_grid.json` + `posterior_grid.json`, `"nested"` writes
  `pf_grid_nested.json` + `posterior_grid_nested.json`. The pairs coexist;
  flipping the flag picks the matching pair (or recomputes if absent).

Any mismatch → recompute. See `spatial/checkpoint.py`.

---

## Module map (`case_studies/ark_main/`)

### Top-level scripts

| File | Purpose |
|------|---------|
| `run.py` | Per-section reliability pipeline (FragilityPipeline). Unchanged. Upstream of spatial. |
| `analysis/run_mc.py` | Per-section MCS with checkpointing + convergence plots + postprocess flag. |
| `run_spatial.py` | **Spatial pipeline entry point.** `analyze()` orchestrates prior + posterior legs, writes plots/forecasts/summary. |
| `run_spatial_sweep.py` | θ_wall × θ_cr sweep driver. Loops over `(theta_wall, theta_cr)` combos calling `analyze(cfg)`. Defaults: `(50, 200, 800) × (50, 200, 800)` m. Prints a summary table at the end. |
| `analysis/analyze_fragility.py`, `analysis/check_lsf.py`, `analysis/eval_g_at_mean.py`, `analysis/run_sensitivity.py` | Per-section analyses. Touched lightly, not part of the spatial work. |

### `spatial/` (the new modules)

| File | Purpose |
|------|---------|
| `__init__.py` | `setup_signature()`, `cache_dir()`, `results_dir()` helpers. |
| `engine.py` | `compute_or_load_pf_grid(spatial_settings)` — prior-leg MCS. Effective Cholesky collapse `C_eff = Σ_v α_v² · C_v`. Common random numbers across cr-points so `Pf(cr)` is smooth. |
| `posterior_mcs.py` | `run_posterior_mcs(...)` — field-sampling MCS per obs scenario. Per-variable Cholesky for soil vars; scalar broadcast for non-soil vars. Per-section FORM coefficients `(β_i, α_i)` interpolated from the fragility cache against the realised local cr. |
| `nested_mcs.py` | Nested-FORM tangent-hyperplane MCS for **both legs**. `run_nested_mcs(...)` — per-obs-scenario posterior sampler: one nested-FORM linearisation per (section, t) using Kriging-conditional cr-field moments. `run_nested_mcs_prior(...)` — unconditional prior sampler: one (β_T, α_cr, α_basic) triple per t (prior is stationary along the wall), tangent-hyperplane MCS on the prior cr-field. Both selected by `mcs_method: "nested"`. |
| `cr_field.py` | Nataf + Kriging utilities. `cdf_on_grid`, `inv_cdf_at`, `z_moments_under_posterior`, `kriged_chol`, `sample_cr_field`, `conditional_cr_quantiles` (closed-form). |
| `covariance.py` | `spatial_covariance(x, theta, rho_0)`, `effective_cholesky`, `resolve_config`, `resolve_cr_config`, `is_soil_variable`. |
| `integration.py` | `over_cr(...)` — integrate `Pf(cr)` against a cr_pdf via trapezoidal rule. Uses `np.trapezoid` (NumPy 2.x; `np.trapz` was removed). |
| `fragility.py` | `load_points(remote, lsf_name)` — read & sort converged `point_*.json` files. `pick_cr_point`. |
| `plots.py` | `pf_vs_cr`, `pf_vs_time`, `pf_vs_time_posterior`, `section_beta`, `cr_along_wall`, `cr_along_wall_violin`, `realizations`. All write via `src.plotting.save_figure`. |
| `checkpoint.py` | `try_load`/`save` for `pf_grid.json`, `try_load_posterior`/`save_posterior` for `posterior_grid.json`. Atomic-rename retry for SMB share. |
| `analyze_sweep.py` | Reads every combo's `forecasts/{prior,posterior}.json` and renders the two sweep plots. **EARLIEST** obs scenario is used in `beta_forecast_grid` (the latest obs is a single-point curve). |

### `io/`

| File | Purpose |
|------|---------|
| `export_cr_pdfs.py` | Produces `<remote>/output/cr_pdfs_<lsf>.json`. Run after any change to the corrosion model or observation data. |
| `export_wall_params.py` | Dumps distribution table CSV. Supports `--lsf <name>` to filter to variables with non-zero α in that LSF's fragility cache. Pretty labels (`Klei φ (°)`), 2-decimal formatting, `utf-8-sig` for Excel/Greek-char compatibility. |
| `generate_obs.py` | Synthetic observation generator (unchanged). |

### `models/`, `reliability/`, `plotting/`

Upstream of the spatial work. The spatial pipeline reuses:

- `reliability/build_fragility.py::load_settings()` — reads `settings.json`.
- `plotting/timeline.py::plot_beta_forecast_at_time` — same renderer
  `run.py` uses; the spatial pipeline reshapes its prior/posterior arrays
  into the per-obs-time dict shape that function expects.
- `src.plotting.{save_figure, make_gifs, collect_pngs_to_pdf}`.

---

## How to run things

```bash
# repo root
cd /path/to/ms5-asset-performance

# Spatial pipeline (reads spatial_settings.json — mcs_method picks
# "field" or "nested")
python -m case_studies.ark_main.run_spatial

# θ_wall × θ_cr sweep (9 combos by default, ~45 min @ 100k samples)
python -m case_studies.ark_main.run_spatial_sweep

# Aggregate the sweep into 2 plots
python -m case_studies.ark_main.spatial.analyze_sweep

# Refresh the cr_pdfs file (after touching corrosion model or data.json)
python -m case_studies.ark_main.io.export_cr_pdfs --lsf-name lsf_wall

# Distribution-table CSVs
python -m case_studies.ark_main.io.export_wall_params --lsf lsf_wall
python -m case_studies.ark_main.io.export_wall_params --lsf lsf_wall_anchor
python -m case_studies.ark_main.io.export_wall_params       # un-pruned table

# --- Mac dev (no D-SheetPiling install) ---
# Generate a realistic-looking mock fragility cache + cr_pdfs + data.json
# under mock/ (the case-study .env points REMOTE_PATH at mock/ already):
python -m case_studies.ark_main.mock.generate_mock_fragility
# Then run_spatial works end-to-end against the mock inputs.
```

### Mock data for Mac dev (`mock/generate_mock_fragility.py`)

The fragility cache requires D-SheetPiling FORM runs that don't ship on the
Mac. The mock generator fills the three things the spatial pipeline reads:

- `mock/output/fragility_curve_lsf_wall/point_*.json` + `manifest.json` —
  18 cr-points with monotone-decreasing `β(cr)` from ~4 at cr=0 to ~0.3 at
  cr=1, unit-norm α dominated by `model_factor_M`, mild α drift with cr.
- `mock/output/cr_pdfs_lsf_wall.json` — power-law corrosion
  `cr_mm(t) = A·t^B` with `A ~ logN(log 0.15, 0.35²)`, `B ~ N(0.7, 0.07²)`,
  5 000 prior samples reweighted by Gaussian likelihood per obs.
- `mock/input/data.json` — three synthetic obs at `t = 10 / 20 / 30`.

Rerun the generator any time you want to change the fragility shape or
add/remove obs. Outputs land directly where the pipeline expects them
because `.env`'s `REMOTE_PATH` already points at `case_studies/ark_main/mock/`.

---

## Method recap (for resuming the model side)

### FORM linearisation (prior leg, `engine.py`)

Per cached fragility point `cr_k` with `(β_k, α_k(v))`:

- Per section `i`, LSF in u-space: `g_i = β_k − α_k · U(x_i)`.
- Each basic variable `v` has its own spatial field `U_v(x) = L_v · Z_v`
  with covariance `C_v(d) = rho_0_v + (1 − rho_0_v) · exp(−(d / θ_v)²)`.
- Effective covariance of `Y(x) = α · U(x)`:
  `C_eff(x, x') = Σ_v α_v² · C_v(|x − x'|)`. One Cholesky per cr-point;
  the inner MC loop is one `Z @ L_eff.T` to draw `Y` directly.
- Common random numbers across cr-points: `Pf(cr)` is smooth-ish.

### Nataf + Kriging (posterior leg, `cr_field.py`)

- `z(x) := Φ⁻¹(F_prior(cr(x); t))`. By construction `z` is N(0,1) marginal
  under the prior at every section.
- Covariance: `C_cr(d) = rho_0_cr + (1 − rho_0_cr) · exp(−(d/θ_cr)²)`.
- At the obs location `x_0`, the file's posterior PDF integrates to
  `(m_post, v_post)` of `z(x_0)` under the posterior.
- Kriging conditional moments at section `i`:
  - `E[z_i | obs] = ρ_i · m_post` where `ρ_i = C_cr(|x_i − x_0|)`.
  - `Cov[z_i, z_j | obs] = ρ_i · ρ_j · v_post + (ρ_ij − ρ_i · ρ_j)`.
- Per MC sample: draw `z_field = ρ · m_post + L_z @ Z`, transform back
  `cr_i = F_prior⁻¹(Φ(z_i); t)`.
- Per `(sample, section)`: linearly interpolate `(β_i, α_i)` from the
  cached fragility against the realised local `cr_i`. Then sample basic-
  variable u-fields **independently** of the cr field, compute
  `g_i = β_i − Σ_v α_i_v · U_v(x_i)`, tally section + system failures.

### Approximations being made

1. α interpolation does not preserve unit norm — `|α_i| ≠ 1` in general.
   `P(section i fails) = P(Y_i > β_i)` then has `Var[Y_i] = |α_i|²`,
   deviating slightly from `Φ(−β_i)`. Optional renorm
   `α_i ← α_i / |α_i|`, `β_i ← β_i / |α_i|` not implemented — defer until
   we see it bite.
2. Nataf reference marginal is the prior — keeps the spatial field
   stationary. The posterior shows up only via `(m_post, v_post)` at `x_0`.
3. cr field and basic-variable u-fields are sampled independently;
   they couple only at the fragility lookup. Consistent with the FORM
   construction (cached α is the design-point direction conditional on cr).
4. No cross-variable u-correlation within a section — same as `engine.py`.

---

## Method flag — what gets swapped (`mcs_method`)

One flag toggles both legs. Both legs always run, and both use the chosen
method. (The flag was previously named `posterior_method` when it only
gated the posterior leg; the legacy key still works with a deprecation
warning.)

### `"field"` (default)

| Leg       | Module                     | What it does |
|-----------|----------------------------|--------------|
| Prior     | `spatial/engine.py`        | Spatial MCS at each cached cr-point (β_k, α_k) via the **uniform-cr-per-sample** Cholesky `L(α_k)`; then `Pf(cr_k) → ∫ Pf(cr) · prior_pdf(cr; t) dcr` per t. |
| Posterior | `spatial/posterior_mcs.py` | Per MC sample, per section: draw local cr from the **Kriging-conditional** cr-field, interpolate `(β(cr_i), α(cr_i))` from the fragility, evaluate `g_i = β_i − Σ_v α_iv · U_v(x_i)`. |

Pointwise-exact use of the fragility curve. Prior cr is uniform within a
sample (only basic vars vary spatially); posterior cr varies along the wall.

### `"nested"` — `spatial/nested_mcs.py` (`run_nested_mcs_prior` + `run_nested_mcs`)

Treats cr as one more basic random variable via Nataf:
`z = Φ⁻¹(F_prior(cr; t))`. The 1-D nested-FORM search

```
ξ* = argmin  ξ²  +  β(cr(ξ))²        with cr(ξ) = F_prior⁻¹(Φ(μ_z + σ_z · ξ))
```

yields

```
β_T       = sqrt(ξ*²  +  β(cr*)²)
α_cr      = − ξ* / β_T
α_basic   = (β(cr*) / β_T) · α(cr*)          (with α_cr² + Σ α_basic² = 1)
```

The MCS then evaluates the **tangent-hyperplane LSF** with no per-sample
fragility interpolation:

```
g_i = β_T  −  α_cr · ξ_i  −  Σ_v α_basic_v · U_v(x_i)
```

Per-leg differences:

| Leg       | `(μ_z, σ_z)`                                  | (β_T, α_*) shape | cr-field sampler                              |
|-----------|-----------------------------------------------|------------------|-----------------------------------------------|
| Prior     | `(0, 1)` everywhere (stationary along wall)   | `(n_t,)`         | Unconditional Cholesky of cr kernel along x.  |
| Posterior | `(ρ_i · m_post, 1 + ρ_i²(v_post − 1))`         | `(n_t, n_sec)`   | Kriging-conditional Cholesky.                 |

Same technique, same model on both legs — the **only** difference is
whether the cr field is conditioned on observations. This is why nested-mode
prior and posterior Pf curves are directly comparable: a β reduction
between prior and posterior is now attributable to observations, not to a
method switch.

#### What "nested" buys / loses

- ✓ Faster inner loop — one matmul per t, no per-sample interp. Prior leg
  at N=5k / 51 t-steps / 11 sections ran in <0.1 s on Mac.
- ✓ Consistent technique across legs — see table above. β_prior and β_post
  differ only because the cr-field is conditioned, not because the integrator
  changed.
- ✓ Exposes an explicit α per (section, t) for **every** variable including
  cr — see `alpha_heatmap` / `alpha_lines` (posterior, per (t, section))
  and `alpha_lines_prior` (single panel, since the prior is stationary).
- ✗ Linearises the fragility surface at the design point `cr*` instead of
  evaluating it per realised `cr_i`. Prior leg also changes models, not
  just method: it goes from uniform-cr-per-sample to a spatially-varying
  cr field, so nested-prior Pf is generally **higher** than field-prior Pf
  (more system-failure opportunities when cr is uncorrelated along the
  wall). Smoke test on mock data (N=5k, θ_cr=50 m, ρ_0_cr=0): nested-prior
  Pf_sys(t=50) = 3.1 × 10⁻² vs field-prior 1.8 × 10⁻² (~75% higher).

#### Cache layout

`pf_grid.json` + `posterior_grid.json` (field) and `pf_grid_nested.json` +
`posterior_grid_nested.json` (nested) coexist in the same signature folder;
the cache validator includes a `method` field. Each nested cache block
carries the per-(section, t) (or per-t for the prior) `beta_T`, `alpha_cr`,
`alpha_basic`, `active_vars`, `xi_star`, `cr_star`, `mu_z`, `sigma_z` so
the α plots render on cache hits without recomputing.

---

## Plot details worth remembering

- **`pf_vs_time.png`** — system curve is **smoother than per-section** in
  the same panel because the system event count is `~7×` larger than the
  per-section count (series inflation), not because of any extra averaging.
- **`pf_vs_time_posterior.png`** — one viridis curve per obs scenario on
  `[t_obs, t_end]`, prior reference in black. `+` marker at each curve's
  leftmost point.
- **`beta_forecast_system/`** — one PNG per obs time, then bundled into
  PDF + GIF by `collect_pngs_to_pdf` / `make_gifs` exactly like `run.py`'s
  per-section beta-forecast plot but driven by **system** β.
- **`cr_along_wall/`** — closed-form prior 90% band (horizontal) + posterior
  90% band (varies along x). Y-axis is **shared** across the GIF: it's set
  from the prior q95 at `t_end` and the largest obs + 1.96σ noise envelope.
- **`cr_violin/`** — KDE split-violin per section. Single normalisation
  factor across the whole figure so the obs-induced narrowing reads
  visually. Sometimes the prior is a near-singular pile of mass at cr ≈ 0
  early in time — the spike fallback in
  `plots.cr_along_wall_violin` handles `std < 1e-9`.
- **Sweep `beta_forecast_grid.png`** — one subplot per θ_wall (3 panels),
  one viridis curve per θ_cr (3 colours per panel). Uses the **EARLIEST**
  obs scenario because the latest obs is single-point (degenerate visually).
- **`alpha_heatmap/`** (nested only) — one PNG per obs scenario. Grid of
  per-variable heatmaps, x=t, y=section, color=signed α (RdBu_r, symmetric
  vmax shared across panels in the figure). `cr` panel is first; the
  remaining panels are the active basic variables in fragility-cache order.
  Dotted vertical line marks `t_obs`. Watch the `cr` panel: near `x = 0` it
  stays pale (obs pins cr); far from `x = 0` it darkens as the prior cr
  takes over and its share of the unit-norm budget grows.
- **`alpha_lines/`** (nested only) — one PNG per obs scenario. Three
  subplots (first / middle / last section), each plotting `α_v(t)` as
  lines: thick dashed black for cr, viridis-ramped colours for the basic
  variables (same colour per variable across all subplots and across obs
  scenarios). y-axis shared across subplots; right-most subplot carries
  the figure legend.
- **`alpha_lines_prior.png`** (nested only) — single panel, `α_v(t)` for
  the unconditional prior leg. No section dimension because the prior is
  stationary along the wall, so the design point only depends on `t`. Same
  colour palette as `alpha_lines/`; the cr curve is what `α_cr_i` collapses
  to at sections far past `θ_cr` in the posterior plots (Kriging at far
  distance == prior).

---

## Recent decisions (don't re-litigate)

- `cr.rho_0 = 0` and `wall.rho_0 = 0` (no spatial floor) — pure exp-squared
  decay. Earlier defaults of `0.3` masked the far-x convergence to the
  prior; the user explicitly fixed both.
- Non-soil variables (loads, water, model factors, wall stiffness) are
  treated as **uniform along the wall**. Encoded by `is_soil_variable()`
  and `rho_0 = 1` in `spec`. Posterior MCS draws one scalar per MC sample
  per non-soil variable and broadcasts to all sections.
- Posterior leg uses **field MCS**, not parametric `pf_grid` integration.
  The parametric grid assumes uniform cr across the wall and can't
  represent a spatially-varying cr field for the system event.
- **Both legs share a method**: `mcs_method` flag governs prior + posterior,
  not just posterior. The user wanted "consistent calculation of beta using
  the same method" — picking method=nested on the posterior while the prior
  stayed on field-integration mixed two different models/techniques and made
  β reductions hard to attribute. So nested-mode was extended to the prior
  too: same Nataf+nested-FORM machinery, only the cr-field conditioning
  differs (unconditional Cholesky for prior, Kriging for posterior). The
  flag was renamed from the old `posterior_method` to `mcs_method` to
  reflect this scope; the old key is still accepted with a deprecation
  warning so existing settings files keep working.
- Sweep at **100k samples** is the current reference (10k → 100k changed
  results by ~3%, matching √10 noise scaling).
- Settings come from `spatial_settings.json` only — no CLI flags. The
  sweep driver passes dicts to `analyze(cfg)` programmatically.
- Output folders are signature-named, not timestamped — same config →
  same folder, hit caches, overwrite results.
- Per-section beta-forecast plot for the **system** uses the same renderer
  as the per-section one (`plotting/timeline.plot_beta_forecast_at_time`),
  just fed with system arrays reshaped into the per-obs-time dict shape.
- **Method coexistence**: `"field"` (default) and `"nested"` are both kept
  rather than replacing one with the other. Different cache files in the
  same signature folder (`pf_grid.json` + `posterior_grid.json` vs
  `pf_grid_nested.json` + `posterior_grid_nested.json`). Nataf reference
  marginal stays the **prior** for nested-mode (same convention for prior
  leg `(μ_z=0, σ_z=1)` and posterior leg `(μ_z=ρ·m_post, σ_z=…)`), so the
  spatial covariance bookkeeping is identical across the two legs.

---

## Known issues / gotchas

- **NumPy 2.x**: `np.trapz` is gone, use `np.trapezoid` (already done in
  `spatial/integration.py` and `cr_field.py`).
- **SMB share atomic rename**: `tmp.replace(p)` occasionally hits
  `PermissionError` on Windows-mounted SMB. `checkpoint.py::_atomic_replace`
  retries with exponential backoff. On the Mac this is usually unnecessary;
  leave the retry in — it's idempotent.
- **`gaussian_kde` rejects (near-)constant samples**: the violin code
  spike-fallback (`if col.std() < 1e-9`) handles this — happens at very
  early forecast times.
- **`float(array_with_one_element)` is a NumPy deprecation** — use `.item()`.
- **Module path**: `python -m case_studies.ark_main.X` requires repo root
  as cwd (the case-study dir is on `sys.path` via `_ARK` insert).
- **Folder name duplication**: `setup_signature` strips the `lsf_` prefix
  so we get `lsf-wall_...` not `lsf-lsf_wall_...`.
- **Posterior cache is independent** of `pf_grid.json`. Changing only the
  cr kernel invalidates `posterior_grid.json` but the prior leg hit-caches.
- **`obs_times` mismatch invalidates the posterior cache** — if `data.json`
  gets a new observation, the cache rightfully recomputes.

---

## Open / potential next steps

(Nothing the user has explicitly queued — this is a parking lot for ideas
the conversation surfaced.)

- Optional `α_i` renormalisation in `posterior_mcs.py` if the interpolated
  norms drift far from 1. Cheap; deferred until we see it matter.
- `beta_along_wall_posterior.png` — overlay per-section β at one t for each
  obs scenario. Mentioned in the original plan, never implemented.
- A "limit check" entry-point that runs the sweep at extremes
  (`θ_cr → ∞` should converge to the parametric posterior;
  `θ_cr → 0, rho_0_cr = 0` should converge to the prior at far x).
  Would be a clean regression test.
- The single-fragility-shared-across-sections assumption is baked in. If we
  ever needed per-section fragility (e.g. different geometry per section),
  it'd touch `engine.py::_run_mcs_over_cr_grid` and `posterior_mcs.py`'s
  per-section interp.

---

## Quick sanity checklist before running on the Mac

1. `.env` points at the right `<remote>` path.
2. `<remote>/output/fragility_curve_lsf_wall/point_*.json` exists.
3. `<remote>/output/cr_pdfs_lsf_wall.json` exists; regen via
   `io/export_cr_pdfs.py` if not.
4. `<remote>/input/spatial_settings.json` matches the values quoted above
   (or whatever the next experiment needs).
5. `pip install -r requirements.txt` (NumPy 2.x compatible — uses
   `np.trapezoid`). `scipy`, `tqdm`, `python-dotenv`, `matplotlib` required.
6. From repo root: `python -m case_studies.ark_main.run_spatial`. First
   run is the full ~5–10 min compute; subsequent runs at the same setup
   hit both caches in seconds.
