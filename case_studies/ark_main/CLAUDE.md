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

Two reliability legs:

1. **Prior leg** — pure parametric: integrate a cached `Pf(cr, section)` grid
   against the time-varying prior cr-PDF from the corrosion model.
2. **Posterior leg** — field-sampling MCS where cr varies along the wall as
   a Kriged Gaussian field anchored on observations at `x = 0` and decaying
   to the prior at far x via the cr spatial kernel.

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
  "wall": { "theta": 200.0, "rho_0": 0.0 },
  "cr":   { "theta":  50.0, "rho_0": 0.0 },
  "per_variable": {}
}
```

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
│       ├── pf_grid.json           # prior leg cache (Pf at every cr-point)
│       └── posterior_grid.json    # posterior leg cache (per t_obs)
├── results/
│   └── <setup_signature>/
│       ├── summary.json
│       ├── forecasts/
│       │   ├── prior.json
│       │   └── posterior.json     # per t_obs block
│       └── plots/
│           ├── pf_vs_cr.png
│           ├── pf_vs_time.png
│           ├── pf_vs_time_posterior.png
│           ├── beta_along_wall.png
│           ├── realizations.png
│           ├── beta_forecast_system/   # PNG per t_obs
│           ├── beta_forecast_system.pdf
│           ├── beta_forecast_system.gif
│           ├── cr_along_wall/          # PNG per t_obs
│           ├── cr_along_wall.pdf
│           ├── cr_along_wall.gif
│           ├── cr_violin/              # PNG per t_obs
│           ├── cr_violin.pdf
│           └── cr_violin.gif
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

Both `pf_grid.json` and `posterior_grid.json` are validated against:

- LSF name, `n_samples`, `seed`, `n_sections`, `L`.
- `wall_theta`, `wall_rho_0`, and (posterior) `cr_theta`, `cr_rho_0`.
- Fragility fingerprint (`cr_values`, `betas`).
- (Posterior) `obs_times`.

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

# Spatial pipeline (reads spatial_settings.json)
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
```

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
- Sweep at **100k samples** is the current reference (10k → 100k changed
  results by ~3%, matching √10 noise scaling).
- Settings come from `spatial_settings.json` only — no CLI flags. The
  sweep driver passes dicts to `analyze(cfg)` programmatically.
- Output folders are signature-named, not timestamped — same config →
  same folder, hit caches, overwrite results.
- Per-section beta-forecast plot for the **system** uses the same renderer
  as the per-section one (`plotting/timeline.plot_beta_forecast_at_time`),
  just fed with system arrays reshaped into the per-obs-time dict shape.

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
