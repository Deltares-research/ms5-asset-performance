"""
Sweep theta_cr x rho_0_cr for the spatial-variability MCS pipeline.

Loads ``<remote>/input/spatial_settings.json`` as a base, then for each
``(theta_cr, rho_0_cr)`` pair overrides only those two fields and calls
``analyze`` programmatically. Every combo writes its own self-contained
``cached/<signature>/`` and ``results/<signature>/`` folder (signature
encodes both via the ``crth`` / ``crrh`` parts), so combos never overwrite
each other and identical re-runs hit the caches.

After the loop, prints a summary table of system Pf and beta at the final
forecast time, both prior and posterior (taking the last obs scenario).

Usage::

    python -m case_studies.ark_main.run_spatial_sweep
    python -m case_studies.ark_main.run_spatial_sweep --theta-cr 50 200 --rho0-cr 0 0.3
    python -m case_studies.ark_main.run_spatial_sweep --mcs-method nested --n-samples 5000

``--theta-cr`` / ``--rho0-cr`` set the sweep grid. The remaining scalar
flags override fields in the base config for the whole sweep (same flag
surface as ``run_spatial.py``, minus ``--cr-theta`` / ``--cr-rho0`` since
those are swept).
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

_ARK = Path(__file__).resolve().parent
sys.path.insert(0, str(_ARK))

# Importing ``run_spatial`` pulls dotenv + sys.path + load_settings via the
# spatial.engine chain.
from run_spatial import _apply_overrides, analyze
from spatial import results_dir as _results_dir
from src.io import get_remote_path

_ENV = _ARK / ".env"
_remote = get_remote_path(_ENV)


def _load_base(path: Path | None = None) -> dict:
    return json.load(open(path or (_remote / "input" / "spatial_settings.json")))


def _override(base: dict, theta_cr: float, rho_0_cr: float) -> dict:
    cfg = json.loads(json.dumps(base))                # deep copy
    cr = cfg.setdefault("cr", {})
    cr["theta"] = float(theta_cr)
    cr["rho_0"] = float(rho_0_cr)
    return cfg


def sweep(
    theta_cr_values: tuple[float, ...] = (50.0, 200.0, 800.0),
    rho_0_cr_values: tuple[float, ...] = (0.0, 0.3, 0.6),
    base: dict | None = None,
) -> None:
    if base is None:
        base = _load_base()
    combos = list(product(theta_cr_values, rho_0_cr_values))

    print(f"\n{'>' * 5}  Spatial-MCS sweep: {len(combos)} combinations  {'<' * 5}")
    print(f"  theta_cr: {list(theta_cr_values)}")
    print(f"  rho_0_cr: {list(rho_0_cr_values)}")
    print(f"  base config: lsf={base['lsf_name']}, "
          f"N={base['n_samples']}, seed={base['seed']}, "
          f"L={base['L']}, n_sections={base['n_sections']}, "
          f"wall.theta={base['wall']['theta']}, "
          f"wall.rho_0={base['wall']['rho_0']}")

    for i, (tc, rc) in enumerate(combos, 1):
        cfg = _override(base, tc, rc)
        print(f"\n{'#' * 70}")
        print(f"#  [{i}/{len(combos)}]  theta_cr = {tc:.0f} m   "
              f"rho_0_cr = {rc:g}")
        print(f"{'#' * 70}")
        analyze(cfg)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  SWEEP SUMMARY  (system metrics at t = t_end, "
          f"posterior from the latest obs scenario)")
    print(f"{'=' * 70}")
    header = "{:>10} | {:>10} | {:>11} | {:>11} | {:>11} | {:>11}".format(
        "theta_cr", "rho_0_cr", "Pf prior", "beta prior", "Pf post", "beta post"
    )
    print(header)
    print("-" * len(header))
    for tc, rc in combos:
        cfg = _override(base, tc, rc)
        rdir = _results_dir(_remote, cfg)
        prior_path = rdir / "forecasts" / "prior.json"
        post_path  = rdir / "forecasts" / "posterior.json"

        if prior_path.exists():
            prior = json.load(open(prior_path))
            pf_prior = float(prior["pf_system_t"][-1])
            b_prior = prior["beta_system_t"][-1]
            b_prior_str = "inf" if b_prior is None else f"{float(b_prior):.3f}"
            pf_prior_str = f"{pf_prior:.3e}"
        else:
            pf_prior_str = "n/a"
            b_prior_str = "n/a"

        if post_path.exists():
            post = json.load(open(post_path))
            last_obs = sorted(post["per_obs"].keys(), key=float)[-1]
            block = post["per_obs"][last_obs]
            pf_post = float(block["pf_system_t"][-1])
            b_post = block["beta_system_t"][-1]
            b_post_str = "inf" if b_post is None else f"{float(b_post):.3f}"
            pf_post_str = f"{pf_post:.3e}"
        else:
            pf_post_str = "n/a"
            b_post_str = "n/a"

        print("{:>10.0f} | {:>10g} | {:>11} | {:>11} | {:>11} | {:>11}".format(
            tc, rc, pf_prior_str, b_prior_str, pf_post_str, b_post_str
        ))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Sweep theta_cr x rho_0_cr for the spatial-variability MCS. "
            "Reads <remote>/input/spatial_settings.json as base; the scalar "
            "flags below override the base config for the whole sweep."
        ),
    )
    p.add_argument("--theta-cr", type=float, nargs="+", default=None,
                   help="cr-field correlation lengths (m). Default: 50 200 800")
    p.add_argument("--rho0-cr", type=float, nargs="+", default=None,
                   help="cr-field correlation floors. Default: 0.0 0.3 0.6")
    p.add_argument("--settings", type=Path, default=None,
                   help="Path to base spatial_settings.json "
                        "(default: <remote>/input/spatial_settings.json)")
    p.add_argument("--lsf-name", type=str, default=None,
                   help="LSF name override applied to every combo")
    p.add_argument("--n-samples", type=int, default=None,
                   help="MCS sample count override applied to every combo")
    p.add_argument("--seed", type=int, default=None,
                   help="MCS RNG seed override applied to every combo")
    p.add_argument("--L", type=float, default=None, help="Wall length (m)")
    p.add_argument("--n-sections", type=int, default=None,
                   help="Number of sections along the wall")
    p.add_argument("--mcs-method",
                   choices=["interpolate", "alphas", "field", "nested"],
                   default=None,
                   help="MCS method for BOTH legs applied to every combo "
                        "('field'/'nested' are deprecated aliases for "
                        "'interpolate'/'alphas')")
    p.add_argument("--wall-theta", type=float, default=None,
                   help="Wall-kernel correlation length (fixed, not swept)")
    p.add_argument("--wall-rho0", type=float, default=None,
                   help="Wall-kernel correlation floor (fixed, not swept)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    # cr.theta and cr.rho_0 are SWEPT, not overridden; null those fields out
    # of args before handing them to _apply_overrides so it doesn't pin them.
    args.cr_theta = None
    args.cr_rho0 = None
    base = _apply_overrides(_load_base(args.settings), args)
    theta_cr_values = tuple(args.theta_cr) if args.theta_cr else (50.0, 200.0, 800.0)
    rho_0_cr_values = tuple(args.rho0_cr)  if args.rho0_cr  else (0.0, 0.3, 0.6)
    sweep(theta_cr_values, rho_0_cr_values, base=base)
