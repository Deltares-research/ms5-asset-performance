"""
Sweep theta_wall x theta_cr for the spatial-variability MCS pipeline.

Loads ``<remote>/input/spatial_settings.json`` as a base, then for each
``(theta_wall, theta_cr)`` pair overrides only those two fields and calls
``analyze`` programmatically. Every combo writes its own self-contained
``cached/<signature>/`` and ``results/<signature>/`` folder (signature
encodes both thetas via the ``wth``/``crth`` parts), so combos never
overwrite each other and identical re-runs hit the caches.

After the loop, prints a summary table of system Pf and beta at the final
forecast time, both prior and posterior (taking the last obs scenario).

Usage::

    python -m case_studies.ark_main.run_spatial_sweep
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

_ARK = Path(__file__).resolve().parent
sys.path.insert(0, str(_ARK))

# Importing ``run_spatial`` pulls dotenv + sys.path + load_settings via the
# spatial.engine chain.
from run_spatial import analyze
from spatial import results_dir as _results_dir
from src.io import get_remote_path

_ENV = _ARK / ".env"
_remote = get_remote_path(_ENV)


def _load_base() -> dict:
    return json.load(open(_remote / "input" / "spatial_settings.json"))


def _override(base: dict, theta_wall: float, theta_cr: float) -> dict:
    cfg = json.loads(json.dumps(base))                # deep copy
    cfg.setdefault("wall", {})["theta"] = float(theta_wall)
    cfg.setdefault("cr",   {})["theta"] = float(theta_cr)
    return cfg


def sweep(
    theta_wall_values: tuple[float, ...] = (50.0, 200.0, 800.0),
    theta_cr_values:   tuple[float, ...] = (50.0, 200.0, 800.0),
) -> None:
    base = _load_base()
    combos = list(product(theta_wall_values, theta_cr_values))

    print(f"\n{'>' * 5}  Spatial-MCS sweep: {len(combos)} combinations  {'<' * 5}")
    print(f"  theta_wall: {list(theta_wall_values)}")
    print(f"  theta_cr:   {list(theta_cr_values)}")
    print(f"  base config: lsf={base['lsf_name']}, "
          f"N={base['n_samples']}, seed={base['seed']}, "
          f"L={base['L']}, n_sections={base['n_sections']}, "
          f"wall.rho_0={base['wall']['rho_0']}, "
          f"cr.rho_0={base['cr']['rho_0']}")

    for i, (tw, tc) in enumerate(combos, 1):
        cfg = _override(base, tw, tc)
        print(f"\n{'#' * 70}")
        print(f"#  [{i}/{len(combos)}]  theta_wall = {tw:.0f} m   "
              f"theta_cr = {tc:.0f} m")
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
        "theta_wall", "theta_cr", "Pf prior", "beta prior", "Pf post", "beta post"
    )
    print(header)
    print("-" * len(header))
    for tw, tc in combos:
        cfg = _override(base, tw, tc)
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

        print("{:>10.0f} | {:>10.0f} | {:>11} | {:>11} | {:>11} | {:>11}".format(
            tw, tc, pf_prior_str, b_prior_str, pf_post_str, b_post_str
        ))


if __name__ == "__main__":
    sweep()
