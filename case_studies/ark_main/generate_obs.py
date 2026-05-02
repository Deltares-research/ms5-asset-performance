"""
Generate synthetic corrosion observations for ark_main.

Reads settings.json for everything that defines the observation set:

- ``model_type``                 -> picks which forward model
- ``t_start`` / ``t_end``        -> time range
- ``n_obs``                      -> number of observation times
- ``obs_error_std``              -> Gaussian noise std on obs
- ``B_mu/std/min/max`` (power) or ``C50_mu/std/min/max`` (linear)
                                 -> prior over the random parameter
- ``true_param_quantile``        -> picks the true value as the q-th
                                    percentile of the prior (default 0.35,
                                    a slightly-pessimistic-than-mean draw)

Writes ``data.json`` in the schema run.py / plotting.save_all_plots expect:
    { "<t>": {"corrosion": <mm>, "corrosion_ratio": <->, "EI_corroded": <kNm2>,
              "true_corrosion": <mm>}, ... }

CLI flags only cover model selection and run-level options; everything that
defines the observation set lives in settings.json.

Usage:
    python generate_obs.py                          # use settings.json as-is
    python generate_obs.py --model-type linear      # override model_type only
    python generate_obs.py --seed 1 --dry-run       # try a different draw, don't write
"""

import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from scipy import stats

from src.io import get_remote_path
from corrosion import CorrosionModel


_ENV = Path(__file__).parent / ".env"


def _prior_quantile(model_type: str, config: dict, q: float) -> float:
    """q-th percentile of the active prior (TruncN over B or C50)."""
    if model_type == "power":
        mu = float(config["B_mu"])
        sigma = float(config["B_std"])
        lo = float(config.get("B_min", 0.4))
        hi = float(config.get("B_max", 1.0))
    else:
        mu = float(config["C50_mu"])
        sigma = float(config["C50_std"])
        lo = float(config.get("C50_min", 0.5))
        hi = float(config.get("C50_max", 2.5))
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    return float(stats.truncnorm.ppf(q, a, b, loc=mu, scale=sigma))


def main(
    model_type: str | None = None,
    seed: int = 42,
    output: str | None = None,
    dry_run: bool = False,
) -> None:
    remote = get_remote_path(_ENV)
    settings_path = remote / "input" / "settings.json"
    with open(settings_path, "r") as f:
        settings = json.load(f)
    config = settings["parameters"]

    # Model type: CLI overrides settings.json
    mt = model_type if model_type is not None else config.get("model_type", "power")
    if mt not in ("power", "linear"):
        raise ValueError(f"--model-type must be 'power' or 'linear', got {mt!r}")
    config["model_type"] = mt

    # Everything else from settings.json
    t_lo = float(config.get("t_start", 0))
    t_hi = float(config.get("t_end", 50))
    if t_hi <= t_lo:
        raise ValueError(f"t_end ({t_hi}) must be > t_start ({t_lo}) in settings.json")

    n_obs = int(config.get("n_obs", 8))
    sigma = float(config["obs_error_std"])
    q = float(config.get("true_param_quantile", 0.35))
    true_param = _prior_quantile(mt, config, q)

    # Build the corrosion model
    cm = CorrosionModel(
        model_type=mt,
        # linear
        C50_mu=config.get("C50_mu", 1.5),
        C50_std=config.get("C50_std", 0.75),
        corrosion_rate=config.get("corrosion_rate", 0.022),
        t_start=config.get("t_start", 50.0),
        C50_min=config.get("C50_min", 0.5),
        C50_max=config.get("C50_max", 2.5),
        # power
        power_A=config.get("power_A", 0.091),
        B_mu=config.get("B_mu", 0.72),
        B_std=config.get("B_std", 0.05),
        B_min=config.get("B_min", 0.4),
        B_max=config.get("B_max", 1.0),
        # common
        wall_thickness=config["wall_thickness"],
        obs_error_std=sigma,
        n_grid=config.get("n_B_grid", config.get("n_C50_grid", 100)),
        n_corrosion_grid=config["n_cr_grid"],
    )

    # Time grid: linear from t_lo to t_hi, n_obs samples (inclusive endpoints).
    # In power mode skip t = 0 (C(0) = 0 → after additive noise + clip the obs
    # would just be |N(0, sigma)|, which is degenerate).
    if mt == "power" and t_lo == 0.0 and n_obs > 1:
        times = np.linspace(t_hi / n_obs, t_hi, n_obs)
    else:
        times = np.linspace(t_lo, t_hi, n_obs)

    # Generate observations
    obs = cm.generate_observations(times, true_param, seed=seed)
    true_mean = cm.mean_corrosion(times, true_param).reshape(-1)

    wall_thickness = float(config["wall_thickness"])
    EI_start = float(config.get("EI_start", config.get("Wall_SheetPilingElementEI", 0.0)))

    # Assemble data.json payload
    setting = {}
    for t, c, c_true in zip(times, obs, true_mean):
        ratio = float(c) / wall_thickness
        setting[f"{float(t):.1f}"] = {
            "time": float(t),
            "corrosion": float(c),
            "corrosion_ratio": float(ratio),
            "EI_corroded": EI_start * (1.0 - ratio) if EI_start else None,
            "true_corrosion": float(c_true),
        }

    # Reporting
    param_name = "B" if mt == "power" else "C50"
    print("=" * 70)
    print(f"Generated {n_obs} corrosion observations")
    print(f"  model_type        : {mt}")
    print(f"  {param_name:<17s} : prior {param_name}_mu={config.get(f'{param_name}_mu')}, "
          f"true = q{q:.2f} prior = {true_param:.4f}")
    print(f"  obs_error_std     : {sigma:.3f} mm")
    print(f"  time range        : [{times.min():.1f}, {times.max():.1f}] yr")
    print(f"  seed              : {seed}")
    print("-" * 70)
    print(f"{'t [yr]':>8s} {'true [mm]':>11s} {'obs [mm]':>10s} {'noise [mm]':>12s}")
    for t, c, c_true in zip(times, obs, true_mean):
        print(f"{t:>8.1f} {c_true:>11.3f} {c:>10.3f} {c - c_true:>12.3f}")
    print("=" * 70)

    if dry_run:
        print("--dry-run: not writing data.json")
        return

    out_path = Path(output) if output else (remote / "input" / "data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(setting, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model-type", type=str, choices=["power", "linear"], default=None,
        help="Corrosion model. Default: settings.json:model_type.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path. Default: <remote>/input/data.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print obs but don't write data.json.")
    args = parser.parse_args()

    main(
        model_type=args.model_type,
        seed=args.seed,
        output=args.output,
        dry_run=args.dry_run,
    )
