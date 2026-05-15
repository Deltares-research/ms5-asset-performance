"""
Export basic FORM-variable distributions (the "wall parameters") to CSV.

Reads ``<remote>/input/settings.json``'s ``variables`` block and produces
one row per variable with the columns

* Variable         — name as it appears in settings.json
* Distribution     — compact descriptor, e.g. ``LogN(3.10, 0.08)``,
                     ``Gumbel(14.0)``, ``N(20.0, 5.0)``, ``Deterministic(3.0)``
* Mean             — distribution mean in x-space
* CoV              — coefficient of variation (std / |mean|) as a %
* 1% / 99%         — distribution quantiles, computed analytically with scipy

The distribution helpers mirror ``analysis/run_mc.py::_build_marginal`` so
the CSV is exactly what the spatial / per-section MCS actually samples
from.

Output: ``<remote>/output/wall_parameters.csv`` (override with ``--out``).

Usage::

    python -m case_studies.ark_main.io.export_wall_params
    python -m case_studies.ark_main.io.export_wall_params --out path/to/file.csv
"""
from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy import stats as st

_ARK = Path(__file__).resolve().parents[1]
if str(_ARK) not in sys.path:
    sys.path.insert(0, str(_ARK))

from dotenv import load_dotenv
load_dotenv(_ARK / "geolib.env")

from src.io import get_remote_path


_ENV = _ARK / ".env"
_EULER = 0.5772156649


# Soil-property suffixes appearing after ``<Layer>_soil``. Each maps to
# (symbol, unit) used to format the human-readable variable label.
_SOIL_PROPERTY_LABEL = {
    "phi":      ("φ",     "°"),
    "cohesion": ("c",     "kPa"),
    "gamdry":   ("γ_dry", "kN/m³"),
    "gamwet":   ("γ_wet", "kN/m³"),
    "curkb1":   ("k_b1",  "kN/m³"),
}

# Non-soil variables get an explicit mapping. Falls back to the raw name
# (with underscores -> spaces) if absent.
_OTHER_LABEL = {
    "model_factor_M":            ("θ_M",            "-"),
    "model_factor_F":            ("θ_F",            "-"),
    "phreatic_level":            ("Phreatic level", "mNAP"),
    "canal_level":               ("Canal level",    "mNAP"),
    "uniform_load_left":         ("Uniform load",   "kN/m"),
    "Wall_SheetPilingElementEI": ("Wall EI",        "kNm²/m"),
}


def _pretty_variable(name: str) -> str:
    """Map an internal variable name to a human-readable label with units."""
    if name in _OTHER_LABEL:
        sym, unit = _OTHER_LABEL[name]
        return f"{sym} ({unit})"
    if "_soil" in name:
        layer, _, prop = name.partition("_soil")
        if prop in _SOIL_PROPERTY_LABEL:
            sym, unit = _SOIL_PROPERTY_LABEL[prop]
            return f"{layer} {sym} ({unit})"
    return name.replace("_", " ")


def _build_marginal(v: dict):
    """Frozen scipy distribution for one variable; ``None`` if deterministic.

    Mirrors ``run_mc._build_marginal`` so the quantiles in the CSV match
    what the MCS samples.
    """
    dist_type = v.get("distribution_type", "normal").lower()
    mean = float(v["mean"])
    std = float(v["standard_deviation"])
    lo = float(v.get("lower_bound", -np.inf))
    hi = float(v.get("upper_bound", np.inf))

    if dist_type in ("deterministic", "constant", "fixed") or std == 0:
        return None

    if dist_type in ("normal", "norm", "n", "gaussian"):
        a = (lo - mean) / std if np.isfinite(lo) else -10.0
        b = (hi - mean) / std if np.isfinite(hi) else 10.0
        return st.truncnorm(a, b, loc=mean, scale=std)

    if dist_type in ("lognormal", "lognorm"):
        if mean <= 0:
            return None
        sigma = float(np.sqrt(np.log(1.0 + (std / mean) ** 2)))
        mu = float(np.log(mean) - 0.5 * sigma ** 2)
        return st.lognorm(sigma, scale=np.exp(mu))

    if dist_type in ("gumbel", "gumbel_max", "gumbel_r"):
        beta = std * np.sqrt(6) / np.pi
        return st.gumbel_r(loc=mean - _EULER * beta, scale=beta)

    if dist_type in ("gumbel_min", "gumbel_l"):
        beta = std * np.sqrt(6) / np.pi
        return st.gumbel_l(loc=mean + _EULER * beta, scale=beta)

    if dist_type in ("uniform", "unif"):
        return st.uniform(loc=lo, scale=hi - lo)

    return st.norm(loc=mean, scale=std)


def _distribution_label(v: dict) -> str:
    """Compact human-readable distribution descriptor.

    Lognormal uses ``LogN(mu, sigma)`` of the log-distribution; Gumbel
    uses ``Gumbel(beta)`` with the scale parameter (``beta = sigma *
    sqrt(6) / pi``). Same conventions as the screenshot the user posted.
    """
    dist_type = v.get("distribution_type", "normal").lower()
    mean = float(v["mean"])
    std = float(v["standard_deviation"])

    if dist_type in ("deterministic", "constant", "fixed") or std == 0:
        return f"Deterministic({mean:.2f})"

    if dist_type in ("normal", "norm", "n", "gaussian"):
        return f"N({mean:.2f}, {std:.2f})"

    if dist_type in ("lognormal", "lognorm"):
        if mean <= 0:
            return "LogN(invalid: mean<=0)"
        sigma = float(np.sqrt(np.log(1.0 + (std / mean) ** 2)))
        mu = float(np.log(mean) - 0.5 * sigma ** 2)
        return f"LogN({mu:.2f}, {sigma:.2f})"

    if dist_type in ("gumbel", "gumbel_max", "gumbel_r"):
        beta = std * np.sqrt(6) / np.pi
        mu = mean - _EULER * beta              # location (mode), Gumbel-right
        return f"Gumbel({mu:.2f}, {beta:.2f})"
    if dist_type in ("gumbel_min", "gumbel_l"):
        beta = std * np.sqrt(6) / np.pi
        mu = mean + _EULER * beta              # location, Gumbel-left
        return f"Gumbel_min({mu:.2f}, {beta:.2f})"

    if dist_type in ("uniform", "unif"):
        lo = float(v.get("lower_bound", 0.0))
        hi = float(v.get("upper_bound", 0.0))
        return f"U({lo:.2f}, {hi:.2f})"

    return dist_type


def _active_vars_for_lsf(remote: Path, lsf_name: str) -> set[str]:
    """Variables with ``|alpha| > 1e-6`` anywhere in ``lsf_name``'s fragility.

    Reads every cached ``point_*.json`` under
    ``<remote>/output/fragility_curve_<lsf>/`` and accumulates the union of
    non-zero-alpha variable names. Anything with alpha = 0 across the entire
    cr-grid contributes nothing to FORM failure for this LSF and is dropped
    from the per-LSF table.
    """
    cache_dir = remote / "output" / f"fragility_curve_{lsf_name}"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"No fragility cache at {cache_dir}. Build it first."
        )
    active: set[str] = set()
    for p in cache_dir.glob("point_*.json"):
        pt = json.load(open(p))
        if not pt.get("convergence"):
            continue
        for v, a in (pt.get("alphas") or {}).items():
            if abs(float(a)) > 1e-6:
                active.add(v)
    return active


def main(
    out_path: Path | None = None,
    settings_file: str = "settings_full.json",
    lsf_name: str | None = None,
) -> None:
    """Write the distribution CSV.

    If ``lsf_name`` is given, the table is filtered to variables that
    actually contribute to FORM failure for that LSF (non-zero alpha
    somewhere in its fragility curve). Default output filename includes
    the LSF stem so files for different LSFs coexist.
    """
    remote = get_remote_path(_ENV)
    settings = json.load(open(remote / "input" / settings_file))
    variables = settings.get("variables", [])

    if lsf_name is not None:
        active = _active_vars_for_lsf(remote, lsf_name)
        variables = [v for v in variables if v["name"] in active]
        suffix = lsf_name
    else:
        suffix = Path(settings_file).stem

    out = Path(out_path) if out_path else (
        remote / "output" / f"wall_parameters_{suffix}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    # utf-8-sig: BOM so Excel renders φ / γ / θ in the variable labels
    # without the user having to set encoding manually.
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Variable", "Distribution", "Mean", "CoV", "1%", "99%"])
        for v in variables:
            name = v["name"]
            mean = float(v["mean"])
            std = float(v["standard_deviation"])
            cov_str = (
                f"{(std / abs(mean) * 100):.2f}%" if mean != 0
                else ("0.00%" if std == 0 else "inf%")
            )

            marg = _build_marginal(v)
            if marg is None:
                q01_str = q99_str = "-"
            else:
                q01_str = f"{float(marg.ppf(0.01)):.2f}"
                q99_str = f"{float(marg.ppf(0.99)):.2f}"

            writer.writerow([
                _pretty_variable(name),
                _distribution_label(v),
                f"{mean:.2f}",
                cov_str,
                q01_str,
                q99_str,
            ])
            n_written += 1

    print(f"Wrote {n_written} rows -> {out}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output CSV path. Defaults to <remote>/output/wall_parameters_<stem>.csv.",
    )
    parser.add_argument(
        "--settings", type=str, default="settings_full.json",
        help="Settings filename under <remote>/input/. Defaults to "
             "settings_full.json (the un-pruned variable set).",
    )
    parser.add_argument(
        "--lsf", type=str, default=None,
        help="If given, filter the table to variables with non-zero alpha "
             "anywhere in the named LSF's fragility cache. Output filename "
             "becomes wall_parameters_<lsf>.csv.",
    )
    args = parser.parse_args()
    main(
        out_path=Path(args.out) if args.out else None,
        settings_file=args.settings,
        lsf_name=args.lsf,
    )
