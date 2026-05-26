"""Export prior + posterior corrosion-ratio PDFs over time, plus Pf(cr).

Produces a single JSON file with everything needed for downstream reliability
post-processing without re-running the pipeline:

  - cr_grid:          common corrosion-ratio grid in [0, 1]
  - forecast_times:   integer grid + exact obs times
  - obs_times:        observation times
  - fragility:        cached fragility points (cr, pf, beta, method, convergence)
  - pf_on_cr_grid:    Pf(cr) linearly interpolated onto cr_grid (loglinear)
  - prior_pdf_per_t:  {t: pdf(cr | prior)} — same for every obs scenario
  - posterior_pdf_per_obs:
        {t_obs: {t: pdf(cr | obs <= t_obs, anchored on last obs)}}
        one block per obs time, mirroring how the pipeline / plots compute
        posteriors as observations accumulate.

Usage:
    python export_cr_pdfs.py
    python export_cr_pdfs.py --lsf-name lsf_wall_anchor --out cr_pdfs.json
"""

import json
import sys
from pathlib import Path
from argparse import ArgumentParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.io import get_remote_path
from models.jpdf import JPDF
from models.corrosion import CorrosionModel


_ENV = Path(__file__).resolve().parents[1] / ".env"


def load_fragility(cache_dir: Path) -> list[dict]:
    pts = []
    for f in sorted(cache_dir.glob("point_*.json")):
        with open(f) as fh:
            rec = json.load(fh)
        if "point" in rec and "pf" in rec and "beta" in rec:
            pts.append(rec)
    pts.sort(key=lambda r: r["index"])
    return pts


def interp_pf_on_grid(cr_grid: np.ndarray, frag_cr: np.ndarray,
                      frag_pf: np.ndarray) -> np.ndarray:
    """Log-linear interpolation of Pf vs cr (Pf can span many orders)."""
    # Clip to a tiny floor to keep log finite, preserve original elsewhere
    pf_safe = np.clip(frag_pf, 1e-300, 1.0)
    log_pf = np.log(pf_safe)
    log_pf_grid = np.interp(cr_grid, frag_cr, log_pf,
                            left=log_pf[0], right=log_pf[-1])
    return np.exp(log_pf_grid)


def build_corrosion_model(config: dict) -> CorrosionModel:
    return CorrosionModel(
        model_type=config.get("model_type", "power"),
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
        obs_error_std=config["obs_error_std"],
        n_grid=config.get("n_B_grid", config.get("n_C50_grid", 100)),
        n_corrosion_grid=config["n_cr_grid"],
    )


def pdf_on_grid(cr_grid_target: np.ndarray, g: np.ndarray, p: np.ndarray) -> list:
    """Interpolate a (g, p) PDF onto cr_grid_target. Returns plain Python list."""
    return np.interp(cr_grid_target, g, p, left=0.0, right=0.0).tolist()


def main(lsf_name: str = "lsf_wall", out_path: str | None = None) -> None:
    remote = get_remote_path(_ENV)
    settings_path = remote / "input" / "settings.json"
    data_path = remote / "input" / "data.json"
    cache_dir = remote / "output" / f"fragility_curve_{lsf_name}"

    with open(settings_path) as f:
        settings = json.load(f)
    config = settings["parameters"]

    with open(data_path) as f:
        data = json.load(f)

    # Forecast grid: integer years from t_start to t_end + exact obs times.
    obs_times = sorted([float(d["time"]) for d in data.values()])
    forecast_interval = int(config.get("forecast_interval", 1))
    t_start = int(config.get("t_start", min(obs_times)))
    t_end = int(config.get("t_end", max(obs_times)))
    forecast_times = sorted(set(
        list(range(t_start, t_end + forecast_interval, forecast_interval))
        + obs_times
    ))
    forecast_times = [float(t) for t in forecast_times]

    # Common cr-grid for ALL exported PDFs.
    n_cr = int(config.get("n_cr_grid", 1000))
    cr_grid = np.linspace(0.0, 1.0, n_cr)

    # Fragility curve.
    if not (cache_dir / "manifest.json").exists():
        raise FileNotFoundError(f"No fragility cache at {cache_dir}")
    points = load_fragility(cache_dir)
    frag_cr = np.array([p["point"]["corrosion_rate"] for p in points])
    frag_pf = np.array([p["pf"] for p in points])
    frag_beta = np.array([p["beta"] for p in points])
    frag_method = [p["method"] for p in points]
    frag_conv = [bool(p["convergence"]) for p in points]
    pf_on_grid = interp_pf_on_grid(cr_grid, frag_cr, frag_pf)

    # Corrosion model + JPDF (prior).
    cm = build_corrosion_model(config)
    jpdf = JPDF(name="dsheet", config=config)
    jpdf.set_prior_from_settings(settings_path)

    # Prior cr PDF per forecast time (same for every obs scenario).
    print(f"Building prior cr-PDFs at {len(forecast_times)} forecast times ...")
    prior_pdf_per_t = {}
    for t in forecast_times:
        g, p = cm.corrosion_ratio_pdf(t=t, param_pdf=jpdf.param_prior)
        prior_pdf_per_t[f"{t:.4f}"] = pdf_on_grid(cr_grid, g, p)

    # Posterior cr PDFs: incrementally update with each new observation, then
    # snapshot the forecast PDF at every t in forecast_times >= t_obs.
    print(f"Building posterior cr-PDFs for {len(obs_times)} obs scenarios ...")
    posterior_pdf_per_obs = {}
    seen_times, seen_values = [], []
    for t_obs in obs_times:
        key = f"{t_obs:.4f}"
        # Find the obs at this time
        rec = next((d for d in data.values() if abs(float(d["time"]) - t_obs) < 1e-6), None)
        if rec is None:
            continue
        c_obs = float(rec["corrosion"])
        seen_times.append(t_obs)
        seen_values.append(c_obs)

        # Update posterior with all obs seen so far.
        jpdf.reset_to_prior()
        jpdf.update(np.asarray(seen_times), np.asarray(seen_values))

        # Snapshot cr PDF at every forecast time >= t_obs.
        block = {}
        for t in forecast_times:
            if t < t_obs:
                continue
            g, p = cm.corrosion_ratio_pdf(
                t=t, param_pdf=jpdf.param_pdf,
                last_obs_time=t_obs, last_obs=c_obs,
            )
            block[f"{t:.4f}"] = pdf_on_grid(cr_grid, g, p)
        posterior_pdf_per_obs[key] = block

    # Reset jpdf to prior afterward so further pipeline use starts clean.
    jpdf.reset_to_prior()

    # Assemble payload.
    payload = {
        "lsf_name": lsf_name,
        "model_type": config.get("model_type", "power"),
        "wall_thickness": float(config["wall_thickness"]),
        "cr_grid": cr_grid.tolist(),
        "forecast_times": forecast_times,
        "obs_times": [float(t) for t in obs_times],
        "fragility": {
            "cr": frag_cr.tolist(),
            "pf": frag_pf.tolist(),
            "beta": frag_beta.tolist(),
            "method": frag_method,
            "convergence": frag_conv,
        },
        "pf_on_cr_grid": pf_on_grid.tolist(),
        "prior_pdf_per_t": prior_pdf_per_t,
        "posterior_pdf_per_obs": posterior_pdf_per_obs,
    }

    out = Path(out_path) if out_path else (
        remote / "output" / f"cr_pdfs_{lsf_name}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    size_mb = out.stat().st_size / 1e6
    print(f"\nWrote {out}  ({size_mb:.2f} MB)")
    print(f"  cr_grid points : {len(cr_grid)}")
    print(f"  forecast times : {len(forecast_times)}")
    print(f"  obs scenarios  : {len(posterior_pdf_per_obs)}")
    print(f"  fragility pts  : {len(points)}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lsf-name", type=str, default="lsf_wall")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path. Default: <remote>/output/cr_pdfs_{lsf}.json")
    args = parser.parse_args()
    main(lsf_name=args.lsf_name, out_path=args.out)
