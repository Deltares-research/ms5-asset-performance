"""
D-Sheet piling reliability analysis script.

Uses the generic FragilityPipeline from src/ with domain-specific
JPDF, corrosion model, and performance function.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from dotenv import load_dotenv
import json
from datetime import datetime

from src import FragilityPipeline
from case_studies.ark_example.jpdf import JPDF
from case_studies.ark_example.corrosion import CorrosionModel
from case_studies.ark_example.performance_function import (
    Performance,
    FragilitySurfaceIndex,
    MLP,
)
from case_studies.ark_example import io
from case_studies.ark_example import plotting


def setup_pipeline(config: Dict[str, Any], settings_path: Path, n_samples: int = 100_000, seed: int = 42) -> FragilityPipeline:
    """Create and set up a FragilityPipeline for D-Sheet piling.

    Args:
        config: Parameters dict from settings JSON.
        settings_path: Path to settings JSON file.
        n_samples: Number of MC samples.
        seed: Random seed.

    Returns:
        Configured FragilityPipeline ready for fragility building and timeline.
    """
    performance = Performance(
        name="dsheet_moment",
        parameters={
            "moment_cap": config["moment_cap"],
            "EI_start": config["EI_start"],
            "ei_column_idx": -2,
        },
    )

    pipeline = FragilityPipeline(
        settings_path=settings_path,
        performance=performance,
        config=config,
    )

    # JPDF
    pipeline.jpdf = JPDF(name="dsheet", config=config)
    pipeline.jpdf.set_prior_from_settings(settings_path)
    pipeline.jpdf.initiate_samples(n_samples=n_samples, seed=seed)
    pipeline.jpdf.add_water_level(water_lvl=-1.0)

    # Corrosion model
    pipeline.corrosion_model = CorrosionModel(
        C50_mu=config.get("C50_mu", 1.5),
        C50_std=config.get("C50_std", 0.75),
        corrosion_rate=config["corrosion_rate"],
        start_thickness=config["start_thickness"],
        obs_error_std=config["obs_error_std"],
        t_start=config["t_start"],
        n_grid=config["n_C50_grid"],
        n_corrosion_grid=config["n_grid"],
    )

    return pipeline


def load_surrogate(pipeline: FragilityPipeline, input_dim=11, hidden_dims=None, output_dim=1):
    """Load trained surrogate model into the pipeline's performance function."""
    hidden_dims = hidden_dims or [1024, 512, 256, 128, 64, 32]
    model_kwargs = {"input_dim": input_dim, "hidden_dims": hidden_dims, "output_dim": output_dim}
    model, scaler_x, scaler_y = io.load_surrogate_model(MLP, model_kwargs)
    pipeline.performance.set_surrogate(model, scaler_x, scaler_y)


def build_fragility_surface(
    pipeline: FragilityPipeline,
    n_cr: int = 100,
    n_moments: int = 50,
    moment_range: Optional[Tuple[float, float]] = None,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> None:
    """Build or load 2D fragility surface."""
    if not force_rebuild and io.fragility_surface_exists():
        if verbose:
            print("Loading cached fragility surface...")
        pipeline.fragility_surface = io.load_fragility_surface()
        if verbose:
            print(pipeline.fragility_surface.summary())
    else:
        if verbose:
            print(f"Building fragility surface ({n_cr} CR x {n_moments} moments)...")
        pipeline.fragility_surface = pipeline.performance.build_fragility_surface(
            x=pipeline.jpdf.X_samples,
            n_cr=n_cr,
            n_moments=n_moments,
            moment_range=moment_range,
            verbose=verbose,
        )
        if verbose:
            print("Saving fragility surface...")
        io.save_fragility_surface(pipeline.fragility_surface)
        if verbose:
            print("Fragility surface saved.")


def save_results(pipeline: FragilityPipeline, filename: str = "reliability_results.json"):
    """Save pipeline results to JSON."""
    results_json = {str(t): data for t, data in pipeline.results.items()}
    load_dotenv(".env")
    username = os.environ.get("USER", "unknown").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    io.save_json(results_json, f"{username}_{timestamp}/{filename}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _get_output_dir() -> Path:
    load_dotenv(".env")
    username = os.environ.get("USER", "unknown").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return io.get_remote_path() / f"output/results/{username}_{timestamp}/plots"


def plot_betas(pipeline: FragilityPipeline, config: Dict, output_dir: Path = None):
    output_dir = Path(output_dir or _get_output_dir())
    png_dir = output_dir / "beta_forecast"
    png_dir.mkdir(parents=True, exist_ok=True)

    times = sorted(pipeline.results.keys())
    for t in times:
        results_up_to_t = {k: v for k, v in pipeline.results.items() if k <= t}
        fig = plotting.plot_beta_forecast_at_time(
            current_time=t, results=results_up_to_t, beta_req=config["beta_req"],
        )
        plotting.save_figure(fig, png_dir / f"beta_forecast_t{int(t):03d}.png")
    plotting.collect_pngs_to_pdf(png_dir, output_dir / "beta_forecast.pdf")
    print(f"Plots saved to {png_dir}")


def plot_corrosion_forecasts(pipeline: FragilityPipeline, config: Dict, setting: Dict, output_dir: Path = None):
    output_dir = Path(output_dir or _get_output_dir())
    png_dir = output_dir / "corrosion"
    png_dir.mkdir(parents=True, exist_ok=True)

    times = sorted([float(k) for k in setting.keys() if k != "metadata"])
    obs_times, obs_corrosion = [], []
    for t in times:
        key = str(t) if str(t) in setting else f"{t:.1f}"
        if key in setting and "corrosion" in setting[key]:
            obs_times.append(t)
            obs_corrosion.append(setting[key]["corrosion"])

    t_start, t_end = times[0], times[-1]
    for current_t in obs_times:
        fig = plotting.plot_corrosion_forecast_at_time(
            cr_grid=pipeline.fragility_surface.corrosion_ratios,
            cr_forecast_prior=pipeline.results[t_start]["prior"]["cr_forecast"],
            cr_forecast_posterior=pipeline.results[current_t]["posterior"]["cr_forecast"],
            obs_times=[t for t in obs_times if t <= current_t],
            obs_values=[c for t, c in zip(obs_times, obs_corrosion) if t <= current_t],
            obs_error_std=config["obs_error_std"],
            start_thickness=config["start_thickness"],
            xlim=(t_start, t_end),
            ylim=(0, config["start_thickness"]),
        )
        plotting.save_figure(fig, png_dir / f"corrosion_t{int(current_t):03d}.png")
    plotting.collect_pngs_to_pdf(png_dir, output_dir / "corrosion.pdf")
    print(f"Corrosion forecast plots saved to {png_dir}")


def plot_moment_forecasts(pipeline: FragilityPipeline, config: Dict, setting: Dict, output_dir: Path = None):
    output_dir = Path(output_dir or _get_output_dir())
    png_dir = output_dir / "moment"
    png_dir.mkdir(parents=True, exist_ok=True)

    start_thickness = config["start_thickness"]
    moment_cap = config["moment_cap"]

    times = sorted([float(k) for k in setting.keys() if k != "metadata"])
    obs_times, obs_moment_cap = [], []
    survived_times, survived_moments = [], []
    for t in times:
        key = str(t) if str(t) in setting else f"{t:.1f}"
        if key in setting and "corrosion" in setting[key]:
            obs_times.append(t)
            cr = setting[key].get("corrosion_ratio", setting[key]["corrosion"] / start_thickness)
            obs_moment_cap.append(moment_cap * (1 - cr))
            ms = setting[key].get("moment_survived")
            if ms is not None:
                survived_times.append(t)
                survived_moments.append(ms)

    t_start, t_end = times[0], times[-1]
    for current_t in obs_times:
        fig = plotting.plot_moment_forecast_at_time(
            cr_grid=pipeline.fragility_surface.corrosion_ratios,
            cr_forecast_prior=pipeline.results[t_start]["prior"]["cr_forecast"],
            cr_forecast_posterior=pipeline.results[current_t]["posterior"]["cr_forecast"],
            moment_cap=moment_cap,
            survived_times=survived_times or None,
            survived_moments=survived_moments or None,
            obs_times=[t for t in obs_times if t <= current_t],
            obs_moment_cap=[m for t, m in zip(obs_times, obs_moment_cap) if t <= current_t],
            xlim=(t_start, t_end),
            ylim=(0, moment_cap * 1.1),
        )
        plotting.save_figure(fig, png_dir / f"moment_t{int(current_t):03d}.png")
    plotting.collect_pngs_to_pdf(png_dir, output_dir / "moment.pdf")
    print(f"Moment capacity forecast plots saved to {png_dir}")


def plot_end_of_life(pipeline: FragilityPipeline, config: Dict, output_dir: Path = None):
    output_dir = Path(output_dir or _get_output_dir())
    png_dir = output_dir / "end_of_life"
    png_dir.mkdir(parents=True, exist_ok=True)

    fig = plotting.plot_end_of_life(
        results=pipeline.results, beta_req=config["beta_req"], t_start=config["t_start"],
    )
    plotting.save_figure(fig, png_dir / "end_of_life.png")
    plotting.collect_pngs_to_pdf(png_dir, output_dir / "end_of_life.pdf")
    print(f"End-of-life plot saved to {png_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv(".env")
    os.environ["REMOTE_DATA_PATH"] = os.environ["REMOTE_PATH"] + r"/input"
    settings_path = Path(os.environ["REMOTE_DATA_PATH"]) / "settings.json"

    with open(settings_path, "r") as f:
        settings = json.load(f)
    config = settings.get("parameters", {})

    # Setup
    pipeline = setup_pipeline(config, settings_path, n_samples=1_000_000, seed=42)

    # Load surrogate
    load_surrogate(pipeline)
    print("Surrogate loaded.")

    # Build fragility surface
    build_fragility_surface(pipeline, n_cr=1000, n_moments=50, force_rebuild=False)

    # Load data and run timeline
    data = io.load_json("data.json")
    obs_times = [int(float(key)) for key in data.keys()]
    forecast_times = list(range(min(obs_times), max(obs_times) + 1, 1))
    pipeline.init_times(obs_times, forecast_times)

    results = pipeline.run_timeline(data, verbose=True)

    save_results(pipeline)
    print("Results saved.")

    # Plots
    plot_betas(pipeline, config)


if __name__ == "__main__":
    main()
