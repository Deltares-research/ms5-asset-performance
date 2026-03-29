"""
Settlement-specific reliability analysis script.

Uses the generic ReliabilityPipeline from src/ with the settlement engine
as the physical model. Handles file I/O, configuration, and plotting.
"""

import numpy as np
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from argparse import ArgumentParser

from src import ReliabilityPipeline
from case_studies.settlement_example.io import load_json, get_remote_path
from performance_function import Performance
from settlement_engine import get_settlement
from plotting import (save_jpdf_plots, save_jpdf_plots_samples,
                      save_settlement_forecast_plots, save_settlement_residual_plots,
                      save_beta_over_time_plot, make_gifs)


# Variable mapping: JPDF variable name -> settlement engine parameter name
VAR_MAP = {"CR": "CR", "k": "k"}


def main(input_file: Optional[str] = None, analysis_method: Optional[str] = None, force_rebuild: bool = False):
    """Run the full settlement reliability analysis pipeline.

    Args:
        input_file: Override for the specs file (without .json extension).
        analysis_method: Override for the analysis method.
        force_rebuild: If True, recompute settlement caches.
    """
    # Paths
    load_dotenv(".env")
    os.environ["REMOTE_DATA_PATH"] = str(get_remote_path() / "input")
    if input_file:
        specs_path = Path(os.environ["REMOTE_DATA_PATH"]) / f"{input_file}.json"
    else:
        specs_path = Path(os.environ["REMOTE_DATA_PATH"]) / "settings.json"

    username = os.environ.get("USER", "unknown").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = get_remote_path() / f"output/results/{username}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config from specs JSON
    with open(specs_path, "r") as f:
        specs = json.load(f)
    config = specs.get("parameters", {})
    if analysis_method:
        config["analysis_method"] = analysis_method

    # Initialize performance function
    performance = Performance(name="settlement", parameters=config)

    # Initialize pipeline
    pipeline = ReliabilityPipeline(
        specs_path=specs_path,
        performance=performance,
        analysis_method=config.get("analysis_method", "semi-analytical"),
        n_samples=config.get("n_samples", 100_000),
        obs_error=config.get("obs_error", 0.1),
    )

    # =========================================================================
    # STEP 1: Setup
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Setup")
    print("=" * 60)
    pipeline.setup(seed=42)

    # =========================================================================
    # STEP 2: Initialize (or load) pre-evaluated model output
    # =========================================================================
    print("=" * 60)
    print("STEP 2: Model evaluation")
    print("=" * 60)

    setting_raw = load_json("data.json")
    if isinstance(setting_raw, list):
        setting = {row["time"]: row["settlement_1"] for row in setting_raw}
    else:
        setting = setting_raw

    obs_times = np.array([float(k) for k in setting.keys()])
    obs_values = np.array([float(v) for v in setting.values()])

    # Build forecast times
    preload_removal_time = config["preload_removal_time"]
    forecast_times = np.arange(0, preload_removal_time, config.get("forecast_interval", 10))
    forecast_times = np.append(forecast_times, [preload_removal_time, config["end_time"]])
    pipeline.init_times(obs_times=obs_times, forecast_times=forecast_times)

    # Settlement engine kwargs (domain-specific parameters)
    model_kwargs = dict(
        RR=config["RR"],
        Ca=config["Ca"],
        h=config["layer_thickness"],
        sigma_0=config["sigma_0"],
        sigma_v=config["sigma_0"] + config["preload"],
        sigma_p=config["sigma_p"],
        method=config.get("doc_method", "Terzaghi"),
    )

    cache_dir = get_remote_path() / "output/cache"
    pipeline.init_model_output(
        model_fn=get_settlement,
        var_map=VAR_MAP,
        model_kwargs=model_kwargs,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
    )

    # =========================================================================
    # STEP 3: Run timeline analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Run timeline analysis")
    print("=" * 60)

    results = pipeline.run_timeline(obs_values=obs_values, verbose=True)

    # =========================================================================
    # STEP 4: Plots
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Plots")
    print("=" * 60)

    variables = {v["name"]: v for v in specs.get("variables", [])}
    var_names = pipeline.jpdf.variable_names
    true_values = {name: variables.get(name, {}).get("true") for name in var_names}

    if pipeline.is_sample_based:
        samples_dict = {name: pipeline.jpdf.get_samples(name) for name in var_names}
        save_jpdf_plots_samples(
            results=results,
            output_dir=output_dir,
            var_names=var_names,
            samples=samples_dict,
            W_prior=pipeline.jpdf.W_prior_samples,
            W_posterior_per_t=pipeline.W_posterior_per_t,
            true_values=true_values,
        )
    else:
        save_jpdf_plots(results=results, output_dir=output_dir, var_names=var_names, true_values=true_values)

    save_settlement_forecast_plots(
        results=results,
        output_dir=output_dir,
        t_max=preload_removal_time + 5,
        obs_error=config.get("obs_error", 0.1),
        y_max=obs_values.max(),
    )
    save_settlement_residual_plots(
        results=results,
        output_dir=output_dir,
        end_settlement_req=config.get("end_settlement_req", 0.05),
    )
    save_beta_over_time_plot(results=results, output_dir=output_dir)
    make_gifs(output_dir)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str, default="settings")
    parser.add_argument("--analysis_method", type=str, default="semi-analytical")
    parser.add_argument("--force_rebuild", action="store_false")
    args = parser.parse_args()

    main(
        input_file=args.input_file,
        analysis_method=args.analysis_method,
        force_rebuild=args.force_rebuild,
    )
