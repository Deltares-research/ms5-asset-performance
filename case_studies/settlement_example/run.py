"""
Settlement-specific reliability analysis script.

Uses the generic GridModelPipeline from src/ with the settlement engine
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
from src.plotting import save_jpdf_plots, make_gifs
from case_studies.settlement_example.io import load_json, get_remote_path
from performance_function import Performance
from settlement_engine import get_settlement
from plotting import (save_settlement_forecast_plots, save_settlement_residual_plots,
                      save_beta_over_time_plot)


def main(input_file: Optional[str] = None, analysis_method: Optional[str] = None, force_rebuild: bool = False):
    # Paths
    load_dotenv(".env")
    os.environ["REMOTE_DATA_PATH"] = str(get_remote_path() / "input")
    if input_file:
        settings_path = Path(os.environ["REMOTE_DATA_PATH"]) / f"{input_file}.json"
    else:
        settings_path = Path(os.environ["REMOTE_DATA_PATH"]) / "settings.json"

    username = os.environ.get("USER", "unknown").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = get_remote_path() / f"output/results/{username}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(settings_path, "r") as f:
        settings = json.load(f)
    config = settings.get("parameters", {})
    if analysis_method:
        config["analysis_method"] = analysis_method

    # Load observations
    setting_raw = load_json("data.json")
    if isinstance(setting_raw, list):
        setting = {row["time"]: row["settlement_1"] for row in setting_raw}
    else:
        setting = setting_raw
    obs_times = np.array([float(k) for k in setting.keys()])
    obs_values = np.array([float(v) for v in setting.values()])

    # Build forecast times
    t_removal = config["preload_removal_time"]
    forecast_times = np.arange(0, t_removal, config.get("forecast_interval", 10))
    forecast_times = np.append(forecast_times, [t_removal, config["end_time"]])

    # Settlement engine kwargs
    model_kwargs = dict(
        RR=config["RR"],
        Ca=config["Ca"],
        h=config["layer_thickness"],
        sigma_0=config["sigma_0"],
        sigma_v=config["sigma_0"] + config["preload"],
        sigma_p=config["sigma_p"],
        method=config.get("doc_method", "Terzaghi"),
    )

    # Run pipeline
    pipeline = ReliabilityPipeline(
        settings_path=settings_path,
        performance=Performance(name="settlement", parameters=config),
        config=config,
    )
    results = pipeline.run(
        obs_times=obs_times,
        obs_values=obs_values,
        model_fn=get_settlement,
        var_map={"CR": "CR", "k": "k"},
        model_kwargs=model_kwargs,
        forecast_times=forecast_times,
        cache_dir=get_remote_path() / "output/cache",
        force_rebuild=force_rebuild,
    )

    # =========================================================================
    # Plots
    # =========================================================================
    variables = {v["name"]: v for v in settings.get("variables", [])}
    var_names = pipeline.jpdf.variable_names
    true_values = {name: variables.get(name, {}).get("true") for name in var_names}

    if pipeline.is_sample_based:
        samples_dict = {name: pipeline.jpdf.get_samples(name) for name in var_names}
        save_jpdf_plots(
            results=results, output_dir=output_dir, var_names=var_names,
            mode="samples", samples=samples_dict,
            W_prior=pipeline.jpdf.W_prior_samples,
            W_posterior_per_t=pipeline.W_posterior_per_t,
            true_values=true_values,
        )
    else:
        save_jpdf_plots(
            results=results, output_dir=output_dir, var_names=var_names,
            mode="grid", true_values=true_values,
        )

    save_settlement_forecast_plots(
        results=results, output_dir=output_dir,
        t_max=t_removal + 5, obs_error=config.get("obs_error", 0.1), y_max=obs_values.max(),
    )
    save_settlement_residual_plots(
        results=results, output_dir=output_dir,
        end_settlement_req=config.get("end_settlement_req", 0.05),
    )
    save_beta_over_time_plot(results=results, output_dir=output_dir)
    make_gifs(output_dir)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str, default="settings_1loc")
    parser.add_argument("--analysis_method", type=str, default="semi-analytical")
    parser.add_argument("--force_rebuild", action="store_false")
    args = parser.parse_args()

    main(
        input_file=args.input_file,
        analysis_method=args.analysis_method,
        force_rebuild=args.force_rebuild,
    )
