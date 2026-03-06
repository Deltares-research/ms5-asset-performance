import numpy as np
from scipy import stats as st
from numpy.typing import NDArray
from case_studies.settlement_example.io import save_json, load_json, get_remote_path
from jpdf import JPDF
from performance_function import Performance
from config import CaseStudyConfig
from typing import Optional, List, Dict, Tuple, Any
from dotenv import load_dotenv
from pathlib import Path
import os
import json
from datetime import datetime
from settlement_engine import get_settlement
from plotting import save_jpdf_plots, save_settlement_forecast_plots, save_settlement_residual_plots, make_gifs


class ReliabilityPipeline:

    def __init__(
        self,
        config: Optional[CaseStudyConfig] = None,
        specs_path: Optional[Path | str] = None,
    ) -> None:

        self.config = config or CaseStudyConfig()
        self.specs_path = specs_path

        # Components
        self.jpdf: Optional[JPDF] = None
        self.performance: Optional[Performance] = None
        self.settlement_at_obs_times: Optional[NDArray] = None
        self.settlement_residual: Optional[NDArray] = None
        self.settlement_forecast: Optional[NDArray] = None
        self.settlement_grid: Optional[NDArray] = None
        self.results: Dict[str, Any] = {}

    def setup(
        self,
        n_samples: int = 100_000,
        seed: int = 42,
    ) -> None:

        # Initialize JPDF
        self.jpdf = JPDF(name="settlement", config=self.config)
        self.jpdf.set_prior_from_specs(self.specs_path)

        # Initialize corrosion model
        with open(self.specs_path, "r") as f:
            specs = json.load(f)

        params = specs.get("parameters", {})
        self.performance = Performance(name="settlement", parameters=params)

    def init_times(self, setting) -> None:

        self.obs_times = np.array([float(key) for key in setting.keys()])

        preload_removal_time = self.config.preload_removal_time
        end_time = self.config.end_time
        forecast_interval = self.config.forecast_interval

        forecast_times = np.arange(0, preload_removal_time, forecast_interval)
        forecast_times = np.append(forecast_times, [preload_removal_time, end_time])
        forecast_times = np.append(forecast_times, self.obs_times)
        self.forecast_times = np.sort(np.unique(forecast_times))

    def init_settlements(
            self,
            setting: Dict[str, float],
            cache_dir: Optional[Path] = None,
            force_rebuild: bool = False
    ) -> None:

        self.init_times(setting)

        cache_file_obs = cache_dir / "settlements_obs.npy"
        cache_file_forecast = cache_dir / "settlement_forecast.npy"

        if not force_rebuild and cache_file_obs.exists() and cache_file_forecast.exists():
            
            settlements_obs = np.load(cache_file_obs)
            settlement_forecast = np.load(cache_file_forecast)
            
            if settlements_obs.shape[-1] != len(self.obs_times):
                raise ValueError(f"""
                Inconsistent times: cached settlements have {settlements_obs.shape[-1]} 
                times while analysis needs {len(self.obs_times)}.
                """)
            
        else:
    
            settlements_obs = get_settlement(
                t=self.obs_times,
                CR=self.jpdf.CR_grid,
                k=self.jpdf.k_grid,
                RR=self.config.RR,
                Ca=self.config.Ca,
                h=self.config.layer_thickness,
                sigma_0=self.config.sigma_0,
                sigma_v=self.config.sigma_0+self.config.preload,
                sigma_p=self.config.sigma_p,
                method=self.config.doc_method,
            )
            
            settlement_forecast = get_settlement(
                t=self.forecast_times,
                CR=self.jpdf.CR_grid,
                k=self.jpdf.k_grid,
                RR=self.config.RR,
                Ca=self.config.Ca,
                h=self.config.layer_thickness,
                sigma_0=self.config.sigma_0,
                sigma_v=self.config.sigma_0+self.config.preload,
                sigma_p=self.config.sigma_p,
                method=self.config.doc_method,
            )
            
            np.save(cache_file_obs, settlements_obs)
            np.save(cache_file_forecast, settlement_forecast)

        # If settlement matrix is takes more than 100MB memory, reduce its accuracy.
        if settlements_obs.nbytes / 1e6 >= 100:
            settlements_obs = settlements_obs.astype(np.float32)
            print("Reducing accuracy of 'settlements' to float32")
            print(f"Current memory for 'settlements'={settlements_obs.nbytes/1e3:.0f}KB")

        # If settlement matrix is takes more than 100MB memory, reduce its accuracy.
        if settlement_forecast.nbytes / 1e6 >= 100:
            settlement_forecast = settlement_forecast.astype(np.float32)
            print("Reducing accuracy of 'settlements' to float32")
            print(f"Current memory for 'settlements'={settlement_forecast.nbytes/1e3:.0f}KB")

        self.settlement_at_obs_times = settlements_obs
        self.settlement_residual = settlement_forecast[..., -1] - settlement_forecast[..., -2]
        self.settlement_forecast = settlement_forecast

        n_settlement_grid = 1_001

        settlement_min = min(np.nanmin(self.settlement_at_obs_times), np.nanmin(self.settlement_forecast))
        settlement_max = max(np.nanmax(self.settlement_at_obs_times), np.nanmax(self.settlement_forecast))
        settlement_grid = np.linspace(settlement_min, settlement_max, n_settlement_grid)
        self.settlement_grid = np.sort(np.unique(np.append(settlement_grid, 0)))

        diff_min = np.nanmin(self.settlement_residual)
        diff_max = np.nanmax(self.settlement_residual)
        diff_grid = np.linspace(diff_min, diff_max, n_settlement_grid)
        self.end_diff_grid = np.sort(np.unique(np.append(diff_grid, 0)))

    def get_settlement_pdf(self, settlement: NDArray, use_prior: bool = False, grid: NDArray = None) -> Tuple[NDArray, NDArray]:

        pdf = self.jpdf.get_prior() if use_prior else self.jpdf.posterior_pdf
        grid = grid if grid is not None else self.settlement_grid

        dCR = np.diff(self.jpdf.CR_grid)
        dk = np.diff(self.jpdf.k_grid)
        pdf_centers = (pdf[:-1, :-1] + pdf[:-1, 1:] + pdf[1:, :-1] + pdf[1:, 1:]) / 4
        prob_mass = pdf_centers * dCR[:, np.newaxis] * dk[np.newaxis, :]

        settlement_centers = (settlement[:-1, :-1] + settlement[:-1, 1:] + settlement[1:, :-1] + settlement[1:, 1:]) / 4

        s_flat = settlement_centers.flatten()
        pm_flat = prob_mass.flatten()
        valid = np.isfinite(s_flat) & np.isfinite(pm_flat)

        idx = np.digitize(s_flat[valid], bins=grid) - 1
        idx = np.clip(idx, 0, len(grid) - 1)

        settlement_prob_mass = np.zeros(len(grid))
        np.add.at(settlement_prob_mass, idx, pm_flat[valid])

        ds = np.diff(grid)
        settlement_pdf = settlement_prob_mass[:-1] / ds

        grid_centers = (grid[:-1] + grid[1:]) / 2
        settlement_pdf /= np.trapezoid(settlement_pdf, grid_centers)

        return grid_centers, settlement_pdf

    def compute_pf_at_time(self, forecast_time: float, use_prior: bool = False) -> None:

        pf = self.performance.failure_probability(
            x=self.settlement_residual,
            pdf=self.jpdf.get_prior() if use_prior else self.jpdf.posterior_pdf,
            CR_grid=self.jpdf.CR_grid,
            k_grid=self.jpdf.k_grid,
        )

        pf_clipped = np.clip(pf, 1e-10, 1 - 1e-10)
        beta = st.norm.ppf(1 - pf_clipped)

        settlement_at_forecast_time = self.settlement_forecast[..., self.forecast_times==forecast_time].squeeze()
        settlement_grid, settlement_pdf = self.get_settlement_pdf(
            settlement=settlement_at_forecast_time,
            use_prior=use_prior
        )

        return {
            "pf": pf.item(),
            "beta": beta.item(),
            "settlement_grid": settlement_grid.tolist(),
            "settlement_pdf": settlement_pdf.tolist(),
        }


    def run_timeline(self, setting: Dict[str, float], verbose=True) -> Dict[str, Any]:

        self.results = {}

        # Collect settlement observations
        def get_observations_up_to(t: float):
            obs_times = self.obs_times[self.obs_times<=t]
            obs_values = [float(v) for v in setting.values()]
            obs_values = np.array(obs_values)[self.obs_times<=t]
            return np.array(obs_times), np.array(obs_values)

        # Reset C50 to prior
        self.jpdf.reset_to_priors()

        for t in self.obs_times.tolist():

            if verbose:
                print(f"Processing t={t:.0f}...")

            # Get observations up to current time
            obs_times, obs_values = get_observations_up_to(t)

            # Update posterior
            if len(obs_times) > 0:
                settlement_at_obs_time = self.settlement_at_obs_times[..., self.obs_times<=t]
                self.jpdf.update(obs_values=obs_values, settlements=settlement_at_obs_time)

            # Compute Pf FORECAST for all future times (from t to t_end)
            prediction_times = [pt for pt in self.forecast_times.tolist()]
            pf_forecast_prior = {}
            pf_forecast_posterior = {}
            beta_forecast_prior = {}
            beta_forecast_posterior = {}
            settlement_prior_grid = {}
            settlement_forecast_prior = {}
            settlement_posterior_grid = {}
            settlement_forecast_posterior = {}

            for pt in prediction_times:

                # Prior forecast
                result_prior = self.compute_pf_at_time(forecast_time=pt, use_prior=True)
                pf_forecast_prior[pt] = result_prior["pf"]
                beta_forecast_prior[pt] = result_prior["beta"]
                settlement_prior_grid[pt] = result_prior["settlement_grid"]
                settlement_forecast_prior[pt] = result_prior["settlement_pdf"]

                # Posterior forecast
                result_posterior = self.compute_pf_at_time(forecast_time=pt, use_prior=False)
                pf_forecast_posterior[pt] = result_posterior["pf"]
                beta_forecast_posterior[pt] = result_posterior["beta"]
                settlement_posterior_grid[pt] = result_posterior["settlement_grid"]
                settlement_forecast_posterior[pt] = result_posterior["settlement_pdf"]

            # End differential settlement PDFs
            diff_grid_prior, diff_pdf_prior = self.get_settlement_pdf(
                settlement=self.settlement_residual,
                use_prior=True,
                grid=self.end_diff_grid,
            )
            diff_grid_posterior, diff_pdf_posterior = self.get_settlement_pdf(
                settlement=self.settlement_residual,
                use_prior=False,
                grid=self.end_diff_grid,
            )

            # Current time results
            t_min = min(beta_forecast_prior)
            pf_current_prior = pf_forecast_prior[t_min]
            pf_current_posterior = pf_forecast_posterior[t_min]
            beta_current_prior = beta_forecast_prior[t_min]
            beta_current_posterior = beta_forecast_posterior[t_min]

            self.results[t] = {
                "time": t,
                "obs_times": obs_times.tolist(),
                "settlement_obs": obs_values.tolist(),
                "prior": {
                    "pf": pf_current_prior,
                    "beta": beta_current_prior,
                    "pf_forecast": pf_forecast_prior,
                    "beta_forecast": beta_forecast_prior,
                    "settlement_prior_grid": settlement_prior_grid,
                    "settlement_forecast": settlement_forecast_prior,
                },
                "posterior": {
                    "pf": pf_current_posterior,
                    "beta": beta_current_posterior,
                    "pf_forecast": pf_forecast_posterior,
                    "beta_forecast": beta_forecast_posterior,
                    "settlement_posterior_grid": settlement_posterior_grid,
                    "settlement_forecast": settlement_forecast_posterior,
                },
                "settlement_residual": {
                    "prior_grid": diff_grid_prior.tolist(),
                    "prior_pdf": diff_pdf_prior.tolist(),
                    "posterior_grid": diff_grid_posterior.tolist(),
                    "posterior_pdf": diff_pdf_posterior.tolist(),
                },
                # Store JPDF state for snapshot generation
                "jpdf_state": {
                    "CR_grid": self.jpdf.CR_grid.tolist(),
                    "CR_prior": self.jpdf.CR_prior.tolist(),
                    "CR_posterior": self.jpdf.CR_pdf.tolist(),
                    "k_grid": self.jpdf.k_grid.tolist(),
                    "k_prior": self.jpdf.k_prior.tolist(),
                    "k_posterior": self.jpdf.k_pdf.tolist(),
                    "prior": self.jpdf.get_prior().tolist(),
                    "posterior": self.jpdf.posterior_pdf.tolist(),
                },
            }

            if verbose:
                print(f"  Prior:     Pf={result_prior['pf']:.2e}, beta={result_prior['beta']:.2f}")
                print(f"  Posterior: Pf={result_posterior['pf']:.2e}, beta={result_posterior['beta']:.2f}")

        return self.results

    def save_results(self, filename: str = "reliability_results.json") -> None:
        """Save results to file."""
        # Convert to JSON-serializable format
        results_json = {}
        for t, data in self.results.items():
            results_json[str(t)] = data

        load_dotenv("ark_example.env")
        username = os.environ.get("USER", "unknown").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_folder = f"{username}_{timestamp}/{filename}"

        save_json(results_json, output_folder)


def main():
    # Paths
    load_dotenv("settlement_example.env")
    os.environ["REMOTE_DATA_PATH"] = str(get_remote_path()/"input")
    specs_path = Path(os.environ["REMOTE_DATA_PATH"]) / "case_study_specifications.json"

    # Initialize pipeline
    config = CaseStudyConfig.from_json(specs_path)
    pipeline = ReliabilityPipeline(config=config, specs_path=specs_path)

    # =========================================================================
    # STEP 1: Setup
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Setup")
    print("=" * 60)
    pipeline.setup(n_samples=1_000_000, seed=42)

    # =========================================================================
    # STEP 2: Initialize (or load) pre-evaluated settlements
    # =========================================================================
    print("=" * 60)
    print("STEP 2: Settlements")
    print("=" * 60)
    setting = load_json("case_study_setting.json")
    cache_dir = get_remote_path() / "output/cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    pipeline.init_settlements(setting=setting, force_rebuild=True, cache_dir=cache_dir)

    # =========================================================================
    # STEP 3: Run timeline analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Run timeline analysis")
    print("=" * 60)

    # Run timeline analysis
    results = pipeline.run_timeline(setting=setting, verbose=True)

    # Save results
    pipeline.save_results()
    print("Results saved.")

    # =========================================================================
    # STEP 4: Plots
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Plots")
    print("=" * 60)

    username = os.environ.get("USER", "unknown").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = get_remote_path() / f"output/results/{username}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(specs_path, "r") as f:
        specs = json.load(f)
    variables = {v["name"]: v for v in specs.get("variables", [])}
    CR_true = variables.get("CR", {}).get("true")
    k_true = variables.get("k", {}).get("true")

    obs_values = np.array([float(v) for v in setting.values()])
    save_jpdf_plots(results=results, output_dir=output_dir, CR_true=CR_true, k_true=k_true)
    save_settlement_forecast_plots(
        results=results,
        output_dir=output_dir,
        t_max=config.preload_removal_time+5,
        obs_error=config.obs_error,
        y_max=obs_values.max()
    )
    save_settlement_residual_plots(
        results=results,
        output_dir=output_dir,
        end_settlement_req=config.end_settlement_req
    )
    make_gifs(output_dir)


if __name__ == "__main__":

    main()

