import numpy as np
from scipy import stats as st
from geolib.models.base_model_structure import settings
from numpy.typing import NDArray
from case_studies.settlement_example.io import load_json, get_remote_path
from jpdf import JPDF
from performance_function import Performance
from config import CaseStudyConfig
from typing import Optional, List, Dict, Tuple
from dotenv import load_dotenv
from pathlib import Path
import os
import json
from settlement_engine import get_settlement


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
        self.end_diff_settlement: Optional[NDArray] = None
        self.settlement_forecast: Optional[NDArray] = None
        self.settlement_grid: Optional[NDArray] = None

        # Results
        self.results: Dict[float, Dict[str, Any]] = {}

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
                sigma_v=self.config.sigma_v,
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
                sigma_v=self.config.sigma_v,
                sigma_p=self.config.sigma_p,
                method=self.config.doc_method,
            )
            
            np.save(cache_file_obs, settlements_obs)
            np.save(cache_file_forecast, settlement_forecast)

        # If settlement matrix is takes more than 100MB memory, reduce its accuracy.
        if settlements_obs.nbytes / 1e6 >= 100:
            settlements_obs = settlements_obs.astype(float32)
            print("Reducing accuracy to of 'settlements' to float32")
            print(f"Current memory for 'settlements'={settlements_obs.nbytes/1e3:.0f}KB")

        # If settlement matrix is takes more than 100MB memory, reduce its accuracy.
        if settlement_forecast.nbytes / 1e6 >= 100:
            settlement_forecast = settlement_forecast.astype(float32)
            print("Reducing accuracy to of 'settlements' to float32")
            print(f"Current memory for 'settlements'={settlement_forecast.nbytes/1e3:.0f}KB")

        self.settlement_at_obs_times = settlements_obs
        self.end_diff_settlement = settlement_forecast[..., -1] - settlement_forecast[..., -2]
        self.settlement_forecast = settlement_forecast

        self.settlement_grid = np.linspace(
            max(self.settlement_at_obs_times.max(), self.settlement_forecast.max()),
            max(self.settlement_at_obs_times.max(), self.settlement_forecast.max()),
            1_000
        )

    def settlement_pdf(self, settlement: NDArray, use_prior: bool = False) -> NDArray:

        pdf = self.jpdf.get_prior() if use_prior else self.jpdf.posterior_pdf

        settlement_pdf = np.zeros_like(self.settlement_grid)

        settlement_pdf /= np.trapezoid(settlement_pdf, self.settlement_grid)


        return settlement_pdf

    def compute_pf_at_time(
            self,
            forecast_time: float,
            use_prior: bool = False,
    ) -> None:

        settlement_at_forecast_time = self.settlement_forecast[..., self.forecast_times==forecast_time].squeeze()

        pf = self.performance.failure_probability(
            x=settlement_at_forecast_time,
            pdf=self.jpdf.get_prior() if use_prior else self.jpdf.posterior_pdf,
            CR_grid=self.jpdf.CR_grid,
            k_grid=self.jpdf.k_grid,
        )

        beta = st.norm.ppf(1-pf)

        self.get_settlement_pdf(
            settleent=settlement_at_forecast_time,
            use_prior=use_prior
        )

        return {
            "pf": pf,
            "beta": beta,

        }


    def run_timeline(self, setting: Dict[str, float], verbose=True) -> None:

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
            future_times = [ft for ft in self.forecast_times if ft >= t]
            beta_forecast_prior = {}
            beta_forecast_posterior = {}
            settlement_forecast_prior = {}
            settlement_forecast_posterior = {}

            for ft in future_times:

                # Prior forecast
                result_prior = self.compute_pf_at_time(forecast_time=ft, use_prior=True)
                beta_forecast_prior[ft] = result_prior["beta"]

                # Posterior forecast
                result_posterior = self.compute_pf_at_time(forecast_time=ft, use_prior=False)
                beta_forecast_posterior[ft] = result_posterior["beta"]

                # Store corrosion ratio PDFs for forecast times
                settlement_forecast_prior[ft] = result_prior["settlement"]
                settlement_forecast_posterior[ft] = result_posterior["settlement"]

            # Current time results
            beta_current_prior = beta_forecast_prior[max(beta_forecast_prior)]
            beta_current_posterior = beta_forecast_posterior[max(beta_forecast_posterior)]
            beta_current_posterior_proven_strength = beta_forecast_posterior_proven_strength[max(beta_forecast_posterior_proven_strength)]

            cr_grid_prior, _ = self.get_corrosion_ratio_pdf(t, self.jpdf.C50_prior)

            self.results[t] = {
                "time": t,
                "settlement_obs": settlement_obs,
                "prior": {
                    "beta": beta_current_prior,
                    "beta_forecast": beta_forecast_prior,
                    "settlement_forecast": settlement_forecast_prior,
                },
                "posterior": {
                    "beta": beta_current_posterior,
                    "beta_forecast": beta_forecast_posterior,
                    "settlement_forecast": settlement_forecast_posterior,
                },
                # Store JPDF state for snapshot generation
                "jpdf_state": {
                    "CR_grid": self.jpdf.CR_grid.tolist(),
                    "CR_prior": self.jpdf.CR_prior.tolist(),
                    "CR_posterior": self.jpdf.CR_pdf.tolist(),
                    "k_grid": self.jpdf.k_grid.tolist(),
                    "k_prior": self.jpdf.k_prior.tolist(),
                    "k_posterior": self.jpdf.k_pdf.tolist(),
                    "prior": self.jpdf.get_prior(),
                    "posterior": self.jpdf.posterior_pdf,
                },
            }

            if verbose:
                print(f"  Prior:     Pf={result_prior['pf']:.2e}, beta={result_prior['beta']:.2f}")
                print(f"  Posterior: Pf={result_posterior['pf']:.2e}, beta={result_posterior['beta']:.2f}")

        return self.results


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
    pipeline.init_settlements(setting=setting, force_rebuild=False, cache_dir=cache_dir)

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
    
    
    pass


if __name__ == "__main__":

    main()

