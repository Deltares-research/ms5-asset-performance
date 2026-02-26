"""
Pipeline for D-Sheet piling reliability analysis.

Workflow:
1. Load configuration and initialize JPDF
2. Build or load fragility curve
3. For each timestep:
   - Update C50 posterior with corrosion observations
   - Compute corrosion ratio PDF
   - Integrate fragility to get Pf
4. Store and visualize results
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from case_studies.dsheet_example.config import CaseStudyConfig
from case_studies.dsheet_example.jpdf import JPDF
from case_studies.dsheet_example.corrosion import CorrosionModel
from case_studies.dsheet_example.performance_function import (
    Performance,
    FragilityCurve,
    FragilitySurfaceIndex,
    MLP,
)
from case_studies.dsheet_example import io
from case_studies.dsheet_example import plotting
from dotenv import load_dotenv


class ReliabilityPipeline:
    """
    Pipeline for time-dependent reliability analysis with corrosion.

    Args:
        config: Case study configuration.
        specs_path: Path to case study specifications JSON.
    """

    def __init__(
        self,
        config: Optional[CaseStudyConfig] = None,
        specs_path: Optional[Path | str] = None,
    ):
        self.config = config or CaseStudyConfig()
        self.specs_path = specs_path

        # Components
        self.jpdf: Optional[JPDF] = None
        self.corrosion_model: Optional[CorrosionModel] = None
        self.performance: Optional[Performance] = None
        self.fragility: Optional[FragilityCurve] = None
        self.fragility_surface: Optional[FragilitySurfaceIndex] = None

        # Results
        self.results: Dict[float, Dict[str, Any]] = {}

    def setup(
        self,
        n_samples: int = 100_000,
        seed: int = 42,
    ) -> None:
        """
        Initialize JPDF, corrosion model, and performance function.

        Args:
            n_samples: Number of MC samples.
            seed: Random seed.
        """
        # Initialize JPDF
        self.jpdf = JPDF(name="dsheet", config=self.config)

        if self.specs_path is not None:
            self.jpdf.set_prior_from_specs(self.specs_path)
        else:
            # Initialize C50 with defaults
            self.jpdf.init_C50_prior(C50_mu=1.5, C50_std=0.75)

        # Generate samples
        self.jpdf.initiate_samples(n_samples=n_samples, seed=seed)
        self.jpdf.add_water_level(water_lvl=-1.0)

        # Initialize corrosion model
        self.corrosion_model = CorrosionModel(
            C50_mu=1.5,
            C50_std=0.75,
            corrosion_rate=self.config.corrosion_rate,
            start_thickness=self.config.start_thickness,
            obs_error_std=self.config.obs_error_std,
            t_ref=self.config.t_ref,
            n_grid=self.config.n_C50_grid,
        )

        # Initialize performance function
        params = {
            "moment_cap": self.config.moment_cap,
            "EI_start": self.config.EI_start,
            "ei_column_idx": -2,
        }
        self.performance = Performance(name="dsheet_moment", parameters=params)

    def load_surrogate(
        self,
        model_path: Optional[Path | str] = None,
        input_dim: int = 11,
        hidden_dims: list = [1024, 512, 256, 128, 64, 32],
        output_dim: int = 1,
    ) -> None:
        """
        Load trained surrogate model.

        Args:
            model_path: Path to model directory. Uses io.load_surrogate_model if None.
            input_dim: MLP input dimension.
            hidden_dims: MLP hidden layer sizes.
            output_dim: MLP output dimension.
        """
        if self.performance is None:
            raise ValueError("Call setup() first.")

        model_kwargs = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
        }

        model, scaler_x, scaler_y = io.load_surrogate_model(MLP, model_kwargs)

        self.performance.set_surrogate(model, scaler_x, scaler_y)

    def build_fragility(
        self,
        n_grid: int = 100,
        cache_moments: bool = True,
        force_rebuild: bool = False,
        verbose: bool = True,
    ) -> FragilityCurve:
        """
        Build or load fragility curve.

        Args:
            n_grid: Number of corrosion ratio grid points.
            cache_moments: Cache max moments per CR (memory intensive).
            force_rebuild: Rebuild even if cached version exists.
            verbose: Print progress.

        Returns:
            FragilityCurve instance.
        """
        if self.performance is None or self.jpdf is None:
            raise ValueError("Call setup() first.")

        # Check for cached fragility
        if not force_rebuild and io.fragility_exists("fragility"):
            if verbose:
                print("Loading cached fragility curve...")
            self.fragility = io.load_fragility_curve("fragility")
            return self.fragility

        if verbose:
            print(f"Building fragility curve ({n_grid} grid points)...")

        self.fragility = self.performance.build_fragility_curve(
            x=self.jpdf.X_samples,
            n_grid=n_grid,
            cache_moments=cache_moments,
            verbose=verbose,
        )

        # Save to cache
        io.save_fragility_curve(self.fragility, "fragility", fmt="npz")

        if verbose:
            print(f"Fragility curve saved.")
            print(self.fragility.summary())

        return self.fragility

    # -------------------------------------------------------------------------
    # Fragility Surface (2D: corrosion_ratio x moment_survived)
    # -------------------------------------------------------------------------

    def build_fragility_surface(
        self,
        n_cr: int = 100,
        n_moments: int = 50,
        moment_range: Optional[Tuple[float, float]] = None,
        force_rebuild: bool = False,
        verbose: bool = True,
    ) -> FragilitySurfaceIndex:
        """
        Build or load 2D fragility surface.

        This is an explicit pipeline step that pre-computes fragility curves
        for a grid of moment_survived values. Each curve is stored separately
        for memory-efficient on-demand loading.

        Args:
            n_cr: Number of corrosion ratio grid points.
            n_moments: Number of moment_survived grid points.
            moment_range: (min, max) for moment_survived. If None, auto-computed.
            force_rebuild: Rebuild even if cached version exists.
            verbose: Print progress.

        Returns:
            FragilitySurfaceIndex instance.
        """
        if self.performance is None or self.jpdf is None:
            raise ValueError("Call setup() first.")

        # Check for cached surface
        if not force_rebuild and io.fragility_surface_exists():
            if verbose:
                print("Loading cached fragility surface...")
            self.fragility_surface = io.load_fragility_surface()
            if verbose:
                print(self.fragility_surface.summary())
            return self.fragility_surface

        if verbose:
            print(f"Building fragility surface ({n_cr} CR x {n_moments} moments)...")

        self.fragility_surface = self.performance.build_fragility_surface(
            x=self.jpdf.X_samples,
            n_cr=n_cr,
            n_moments=n_moments,
            moment_range=moment_range,
            verbose=verbose,
        )

        # Save to cache
        if verbose:
            print("Saving fragility surface...")
        io.save_fragility_surface(self.fragility_surface)

        if verbose:
            print("Fragility surface saved.")

        return self.fragility_surface

    def load_fragility_surface(
        self,
        name: str = "fragility_surface",
        verbose: bool = True,
    ) -> FragilitySurfaceIndex:
        """
        Load pre-computed fragility surface from disk.

        This is an explicit pipeline step that loads only the manifest.
        Individual curves are loaded on-demand when queried.

        Args:
            name: Name of the fragility surface directory.
            verbose: Print progress.

        Returns:
            FragilitySurfaceIndex instance.
        """
        if not io.fragility_surface_exists(name):
            raise FileNotFoundError(
                f"Fragility surface '{name}' not found. Run build_fragility_surface() first."
            )

        if verbose:
            print(f"Loading fragility surface '{name}'...")

        self.fragility_surface = io.load_fragility_surface(name)

        if verbose:
            print(self.fragility_surface.summary())

        return self.fragility_surface

    def fragility_surface_exists(self, name: str = "fragility_surface") -> bool:
        """Check if a fragility surface exists."""
        return io.fragility_surface_exists(name)

    def get_fragility_curve_at(
        self,
        moment_survived: float,
    ) -> FragilityCurve:
        """
        Get fragility curve for a specific moment_survived threshold.

        Loads the curve from disk if not already cached. Interpolates
        between adjacent curves if exact moment not in grid.

        Args:
            moment_survived: Proven moment capacity threshold [kNm].

        Returns:
            FragilityCurve instance.
        """
        if self.fragility_surface is None:
            raise ValueError(
                "Call build_fragility_surface() or load_fragility_surface() first."
            )

        return self.fragility_surface.get_curve_at(moment_survived)

    def compute_pf_with_proven_strength(
        self,
        t: float,
        moment_survived: float,
        use_posterior: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute Pf at time t conditioned on proven strength.

        Uses the fragility surface for fast lookup.

        Args:
            t: Time [years].
            moment_survived: Proven moment capacity threshold [kNm].
            use_posterior: Use posterior C50 (True) or prior (False).

        Returns:
            Dict with pf, beta, and intermediate values.
        """
        if self.fragility_surface is None:
            raise ValueError(
                "Call build_fragility_surface() or load_fragility_surface() first."
            )

        # Get C50 PDF
        C50_pdf = self.jpdf.C50_pdf if use_posterior else self.jpdf.C50_prior

        # Get corrosion ratio PDF
        cr_grid, cr_pdf = self.get_corrosion_ratio_pdf(t, C50_pdf)

        # Get fragility curve at moment_survived (loads on-demand)
        fragility = self.fragility_surface.get_curve_at(moment_survived)

        # Interpolate PDF to fragility grid
        cr_pdf_interp = np.interp(
            fragility.corrosion_ratios, cr_grid, cr_pdf, left=0, right=0
        )
        # Renormalize
        cr_pdf_interp /= np.trapezoid(cr_pdf_interp, fragility.corrosion_ratios) + 1e-10

        # Integrate fragility
        pf, beta = self.performance.pf_from_fragility(fragility, cr_pdf_interp)

        return {
            "time": t,
            "moment_survived": moment_survived,
            "pf": pf,
            "beta": beta,
            "C50_stats": self.jpdf.get_C50_stats(),
            "cr_pdf": cr_pdf_interp.tolist(),
        }

    def get_corrosion_ratio_pdf(
        self,
        t: float,
        C50_pdf: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute corrosion ratio PDF at time t given C50 distribution.

        Args:
            t: Time [years].
            C50_pdf: PDF over C50 grid. Uses jpdf.C50_pdf if None.

        Returns:
            Tuple of (corrosion_ratio_grid, pdf).
        """
        if self.corrosion_model is None or self.jpdf is None:
            raise ValueError("Call setup() first.")

        if C50_pdf is None:
            C50_pdf = self.jpdf.C50_pdf

        # Use corrosion model to get PDF
        ratio_grid, ratio_pdf = self.corrosion_model.corrosion_ratio_pdf(
            t=t, C50_pdf=C50_pdf
        )

        return ratio_grid, ratio_pdf

    def compute_pf_at_time(
        self,
        t: float,
        use_posterior: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute failure probability at a single timestep.

        Args:
            t: Time [years].
            use_posterior: Use posterior C50 (True) or prior (False).

        Returns:
            Dict with pf, beta, and intermediate values.
        """
        if self.fragility is None:
            raise ValueError("Call build_fragility() first.")

        # Get C50 PDF
        C50_pdf = self.jpdf.C50_pdf if use_posterior else self.jpdf.C50_prior

        # Get corrosion ratio PDF
        cr_grid, cr_pdf = self.get_corrosion_ratio_pdf(t, C50_pdf)

        # Interpolate PDF to fragility grid
        cr_pdf_interp = np.interp(
            self.fragility.corrosion_ratios, cr_grid, cr_pdf, left=0, right=0
        )
        # Renormalize
        cr_pdf_interp /= np.trapezoid(cr_pdf_interp, self.fragility.corrosion_ratios) + 1e-10

        # Integrate fragility
        pf, beta = self.performance.pf_from_fragility(self.fragility, cr_pdf_interp)

        return {
            "time": t,
            "pf": pf,
            "beta": beta,
            "C50_stats": self.jpdf.get_C50_stats(),
            "cr_pdf": cr_pdf_interp.tolist(),
        }

    def run_timeline(
        self,
        setting: Dict[str, Any],
        verbose: bool = True,
    ) -> Dict[float, Dict[str, Any]]:
        """
        Run reliability analysis over timeline.

        Args:
            setting: Case study setting with time-series data.
            verbose: Print progress.

        Returns:
            Results dict keyed by time.
        """
        if self.fragility is None:
            raise ValueError("Call build_fragility() first.")

        self.results = {}

        # Get times from setting
        times = sorted([float(k) for k in setting.keys() if k != "metadata"])

        # Collect corrosion observations
        def get_observations_up_to(t: float):
            obs_times = []
            obs_values = []
            for time_key in sorted(setting.keys()):
                if time_key == "metadata":
                    continue
                time = float(time_key)
                if time <= t:
                    obs_times.append(time)
                    obs_values.append(setting[time_key]["corrosion"])
            return np.array(obs_times), np.array(obs_values)

        # Reset C50 to prior
        self.jpdf.reset_C50_to_prior()

        for t in times:
            if verbose:
                print(f"Processing t={t:.0f}...")

            # Get observations up to current time
            obs_times, obs_values = get_observations_up_to(t)

            # Update C50 posterior
            if len(obs_times) > 0:
                self.jpdf.update_C50(obs_times, obs_values)

            # Get observed corrosion ratio
            key = str(t) if str(t) in setting else f"{t:.1f}"
            corrosion_obs = setting[key]["corrosion"] if key in setting else None
            cr_obs = corrosion_obs / self.config.start_thickness if corrosion_obs else None

            # Compute Pf FORECAST for all future times (from t to t_end)
            future_times = [ft for ft in times if ft >= t]
            pf_forecast_prior = {}
            beta_forecast_prior = {}
            pf_forecast_posterior = {}
            beta_forecast_posterior = {}
            cr_forecast_prior = {}
            cr_forecast_posterior = {}

            for ft in future_times:
                # Prior forecast (using fixed prior C50)
                result_prior = self.compute_pf_at_time(ft, use_posterior=False)
                pf_forecast_prior[ft] = result_prior["pf"]
                beta_forecast_prior[ft] = result_prior["beta"]

                # Posterior forecast (using current C50 posterior)
                result_posterior = self.compute_pf_at_time(ft, use_posterior=True)
                pf_forecast_posterior[ft] = result_posterior["pf"]
                beta_forecast_posterior[ft] = result_posterior["beta"]

                # Store corrosion ratio PDFs for forecast times
                cr_grid, cr_pdf_prior = self.get_corrosion_ratio_pdf(ft, self.jpdf.C50_prior)
                _, cr_pdf_post = self.get_corrosion_ratio_pdf(ft, self.jpdf.C50_pdf)
                cr_forecast_prior[ft] = cr_pdf_prior.tolist()
                cr_forecast_posterior[ft] = cr_pdf_post.tolist()

            # Current time results
            result_prior_current = self.compute_pf_at_time(t, use_posterior=False)
            result_posterior_current = self.compute_pf_at_time(t, use_posterior=True)

            # Get corrosion ratio PDFs at current time
            cr_grid_prior, cr_pdf_prior = self.get_corrosion_ratio_pdf(t, self.jpdf.C50_prior)
            cr_grid_post, cr_pdf_post = self.get_corrosion_ratio_pdf(t, self.jpdf.C50_pdf)

            self.results[t] = {
                "time": t,
                "corrosion": corrosion_obs,
                "corrosion_ratio": cr_obs,
                "prior": {
                    "pf": result_prior_current["pf"],
                    "beta": result_prior_current["beta"],
                    "cr_pdf": cr_pdf_prior.tolist(),
                    "pf_forecast": pf_forecast_prior,
                    "beta_forecast": beta_forecast_prior,
                    "cr_forecast": cr_forecast_prior,
                },
                "posterior": {
                    "pf": result_posterior_current["pf"],
                    "beta": result_posterior_current["beta"],
                    "C50_stats": result_posterior_current["C50_stats"],
                    "cr_pdf": cr_pdf_post.tolist(),
                    "pf_forecast": pf_forecast_posterior,
                    "beta_forecast": beta_forecast_posterior,
                    "cr_forecast": cr_forecast_posterior,
                },
                # Store JPDF state for snapshot generation
                "jpdf_state": {
                    "C50_grid": self.jpdf.C50_grid.tolist(),
                    "C50_prior": self.jpdf.C50_prior.tolist(),
                    "C50_posterior": self.jpdf.C50_pdf.tolist(),
                    "cr_grid": cr_grid_prior.tolist(),
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

        io.save_json(results_json, f"results/{filename}")

    def save_jpdf_snapshots(
        self,
        setting: Dict[str, Any],
        output_dir: Optional[Path | str] = None,
        save_pdf: bool = True,
    ) -> None:
        """
        Generate and save JPDF snapshot images for each timestep.

        Args:
            setting: Case study setting with observations.
            output_dir: Directory for snapshot images.
            save_pdf: Also save all snapshots to a single PDF.
        """
        if not self.results:
            print("No results to plot. Run run_timeline() first.")
            return

        if self.fragility is None:
            print("No fragility curve. Run build_fragility() first.")
            return

        if output_dir is None:
            output_dir = io.get_remote_path() / "results/jpdf_snapshots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all observations
        times = sorted(self.results.keys())
        obs_times = []
        obs_corrosion = []
        for t in times:
            key = str(t) if str(t) in setting else f"{t:.1f}"
            if key in setting and "corrosion" in setting[key]:
                obs_times.append(t)
                obs_corrosion.append(setting[key]["corrosion"])

        figs = []

        for t in times:
            result = self.results[t]
            jpdf_state = result["jpdf_state"]

            # Extract data
            C50_grid = np.array(jpdf_state["C50_grid"])
            C50_prior = np.array(jpdf_state["C50_prior"])
            C50_posterior = np.array(jpdf_state["C50_posterior"])
            cr_grid = np.array(jpdf_state["cr_grid"])
            cr_pdf_prior = np.array(result["prior"]["cr_pdf"])
            cr_pdf_posterior = np.array(result["posterior"]["cr_pdf"])

            # Current observed corrosion ratio
            current_cr = result.get("corrosion_ratio")

            # Generate snapshot
            fig = plotting.plot_jpdf_snapshot(
                time=t,
                C50_grid=C50_grid,
                C50_prior=C50_prior,
                C50_posterior=C50_posterior,
                corrosion_ratio_grid=cr_grid,
                cr_pdf_prior=cr_pdf_prior,
                cr_pdf_posterior=cr_pdf_posterior,
                fragility_cr=self.fragility.corrosion_ratios,
                fragility_pf=self.fragility.pf,
                pf_prior=result["prior"]["pf"],
                pf_posterior=result["posterior"]["pf"],
                beta_prior=result["prior"]["beta"],
                beta_posterior=result["posterior"]["beta"],
                pf_forecast_prior=result["prior"].get("pf_forecast"),
                pf_forecast_posterior=result["posterior"].get("pf_forecast"),
                obs_times=obs_times,
                obs_corrosion=obs_corrosion,
                current_cr=current_cr,
                moment_cap=self.config.moment_cap,
                start_thickness=self.config.start_thickness,
            )

            # Save individual PNG
            plotting.save_figure(fig, output_dir / f"jpdf_t{int(t):03d}.png")
            figs.append(fig)

        # Save all to PDF
        if save_pdf:
            # Need to regenerate figures since save_figure closes them
            figs_for_pdf = []
            for t in times:
                result = self.results[t]
                jpdf_state = result["jpdf_state"]

                fig = plotting.plot_jpdf_snapshot(
                    time=t,
                    C50_grid=np.array(jpdf_state["C50_grid"]),
                    C50_prior=np.array(jpdf_state["C50_prior"]),
                    C50_posterior=np.array(jpdf_state["C50_posterior"]),
                    corrosion_ratio_grid=np.array(jpdf_state["cr_grid"]),
                    cr_pdf_prior=np.array(result["prior"]["cr_pdf"]),
                    cr_pdf_posterior=np.array(result["posterior"]["cr_pdf"]),
                    fragility_cr=self.fragility.corrosion_ratios,
                    fragility_pf=self.fragility.pf,
                    pf_prior=result["prior"]["pf"],
                    pf_posterior=result["posterior"]["pf"],
                    beta_prior=result["prior"]["beta"],
                    beta_posterior=result["posterior"]["beta"],
                    pf_forecast_prior=result["prior"].get("pf_forecast"),
                    pf_forecast_posterior=result["posterior"].get("pf_forecast"),
                    obs_times=obs_times,
                    obs_corrosion=obs_corrosion,
                    current_cr=result.get("corrosion_ratio"),
                    moment_cap=self.config.moment_cap,
                    start_thickness=self.config.start_thickness,
                )
                figs_for_pdf.append(fig)

            plotting.save_figures_to_pdf(figs_for_pdf, output_dir / "jpdf_snapshots.pdf")

        print(f"JPDF snapshots saved to {output_dir}")

    def plot_results(self, output_dir: Optional[Path | str] = None) -> None:
        """
        Generate plots of results with forecasts.

        For each observation time t, generates a plot showing:
        - Prior forecast (gray dashed, no updating)
        - Posterior forecasts from all past observation times
        - Current observation time's forecast (emphasized)

        Also generates:
        - Grid plot with all observation times in subplots
        - Evolution plot showing all forecasts on one figure

        Args:
            output_dir: Directory for plot files. Uses remote path if None.
        """
        if not self.results:
            print("No results to plot.")
            return

        if output_dir is None:
            output_dir = io.get_remote_path() / "results/plots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        times = sorted(self.results.keys())

        # Generate one plot per timestep showing cumulative forecasts
        for t in times:
            # Get all results up to and including time t
            results_up_to_t = {k: v for k, v in self.results.items() if k <= t}

            fig = plotting.plot_beta_forecast_at_time(
                current_time=t,
                results=results_up_to_t,
                beta_req=3.8,
            )
            plotting.save_figure(fig, output_dir / f"beta_forecast_t{int(t):03d}.png")

        # Save all individual plots to PDF
        figs_for_pdf = []
        for t in times:
            results_up_to_t = {k: v for k, v in self.results.items() if k <= t}
            fig = plotting.plot_beta_forecast_at_time(
                current_time=t,
                results=results_up_to_t,
                beta_req=3.8,
            )
            figs_for_pdf.append(fig)
        plotting.save_figures_to_pdf(figs_for_pdf, output_dir / "beta_forecasts.pdf")

        # Generate grid plot with all observation times
        fig_grid = plotting.plot_beta_forecast_grid(
            results=self.results,
            beta_req=3.8,
            ncols=3,
            title="Reliability Index Forecasts per Observation Time",
        )
        plotting.save_figure(fig_grid, output_dir / "beta_forecast_grid.png")

        # Generate evolution plot (all forecasts on one figure)
        fig_evolution = plotting.plot_posterior_forecast_evolution(
            results=self.results,
            beta_req=3.8,
            title="Posterior Forecast Evolution with Observations",
        )
        plotting.save_figure(fig_evolution, output_dir / "beta_forecast_evolution.png")

        print(f"Plots saved to {output_dir}")

    def plot_prior_posterior_pdfs(
        self,
        setting: Dict[str, Any],
        output_dir: Optional[Path | str] = None,
    ) -> None:
        """
        Generate prior vs posterior plots for corrosion and moment capacity.

        Args:
            setting: Case study setting with time-series data.
            output_dir: Directory for plot files.
        """
        if output_dir is None:
            output_dir = io.get_remote_path() / "results/plots_prior_posterior"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        times = sorted([float(k) for k in setting.keys() if k != "metadata"])
        metadata = setting.get("metadata", {})

        # Extract corrosion observations
        obs_times = []
        obs_corrosion = []
        obs_moment_cap = []

        moment_cap_start = self.config.moment_cap
        start_thickness = self.config.start_thickness

        for t in times:
            key = str(t) if str(t) in setting else f"{t:.1f}"
            if key in setting:
                obs_times.append(t)
                obs_corrosion.append(setting[key]["corrosion"])
                cr = setting[key].get("corrosion_ratio", setting[key]["corrosion"] / start_thickness)
                obs_moment_cap.append(moment_cap_start * (1 - cr))

        # Compute corrosion and moment capacity forecasts
        # For simplicity, use linear model based on C50 stats
        C50_stats_prior = {"mean": 1.5, "std": 0.75}

        # Forecast time grid: from first to last observation with specified interval
        t_start = times[0]
        t_end = times[-1]
        interval = self.config.forecast_interval
        times_forecast = np.arange(t_start, t_end + interval, interval)

        # Prior forecasts
        # Corrosion model: corrosion(t) = C50 + rate * (t - t_ref)
        corrosion_rate = self.config.corrosion_rate
        t_ref = 50.0
        C50_mu = C50_stats_prior["mean"]
        C50_std = C50_stats_prior["std"]

        corrosion_mean_prior = C50_mu + corrosion_rate * (times_forecast - t_ref)
        corrosion_std_prior = np.full_like(times_forecast, C50_std)  # std doesn't change with time
        corrosion_q05_prior = corrosion_mean_prior - 1.645 * corrosion_std_prior
        corrosion_q95_prior = corrosion_mean_prior + 1.645 * corrosion_std_prior

        # Posterior: condition on observations up to each time
        # The posterior mean passes exactly through the last observation
        if self.results and obs_times:
            result_times = sorted(self.results.keys())
            corrosion_mean_post = np.zeros_like(times_forecast)
            corrosion_std_post = np.zeros_like(times_forecast)

            for i, t_forecast in enumerate(times_forecast):
                # Find the latest observation time <= t_forecast
                relevant_obs_times = [t for t in obs_times if t <= t_forecast]
                if relevant_obs_times:
                    latest_obs_t = max(relevant_obs_times)
                    latest_obs_idx = obs_times.index(latest_obs_t)
                    latest_obs_corrosion = obs_corrosion[latest_obs_idx]

                    # Infer C50 from the last observation: C50 = obs - rate * (t_obs - t_ref)
                    C50_inferred = latest_obs_corrosion - corrosion_rate * (latest_obs_t - t_ref)

                    # Use inferred C50 to compute corrosion at forecast time
                    corrosion_mean_post[i] = C50_inferred + corrosion_rate * (t_forecast - t_ref)

                    # Get std from results if available, else use prior std
                    if latest_obs_t in self.results:
                        C50_stats_t = self.results[latest_obs_t]["posterior"].get("C50_stats", C50_stats_prior)
                        corrosion_std_post[i] = C50_stats_t.get("std", C50_std)
                    else:
                        corrosion_std_post[i] = C50_std
                else:
                    # Before first observation, use prior
                    corrosion_mean_post[i] = C50_mu + corrosion_rate * (t_forecast - t_ref)
                    corrosion_std_post[i] = C50_std
        else:
            corrosion_mean_post = C50_mu + corrosion_rate * (times_forecast - t_ref)
            corrosion_std_post = np.full_like(times_forecast, C50_std)

        corrosion_q05_post = corrosion_mean_post - 1.645 * corrosion_std_post
        corrosion_q95_post = corrosion_mean_post + 1.645 * corrosion_std_post

        # Corrosion plot
        fig = plotting.plot_corrosion_prior_posterior(
            times_forecast=times_forecast,
            corrosion_mean_prior=corrosion_mean_prior,
            corrosion_q05_prior=np.maximum(corrosion_q05_prior, 0),
            corrosion_q95_prior=corrosion_q95_prior,
            corrosion_mean_posterior=corrosion_mean_post,
            corrosion_q05_posterior=np.maximum(corrosion_q05_post, 0),
            corrosion_q95_posterior=corrosion_q95_post,
            obs_times=obs_times,
            obs_values=obs_corrosion,
            obs_error_std=self.config.obs_error_std,
            title="Corrosion progression: Prior vs Posterior",
            xlim=(times[0], times[-1]),
            ylim=(0, min(start_thickness, max(corrosion_q95_prior) * 1.2)),
        )
        plotting.save_figure(fig, output_dir / "corrosion_prior_posterior.png")

        # Moment capacity from corrosion ratio
        cr_mean_prior = corrosion_mean_prior / start_thickness
        cr_q05_prior = np.maximum(corrosion_q05_prior, 0) / start_thickness
        cr_q95_prior = corrosion_q95_prior / start_thickness

        cr_mean_post = corrosion_mean_post / start_thickness
        cr_q05_post = np.maximum(corrosion_q05_post, 0) / start_thickness
        cr_q95_post = corrosion_q95_post / start_thickness

        moment_mean_prior = moment_cap_start * (1 - cr_mean_prior)
        moment_q95_prior = moment_cap_start * (1 - cr_q05_prior)  # Flip quantiles
        moment_q05_prior = moment_cap_start * (1 - cr_q95_prior)

        moment_mean_post = moment_cap_start * (1 - cr_mean_post)
        moment_q95_post = moment_cap_start * (1 - cr_q05_post)
        moment_q05_post = moment_cap_start * (1 - cr_q95_post)

        # Get moment_survived from setting
        moment_survived = None
        for key in setting:
            if key != "metadata" and "moment_survived" in setting[key]:
                moment_survived = setting[key]["moment_survived"]
                break

        fig = plotting.plot_moment_capacity_prior_posterior(
            times_forecast=times_forecast,
            moment_mean_prior=moment_mean_prior,
            moment_q05_prior=moment_q05_prior,
            moment_q95_prior=moment_q95_prior,
            moment_mean_posterior=moment_mean_post,
            moment_q05_posterior=moment_q05_post,
            moment_q95_posterior=moment_q95_post,
            moment_survived=moment_survived,
            obs_times=obs_times,
            obs_moment_cap=obs_moment_cap,
            title="Moment capacity: Prior vs Posterior",
            xlim=(times[0], times[-1]),
            ylim=(0, moment_cap_start * 1.1),
        )
        plotting.save_figure(fig, output_dir / "moment_prior_posterior.png")

        print(f"Prior/Posterior plots saved to {output_dir}")


def main():
    """Load data path"""
    load_dotenv("dsheet_example.env")
    specs_path = Path(os.environ.get("REMOTE_DATA_PATH", None)) / "case_study_specifications.json"

    # Initialize pipeline
    config = CaseStudyConfig.from_json(specs_path)
    pipeline = ReliabilityPipeline(config=config, specs_path=specs_path)

    # =========================================================================
    # STEP 1: Setup
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Setup")
    print("=" * 60)
    pipeline.setup(n_samples=10_000, seed=42)

    print(f"JPDF: {pipeline.jpdf.nvar} variables, {pipeline.jpdf.n_samples} samples")
    print(f"C50 prior: {pipeline.jpdf.get_C50_stats()}")

    # =========================================================================
    # STEP 2: Load surrogate model
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Load surrogate model")
    print("=" * 60)
    pipeline.load_surrogate()
    print("Surrogate loaded.")

    # =========================================================================
    # STEP 3: Build/load fragility surface (2D cached curves)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Build fragility surface (2D: CR x moment_survived)")
    print("=" * 60)

    # Build the 2D surface (expensive, done once)
    # Creates: results/fragility_surface/manifest.json
    #          results/fragility_surface/moment_XXX.X.npz (one per moment value)
    surface = pipeline.build_fragility_surface(
        n_cr=1000,
        n_moments=50,
        force_rebuild=False,  # Set False to use cached version
        verbose=True,
    )

    # =========================================================================
    # STEP 4: Query fragility surface (fast, loads curves on-demand)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Query fragility surface")
    print("=" * 60)

    # Get a specific fragility curve (loads from disk on first access)
    moment_survived = 113.16  # From case study setting
    curve = pipeline.get_fragility_curve_at(moment_survived)
    print(f"Fragility curve at moment_survived={moment_survived:.1f} kNm:")
    print(f"  Pf at CR=0.2: {curve.pf_at(0.2):.4e}")
    print(f"  Pf at CR=0.3: {curve.pf_at(0.3):.4e}")

    # Query at arbitrary moment (interpolates between curves)
    moment_test = 125.0
    print(f"\nInterpolated query at moment_survived={moment_test:.1f} kNm:")
    print(f"  Pf at CR=0.2: {surface.pf_at(0.2, moment_test):.4e}")
    print(f"  Beta at CR=0.2: {surface.beta_at(0.2, moment_test):.2f}")

    # =========================================================================
    # STEP 5: Compute Pf with proven strength
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Compute Pf with proven strength")
    print("=" * 60)

    # Compute Pf at t=60 given structure survived with moment_survived=113.16
    result = pipeline.compute_pf_with_proven_strength(
        t=60.0,
        moment_survived=moment_survived,
        use_posterior=False,  # Using prior for now (no observations yet)
    )
    print(f"Pf at t=60 with proven strength ({moment_survived:.1f} kNm):")
    print(f"  Pf: {result['pf']:.4e}")
    print(f"  Beta: {result['beta']:.2f}")

    # =========================================================================
    # STEP 6: Build standard fragility curve (for backward compatibility)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Build standard fragility curve")
    print("=" * 60)
    fragility = pipeline.build_fragility(n_grid=1000, cache_moments=True, force_rebuild=True)
    print(fragility.summary())

    # =========================================================================
    # STEP 7: Run timeline analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Run timeline analysis")
    print("=" * 60)

    # Load case study setting
    setting = io.load_json("case_study_setting.json")

    # Run timeline analysis
    results = pipeline.run_timeline(setting, verbose=True)

    # Save results
    pipeline.save_results()
    print("Results saved.")

    # =========================================================================
    # STEP 8: Generate plots
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 8: Generate plots")
    print("=" * 60)
    pipeline.plot_results()
    pipeline.plot_prior_posterior_pdfs(setting)
    pipeline.save_jpdf_snapshots(setting)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    times = sorted(results.keys())
    print(f"{'Time':>6} | {'Beta (prior)':>12} | {'Beta (post)':>12} | {'Pf (post)':>12}")
    print("-" * 60)
    for t in times:
        r = results[t]
        print(
            f"{t:>6.0f} | {r['prior']['beta']:>12.2f} | "
            f"{r['posterior']['beta']:>12.2f} | {r['posterior']['pf']:>12.2e}"
        )


if __name__ == "__main__":
    main()
