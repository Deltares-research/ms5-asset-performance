"""
D-Sheet piling reliability analysis script.

Uses the generic FragilityPipeline from src/ with domain-specific
JPDF, corrosion model, and performance function. Handles surrogate
loading, fragility building, plotting, and I/O.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
import json
from datetime import datetime
from src import FragilityPipeline
from case_studies.ark_example.jpdf import JPDF
from case_studies.ark_example.corrosion import CorrosionModel
from case_studies.ark_example.performance_function import (
    Performance,
    FragilityCurve,
    FragilitySurfaceIndex,
    MLP,
)
from case_studies.ark_example import io
from case_studies.ark_example import plotting


class ReliabilityPipeline(FragilityPipeline):
    """D-Sheet piling reliability pipeline.

    Inherits the fragility-based Bayesian loop from FragilityPipeline.
    Adds domain-specific setup (corrosion model, MC sampling, surrogate),
    fragility building, and plotting.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        settings_path: Optional[Path | str] = None,
    ):
        self.config = config or {}

        # Initialize performance function
        perf_params = {
            "moment_cap": self.config["moment_cap"],
            "EI_start": self.config["EI_start"],
            "ei_column_idx": -2,
        }
        performance = Performance(name="dsheet_moment", parameters=perf_params)

        super().__init__(
            settings_path=settings_path,
            performance=performance,
            obs_error=self.config["obs_error_std"],
        )

        # Domain components
        self.fragility: Optional[FragilityCurve] = None

    def setup(
        self,
        n_samples: int = 100_000,
        seed: int = 42,
        **kwargs,
    ) -> None:
        """Initialize JPDF, corrosion model, and performance function.

        Args:
            n_samples: Number of MC samples.
            seed: Random seed.
        """
        # Initialize domain-specific JPDF
        self.jpdf = JPDF(name="dsheet", config=self.config)
        self.jpdf.set_prior_from_settings(self.settings_path)

        # Generate correlated MC samples
        self.jpdf.initiate_samples(n_samples=n_samples, seed=seed)
        self.jpdf.add_water_level(water_lvl=-1.0)

        # Initialize corrosion model
        with open(self.settings_path, "r") as f:
            settings = json.load(f)
        params = settings.get("parameters", {})
        C50_mu = params.get("C50_mu", 1.5)
        C50_std = params.get("C50_std", 0.75)
        self.corrosion_model = CorrosionModel(
            C50_mu=C50_mu,
            C50_std=C50_std,
            corrosion_rate=self.config["corrosion_rate"],
            start_thickness=self.config["start_thickness"],
            obs_error_std=self.config["obs_error_std"],
            t_start=self.config["t_start"],
            n_grid=self.config["n_C50_grid"],
            n_corrosion_grid=self.config["n_grid"],
        )

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

        if model_path is not None:
            model, scaler_x, scaler_y = io.load_surrogate_model(
                MLP, model_kwargs
            )
        else:
            # Load from remote path
            model, scaler_x, scaler_y = io.load_surrogate_model(
                MLP, model_kwargs
            )

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
    ) -> None:
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
        else:

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
            raise ValueError("Call build_fragility_surface() or load_fragility_surface() first.")

        return self.fragility_surface.get_curve_at(moment_survived)

    def save_results(self, filename: str = "reliability_results.json") -> None:
        """Save results to file."""
        # Convert to JSON-serializable format
        results_json = {}
        for t, data in self.results.items():
            results_json[str(t)] = data

        load_dotenv(".env")
        username = os.environ.get("USER", "unknown").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_folder = f"{username}_{timestamp}/{filename}"

        io.save_json(results_json, output_folder)

    def save_jpdf_snapshots(
        self,
        setting: Dict[str, Any],
        output_dir: Optional[Path | str] = None,
        save_pdf: bool = True,
    ) -> None:
        """
        Generate and save JPDF snapshot images for each timestep.

        PNGs are saved into ``output_dir/jpdf_snapshots/`` and a companion
        PDF is written to ``output_dir/jpdf_snapshots.pdf``.

        Args:
            setting: Case study setting with observations.
            output_dir: Directory for snapshot images.
            save_pdf: Also save all snapshots to a single PDF.
        """
        if not self.results:
            print("No results to plot. Run run_timeline() first.")
            return

        if self.fragility_surface is None:
            print("No fragility curve. Run build_fragility_surface() first.")
            return

        if output_dir is None:
            output_dir = io.get_remote_path() / "results/plots"
        output_dir = Path(output_dir)

        png_dir = output_dir / "jpdf_snapshots"
        png_dir.mkdir(parents=True, exist_ok=True)

        # Collect all observations
        times = sorted(self.results.keys())
        obs_times = []
        obs_corrosion = []
        for t in times:
            key = str(t) if str(t) in setting else f"{t:.1f}"
            if key in setting and "corrosion" in setting[key]:
                obs_times.append(t)
                obs_corrosion.append(setting[key]["corrosion"])

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

            #TODO: Fix
            # Generate snapshot
            # fig = plotting.plot_jpdf_snapshot(
            #     time=t,
            #     C50_grid=C50_grid,
            #     C50_prior=C50_prior,
            #     C50_posterior=C50_posterior,
            #     corrosion_ratio_grid=cr_grid,
            #     cr_pdf_prior=cr_pdf_prior,
            #     cr_pdf_posterior=cr_pdf_posterior,
            #     fragility_cr=self.fragility_surface.corrosion_ratios,
            #     fragility_pf=self.fragility_surface.pf,
            #     beta_prior=result["prior"]["beta"],
            #     beta_posterior=result["posterior"]["beta"],
            #     beta_forecast_prior=result["prior"].get("beta_forecast"),
            #     beta_forecast_posterior=result["posterior"].get("beta_forecast"),
            #     obs_times=obs_times,
            #     obs_corrosion=obs_corrosion,
            #     current_cr=current_cr,
            #     moment_cap=self.config["moment_cap"],
            #     start_thickness=self.config["start_thickness"],
            # )
            # Save individual PNG
            # plotting.save_figure(fig, png_dir / f"jpdf_t{int(t):03d}.png")

        # Collect all PNGs into a sibling PDF
        if save_pdf:
            plotting.collect_pngs_to_pdf(png_dir, output_dir / "jpdf_snapshots.pdf")

        print(f"JPDF snapshots saved to {png_dir}")

    def plot_betas(self, output_dir: Optional[Path | str] = None) -> None:
        """
        Generate plots of results with forecasts.

        For each observation time t, generates a plot showing:
        - Prior forecast (gray dashed, no updating)
        - Posterior forecasts from all past observation times
        - Current observation time's forecast (emphasized)

        Also generates:
        - Grid plot with all observation times in subplots
        - Evolution plot showing all forecasts on one figure

        PNGs are saved into ``output_dir/beta_forecast/`` and a companion
        PDF is written to ``output_dir/beta_forecast.pdf``.

        Args:
            output_dir: Directory for plot files. Uses remote path if None.
        """
        if not self.results:
            print("No results to plot.")
            return

        if output_dir is None:
            load_dotenv(".env")
            username = os.environ.get("USER", "unknown").lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_dir = io.get_remote_path() / f"output/results/{username}_{timestamp}/plots"
        output_dir = Path(output_dir)

        png_dir = output_dir / "beta_forecast"
        png_dir.mkdir(parents=True, exist_ok=True)

        times = sorted(self.results.keys())

        # Generate one plot per timestep showing cumulative forecasts
        for t in times:
            results_up_to_t = {k: v for k, v in self.results.items() if k <= t}
            fig = plotting.plot_beta_forecast_at_time(
                current_time=t,
                results=results_up_to_t,
                beta_req=self.config["beta_req"],
            )
            plotting.save_figure(fig, png_dir / f"beta_forecast_t{int(t):03d}.png")

        # Collect all PNGs into a sibling PDF
        plotting.collect_pngs_to_pdf(png_dir, output_dir / "beta_forecast.pdf")

        print(f"Plots saved to {png_dir}")

    def plot_corrosion_forecasts(
        self,
        setting: Dict[str, Any],
        output_dir: Optional[Path | str] = None,
    ) -> None:
        """
        Generate one corrosion forecast plot per observation time.

        Each plot shows:
        - Prior corrosion band (full time range)
        - Observations up to the current observation time
        - Posterior forecast from the current observation time onwards

        PNGs are saved into ``output_dir/corrosion/`` and a companion
        PDF is written to ``output_dir/corrosion.pdf``.

        Args:
            setting: Case study setting with time-series data.
            output_dir: Directory for plot files.
        """
        if not self.results:
            print("No results to plot.")
            return

        if output_dir is None:
            load_dotenv(".env")
            username = os.environ.get("USER", "unknown").lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_dir = io.get_remote_path() / f"output/results/{username}_{timestamp}/plots"
        output_dir = Path(output_dir)

        png_dir = output_dir / "corrosion"
        png_dir.mkdir(parents=True, exist_ok=True)

        # Read settings for C50 prior parameters
        with open(self.settings_path, "r") as f:
            settings = json.load(f)
        params = settings.get("parameters", {})
        C50_mu = params.get("C50_mu", 1.5)
        C50_std = params.get("C50_std", 0.75)

        corrosion_rate = self.config["corrosion_rate"]
        t_start = self.config["t_start"]
        start_thickness = self.config["start_thickness"]
        obs_error_std = self.config["obs_error_std"]

        # Extract observations from setting
        times = sorted([float(k) for k in setting.keys() if k != "metadata"])
        obs_times = []
        obs_corrosion = []
        for t in times:
            key = str(t) if str(t) in setting else f"{t:.1f}"
            if key in setting and "corrosion" in setting[key]:
                obs_times.append(t)
                obs_corrosion.append(setting[key]["corrosion"])

        # Forecast time grid
        t_start = times[0]
        t_end = times[-1]

        # One plot per observation time
        for current_t in obs_times:
            # Posterior forecast conditioned on observations up to current_t

            fig = plotting.plot_corrosion_forecast_at_time(
                cr_grid=self.fragility_surface.corrosion_ratios,
                cr_forecast_prior=self.results[t_start]["prior"]["cr_forecast"],
                cr_forecast_posterior=self.results[current_t]["posterior"]["cr_forecast"],
                obs_times=[obs_time for obs_time in obs_times if obs_time <= current_t],
                obs_values=[obs_corr for (obs_time, obs_corr) in zip(obs_times, obs_corrosion) if obs_time <= current_t],
                obs_error_std=obs_error_std,
                start_thickness=self.config["start_thickness"],
                xlim=(t_start, t_end),
                ylim=(0, self.config["start_thickness"]),
            )
            plotting.save_figure(fig, png_dir / f"corrosion_t{int(current_t):03d}.png")

        # Collect all PNGs into a sibling PDF
        plotting.collect_pngs_to_pdf(png_dir, output_dir / "corrosion.pdf")

        print(f"Corrosion forecast plots saved to {png_dir}")

    def plot_moment_forecasts(
        self,
        setting: Dict[str, Any],
        output_dir: Optional[Path | str] = None,
    ) -> None:
        """
        Generate one moment capacity forecast plot per observation time.

        Each plot shows:
        - Prior moment capacity band (full time range)
        - Observed moment capacity up to the current observation time
        - Posterior forecast from the current observation time onwards
        - Survived moment line

        PNGs are saved into ``output_dir/moment/`` and a companion
        PDF is written to ``output_dir/moment.pdf``.

        Args:
            setting: Case study setting with time-series data.
            output_dir: Directory for plot files.
        """
        if not self.results:
            print("No results to plot.")
            return

        if output_dir is None:
            load_dotenv(".env")
            username = os.environ.get("USER", "unknown").lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_dir = io.get_remote_path() / f"output/results/{username}_{timestamp}/plots"
        output_dir = Path(output_dir)

        png_dir = output_dir / "moment"
        png_dir.mkdir(parents=True, exist_ok=True)

        start_thickness = self.config["start_thickness"]
        moment_cap_start = self.config["moment_cap"]

        # Extract observations from setting
        times = sorted([float(k) for k in setting.keys() if k != "metadata"])
        obs_times = []
        obs_corrosion = []
        obs_moment_cap = []
        for t in times:
            key = str(t) if str(t) in setting else f"{t:.1f}"
            if key in setting and "corrosion" in setting[key]:
                obs_times.append(t)
                obs_corrosion.append(setting[key]["corrosion"])
                cr = setting[key].get("corrosion_ratio", setting[key]["corrosion"] / start_thickness)
                obs_moment_cap.append(moment_cap_start * (1 - cr))

        t_start = times[0]
        t_end = times[-1]

        # Collect survived moments from setting
        survived_times = []
        survived_moments = []
        for t in obs_times:
            key = str(t) if str(t) in setting else f"{t:.1f}"
            if key in setting:
                ms = setting[key].get("moment_survived")
                if ms is not None:
                    survived_times.append(t)
                    survived_moments.append(ms)

        # One plot per observation time
        for current_t in obs_times:
            fig = plotting.plot_moment_forecast_at_time(
                cr_grid=self.fragility_surface.corrosion_ratios,
                cr_forecast_prior=self.results[t_start]["prior"]["cr_forecast"],
                cr_forecast_posterior=self.results[current_t]["posterior"]["cr_forecast"],
                moment_cap=moment_cap_start,
                survived_times=survived_times or None,
                survived_moments=survived_moments or None,
                obs_times=[t for t in obs_times if t <= current_t],
                obs_moment_cap=[m for t, m in zip(obs_times, obs_moment_cap) if t <= current_t],
                xlim=(t_start, t_end),
                ylim=(0, moment_cap_start * 1.1),
            )
            plotting.save_figure(fig, png_dir / f"moment_t{int(current_t):03d}.png")

        # Collect all PNGs into a sibling PDF
        plotting.collect_pngs_to_pdf(png_dir, output_dir / "moment.pdf")

        print(f"Moment capacity forecast plots saved to {png_dir}")

    def plot_end_of_life(self, output_dir: Optional[Path | str] = None) -> None:
        """
        Generate end-of-life bar plot.

        Shows EOL (time beta drops below requirement) per observation time,
        plus the change in EOL between consecutive observations.

        PNGs are saved into ``output_dir/end_of_life/`` and a companion
        PDF is written to ``output_dir/end_of_life.pdf``.

        Args:
            output_dir: Directory for plot files. Uses remote path if None.
        """
        if not self.results:
            print("No results to plot.")
            return

        if output_dir is None:
            load_dotenv(".env")
            username = os.environ.get("USER", "unknown").lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_dir = io.get_remote_path() / f"output/results/{username}_{timestamp}/plots"
        output_dir = Path(output_dir)

        png_dir = output_dir / "end_of_life"
        png_dir.mkdir(parents=True, exist_ok=True)

        fig = plotting.plot_end_of_life(
            results=self.results,
            beta_req=self.config["beta_req"],
            t_start=self.config["t_start"],
        )
        plotting.save_figure(fig, png_dir / "end_of_life.png")

        plotting.collect_pngs_to_pdf(png_dir, output_dir / "end_of_life.pdf")

        print(f"End-of-life plot saved to {png_dir}")

def main():
    # Paths
    load_dotenv(".env")
    os.environ["REMOTE_DATA_PATH"] = os.environ["REMOTE_PATH"] + r"/input"
    settings_path = Path(os.environ["REMOTE_DATA_PATH"]) / "settings.json"

    # Initialize pipeline
    with open(settings_path, "r") as f:
        settings = json.load(f)
    config = settings.get("parameters", {})
    pipeline = ReliabilityPipeline(config=config, settings_path=settings_path)

    # =========================================================================
    # STEP 1: Setup
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Setup")
    print("=" * 60)
    pipeline.setup(n_samples=1_000_000, seed=42)

    # =========================================================================
    # STEP 2: Load surrogate model
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Load surrogate model")
    print("=" * 60)
    pipeline.load_surrogate()
    print("Surrogate loaded.")

    # =========================================================================
    # STEP 3: Run timeline analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Run timeline analysis")
    print("=" * 60)

    pipeline.build_fragility_surface(
        n_cr=1000,
        n_moments=50,
        force_rebuild=False,
        verbose=True,
    )

    # Load case study data
    data = io.load_json("data.json")

    obs_times = [int(float(key)) for key in data.keys()]
    forecast_times = list(range(min(obs_times), max(obs_times)+1, 1))
    pipeline.init_times(obs_times, forecast_times)

    # Run timeline analysis
    results = pipeline.run_timeline(data, verbose=True)

    # Save results
    pipeline.save_results()
    print("Results saved.")

    # =========================================================================
    # STEP 4: Generate plots
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Generate plots")
    print("=" * 60)
    pipeline.plot_betas()
    # pipeline.plot_corrosion_forecasts(settings)
    # pipeline.plot_moment_forecasts(settings)
    # pipeline.plot_end_of_life()
    # pipeline.save_jpdf_snapshots(setting)


if __name__ == "__main__":

    main()

