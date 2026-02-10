"""
Joint probability distribution for D-Sheet piling case study.

Manages:
- Soil and wall parameter samples (X_samples)
- C50 as a numerical PDF on a grid (updatable via Bayesian inference)
- Performance function outputs (G_samples)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import scipy.stats as st
from numpy.typing import NDArray

from case_studies.ark_example.config import CaseStudyConfig


class JPDF:
    """
    Joint probability distribution for D-Sheet piling case study.

    Manages:
    - Soil and wall parameter samples (X_samples)
    - C50 as a numerical PDF on a grid (updatable via Bayesian inference)
    - Performance function outputs (G_samples)

    Args:
        name: Name identifier.
        config: CaseStudyConfig instance.
    """

    def __init__(self, name: str = "", config: Optional[CaseStudyConfig] = None):
        self.name = name
        self.config = config or CaseStudyConfig()

        # Variables (loaded from specs)
        self.nvar: int = 0
        self.variable_names: List[str] = []
        self.variables: List[st.rv_continuous] = []
        self.correlation_matrix: Optional[NDArray] = None

        # Samples
        self.X_samples: Optional[NDArray] = None  # (n_samples, nvar)
        self.U_samples: Optional[NDArray] = None  # Standard normal samples
        self.n_samples: int = 0

        # C50 as numerical PDF on grid
        self.C50_grid: Optional[NDArray] = None
        self.C50_prior: Optional[NDArray] = None  # Fixed prior
        self.C50_pdf: Optional[NDArray] = None    # Current (updated) PDF

        # Performance outputs
        self.G_samples: Optional[Dict[float, NDArray]] = None  # g values per time
        self.Y_samples: Optional[Dict[float, Dict]] = None     # metadata per time

    def set_prior_from_specs(self, filepath: Path | str) -> None:
        """
        Load variable definitions from case study specifications JSON.

        Args:
            filepath: Path to specifications JSON file.
        """
        with open(filepath, "r") as f:
            specs = json.load(f)

        self.nvar = specs.get("number_of_variables", 0)
        self.variable_names = []
        self.variables = []

        for i, v in enumerate(specs.get("variables", [])):
            name = v.get("name", f"var_{i+1}")
            self.variable_names.append(name)

            dist_type = v.get("distribution_type", "normal").lower()

            if dist_type in ["normal", "norm", "n", "gaussian"]:
                mean = v.get("mean", 0.0)
                std = v.get("standard_deviation", 1.0)
                var = st.norm(loc=mean, scale=std)
            elif dist_type in ["lognormal", "lognorm"]:
                mean = v.get("mean", 1.0)
                std = v.get("standard_deviation", 0.5)
                # Convert to lognormal parameters
                sigma = np.sqrt(np.log(1 + (std/mean)**2))
                mu = np.log(mean) - 0.5 * sigma**2
                var = st.lognorm(s=sigma, scale=np.exp(mu))
            elif dist_type in ["uniform", "unif"]:
                lower = v.get("lower_bound", 0.0)
                upper = v.get("upper_bound", 1.0)
                var = st.uniform(loc=lower, scale=upper - lower)
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")

            self.variables.append(var)

        # Correlation matrix
        corr = specs.get("correlation_in_u_space")
        if corr is not None:
            self.correlation_matrix = np.array(corr)
        else:
            self.correlation_matrix = np.eye(self.nvar)

        # Initialize C50 prior from specs
        params = specs.get("parameters", {})
        C50_mu = params.get("C50_mu", 1.5)
        C50_std = params.get("C50_std", 0.75)
        self.init_C50_prior(C50_mu, C50_std)

    def init_C50_prior(self, C50_mu: float, C50_std: float) -> None:
        """
        Initialize C50 grid and prior PDF.

        Args:
            C50_mu: Mean of C50.
            C50_std: Standard deviation of C50.
        """
        n_grid = self.config.n_C50_grid

        # Grid from ~0 to reasonable upper bound
        lower = max(0.01, C50_mu - 4 * C50_std)
        upper = C50_mu + 4 * C50_std
        self.C50_grid = np.linspace(lower, upper, n_grid)

        # Use uniform prior for C50
        a = self.C50_grid.min()
        b = self.C50_grid.max() - a
        self.C50_prior = st.uniform(loc=a, scale=b).pdf(self.C50_grid)

        # Initialize current PDF to prior
        self.C50_pdf = self.C50_prior.copy()

    def reset_C50_to_prior(self) -> None:
        """Reset C50 PDF to the original prior."""
        if self.C50_prior is not None:
            self.C50_pdf = self.C50_prior.copy()

    def update_C50(
        self,
        obs_times: NDArray,
        obs_values: NDArray,
    ) -> None:
        """
        Update C50 PDF via Bayesian inference given corrosion observations.

        Args:
            obs_times: Times of observations [years].
            obs_values: Observed corrosion values [mm].
        """
        if self.C50_pdf is None or self.C50_grid is None:
            raise ValueError("C50 not initialized. Call set_prior_from_specs first.")

        C50_mu = self.config.C50_mu
        corrosion_rate = self.config.corrosion_rate
        obs_error_std = self.config.obs_error_std
        t_ref = self.config.t_ref

        log_prior = np.log(self.C50_prior + 1e-10)

        # Compute likelihood
        C50_grid = self.C50_grid[:, np.newaxis]  # (n_grid, 1)
        obs_times = np.asarray(obs_times)[np.newaxis, :]  # (1, n_obs)
        obs_values = np.asarray(obs_values)  # (n_obs,)

        corr_mu = C50_grid * (1 + corrosion_rate / C50_mu * (obs_times - self.config.t_ref))
        corr_deviations = (obs_values - corr_mu) / obs_error_std
        corr_deviations = np.concatenate((corr_deviations[:, 0][:, np.newaxis], np.diff(corr_deviations, axis=1)), axis=1)

        loglikes = st.norm(loc=0, scale=1).logpdf(corr_deviations)

        loglike = loglikes.sum(axis=1)

        log_post = log_prior + loglike
        post = np.exp(log_post)
        post /= np.trapezoid(post, C50_grid.squeeze())

        self.C50_pdf = post

    def get_C50_stats(self) -> Dict[str, float]:
        """
        Get statistics of current C50 distribution.

        Returns:
            Dict with mean, std, and quantiles.
        """
        if self.C50_pdf is None or self.C50_grid is None:
            return {}

        mean = np.trapezoid(self.C50_grid * self.C50_pdf, self.C50_grid)
        var = np.trapezoid((self.C50_grid - mean)**2 * self.C50_pdf, self.C50_grid)
        std = np.sqrt(var)

        # CDF for quantiles
        cdf = np.cumsum(self.C50_pdf) * np.diff(self.C50_grid, prepend=self.C50_grid[0])
        cdf /= cdf[-1]

        q05 = np.interp(0.05, cdf, self.C50_grid)
        q50 = np.interp(0.50, cdf, self.C50_grid)
        q95 = np.interp(0.95, cdf, self.C50_grid)

        return {
            "mean": mean,
            "std": std,
            "q05": q05,
            "median": q50,
            "q95": q95,
        }

    def initiate_samples(self, n_samples: int, seed: Optional[int] = None) -> None:
        """
        Generate Monte Carlo samples for soil/wall parameters.

        Args:
            n_samples: Number of samples to generate.
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        self.n_samples = n_samples

        # Sample in U-space (standard normal)
        self.U_samples = st.multivariate_normal(
            mean=np.zeros(self.nvar),
            cov=self.correlation_matrix,
        ).rvs(n_samples)

        # Transform to X-space
        self.X_samples = np.zeros_like(self.U_samples)
        for i in range(self.nvar):
            u_marginal = st.norm.cdf(self.U_samples[:, i])
            self.X_samples[:, i] = self.variables[i].ppf(u_marginal)

        # Initialize output storage
        self.G_samples = {}
        self.Y_samples = {}

    def add_water_level(self, water_lvl: float = -1.0) -> None:
        """
        Append deterministic water level column to X_samples.

        Args:
            water_lvl: Water level value [m].
        """
        if self.X_samples is None:
            raise ValueError("Samples not initialized. Call initiate_samples first.")

        water_col = np.full((self.n_samples, 1), water_lvl)
        self.X_samples = np.hstack([self.X_samples, water_col])
        self.variable_names.append("water_lvl")

    def store_results(self, t: float, g: NDArray, metadata: Optional[Dict] = None) -> None:
        """
        Store performance function results for a timestep.

        Args:
            t: Time [years].
            g: Performance function values.
            metadata: Additional metadata.
        """
        self.G_samples[t] = g
        self.Y_samples[t] = metadata or {}

    def get_samples_with_EI(self, EI_values: NDArray, ei_column_idx: int = -2) -> NDArray:
        """
        Get X_samples with specified EI values (for degraded stiffness).

        Args:
            EI_values: EI values to use (n_samples,).
            ei_column_idx: Index of EI column in samples.

        Returns:
            Modified X_samples with updated EI column.
        """
        if self.X_samples is None:
            raise ValueError("Samples not initialized.")

        x = self.X_samples.copy()
        x[:, ei_column_idx] = EI_values
        return x


if __name__ == "__main__":
    # Example usage
    config = CaseStudyConfig()
    jpdf = JPDF(name="test", config=config)

    # Load from mock specs
    specs_path = Path(__file__).parent / "mock/data/case_study_specifications.json"
    if specs_path.exists():
        jpdf.set_prior_from_specs(specs_path)
        print(f"Loaded {jpdf.nvar} variables: {jpdf.variable_names}")

        # Generate samples
        jpdf.initiate_samples(n_samples=1000, seed=42)
        jpdf.add_water_level(-1.0)
        print(f"X_samples shape: {jpdf.X_samples.shape}")

        # C50 prior stats
        stats = jpdf.get_C50_stats()
        print(f"C50 prior: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

        # Simulate Bayesian update with mock observations
        obs_times = np.array([50., 55., 60.])
        obs_values = np.array([1.5, 1.8, 2.1])
        jpdf.update_C50(obs_times, obs_values, C50_mu=1.5)

        stats = jpdf.get_C50_stats()
        print(f"C50 posterior: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
