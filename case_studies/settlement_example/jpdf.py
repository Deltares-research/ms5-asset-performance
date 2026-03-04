import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import scipy.stats as st
from numpy.typing import NDArray
from case_studies.settlement_example.config import CaseStudyConfig


class JPDF:

    def __init__(
            self,
            name: str = "",
            config: Optional[CaseStudyConfig] = None
    ) -> None:
        self.name = name
        self.config = config or CaseStudyConfig()

        # Variables (loaded from specs)
        self.nvar: int = 0
        self.variable_names: List[str] = []
        self.variables: Dict[st.rv_continuous] = []
        self.correlation_matrix: Optional[NDArray] = None

        # Samples
        self.X_samples: Optional[NDArray] = None  # (n_samples, nvar)
        self.U_samples: Optional[NDArray] = None  # Standard normal samples
        self.n_samples: int = 0

        # Numerical PDF on grid
        self.CR_grid: Optional[NDArray] = None
        self.CR_prior: Optional[NDArray] = None
        self.CR_pdf: Optional[NDArray] = None

        self.k_grid: Optional[NDArray] = None
        self.k_prior: Optional[NDArray] = None
        self.k_pdf: Optional[NDArray] = None

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
        self.variables = {}

        for i, v in enumerate(specs.get("variables", [])):
            name = v.get("name", f"var_{i+1}")

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

            self.variable_names.append(name)
            self.variables[name] = var

        # Correlation matrix
        corr = specs.get("correlation_in_u_space")
        if corr is not None:
            self.correlation_matrix = np.array(corr)
        else:
            self.correlation_matrix = np.eye(self.nvar)

        # Initialize priors from specs
        self.init_priors()
        
    def init_priors(self):

        n_grid = self.config.n_CR_grid
        self.CR_grid = np.linspace(0.5, 2.5, n_grid)
        self.CR_prior = self.variables["CR"].pdf
        self.CR_pdf = self.CR_prior.copy()
        
        n_grid = self.config.n_k_grid
        self.k_grid = np.linspace(0.5, 2.5, n_grid)
        self.k_prior = self.variable_names["k"].pdf
        self.k_pdf = self.k_prior.copy()

    def reset_priors(self) -> None:
        if self.CR_prior is not None:
            self.CR_pdf = self.CR_prior.copy()
        if self.k_prior is not None:
            self.k_pdf = self.k_prior.copy()

    def update(self, loglikes: NDArray) -> None:

        if self.CR_pdf is None or self.CR_grid is None or self.k_pdf is None or self.k_grid is None:
            raise ValueError("Variables not initialized. Call set_prior_from_specs first.")

        def update_variable(
                grid: NDArray,
                prior: NDArray,
                loglikes: NDArray,
        ) -> NDArray:
            log_prior = np.log(prior + 1e-10)
            log_post = log_prior + loglikes
            post = np.exp(log_post)
            post /= np.trapezoid(post, grid.squeeze())
            return post.copy()

        self.CR_pdf = update_variable(self.CR_grid, self.CR_prior, loglikes)
        self.k_pdf = update_variable(self.k_grid, self.k_prior, loglikes.T)

    def get_stats(self) -> Dict[str, float]:

        if self.CR_pdf is None or self.CR_grid is None or self.k_pdf is None or self.k_grid is None:
            return {}

        CR_mean = np.trapezoid(self.CR_grid * self.CR_pdf, self.CR_grid)
        CR_var = np.trapezoid((self.CR_grid - mean)**2 * self.CR_pdf, self.CR_grid)
        CR_std = np.sqrt(var)

        # CDF for quantiles
        cdf = np.cumsum(self.CR_pdf) * np.diff(self.CR_grid, prepend=self.CR_grid[0])
        cdf /= cdf[-1]

        CR_q05 = np.interp(0.05, cdf, self.CR_grid)
        CR_q50 = np.interp(0.50, cdf, self.CR_grid)
        CR_q95 = np.interp(0.95, cdf, self.CR_grid)
        
        k_mean = np.trapezoid(self.k_grid * self.k_pdf, self.k_grid)
        k_var = np.trapezoid((self.k_grid - mean)**2 * self.k_pdf, self.k_grid)
        k_std = np.sqrt(var)

        # CDF for quantiles
        cdf = np.cumsum(self.k_pdf) * np.diff(self.k_grid, prepend=self.k_grid[0])
        cdf /= cdf[-1]

        k_q05 = np.interp(0.05, cdf, self.k_grid)
        k_q50 = np.interp(0.50, cdf, self.k_grid)
        k_q95 = np.interp(0.95, cdf, self.k_grid)
        
        return {
            "CR": {
                "mean": CR_mean,
                "std": CR_std,
                "q05": CR_q05,
                "median": CR_q50,
                "q95": CR_q95,
            },
            "k": {
                "mean": k_mean,
                "std": k_std,
                "q05": k_q05,
                "median": k_q50,
                "q95": k_q95,
            },
        }


if __name__ == "__main__":

    pass
