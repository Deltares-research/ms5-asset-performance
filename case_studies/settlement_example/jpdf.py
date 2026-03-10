"""
Joint probability density function (JPDF) for soil parameters CR and k.

Manages the prior and posterior distributions on a 2D grid of compression
ratio (CR) and permeability (k). Supports Bayesian updating given settlement
observations with measurement error.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import scipy.stats as st
from numpy.typing import NDArray, ArrayLike
from case_studies.settlement_example.config import CaseStudyConfig


class JPDF:

    """Joint PDF of CR (compression ratio) and k (permeability) on a discrete grid.

    The joint distribution is stored as a 2D array of shape (n_CR, n_k).
    The prior is constructed from marginal distributions loaded from a JSON
    specs file, and the posterior is updated via Bayesian inference using
    settlement observations.

    Attributes:
        CR_grid: 1D array of CR grid values.
        CR_prior: 1D array of prior marginal PDF of CR.
        CR_pdf: 1D array of current (posterior) marginal PDF of CR.
        k_grid: 1D array of k grid values.
        k_prior: 1D array of prior marginal PDF of k.
        k_pdf: 1D array of current (posterior) marginal PDF of k.
        pdf: 2D array (n_CR, n_k) of current joint PDF.
    """

    def __init__(
            self,
            name: str = "",
            config: Optional[CaseStudyConfig] = None
    ) -> None:
        self.name = name
        self.config = config or CaseStudyConfig()
        self.grid_based = self.config.analysis_method == "semi-analytical"

        # Variables (loaded from specs)
        self.nvar: int = 0
        self.variable_names: List[str] = []
        self.variables: Dict[st.rv_continuous] = []
        self.correlation_matrix: Optional[NDArray] = None

        # Samples
        self.n_samples: int = 0
        self.X_samples: Optional[NDArray] = None  # (n_samples, nvar)
        self.U_samples: Optional[NDArray] = None  # Standard normal samples
        self.W_prior: Optional[Dict[float, NDArray]] = None  # g values per time

        # Numerical PDF on grid
        self.CR_grid: Optional[NDArray] = None
        self.CR_prior: Optional[NDArray] = None
        self.CR_pdf: Optional[NDArray] = None

        self.k_grid: Optional[NDArray] = None
        self.k_prior: Optional[NDArray] = None
        self.k_pdf: Optional[NDArray] = None

        self.pdf: Optional[NDArray] = None

    def set_prior_from_specs(self, filepath: Path | str) -> None:
        """Load variable definitions from a JSON specs file and initialize priors.

        Reads distribution type, mean, and standard deviation for each variable
        (CR, k) from the specs file. Supported distributions: normal, lognormal,
        uniform. After loading, calls init_priors() to build the discrete grids
        and marginal PDFs.

        Args:
            filepath: Path to the case study specifications JSON file.
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

        # Initialize priors from specs
        if self.grid_based:
            self.init_priors()
            self.pdf = self.get_prior()
        else:
            self.init_prior_samples()

    def init_prior_samples(self, mean_IS: ArrayLike | None = None, covar_IS: ArrayLike | None = None) -> None:
        '''
        Initiates a set of `N` samples using Importance Sampling by default
        '''

        N = self.config.n_samples

        # sampling weights
        if isinstance(mean_IS,type(None)):
            mean_IS = np.zeros(self.nvar)
        elif isinstance(mean_IS,(int,float)):
            mean_IS = mean_IS * np.ones(self.nvar)


        if isinstance(covar_IS,type(None)):
            covar_IS = np.eye(self.nvar)
        elif isinstance(covar_IS,(int,float)):
            covar_IS = covar_IS * np.eye(self.nvar)


        print('>>>',self.nvar, np.shape(mean_IS))

        # self.U_samples = st.multivariate_normal(mean = mean_IS,cov = covar_IS).rvs(N)
        # self.q = st.multivariate_normal(mean = mean_IS, cov = covar_IS).pdf(self.U_samples)
        # self.p = st.multivariate_normal(mean = np.zeros(self.nvar), cov = np.eye(self.nvar)).pdf(self.U_samples)
        # self.W_samples = self.p / self.q
        # self.W_prior = self.p / self.q

        # # Transform correlated
        # self.X_samples = (np.linalg.cholesky(self.correlation_matrix) @ self.U_samples.T).T
        # for i,name in enumerate(self.variable_names):
        #     self.X_samples[:,i] = self.variables[name].ppf(st.norm.cdf(self.X_samples[:,i]))
        #
        # self.Y_samples = [None]*N
        # self.G_samples = [None]*N

        self.U_samples = np.stack([
            st.norm(loc=0, scale=1).rvs(N)
            for _ in range(self.nvar)
        ], axis=-1)

        self.X_samples = np.stack([
            var.ppf(st.norm.cdf(self.U_samples[:, i]))
            for i, var in enumerate(self.variables.values())
        ], axis=-1)


    def init_priors(self) -> None:
        """Initialize discrete grids and marginal prior PDFs for CR and k.

        CR grid: uniform spacing from 0.01 to 4.01.
        k grid: geometric spacing from the 0.1th to 99.9th percentile of the
        prior distribution, capturing the bulk of the lognormal mass while
        providing finer resolution at small k values.
        """

        n_grid = self.config.n_CR_grid
        self.CR_grid = np.linspace(0.01, 4.01, n_grid)
        self.CR_prior = self.variables["CR"].pdf(self.CR_grid)
        self.CR_pdf = self.CR_prior.copy()
        
        n_grid = self.config.n_k_grid
        k_lo = self.variables["k"].ppf(0.001)
        k_hi = self.variables["k"].ppf(0.999)
        self.k_grid = np.geomspace(k_lo, k_hi, n_grid)
        self.k_prior = self.variables["k"].pdf(self.k_grid)
        self.k_pdf = self.k_prior.copy()

    def get_prior(self) -> NDArray:
        """Compute the normalized joint prior PDF from marginal priors.

        Assumes independence: f(CR, k) = f(CR) * f(k). The product is computed
        in log-space for numerical stability, then normalized by double
        integration over the (CR, k) grid.

        Returns:
            2D array of shape (n_CR, n_k) representing the normalized joint prior.
        """

        if self.grid_based:
            prior = np.exp(np.log(self.CR_prior)[:, np.newaxis] + np.log(self.k_prior)[np.newaxis, :])
            integral = np.trapezoid(prior, self.k_grid, axis=1)
            integral = np.trapezoid(integral, self.CR_grid)
            prior /= integral

        else:
            self.init_prior_samples
            
            prior = self.W_prior / np.sum(self.W_prior)


        return prior

    def get_pdf(self) -> NDArray:
        pdf = self.W_samples / np.sum(self.W_samples)


    def reset_to_priors(self) -> None:
        """Reset the joint and marginal PDFs back to their prior state."""
        if self.pdf is not None:
            self.pdf = self.get_prior()
        if self.CR_prior is not None:
            self.CR_pdf = self.CR_prior.copy()
        if self.k_prior is not None:
            self.k_pdf = self.k_prior.copy()

    def update(self, obs_values: NDArray, settlements: NDArray) -> None:
        """Bayesian update of the joint PDF given settlement observations.

        Computes the posterior using Bayes' theorem on the discrete grid:

            log_posterior = log_prior + sum(log_likelihood)

        where the likelihood for each observation is a normal distribution
        centered on the predicted settlement with standard deviation equal
        to the configured observation error.

        A log-sum-exp trick is applied (subtracting the maximum log-posterior)
        before exponentiation to prevent numerical underflow when many
        observations are used simultaneously.

        After updating, the marginal PDFs (CR_pdf, k_pdf) are recomputed
        by integrating the joint posterior over the other axis.

        Args:
            obs_values: 1D array of observed settlement values [m].
            settlements: 3D array of shape (n_CR, n_k, n_obs) with predicted
                settlements at the observation times for each (CR, k) pair.
        """

        if self.CR_pdf is None or self.CR_grid is None or self.k_pdf is None or self.k_grid is None:
            raise ValueError("Variables not initialized. Call set_prior_from_specs first.")

        obs_values_ = obs_values#.reshape(1, 1, -1)
        loglikes = st.norm(loc=settlements, scale=self.config.obs_error).logpdf(obs_values_)#.sum(axis=-1)

        log_prior = np.log(self.get_prior())
        log_post = log_prior + loglikes
        log_post -= np.nanmax(log_post)

        post = np.exp(log_post)
        integral = np.trapezoid(post, self.k_grid, axis=1)
        integral = np.trapezoid(integral, self.CR_grid, axis=0)
        post /= integral

        self.pdf = post.copy()
        self.CR_pdf = np.trapezoid(self.pdf, self.k_grid, axis=1)
        self.k_pdf = np.trapezoid(self.pdf, self.CR_grid, axis=0)

    def get_stats(self) -> Dict[str, float]:
        """Compute summary statistics (mean, std, quantiles) for CR and k."""

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
