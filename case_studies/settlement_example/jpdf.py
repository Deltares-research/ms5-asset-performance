"""
Joint probability density function (JPDF) for soil parameters CR and k.

Manages the prior and posterior distributions of compression ratio (CR) and
permeability (k). Supports two modes: grid-based (discrete 2D PDF on a CR×k
grid) and sample-based (importance sampling with weighted particles). Both
modes support Bayesian updating given settlement observations with
measurement error.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import scipy.stats as st
from numpy.typing import NDArray
from case_studies.settlement_example.config import CaseStudyConfig


class JPDF:

    """Joint PDF of CR (compression ratio) and k (permeability).

    Supports two representations:

    Grid-based: the joint distribution is stored as a 2D array of shape
    (n_CR, n_k). The prior is constructed from marginal distributions and
    the posterior is updated via discrete Bayes on the grid.

    Sample-based: importance samples are drawn from a widened proposal
    distribution and reweighted to represent the prior and posterior.

    Attributes:
        CR_grid: 1D array of CR grid values.
        CR_prior: 1D array of prior marginal PDF of CR.
        CR_pdf: 1D array of current (posterior) marginal PDF of CR.
        k_grid: 1D array of k grid values.
        k_prior: 1D array of prior marginal PDF of k.
        k_pdf: 1D array of current (posterior) marginal PDF of k.
        pdf: 2D array (n_CR, n_k) of current joint PDF (grid-based only).
        X_samples: 2D array (n_samples, nvar) of IS samples.
        IS_dists: Dict of importance sampling proposal distributions.
        log_W_IS: 1D array of log importance weights log(prior/IS) per sample.
        log_likelihood: 1D array of log-likelihood per sample.
        W_prior_samples: 1D array of normalized prior importance weights.
        W_samples: 1D array of normalized posterior importance weights.
    """

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

        self.pdf: Optional[NDArray] = None

        # Sample-based (importance sampling)
        self.IS_dists: Optional[Dict] = None
        self.log_W_IS: Optional[NDArray] = None  # log(prior/IS) per sample
        self.log_likelihood: Optional[NDArray] = None  # log-likelihood per sample
        self.W_prior_samples: Optional[NDArray] = None  # normalized prior weights
        self.W_samples: Optional[NDArray] = None  # normalized posterior weights

        # Performance outputs
        self.G_samples: Optional[Dict[float, NDArray]] = None  # g values per time

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

        # Correlation matrix
        corr = specs.get("correlation_in_u_space")
        if corr is not None:
            self.correlation_matrix = np.array(corr)
        else:
            self.correlation_matrix = np.eye(self.nvar)

        # Initialize priors from specs
        self.init_priors()
        self.pdf = self.get_prior()
        self.CR_pdf = self.CR_prior
        self.k_pdf = self.k_prior

    def init_priors(self) -> None:
        """Build discrete grids and evaluate marginal prior PDFs for CR and k.

        CR uses a linear grid; k uses a geometric (log-spaced) grid. Both span
        the 0.1st to 99.9th percentile of the respective marginal distribution.
        """
        n_grid = self.config.n_CR_grid
        CR_lo = self.variables["CR"].ppf(0.001)
        CR_hi = self.variables["CR"].ppf(0.999)
        self.CR_grid = np.linspace(CR_lo, CR_hi, n_grid)
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
        prior = np.exp(np.log(self.CR_prior)[:, np.newaxis] + np.log(self.k_prior)[np.newaxis, :])
        integral = np.trapezoid(prior, self.k_grid, axis=1)
        integral = np.trapezoid(integral, self.CR_grid)
        prior /= integral
        return prior

    def init_prior_samples(self, covar_IS: float = 2.0, seed: int = 42) -> None:
        """Draw importance samples from a widened prior and compute IS weights.

        The importance sampling (IS) distribution uses the same family as each
        marginal prior but with standard deviation scaled by covar_IS, giving
        broader tails to ensure adequate coverage of the posterior.

        Args:
            covar_IS: Factor by which to widen the prior std for the IS
                proposal distribution.
            seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        n = self.config.n_samples

        # Build IS distributions: same family, wider std
        IS_dists = {}
        for name, var in self.variables.items():
            mean = var.mean()
            std = var.std()
            if var.dist.name == "lognorm":
                std_IS = std * covar_IS
                sigma_IS = np.sqrt(np.log(1 + (std_IS / mean) ** 2))
                mu_IS = np.log(mean) - 0.5 * sigma_IS ** 2
                IS_dists[name] = st.lognorm(s=sigma_IS, scale=np.exp(mu_IS))
            elif var.dist.name == "norm":
                IS_dists[name] = st.norm(loc=mean, scale=std * covar_IS)
            else:
                IS_dists[name] = var
        self.IS_dists = IS_dists

        # Draw samples from IS distribution
        samples = {name: IS_dists[name].rvs(n, random_state=rng) for name in self.variable_names}
        self.X_samples = np.column_stack([samples[name] for name in self.variable_names])
        self.n_samples = n

        # Log IS weights: log p(x) - log q(x)
        log_prior = sum(self.variables[name].logpdf(samples[name]) for name in self.variable_names)
        log_IS = sum(IS_dists[name].logpdf(samples[name]) for name in self.variable_names)
        self.log_W_IS = log_prior - log_IS

        # Normalized prior weights
        log_w = self.log_W_IS - np.max(self.log_W_IS)
        w = np.exp(log_w)
        self.W_prior_samples = w / w.sum()

        # No likelihood yet — posterior = prior
        self.log_likelihood = np.zeros(n)
        self.W_samples = self.W_prior_samples.copy()

    def reset_sample_weights(self) -> None:
        """Reset sample weights to prior (discard likelihood)."""
        if self.log_W_IS is not None:
            self.log_likelihood = np.zeros(self.n_samples)
            self.W_samples = self.W_prior_samples.copy()

    def reset_to_priors(self) -> None:
        """Reset the joint and marginal PDFs back to their prior state."""
        if self.pdf is not None:
            self.pdf = self.get_prior()
        if self.CR_prior is not None:
            self.CR_pdf = self.CR_prior.copy()
        if self.k_prior is not None:
            self.k_pdf = self.k_prior.copy()
        self.reset_sample_weights()

    def update(self, obs_values: NDArray, settlements: NDArray) -> None:
        """Bayesian update given settlement observations.

        Dispatches between sample-based (importance weight reweighting) and
        grid-based (discrete Bayes on 2D grid) depending on whether IS
        samples have been initialized.

        Sample-based: posterior weight_i ∝ (prior_i / IS_i) * likelihood_i.
        Grid-based: log_posterior = log_prior + sum(log_likelihood),
        normalized by trapezoidal integration.

        Args:
            obs_values: 1D array of observed settlement values [m].
            settlements: Sample-based: 2D array (n_samples, n_obs).
                Grid-based: 3D array (n_CR, n_k, n_obs).
        """

        if self.log_W_IS is not None:
            # Sample-based: reweight importance samples
            log_likes = st.norm(loc=obs_values, scale=self.config.obs_error).logpdf(settlements).sum(axis=-1)
            self.log_likelihood = log_likes

            log_w = self.log_W_IS + log_likes
            log_w -= np.max(log_w)
            w = np.exp(log_w)
            self.W_samples = w / w.sum()
        else:
            # Grid-based: discrete Bayes
            if self.CR_pdf is None or self.CR_grid is None or self.k_pdf is None or self.k_grid is None:
                raise ValueError("Variables not initialized. Call set_prior_from_specs first.")

            loglikes = self.get_loglikes(obs_values, settlements)

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

    def get_loglikes(self, obs_values: NDArray, settlements: NDArray) -> NDArray:
        """Compute the sum of log-likelihoods on the (CR, k) grid.

        Assumes independent Gaussian measurement error with std = config.obs_error.

        Args:
            obs_values: 1D array of observed settlement values [m].
            settlements: 3D array (n_CR, n_k, n_obs) of modelled settlements.

        Returns:
            2D array (n_CR, n_k) of summed log-likelihood values.
        """
        obs_values_ = obs_values.reshape(1, 1, -1)
        return st.norm(loc=settlements, scale=self.config.obs_error).logpdf(obs_values_).sum(axis=-1)

    def get_stats(self) -> Dict[str, Any]:
        """Compute summary statistics (mean, std, quantiles) for CR and k.

        Uses the current marginal PDFs (CR_pdf, k_pdf) on their respective
        grids. Quantiles are obtained by inverting the discrete CDF.

        Returns:
            Nested dict with keys "CR" and "k", each containing
            "mean", "std", "q05", "median", "q95".
        """

        if self.CR_pdf is None or self.CR_grid is None or self.k_pdf is None or self.k_grid is None:
            return {}

        CR_mean = np.trapezoid(self.CR_grid * self.CR_pdf, self.CR_grid)
        CR_var = np.trapezoid((self.CR_grid - CR_mean)**2 * self.CR_pdf, self.CR_grid)
        CR_std = np.sqrt(CR_var)

        # CDF for quantiles
        cdf = np.cumsum(self.CR_pdf) * np.diff(self.CR_grid, prepend=self.CR_grid[0])
        cdf /= cdf[-1]

        CR_q05 = np.interp(0.05, cdf, self.CR_grid)
        CR_q50 = np.interp(0.50, cdf, self.CR_grid)
        CR_q95 = np.interp(0.95, cdf, self.CR_grid)

        k_mean = np.trapezoid(self.k_grid * self.k_pdf, self.k_grid)
        k_var = np.trapezoid((self.k_grid - k_mean)**2 * self.k_pdf, self.k_grid)
        k_std = np.sqrt(k_var)

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
