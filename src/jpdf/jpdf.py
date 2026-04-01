"""
Generic joint probability density function (JPDF) for N random variables.

Manages the prior and posterior distributions of arbitrary random variables.
Supports two modes:
- Grid-based: discrete N-D PDF on a tensor product grid with Bayesian
  updating via discrete Bayes.
- Sample-based: importance sampling with weighted particles and Bayesian
  updating via likelihood reweighting.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import scipy.stats as st
from numpy.typing import NDArray


class JPDF:
    """Joint PDF of N random variables.

    Supports two representations:

    Grid-based: the joint distribution is stored as an N-D array on a tensor
    product grid. The prior is constructed from independent marginals and
    the posterior is updated via discrete Bayes.

    Sample-based: importance samples are drawn from a widened proposal
    distribution and reweighted to represent the prior and posterior.

    Attributes:
        variable_names: Ordered list of variable names.
        variables: Dict mapping variable name to scipy rv_continuous.
        grids: Dict mapping variable name to 1D grid array.
        priors: Dict mapping variable name to 1D prior marginal PDF array.
        marginals: Dict mapping variable name to 1D current marginal PDF array.
        grid_config: Dict mapping variable name to grid config (n_grid, grid_type).
        pdf: N-D array of current joint PDF (grid-based only).
        X_samples: 2D array (n_samples, nvar) of IS samples.
        IS_dists: Dict of importance sampling proposal distributions.
        log_W_IS: 1D array of log importance weights log(prior/IS).
        log_likelihood: 1D array of log-likelihood per sample.
        W_prior_samples: 1D array of normalized prior importance weights.
        W_samples: 1D array of normalized posterior importance weights.
    """

    def __init__(self, name: str = "", config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name

        # Variables (loaded from settings)
        self.nvar: int = 0
        self.variable_names: List[str] = []
        self.variables: Dict[str, st.rv_continuous] = {}
        self.correlation_matrix: Optional[NDArray] = None

        # Per-variable grid configuration
        self.grid_config: Dict[str, dict] = {}

        # Samples
        self.X_samples: Optional[NDArray] = None  # (n_samples, nvar)
        self.n_samples: int = 0

        # Numerical PDF on grid (dict-based)
        self.grids: Dict[str, NDArray] = {}
        self.priors: Dict[str, NDArray] = {}
        self.marginals: Dict[str, NDArray] = {}

        self.pdf: Optional[NDArray] = None

        # Sample-based (importance sampling)
        self.IS_dists: Optional[Dict] = None
        self.log_W_IS: Optional[NDArray] = None
        self.log_likelihood: Optional[NDArray] = None
        self.W_prior_samples: Optional[NDArray] = None
        self.W_samples: Optional[NDArray] = None

    def set_variables(self, filepath: Path | str) -> None:

        with open(filepath, "r") as f:
            settings = json.load(f)

        self.variable_names = []
        self.variables = {}
        self.grid_config = {}

        for i, v in enumerate(settings.get("variables", [])):
            name = v.get("name", f"var_{i+1}")
            dist_type = v.get("distribution_type", "normal").lower()

            if dist_type in ["normal", "norm", "n", "gaussian"]:
                mean = v.get("mean", 0.0)
                std = v.get("standard_deviation", 1.0)
                var = st.norm(loc=mean, scale=std)
            elif dist_type in ["lognormal", "lognorm"]:
                mean = v.get("mean", 1.0)
                std = v.get("standard_deviation", 0.5)
                sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
                mu = np.log(mean) - 0.5 * sigma ** 2
                var = st.lognorm(s=sigma, scale=np.exp(mu))
            elif dist_type in ["uniform", "unif"]:
                lower = v.get("lower_bound", 0.0)
                upper = v.get("upper_bound", 1.0)
                var = st.uniform(loc=lower, scale=upper - lower)
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")

            self.variable_names.append(name)
            self.variables[name] = var
            self.grid_config[name] = {
                "n_grid": v.get("n_grid", 100),
                "grid_type": v.get("grid_type", "linear"),
            }

        self.nvar = len(self.variable_names)

        # Correlation matrix
        corr = settings.get("correlation_in_u_space")
        if corr is not None:
            self.correlation_matrix = np.array(corr)
        else:
            self.correlation_matrix = np.eye(self.nvar)

    def set_prior_from_settings(self) -> None:
        """Load variable definitions from a JSON settings file and initialize priors.

        Reads distribution type, mean, standard deviation, grid size, and grid
        type for each variable. Supported distributions: normal, lognormal,
        uniform.

        Args:
            filepath: Path to the specifications JSON file.
        """
        self.init_priors()
        self.pdf = self.get_prior()
        for name in self.variable_names:
            self.marginals[name] = self.priors[name].copy()

    def init_priors(self) -> None:
        """Build discrete grids and evaluate marginal prior PDFs for each variable.

        Grid type per variable is read from grid_config: "linear" uses
        np.linspace, "log" uses np.geomspace. Both span the 0.1st to 99.9th
        percentile of the marginal distribution.
        """
        for name in self.variable_names:
            cfg = self.grid_config[name]
            n_grid = cfg["n_grid"]
            lo = self.variables[name].ppf(0.001)
            hi = self.variables[name].ppf(0.999)

            if cfg["grid_type"] == "log":
                self.grids[name] = np.geomspace(lo, hi, n_grid)
            else:
                self.grids[name] = np.linspace(lo, hi, n_grid)

            self.priors[name] = self.variables[name].pdf(self.grids[name])
            self.marginals[name] = self.priors[name].copy()

    def get_prior(self) -> NDArray:
        """Compute the normalized joint prior PDF from independent marginals.

        Builds the N-D outer product of marginal priors in log-space, then
        normalizes by integrating over all axes.

        Returns:
            N-D array of shape (n_0, n_1, ...) representing the joint prior.
        """
        shape = [len(self.grids[name]) for name in self.variable_names]
        log_prior = np.zeros(shape)
        for i, name in enumerate(self.variable_names):
            s = [1] * self.nvar
            s[i] = len(self.grids[name])
            log_prior += np.log(self.priors[name]).reshape(s)

        prior = np.exp(log_prior)
        prior /= self._integrate(prior)
        return prior

    def _integrate(self, arr: NDArray) -> float:
        """Integrate an N-D array over all grid axes.

        Args:
            arr: N-D array matching the grid shape.

        Returns:
            Scalar integral value.
        """
        result = arr.copy()
        for i in reversed(range(self.nvar)):
            result = np.trapezoid(result, self.grids[self.variable_names[i]], axis=i)
        return float(result)

    def _marginalize(self, arr: NDArray, keep_axis: int) -> NDArray:
        """Integrate an N-D array over all axes except one.

        Args:
            arr: N-D array matching the grid shape.
            keep_axis: The axis index to keep (not integrate over).

        Returns:
            1D array — the marginal along the kept axis.
        """
        result = arr.copy()
        for i in reversed(range(self.nvar)):
            if i != keep_axis:
                result = np.trapezoid(
                    result, self.grids[self.variable_names[i]], axis=i
                )
        return result

    def init_prior_samples(
        self, n_samples: int, covar_IS: float = 2.0, seed: int = 42
    ) -> None:
        """Draw importance samples from a widened prior and compute IS weights.

        The IS proposal uses the same family as each marginal prior but with
        standard deviation scaled by covar_IS.

        Args:
            n_samples: Number of importance samples to draw.
            covar_IS: Factor by which to widen the prior std for the proposal.
            seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)

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

        samples = {
            name: IS_dists[name].rvs(n_samples, random_state=rng)
            for name in self.variable_names
        }
        self.X_samples = np.column_stack(
            [samples[name] for name in self.variable_names]
        )
        self.n_samples = n_samples

        log_prior = sum(
            self.variables[name].logpdf(samples[name])
            for name in self.variable_names
        )
        log_IS = sum(
            IS_dists[name].logpdf(samples[name])
            for name in self.variable_names
        )
        self.log_W_IS = log_prior - log_IS

        log_w = self.log_W_IS - np.max(self.log_W_IS)
        w = np.exp(log_w)
        self.W_prior_samples = w / w.sum()

        self.log_likelihood = np.zeros(n_samples)
        self.W_samples = self.W_prior_samples.copy()

    def get_samples(self, name: str) -> NDArray:
        """Get the 1D sample array for a variable by name.

        Args:
            name: Variable name.

        Returns:
            1D array of shape (n_samples,).
        """
        idx = self.variable_names.index(name)
        return self.X_samples[:, idx]

    def reset_sample_weights(self) -> None:
        """Reset sample weights to prior (discard likelihood)."""
        if self.log_W_IS is not None:
            self.log_likelihood = np.zeros(self.n_samples)
            self.W_samples = self.W_prior_samples.copy()

    def reset_to_priors(self) -> None:
        """Reset the joint and marginal PDFs back to their prior state."""
        if self.pdf is not None:
            self.pdf = self.get_prior()
        for name in self.variable_names:
            if name in self.priors:
                self.marginals[name] = self.priors[name].copy()
        self.reset_sample_weights()

    def update(
        self, obs_values: NDArray, model_output: NDArray, obs_error: float
    ) -> None:
        """Bayesian update given observations.

        Dispatches between sample-based (IS weight reweighting) and grid-based
        (discrete Bayes on N-D grid).

        Args:
            obs_values: 1D array of observed values.
            model_output: Model predictions at parameter points.
                Sample-based: shape (n_samples, n_obs).
                Grid-based: shape (n_0, n_1, ..., n_obs).
            obs_error: Standard deviation of observation measurement error.
        """
        if self.log_W_IS is not None:
            # Sample-based: reweight importance samples
            log_likes = (
                st.norm(loc=obs_values, scale=obs_error)
                .logpdf(model_output)
                .sum(axis=-1)
            )
            self.log_likelihood = log_likes

            log_w = self.log_W_IS + log_likes
            log_w -= np.max(log_w)
            w = np.exp(log_w)
            self.W_samples = w / w.sum()
        else:
            # Grid-based: discrete Bayes
            if not self.grids or not self.marginals:
                raise ValueError(
                    "Variables not initialized. Call set_prior_from_settings first."
                )

            loglikes = self.get_loglikes(obs_values, model_output, obs_error)

            log_prior = np.log(self.get_prior())
            log_post = log_prior + loglikes
            log_post -= np.nanmax(log_post)

            post = np.exp(log_post)
            post /= self._integrate(post)

            self.pdf = post.copy()
            for i, name in enumerate(self.variable_names):
                self.marginals[name] = self._marginalize(self.pdf, keep_axis=i)

    def get_loglikes(
        self, obs_values: NDArray, model_output: NDArray, obs_error: float
    ) -> NDArray:
        """Compute the sum of log-likelihoods on the N-D grid.

        Assumes independent Gaussian measurement error.

        Args:
            obs_values: 1D array of observed values.
            model_output: N-D+1 array (...grid_shape..., n_obs).
            obs_error: Standard deviation of measurement error.

        Returns:
            N-D array of summed log-likelihood values.
        """
        shape = (1,) * self.nvar + (-1,)
        obs_values_ = obs_values.reshape(shape)
        return (
            st.norm(loc=model_output, scale=obs_error)
            .logpdf(obs_values_)
            .sum(axis=-1)
        )

    def get_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for each variable.

        Uses the current marginal PDFs on their respective grids.

        Returns:
            Nested dict keyed by variable name, each containing
            "mean", "std", "q05", "median", "q95".
        """
        if not self.grids or not self.marginals:
            return {}

        result = {}
        for name in self.variable_names:
            grid = self.grids[name]
            pdf = self.marginals[name]

            mean = np.trapezoid(grid * pdf, grid)
            var = np.trapezoid((grid - mean) ** 2 * pdf, grid)
            std = np.sqrt(var)

            cdf = np.cumsum(pdf) * np.diff(grid, prepend=grid[0])
            cdf /= cdf[-1]

            result[name] = {
                "mean": mean,
                "std": std,
                "q05": np.interp(0.05, cdf, grid),
                "median": np.interp(0.50, cdf, grid),
                "q95": np.interp(0.95, cdf, grid),
            }

        return result
