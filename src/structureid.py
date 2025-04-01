import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional


@dataclass
class RV:
    """
    Class to define a random variable with a specific probability distribution.

    Attributes:
        name: Name of the random variable.
        dist: SciPy probability distribution that can be either continuous or discrete.
        rv_grid: Optional grid of values for which the probability density function is calculated.
        pdf_grid: Probability density function of the random variable.
        logpdf_grid: Log probability density function of the random variable.
    """
    name: str
    dist: Union[stats.rv_continuous, stats.rv_discrete]
    rv_grid: Optional[Union[list, np.ndarray[float, "grid_size"]]] = None
    pdf_grid: Union[list, np.ndarray[float, "grid_size"]] = field(init=False)
    logpdf_grid: Union[list, np.ndarray[float, "grid_size"]] = field(init=False)

    def __post_init__(self):
        """
        Post-initialization method to calculate the probability and log probability grids.
        """
        if self.rv_grid is not None:
            self.pdf_grid = self.prob(self.rv_grid)
            self.logpdf_grid = self.logprob(self.rv_grid)

    def prob(self, x: Union[list, np.ndarray[float, "x_size"]]) -> np.ndarray[float, "x_size"]:
        """
        Calculate the probability of the random variable at the given points.
        """
        is_lst = isinstance(x, list)
        if is_lst:
            x = np.asarray(x)
        prob = self.dist.pdf(x)
        if is_lst:
            prob = prob.tolist()
        return prob

    def logprob(self, x: Union[list, np.ndarray[float, "x_size"]]) -> np.ndarray[float, "x_size"]:
        """
        Calculate the log probability of the random variable at the given points.
        """
        is_lst = isinstance(x, list)
        if is_lst:
            x = np.asarray(x)
        log_prob = self.dist.logpdf(x)
        if is_lst:
            log_prob = log_prob.tolist()
        return log_prob

    def sample(self, n_samples: int = 1, seed: int = 42) -> np.ndarray[float, "n_samples"]:
        """
        Sample the random variable.
        """
        np.random.seed(seed)
        return self.dist.rvs(n_samples)

    def read_grid(self, rv_grid: Union[list, np.ndarray[float, "grid_size"]]) -> None:
        """
        Read the grid of the random variable.
        """
        self.rv_grid = rv_grid
        self.pdf_grid = self.prob(self.rv_grid)
        self.logpdf_grid = self.logprob(self.rv_grid)


class State:
    """
    State of the structure at a given timestep with a list of random variables.
    """

    def __init__(self, rvs: Union[List[RV], Tuple[RV]], time_step: int) -> None:
        """
        Initialize the state with a list of random variables.
        """
        self.rvs = rvs
        self.rvs_dict = {rv.name: rv for rv in self.rvs}
        self.n_rvs = len(self.rvs)
        self.time_step = time_step

    def add_rv(self, rv: RV):
        """
        Add a random variable to the state.
        """
        self.rvs += [rv]
        self.rvs_dict.update({rv.name: rv})

    def joint_log_prob(self, x: list) -> float:
        """
        Calculate the joint log probability of the state.
        """
        log_probs = [rv.logprob(x_rv) for (rv, x_rv) in zip(self.rvs, x)]
        return sum(log_probs)


if __name__ == "__main__":

    rv = RV("dummy", stats.norm(5, 1))

