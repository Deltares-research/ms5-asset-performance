import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional


@dataclass
class RV:
    name: str
    dist: Union[stats.rv_continuous, stats.rv_discrete]
    rv_grid: Optional[Union[list, np.ndarray[float, "grid_size"]]] = None
    pdf_grid: Union[list, np.ndarray[float, "grid_size"]] = field(init=False)
    logpdf_grid: Union[list, np.ndarray[float, "grid_size"]] = field(init=False)

    def __post_init__(self):
        if self.rv_grid is not None:
            self.pdf_grid = self.prob(self.rv_grid)
            self.logpdf_grid = self.logprob(self.rv_grid)

    def prob(self, x: Union[list, np.ndarray[float, "x_size"]]) -> np.ndarray[float, "x_size"]:
        is_lst = isinstance(x, list)
        if is_lst:
            x = np.asarray(x)
        prob = self.dist.pdf(x)
        if is_lst:
            prob = prob.tolist()
        return prob

    def logprob(self, x: Union[list, np.ndarray[float, "x_size"]]) -> np.ndarray[float, "x_size"]:
        is_lst = isinstance(x, list)
        if is_lst:
            x = np.asarray(x)
        log_prob = self.dist.logpdf(x)
        if is_lst:
            log_prob = log_prob.tolist()
        return log_prob

    def sample(self, n_samples: int = 1, seed: int = 42) -> np.ndarray[float, "n_samples"]:
        np.random.seed(seed)
        return self.dist.rvs(n_samples)

    def read_grid(self, rv_grid: Union[list, np.ndarray[float, "grid_size"]]) -> None:
        self.rv_grid = rv_grid
        self.pdf_grid = self.prob(self.rv_grid)
        self.logpdf_grid = self.logprob(self.rv_grid)


class State:

    def __init__(self, rvs: Union[List[RV], Tuple[RV]]) -> None:
        self.rvs = rvs
        self.rvs_dict = {rv.name: rv for rv in self.rvs}
        self.n_rvs = len(self.rvs)

    def add_rv(self, rv: RV):
        self.rvs += [rv]
        self.rvs_dict.update({rv.name: rv})

    def joint_log_prob(self, x: list) -> float:
        log_probs = [rv.logprob(x_rv) for (rv, x_rv) in zip(self.rvs, x)]
        return sum(log_probs)


if __name__ == "__main__":

    rv = RV("dummy", stats.norm(5, 1))

