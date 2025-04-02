import numpy as np
from numpy.typing import NDArray
from scipy import stats
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional, Annotated


@dataclass
class RV:
    name: str
    dist: stats.rv_continuous | stats.rv_discrete
    n_dim: int = 1
    rv_grid: Optional[List[float] | Annotated[NDArray[np.float64], ("grid_size")]] = None
    pdf_grid: List[float] | Annotated[NDArray[np.float64], ("grid_size")]= field(init=False)
    logpdf_grid: List[float] | Annotated[NDArray[np.float64], ("grid_size")] = field(init=False)

    def __post_init__(self):
        if self.rv_grid is not None:
            self.pdf_grid = self.prob(self.rv_grid)
            self.logpdf_grid = self.logprob(self.rv_grid)

    def prob(self, x: float | List[float] | Annotated[NDArray[np.float64], ("x_size")]
             ) -> float | List[float] | Annotated[NDArray[np.float64], ("x_size")]:
        prob = self.dist.logpdf(x)
        is_lst = isinstance(x, list)
        if is_lst: prob = prob.tolist()
        return prob

    def logprob(self, x: float | List[float] | Annotated[NDArray[np.float64], ("x_size")]
                ) -> float | List[float] | Annotated[NDArray[np.float64], ("x_size")]:
        log_prob = self.dist.logpdf(x)
        is_lst = isinstance(x, list)
        if is_lst: log_prob = log_prob.tolist()
        return log_prob

    def sample(self, sample_size: int = 1, seed: int = 42) -> Annotated[NDArray[np.float64], ("sample_size")]:
        np.random.seed(seed)
        return self.dist.rvs(sample_size)

    def read_grid(self, rv_grid: List[float] | Annotated[NDArray[np.float64], ("grid_size")]) -> None:
        self.rv_grid = rv_grid
        self.pdf_grid = self.prob(self.rv_grid)
        self.logpdf_grid = self.logprob(self.rv_grid)


class State:

    def __init__(self, rvs: List[RV] | Tuple[RV]) -> None:
        self.rvs = rvs
        self.rvs_dict = {rv.name: rv for rv in self.rvs}
        self.n_rvs = len(self.rvs)

    def add_rv(self, rv: RV) -> None:
        self.rvs += [rv]
        self.rvs_dict.update({rv.name: rv})

    def joint_log_prob(self, x: List[float] | Annotated[NDArray[np.float64], ("n_rvs")]) -> float:
        log_probs = [rv.logprob(x_rv) for (rv, x_rv) in zip(self.rvs, x)]  # Cannot be vectorized in this structure
        return sum(log_probs)


if __name__ == "__main__":

    # rv = RV("dummy", stats.norm(5, 1))

    rvs = [
        RV("dummy_1", stats.norm(5, 1)),
        RV("dummy_2", stats.norm(5, 1)),
        RV("dummy_3", stats.norm(5, 1)),
    ]

    state = State(rvs)
    x = [5, 5, 5]
    jlogprob = state.joint_log_prob(x)

