import numpy as np
from numpy.typing import NDArray
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Annotated
from abc import abstractmethod, ABC

GridType = Annotated[list[float] | tuple[float, ...] | NDArray[np.float64], "grid_size"]
EvalInType = float | Annotated[list[float] | tuple[float, ...] | NDArray[np.float64], "eval_size"]
EvalOutType = float | Annotated[list[float] | tuple[float, ...] | NDArray[np.float64], "eval_size"]


class RV(ABC):

    @abstractmethod
    def prob(self, x: EvalInType) -> EvalOutType:
        raise NotImplementedError()

    @abstractmethod
    def logprob(self, x: EvalInType) -> EvalOutType:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, sample_size: int = 1, seed: int = 42) -> float | Annotated[NDArray[np.float64], "sample_size"]:
        raise NotImplementedError()

    def read_grid(self, rv_grid: GridType) -> None:
        raise NotImplementedError()


class RVD(RV):

    def __init__(self,
                 name: str,
                 dist: stats.rv_continuous | stats.rv_discrete,
                 n_dim: int = 1,
                 rv_grid: Optional[GridType] = None,
                 ) -> None:
        self.name = name
        self.dist = dist
        self.n_dim = n_dim
        self.rv_grid = rv_grid
        if self.rv_grid is not None: self.read_grid(self.rv_grid)

    def prob(self, x: EvalInType) -> EvalOutType:
        prob = self.dist.logpdf(x)
        if isinstance(x, list): prob = prob.tolist()
        if isinstance(x, tuple): prob = tuple(prob)
        return prob

    def logprob(self, x: EvalInType) -> EvalOutType:
        log_prob = self.dist.logpdf(x)
        if isinstance(x, list): log_prob = log_prob.tolist()
        if isinstance(x, tuple): log_prob = tuple(log_prob)
        return log_prob

    def sample(self, sample_size: int = 1, seed: int = 42) -> float | Annotated[NDArray[np.float64], "sample_size"]:
        np.random.seed(seed)
        return self.dist.rvs(sample_size)

    def read_grid(self, rv_grid: GridType) -> None:
        self.rv_grid = rv_grid
        self.pdf_grid = self.prob(self.rv_grid)
        self.logpdf_grid = self.logprob(self.rv_grid)


class RVS(RV):

    def __init__(self,
                 name: str,
                 rv_sample: Annotated[list[float] | tuple[float, ...] | NDArray[np.float64], "n_samples"],
                 ) -> None:
        self.name= name
        self.rv_sample= rv_sample if isinstance(rv_sample, np.ndarray) else np.asarray(rv_sample)
        self.n_dim = 1
        grid = np.linspace(self.rv_sample.min(), self.rv_sample.max(), self.rv_sample.size//10+1)
        self.rv_grid = (grid[1:] + grid[:-1]) / 2
        self.freqs = np.histogram(self.rv_sample, grid, density=True)[0]
        if self.rv_grid is not None: self.read_grid(self.rv_grid)

    def prob(self, x: EvalInType) -> EvalOutType:
        prob = np.interp(x, self.rv_grid, self.freqs)
        if isinstance(x, list): prob = prob.tolist()
        if isinstance(x, tuple): prob = tuple(prob)
        return prob

    def logprob(self, x: EvalInType) -> EvalOutType:
        prob = np.interp(x, self.rv_grid, self.freqs)
        return np.log(prob)

    def sample(self, sample_size: int = 1, seed: int = 42) -> float | Annotated[NDArray[np.float64], "sample_size"]:
        np.random.seed(seed)
        return np.random.choice(self.rv_sample, replace=True, size=sample_size)

    def read_grid(self, rv_grid: GridType) -> None:
        self.rv_grid = rv_grid
        self.pdf_grid = self.prob(self.rv_grid)
        self.logpdf_grid = self.logprob(self.rv_grid)


if __name__ == "__main__":

    pass

