import numpy as np
from numpy.typing import NDArray
from scipy import stats
from typing import Optional, Annotated, List, Tuple
from abc import abstractmethod, ABC

GridType = Annotated[List[float] | Tuple[float, ...] | NDArray[np.float64], "grid_size"]
EvalInType = float | Annotated[List[float] | Tuple[float, ...] | NDArray[np.float64], "eval_size"]
EvalOutType = float | Annotated[List[float] | Tuple[float, ...] | NDArray[np.float64], "eval_size"]


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

    @abstractmethod
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
                 rv_sample: Annotated[List[float] | Tuple[float, ...] | NDArray[np.float64], "n_samples"],
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


class MvnRV(RV):

    def __init__(
            self,
            mus: Tuple[float,...] | List[float] | Annotated[NDArray[np.float64], "ndims"],
            stds: Optional[Tuple[float,...] | List[float] | Annotated[NDArray[np.float64], "ndims"]] = None,
            cov: Optional[Annotated[NDArray[np.float64], "ndims x ndims"]] = None
    ) -> None:
        if (stds is None) == (cov is None):
            raise ValueError("You must provide exactly one of 'stds' or 'cov'")
        self.assert_dimensions(mus, stds, cov)

        self.mus = mus
        self.ndims = len(mus)

        if stds is not None:
            corr = np.eye(self.ndims)
            if not isinstance(stds, np.ndarray): stds = np.asarray(stds)
            cov = stds.dot(corr.dot(stds.T))
            self.dist = stats.multivariate_normal(mus, cov)

        if stds is not None:
            stds = np.sqrt(np.diag(cov))
            self.dist = stats.multivariate_normal(mus, cov)

        self.stds = stds
        self.cov = cov

        self.st_dist = stats.multivariate_normal(np.zeros(self.ndims), np.eye(self.ndims))

    def assert_dimensions(
            self,
            mus: Tuple[float, ...] | List[float] | Annotated[NDArray[np.float64], "ndims"],
            stds: Optional[Tuple[float, ...] | List[float] | Annotated[NDArray[np.float64], "ndims"]] = None,
            cov: Optional[Annotated[NDArray[np.float64], "ndims x ndims"]] = None
    ) -> None:
        input_type = None
        if stds is not None: input_type = "std"
        if cov is not None: input_type = "cov"
        if input_type == "std":
            if len(mus) != len(stds):
                raise ValueError("The size of 'mus' is different from the size of 'stds'.")
        else:
            if len(mus) != cov.shape[0]:
                raise ValueError("The size of 'mus' is different from the number of dimensions in 'cov'.")

    def transform(self, x: EvalInType) -> EvalInType:
        pass

    def detransform(self, x: EvalInType) -> EvalInType:
        pass

    def standard_prob(self, x: EvalInType) -> EvalOutType:
        x_st = self.detransform(x)
        prob = self.st_dist.logpdf(x_st)
        if isinstance(x, list): prob = prob.tolist()
        if isinstance(x, tuple): prob = tuple(prob)
        return prob

    def actual_prob(self, x: EvalInType) -> EvalOutType:
        prob = self.dist.logpdf(x)
        if isinstance(x, list): prob = prob.tolist()
        if isinstance(x, tuple): prob = tuple(prob)
        return prob

    def prob(self, x: EvalInType, standard_domain: bool = False) -> EvalOutType:
        if standard_domain:
            return self.standard_prob(x)
        else:
            return self.actual_prob(x)

    def logprob(self, x: EvalInType) -> EvalOutType:
        raise NotImplementedError()

    def sample(self, sample_size: int = 1, seed: int = 42) -> float | Annotated[NDArray[np.float64], "sample_size"]:
        raise NotImplementedError()

    def read_grid(self, rv_grid: GridType) -> None:
        raise NotImplementedError()


if __name__ == "__main__":

    pass

