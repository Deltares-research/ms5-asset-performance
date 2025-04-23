import numpy as np
from numpy.typing import NDArray
from src.rvs.utils import *
from scipy import stats
from typing import Optional, Annotated, Type
from abc import ABC, abstractmethod
import itertools


class StateBase(ABC):

    pass


class State(StateBase):

    def __init__(self, rvs: list[RVD | RVS] | tuple[RVD | RVS, ...]) -> None:
        self.rvs = rvs
        self.rvs_dict = {rv.name: rv for rv in self.rvs}
        self.n_rvs = len(self.rvs)

    def check_rvs(self) -> None:
        rv_types = []
        for rv in self.rvs:
            if isinstance(rv, RVD):
                rv_type = "distributed"
            elif isinstance(rv, RVS):
                rv_type = "sampled"
            else:
                rv_type = "Unknown"
            rv_types.append(rv_type)

        if len(set(rv_types)) == len(rv_types):
            raise ValueError("The types of the provided RVs are not the same.")

        if "Unknown" in rv_types:
            raise ValueError("An unknown type of RV has been provided.")

    def add_rv(self, rv: Type[RV]) -> None:
        self.rvs += [rv]
        self.rvs_dict.update({rv.name: rv})

    def joint_log_prob(self, x: Annotated[list[float] | tuple[float, ...] | NDArray[np.float64], ("n_rvs")]) -> float:
        log_probs = [rv.logprob(x_rv) for (rv, x_rv) in zip(self.rvs, x)]  # Cannot be vectorized in this structure
        return sum(log_probs)


class GaussianState(StateBase):

    def __init__(self, rvs: Annotated[List[MvnRV] | Tuple[MvnRV, ...], "nrvs"]) -> None:
        self.rvs = rvs
        self.names = [*itertools.chain.from_iterable(rv.names for rv in rvs)]
        self.ndims = len(self.names)
        self.compile_params()

        self.jpdf = stats.multivariate_normal(self.mus, self.cov)
        self.marginal_pdf = {name: stats.norm(mu, std) for (name, mu, std) in zip(self.names, self.mus, self.stds)}
        self.marginal_pdf_type = {name: "normal" for name in self.names}

    def compile_params(self) -> None:
        self.mus = np.asarray([*itertools.chain.from_iterable(rv.mus for rv in self.rvs)])
        self.stds = np.asarray([*itertools.chain.from_iterable(rv.stds for rv in self.rvs)])
        cov = np.eye(self.ndims)
        start = 0
        for rv in self.rvs:
            end = start + rv.ndims
            cov[start: end][:, start: end] = rv.cov
            start = end
        self.cov = cov
        self.chol = np.linalg.cholesky(self.cov)

    def check_rvs(self) -> None:
        pass

    def add_rv(self, rv: Type[RV]) -> None:
        pass

    def joint_prob(self, x: EvalInNpType, standard_domain: bool = False) -> float:
        return self.jpdf.pdf(x)

    def joint_log_prob(self, x: EvalInNpType, standard_domain: bool = False) -> float:
        return self.jpdf.logpdf(x)

    def transform(self, x: EvalInNpType) -> EvalInNpType:
        return self.mus + self.chol.dot(x)

    def detransform(self, x: EvalInNpType) -> EvalInNpType:
        return np.linalg.inv(self.chol).dot(x-self.mus)


if __name__ == "__main__":

    pass

