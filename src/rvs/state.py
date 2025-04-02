import numpy as np
from numpy.typing import NDArray
from utils import RVS, RVD
from scipy import stats
from typing import Optional, Annotated, Type


class State:

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

    def add_rv(self, rv: RVD | RVS) -> None:
        self.rvs += [rv]
        self.rvs_dict.update({rv.name: rv})

    def joint_log_prob(self, x: Annotated[list[float] | tuple[float, ...] | NDArray[np.float64], ("n_rvs")]) -> float:
        log_probs = [rv.logprob(x_rv) for (rv, x_rv) in zip(self.rvs, x)]  # Cannot be vectorized in this structure
        return sum(log_probs)


if __name__ == "__main__":

    rvs = [RVD("dummy_1", stats.norm(5, 1))] * 3
    state = State(rvs)
    x = [5, 5, 5]
    jlogprob = state.joint_log_prob(x)

