from abc import ABC, abstractmethod
from typing import List, TypeAlias
import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.floating]
BoolArray: TypeAlias = NDArray[np.bool_]


class BasePerformance(ABC):

    def __init__(
            self,
            name: str = "test",
            threshold: float = 0.0,  # g(x) < threshold --> failure
            has_gradient: bool = False,  # Closed-form gradients?
    ):
        self.name = name
        self.threshold = threshold
        self.has_gradient = has_gradient

    def __repr__(self) -> str:
        # TODO: To be extended
        return f"{self.__class__.__name__}(name={self.name!r})"

    def lsf(self, x: float | int | ArrayLike) -> float | FloatArray:
        if isinstance(x, int) or isinstance(x, float):
            x = [x]
        return self._lsf(x)

    def is_safe(self, x: ArrayLike) -> bool | BoolArray:
        return self.lsf(x) >= self.threshold

    def grad(self, x: float | int | ArrayLike) -> float | FloatArray:
        if isinstance(x, int) or isinstance(x, float):
            x = [x]
        return self._grad(x)

    @abstractmethod
    def _lsf(self, x: ArrayLike) -> float | FloatArray:
        pass

    @abstractmethod
    def _grad(self, x: float | int | ArrayLike) -> float | FloatArray:
        pass


if __name__ == "__main__":

    # Simple example
    class DummyPerformance(BasePerformance):

        def __init__(self, name):
            super().__init__(name)
            pass

        def _lsf(self, x):
            return sum(x) / len(x) / 3.

        def _grad(self):
            pass


    performance = DummyPerformance(name="example")
    x = [1., 3., .5]
    print(f"LSF = {performance.lsf(x): .2f}")

