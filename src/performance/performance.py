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
            parameters: dict = {},
            threshold: float = 0.0,  # g(x) < threshold --> failure
            has_gradient: bool = False,  # Closed-form gradients?
    ):
        self.name = name
        self.parameters = parameters
        self.threshold = threshold
        self.has_gradient = has_gradient

    def __repr__(self) -> str:
        # TODO: To be extended
        return f"{self.__class__.__name__}(name={self.name!r})"

    def lsf(self, x: float | int | ArrayLike, t: float | int = 0.) -> float | FloatArray:
        if isinstance(x, int) or isinstance(x, float):
            x = [x]
        return self._lsf(x,t)

    def is_safe(self, x: ArrayLike, t) -> bool | BoolArray:
        return self.lsf(x) >= self.threshold

    def grad(self, x: float | int | ArrayLike, t: int | float = 0.) -> float | FloatArray:
        if isinstance(x, int) or isinstance(x, float):
            x = [x]
        return self._grad(x, t)

    @abstractmethod
    def _lsf(self, x: ArrayLike, t: int | float) -> float | FloatArray:
        pass

    @abstractmethod
    def _grad(self, x: ArrayLike, t: int | float) -> float | FloatArray:
        pass


if __name__ == "__main__":

    # Simple example
    class DummyPerformance(BasePerformance):

        def __init__(self, name, parameters: dict = {}):
            super().__init__(name,parameters)
            pass

        def _lsf(self, x, t = 0):
            return sum(x) / len(x) / (t+1) - 1

        def _grad(self):
            pass


    performance = DummyPerformance(name="example")
    x = [1., 3., .5]
    print(f"LSF = {performance.lsf(x,t = 1):.2f}")





