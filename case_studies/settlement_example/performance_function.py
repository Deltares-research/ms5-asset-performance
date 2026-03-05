from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Type
import json
import numpy as np
from numpy._typing import ArrayLike
from numpy.typing import NDArray
from scipy import stats
from src.performance import BasePerformance
from src.performance.performance import FloatArray
from jpdf import JPDF


class Performance(BasePerformance):

    def __init__(self, name: str, parameters: dict = {}):
        super().__init__(name, parameters)

    def _lsf(self, x: NDArray, t: float = 0) -> NDArray:
        x_ = np.atleast_2d(x)
        g = x_ <= self.parameters.end_settlement_req
        return g

    def lsf(self, x: NDArray, t: float = 0) -> NDArray:
        return self._lsf(x, t)

    def _grad(self, x: NDArray, t: float = 0) -> NDArray:
        """Gradient not implemented for surrogate-based model."""
        raise NotImplementedError("Gradients not available for surrogate model.")

    def failure_probability(
        self,
        x: NDArray,
        jpdf: Type[JPDF],
    ) -> float:
        CR_grid = jpdf.CR_grid
        CR_pdf = jpdf.CR_pdf[:, np.newaxis]

        k_grid = jpdf.k_grid
        k_pdf = jpdf.k_pdf[np.newaxis, :]

        g = self.lsf(x, corrosion_ratio)
        lsf_mask = g * CR_pdf * k_pdf
        pf = np.trapezoid(lsf_mask, k_grid, axis=1)
        pf = np.trapezoid(pf, CR_grid, axis=0)

        return pf


if __name__ == "__main__":

    pass

