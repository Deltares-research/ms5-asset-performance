"""
Performance function for the settlement reliability analysis.

Defines the limit state: failure occurs when the residual settlement
(settlement after preload removal minus settlement at preload removal)
exceeds the allowable requirement.
"""

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
    """Limit-state function for residual settlement exceedance.

    Failure is defined as residual settlement >= end_settlement_req.
    The failure probability is computed by integrating the joint PDF
    over the failure domain on the (CR, k) grid.
    """

    def __init__(self, name: str, parameters: dict = {}):
        super().__init__(name, parameters)

    def _lsf(self, x: NDArray, t: float = 0) -> NDArray:
        """Evaluate the limit-state function.

        Args:
            x: Array of residual settlement values on the (CR, k) grid.
            t: Time (unused, kept for interface compatibility).

        Returns:
            Boolean array: True where settlement >= requirement (failure).
        """
        x_ = np.atleast_2d(x)
        g = x_ >= self.parameters["end_settlement_req"]
        return g

    def lsf(self, x: NDArray, t: float = 0) -> NDArray:
        """Public interface for the limit-state function."""
        return self._lsf(x, t)

    def _grad(self, x: NDArray, t: float = 0) -> NDArray:
        """Gradient not implemented for this discrete model."""
        raise NotImplementedError("Gradients not available for surrogate model.")

    def failure_probability(
            self,
            x: NDArray,
            pdf: NDArray,
            CR_grid: NDArray,
            k_grid: NDArray,
        ) -> float:
        """Compute failure probability by integrating the PDF over the failure domain.

        Multiplies the joint PDF by the limit-state indicator (1 in failure
        domain, 0 otherwise) and integrates over the (CR, k) grid using
        the trapezoidal rule.

        Args:
            x: Residual settlement array of shape (n_CR, n_k).
            pdf: Joint PDF array of shape (n_CR, n_k).
            CR_grid: 1D array of CR grid values.
            k_grid: 1D array of k grid values.

        Returns:
            Scalar failure probability.
        """
        g = self.lsf(x, None)
        lsf_mask = g * pdf
        pf = np.trapezoid(lsf_mask, k_grid, axis=1)
        pf = np.trapezoid(pf, CR_grid)
        return pf


if __name__ == "__main__":

    pass

