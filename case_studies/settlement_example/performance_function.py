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
    Supports two modes for failure probability computation: grid-based
    (trapezoidal integration of indicator × PDF over the CR×k grid) and
    sample-based (weighted sum of failure indicators using IS weights).
    """

    def __init__(self, name: str, parameters: dict = {}):
        super().__init__(name, parameters)

    def _lsf(self, x: NDArray, t: float = 0) -> NDArray:
        """Evaluate the limit-state function.

        Args:
            x: Array of residual settlement values. Shape (n_CR, n_k) for
                grid-based or (n_samples,) for sample-based.
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
            pdf: NDArray = None,
            CR_grid: NDArray = None,
            k_grid: NDArray = None,
            weights: NDArray = None,
        ) -> float:
        """Compute failure probability.

        Two modes controlled by which arguments are provided:

        Grid-based (pdf, CR_grid, k_grid): integrates indicator × pdf over
        the (CR, k) grid using the trapezoidal rule.

        Sample-based (weights): computes Pf = sum(weights * indicator) where
        weights are normalized importance sampling weights.

        Args:
            x: Residual settlement array. Shape (n_CR, n_k) for grid-based,
                (n_samples,) for sample-based.
            pdf: Joint PDF array of shape (n_CR, n_k). Grid-based only.
            CR_grid: 1D array of CR grid values. Grid-based only.
            k_grid: 1D array of k grid values. Grid-based only.
            weights: Normalized IS weights of shape (n_samples,). Sample-based only.

        Returns:
            Scalar failure probability.
        """
        if weights is not None:
            I_fail = x >= self.parameters["end_settlement_req"]
            return float(np.sum(weights * I_fail))
        else:
            g = self.lsf(x, None)
            lsf_mask = g * pdf
            pf = np.trapezoid(lsf_mask, k_grid, axis=1)
            pf = np.trapezoid(pf, CR_grid)
            return pf


if __name__ == "__main__":

    pass

