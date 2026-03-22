"""
Performance function for the settlement reliability analysis.

Defines the limit state: failure occurs when the residual settlement
exceeds the allowable requirement.
"""

import numpy as np
from numpy.typing import NDArray
from src import BasePerformance


class Performance(BasePerformance):
    """Limit-state function for residual settlement exceedance.

    Failure is defined as residual settlement >= end_settlement_req.
    Inherits failure_probability() from BasePerformance.
    """

    def __init__(self, name: str, parameters: dict = {}):
        super().__init__(name, parameters)

    def _lsf(self, x: NDArray, t: float = 0) -> NDArray:
        """Evaluate the limit-state function.

        Args:
            x: Array of residual settlement values.
            t: Time (unused, kept for interface compatibility).

        Returns:
            Boolean array: True where settlement >= requirement (failure).
        """
        x_ = np.atleast_2d(x)
        return x_ >= self.parameters["end_settlement_req"]

    def _grad(self, x: NDArray, t: float = 0) -> NDArray:
        """Gradient not implemented for this discrete model."""
        raise NotImplementedError("Gradients not available for this model.")
