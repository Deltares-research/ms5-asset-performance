"""
Fragility curve data structures.

FragilityPoint stores the result of a single reliability analysis at one
grid point. FragilityCurve collects points into arrays for plotting and
integration.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Annotated
from dataclasses import dataclass, asdict, field
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class FragilityPoint(NamedTuple):
    """Result of a single reliability analysis at one grid point."""
    point: List[float]              # deterministic variable values
    pf: float                       # probability of failure
    beta: float                     # reliability index
    design_point: Dict[str, float]  # optimal point in stochastic space
    alphas: Dict[str, float]        # sensitivity factors
    logpf: float                    # log(pf)
    convergence: bool               # did the solver converge?
    method: str = "form"            # which method produced this result


@dataclass
class FragilityCurve:
    """Collection of FragilityPoints with array accessors.

    Attributes:
        fragility_points: List of FragilityPoint objects.
        points: 2D array (n_points, n_det_vars).
        pfs, betas, logpfs: 1D arrays (n_points,).
        convergences: 1D bool array (n_points,).
    """
    fragility_points: Optional[List[FragilityPoint]] = None

    points: NDArray = field(init=False, default=None)
    design_points: NDArray = field(init=False, default=None)
    pfs: NDArray = field(init=False, default=None)
    betas: NDArray = field(init=False, default=None)
    logpfs: NDArray = field(init=False, default=None)
    alphas: NDArray = field(init=False, default=None)
    convergences: NDArray = field(init=False, default=None)

    def __post_init__(self):
        if self.fragility_points is not None:
            self._parse(self.fragility_points)

    def _parse(self, fps: List[FragilityPoint]) -> None:
        self.points = np.asarray([fp.point for fp in fps])
        self.design_points = np.asarray([fp.design_point for fp in fps])
        self.pfs = np.asarray([fp.pf for fp in fps])
        self.betas = np.asarray([fp.beta for fp in fps])
        self.logpfs = np.asarray([fp.logpf for fp in fps])
        self.alphas = np.asarray([fp.alphas for fp in fps])
        self.convergences = np.asarray([fp.convergence for fp in fps])

    def save(self, path: Path) -> None:
        """Save fragility curve to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "fragility_points": [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in fp._asdict().items()}
                for fp in self.fragility_points
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FragilityCurve":
        """Load fragility curve from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        fps = [FragilityPoint(**fp) for fp in data["fragility_points"]]
        return cls(fragility_points=fps)
