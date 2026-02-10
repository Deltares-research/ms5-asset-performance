"""
Configuration for D-Sheet piling case study.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class CaseStudyConfig:
    """
    Configuration parameters for D-Sheet piling case study.

    Attributes:
        moment_cap: Initial moment capacity [kNm].
        EI_start: Initial flexural stiffness [kNm²].
        start_thickness: Initial wall thickness [mm].
        corrosion_rate: Annual corrosion rate coefficient.
        obs_error_std: Observation error standard deviation [mm].
        t_ref: Reference time for C50 [years].
        t_start: Analysis start time [years].
        t_end: Analysis end time [years].
        n_mcs: Number of Monte Carlo samples.
        n_grid: Number of grid points for corrosion ratio.
        n_C50_grid: Number of grid points for C50 PDF.
        forecast_interval: Time interval for forecast grid [years].
    """

    moment_cap: float = 750.0
    EI_start: float = 30000.0
    start_thickness: float = 9.5
    C50_mu: float = 1.0
    C50_std: float = 0.75
    corrosion_rate: float = 0.022
    obs_error_std: float = 0.4
    t_ref: float = 50.0
    t_start: float = 50.0
    t_end: float = 80.0
    n_mcs: int = 100_000
    n_grid: int = 1000
    n_C50_grid: int = 100
    forecast_interval: int = 1
    beta_req: float = 2.3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseStudyConfig":
        """Create config from dictionary."""
        params = data.get("parameters", data)
        return cls(
            moment_cap=params.get("moment_cap", 750.0),
            EI_start=params.get("EI_start", 30000.0),
            start_thickness=params.get("start_thickness", 9.5),
            C50_std=params.get("C50_std", 0.75),
            corrosion_rate=params.get("corrosion_rate", 0.022),
            obs_error_std=params.get("obs_error_std", 0.4),
            t_ref=params.get("t_ref", 50.0),
            t_start=params.get("t_start", 50.0),
            t_end=params.get("t_end", 80.0),
            n_mcs=params.get("n_mcs", 100_000),
            n_grid=params.get("n_grid", 100),
            n_C50_grid=params.get("n_C50_grid", 100),
            forecast_interval=params.get("forecast_interval", 2.0),
            beta_req=params.get("beta_req", 2.3),
        )

    @classmethod
    def from_json(cls, filepath: Path | str) -> "CaseStudyConfig":
        """Load config from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "moment_cap": self.moment_cap,
            "EI_start": self.EI_start,
            "start_thickness": self.start_thickness,
            "C50_std": self.C50_std,
            "corrosion_rate": self.corrosion_rate,
            "obs_error_std": self.obs_error_std,
            "t_ref": self.t_ref,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "n_mcs": self.n_mcs,
            "n_grid": self.n_grid,
            "n_C50_grid": self.n_C50_grid,
            "forecast_interval": self.forecast_interval,
            "beta_req": self.beta_req,
        }
