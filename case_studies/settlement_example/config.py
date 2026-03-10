"""
Configuration for the settlement reliability case study.

Stores all physical, numerical, and analysis parameters as a dataclass.
Parameters can be loaded from a JSON specifications file or set directly.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any


@dataclass
class CaseStudyConfig:
    """Configuration parameters for the settlement reliability analysis.

    Attributes:
        layer_thickness: Compressible layer thickness [m].
        preload: Preload stress applied during construction [kPa].
        permanent_load: Permanent load after preload removal [kPa].
        preload_removal_time: Time at which preload is removed [days].
        end_time: End time of the analysis [days].
        sigma_0: Initial effective vertical stress [kPa].
        sigma_v: Current effective vertical stress [kPa].
        sigma_p: Preconsolidation stress [kPa]. 0 means normally consolidated.
        RR: Recompression ratio [-].
        Ca: Secondary compression coefficient [-].
        obs_error: Standard deviation of observation measurement error [m].
        n_CR_grid: Number of grid points for the CR (compression ratio) axis.
        n_k_grid: Number of grid points for the k (permeability) axis.
        doc_method: Method to compute degree of consolidation (e.g. "Terzaghi").
        end_settlement_req: Allowable residual settlement after preload removal [m].
        forecast_interval: Time interval between forecast evaluation points [days].
    """

    layer_thickness: float = 5.0
    preload: float = 50.0
    permanent_load: float = 50.0
    preload_removal_time: float = 365.0
    end_time: float = 25550.0
    sigma_0: float = 10.0
    sigma_v: float = 20.0
    sigma_p: float = 0.0
    RR: float = 0.02
    Ca: float = 0.0
    obs_error: float = 0.1
    n_CR_grid: int = 100
    n_k_grid: int = 100
    end_settlement_req: float = 0.05
    forecast_interval: int = 10
    doc_method: str = "Terzaghi"
    analysis_method: str = "semi-analytical"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseStudyConfig":
        """Create config from dictionary."""
        params = data.get("parameters", data)
        valid_fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in params.items() if k in valid_fields})

    @classmethod
    def from_json(cls, filepath: Path | str) -> "CaseStudyConfig":
        """Load config from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
