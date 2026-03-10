"""
Configuration for the settlement reliability case study.

Stores all physical, numerical, and analysis parameters as a dataclass.
Parameters can be loaded from a JSON specifications file or set directly.
"""

import json
from dataclasses import dataclass
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
        return cls(
            layer_thickness=params.get("layer_thickness", self.layer_thickness),
            preload=params.get("preload", self.preload),
            permanent_load=params.get("permanent_load", self.permanent_load),
            preload_removal_time=params.get("preload_removal_time", self.preload_removal_time),
            end_time=params.get("end_time", self.end_time),
            sigma_0=params.get("sigma_0", self.sigma_0),
            sigma_v=params.get("sigma_v", self.sigma_v),
            sigma_p=params.get("sigma_p", self.sigma_p),
            RR=params.get("RR", self.RR),
            Ca=params.get("Ca", self.Ca),
            obs_error=params.get("obs_error", self.obs_error),
            n_CR_grid=params.get("n_CR_grid", self.n_CR_grid),
            n_k_grid=params.get("n_k_grid", self.n_k_grid),
            doc_method=params.get("doc_method", self.doc_method),
            end_settlement_req=params.get("end_settlement_req", self.end_settlement_req),
            forecast_interval=params.get("forecast_interval", self.forecast_interval),
            analysis_method=params.get("analysis_method", self.analysis_method),
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
            "layer_thickness": self.layer_thickness,
            "preload": self.preload,
            "permanent_load": self.permanent_load,
            "preload_removal_time": self.preload_removal_time,
            "end_time": self.end_time,
            "sigma_0": self.sigma_0,
            "sigma_v": self.sigma_v,
            "sigma_p": self.sigma_p,
            "RR": self.RR,
            "Ca": self.Ca,
            "obs_error": self.obs_error,
            "n_CR_grid": self.n_CR_grid,
            "n_k_grid": self.n_k_grid,
            "doc_method": self.doc_method,
            "end_settlement_req": self.end_settlement_req,
            "forecast_interval": self.forecast_interval,
            "analysis_method": self.analysis_method,
        }
