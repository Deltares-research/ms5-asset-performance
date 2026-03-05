import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class CaseStudyConfig:
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
    doc_method: str = "Terzaghi"
    end_settlement_req: float = 0.05
    forecast_interval: int = 10

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseStudyConfig":
        """Create config from dictionary."""
        params = data.get("parameters", data)
        return cls(
            layer_thickness=params.get("layer_thickness", None),
            preload=params.get("preload", None),
            permanent_load=params.get("permanent_load", None),
            preload_removal_time=params.get("preload_removal_time", None),
            end_time=params.get("end_time", None),
            sigma_0=params.get("sigma_0", None),
            sigma_v=params.get("sigma_v", None),
            sigma_p=params.get("sigma_p", None),
            RR=params.get("RR", None),
            Ca=params.get("Ca", None),
            obs_error=params.get("obs_error", None),
            n_CR_grid=params.get("n_CR_grid", None),
            n_k_grid=params.get("n_k_grid", None),
            doc_method=params.get("doc_method", None),
            end_settlement_req=params.get("end_settlement_req", None),
            forecast_interval=params.get("forecast_interval", None),
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
        }
