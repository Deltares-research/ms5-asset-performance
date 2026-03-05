import numpy as np
from geolib.models.base_model_structure import settings
from numpy.typing import NDArray
from case_studies.settlement_example.io import load_json, get_remote_path
from jpdf import JPDF
from performance_function import Performance
from config import CaseStudyConfig
from typing import Optional, List, Dict, Tuple
from dotenv import load_dotenv
from pathlib import Path
import os
import json
from settlement_engine import get_settlement


class ReliabilityPipeline:

    def __init__(
        self,
        config: Optional[CaseStudyConfig] = None,
        specs_path: Optional[Path | str] = None,
    ) -> None:

        self.config = config or CaseStudyConfig()
        self.specs_path = specs_path

        # Components
        self.jpdf: Optional[JPDF] = None
        self.performance: Optional[Performance] = None
        self.settlement_at_obs_times: Optional[NDArray] = None
        self.settlement_at_eval_time: Optional[NDArray] = None

        # Results
        self.results: Dict[float, Dict[str, Any]] = {}

    def setup(
        self,
        n_samples: int = 100_000,
        seed: int = 42,
    ) -> None:

        # Initialize JPDF
        self.jpdf = JPDF(name="settlement", config=self.config)
        self.jpdf.set_prior_from_specs(self.specs_path)

        # Initialize corrosion model
        with open(self.specs_path, "r") as f:
            specs = json.load(f)

        params = specs.get("parameters", {})
        self.performance = Performance(name="settlement", parameters=params)

    def init_settlements(
            self,
            setting: Dict[str, float],
            cache_dir: Optional[Path] = None,
            cache: bool = True
    ) -> None:

        obs_times = [float(key) for key in setting.keys()]
        preload_removal_time = self.config.preload_removal_time
        end_time = self.config.end_time
        times = obs_times + [preload_removal_time, end_time]

        cache_file = cache_dir / "settlements.npy"

        if cache and cache_file.exists():
            
            settlements = np.load(cache_file)
            
            if settlements.shape[-1] != len(times):
                raise ValueError(f"""
                Inconsistent times: cached settlements have {settlements.shape[-1]} 
                times while analysis needs {len(times)}.
                """)
            
        else:
    
            settlements = get_settlement(
                t=times,
                CR=self.jpdf.CR_grid,
                k=self.jpdf.k_grid,
                RR=self.config.RR,
                Ca=self.config.Ca,
                h=self.config.layer_thickness,
                sigma_0=self.config.sigma_0,
                sigma_v=self.config.sigma_v,
                sigma_p=self.config.sigma_p,
                method=self.config.doc_method,
            )

            np.save(cache_file, settlements)

        # If settlement matrix is takes more than 100MB memory, reduce its accuracy.
        if settlements.nbytes / 1e6 >= 100:
            settlements = settlements.astype(float32)
            print("Reducing accuracy to of 'settlements' to float32")
            print(f"Current memory for 'settlements'={settlements.nbytes/1e3:.0f}KB")
        
        self.settlement_at_obs_times = settlements[..., :-2]
        self.settlement_at_eval_time = settlements[..., -1] - settlements[..., -2]


def main():
    # Paths
    load_dotenv("settlement_example.env")
    os.environ["REMOTE_DATA_PATH"] = str(get_remote_path()/"input")
    specs_path = Path(os.environ["REMOTE_DATA_PATH"]) / "case_study_specifications.json"

    # Initialize pipeline
    config = CaseStudyConfig.from_json(specs_path)
    pipeline = ReliabilityPipeline(config=config, specs_path=specs_path)

    # =========================================================================
    # STEP 1: Setup
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Setup")
    print("=" * 60)
    pipeline.setup(n_samples=1_000_000, seed=42)

    # =========================================================================
    # STEP 2: Initialize (or load) pre-evaluated settlements
    # =========================================================================
    print("=" * 60)
    print("STEP 2: Settlements")
    print("=" * 60)
    setting = load_json("case_study_setting.json")
    cache_dir = get_remote_path() / "output/cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    pipeline.init_settlements(setting=setting, cache=True, cache_dir=cache_dir)


    pass


if __name__ == "__main__":

    main()

