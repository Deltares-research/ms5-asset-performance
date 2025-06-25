from typing import Annotated, List, Tuple, Dict, NamedTuple
import numpy as np
from scipy.special import erfinv, erf
from src.geotechnical_models.gpr.gpr_classes import DependentGPRModels, MultitaskGPModel, load_gpr_model
from src.geotechnical_models.mlp.MLP_class import MLP, inference
from pathlib import Path
import torch

class FragilityPoint(NamedTuple):
    point: Annotated[List[float] | Tuple[float, ...], "integration_dims"]
    pf: float
    beta: float
    design_point: Dict[str, float]
    alphas: Dict[str, float]
    logpf: float
    convergence: bool
    
class SurrogateReliability:

    def __init__(self, max_moment: float = 23, max_displacement: float = 10):
        self.max_moment = max_moment
        self.max_displacement = max_displacement

    def get_moment_lsf(self, predicted_moment: float):
        return self.max_moment / (predicted_moment + 1e-5)

    def probability_of_failure_moment(self, load: List[float], capacity: List[float]) -> float:
        """
        Calculate probability of failure analytically for Gaussian load and capacity distributions.
        
        For independent Gaussian random variables Load ~ N(μ_L, σ²_L) and Capacity ~ N(μ_C, σ²_C):
        - Safety margin Z = Capacity - Load ~ N(μ_C - μ_L, σ²_C + σ²_L)
        - Probability of failure: P(Load > Capacity) = P(Z < 0) = Φ(-μ_Z/σ_Z)
        
        Args:
            load: List of n load samples from Gaussian distribution
            capacity: List of n capacity samples from Gaussian distribution
            
        Returns:
            float: Probability of failure (between 0 and 1)
        """
        # Calculate statistics from samples
        load_mu = np.mean(load)
        load_std = np.std(load)
        capacity_mu = np.mean(capacity)
        capacity_std = np.std(capacity)

        # For independent Gaussian variables X (load) and Y (capacity):
        # Z = Y - X ~ N(μ_Y - μ_X, σ²_Y + σ²_X)
        # P(failure) = P(X > Y) = P(Y - X < 0) = P(Z < 0)
        
        margin_mu = capacity_mu - load_mu  # Mean of safety margin
        margin_std = np.sqrt(load_std**2 + capacity_std**2)  # Std dev of safety margin
        
        # P(Z < 0) = Φ(-μ_Z/σ_Z) = 0.5 * (1 + erf((-μ_Z)/(√2 * σ_Z)))
        pf = 0.5 * (1 + erf(-margin_mu / (np.sqrt(2) * margin_std)))
        return pf
    
    def get_displacement_lsf(self, predicted_displacement: np.ndarray):
        return self.max_displacement / (predicted_displacement + 1e-5)
    
    def probability_of_failure_displacement(self, predicted_displacements: np.ndarray) -> float:
        lsf =  self.get_displacement_lsf(predicted_displacements)
        # pf_mask = lsf < 0
        # pf = np.sum(pf_mask) / len(predicted_displacements)
        return lsf

    def calculate_reliability_index(self, pf: float) -> float:
        """
        Calculate the reliability index (beta) from probability of failure.
        
        The reliability index β is related to the probability of failure (pf) through:
        β = -Φ^(-1)(pf)
        where Φ^(-1) is the inverse of the standard normal cumulative distribution function.
        
        Args:
            pf (float): Probability of failure between 0 and 1
            
        Returns:
            float: Reliability index β
        """
        pf = np.array(pf)
        if pf < 1e-5:
            return 5.
        if pf > 1-1e-5:
            return 0.
        beta = -np.sqrt(2) * erfinv(2 * pf - 1)
        return beta

def load_GPR_surrogate(model_path: str):
    """
    """
    # Ensure model path exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"GPR model file not found: {model_path}")

    cur_model = load_gpr_model(model_path)
    return cur_model

if __name__ == "__main__":
    surrogate_reliability = SurrogateReliability()
    surrogate_model = load_GPR_surrogate("main/case_study_2025/train/srg/srg_model.pkl")