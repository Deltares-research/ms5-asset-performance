from typing import List, Union, Optional
import numpy as np
from scipy.stats import norm


class BaseLikelihood:
    """Base class for likelihood functions."""
    
    def __init__(self):
        """Initialize the base likelihood class."""
        self.parameter_value = None
        self.max_paramater_value = None
        self.measured_mean = None
        self.measured_sigma = None
        pass
    
    def compute_likelihood_for_equality_information(self, *args, **kwargs) -> float:
        """
        Compute the likelihood value.
        
        This is a base method that should be implemented by derived classes.
        """
        raise NotImplementedError("This method should be implemented by derived classes")
    
    def compute_log_likelihood(self, *args, **kwargs) -> float:
        """
        Compute the log likelihood value.
        
        By default, it uses the compute_likelihood method and takes the natural logarithm.
        """
        return np.log(self.compute_likelihood_for_equality_information(*args, **kwargs))


class DisplacementLikelihood(BaseLikelihood):

    """Likelihood function for displacement measurements."""
    def __init__(self,
                 model_path: str,
                 parameter_names: List[str] = None,
                 use_surrogate: bool = False):
        """
        Initialize the displacement likelihood.
        
        Args:
            measured_displacement_mean (np.ndarray): Mean values of measured displacements
            measured_displacement_sigma (Union[float, List[float]]): Standard deviations of measured displacements
        """
        super().__init__()
        self.use_surrogate = use_surrogate
        if not use_surrogate:
            # Only import and initialize DSheetPiling if we're not using the surrogate
            from src.geotechnical_models.dsheetpiling.model import DSheetPiling
            self.model = DSheetPiling(model_path)
            self.parameter_names = parameter_names

    def set_measured_mean_and_sigma(self, measured_displacement_mean: np.ndarray, 
                                    measured_displacement_sigma: Union[float, List[float]]):
        self.measured_mean = measured_displacement_mean
        self.measured_sigma = measured_displacement_sigma
        return self.measured_mean, self.measured_sigma
    
    def set_max_parameter_value(self, max_parameter_value: float):
        self.max_paramater_value = max_parameter_value
        return self.max_paramater_value

    def generate_synthetic_measurement(self, parameters: List[float], sigma: float = 0.3):
        """
        Generate a synthetic measurement for given parameters and sigma
        """
        self.measured_mean = []
        self.measured_sigma = []
        for parameter in parameters:
            displacement = self.get_displacement_from_dsheet_model(parameter)
            self.measured_mean.append(displacement)
            self.measured_sigma.append(sigma)

        self.measured_mean = np.array(self.measured_mean)
        self.measured_sigma = np.array(self.measured_sigma)
        return self.measured_mean, self.measured_sigma

    def compute_total_likelihood(self, data: List[float]) -> float:
        """
        Compute the total likelihood for a list of data.
        """
        total_likelihood = 1
        weight = 1 / len(data)
        for i in range(len(data)):
            if data[i]["type"] == "equality":
                likelihood = self.compute_likelihood_for_equality_information(data[i]["value"])
            elif data[i]["type"] == "inequality":
                likelihood = self.compute_likelihood_for_inequality_information(data[i]["value"], data[i]["performance_function"])
            else:
                raise ValueError("Invalid data type")
            total_likelihood *= likelihood * weight
        return total_likelihood


    def compute_likelihood_for_equality_information(self, displacement_sample: float) -> float:
        """
        Compute likelihood for a displacement sample.
        
        Args:
            displacement_sample (float): The displacement value to evaluate
            
        Returns:
            float: Likelihood value
        """
        n = self.measured_mean.shape[0]
        w = 1 / n   # uniform weights
        return float(sum(w * norm.pdf(displacement_sample, self.measured_mean[i], 
                                         self.measured_sigma[i if isinstance(self.measured_sigma, list) else 0])
                         for i in range(n)))
    
    def safety_factor(self, displacement_sample: float) -> float:
        '''
        Safety factor for the displacement sample given as the ratio of the max parameter value to the measured displacement value
        '''
        return self.max_paramater_value / (displacement_sample + 1e-5)
    
    def limit_state_function(self, displacement_sample: float, model_uncertainty: Optional[float] = None) -> float:
        '''
        Limit state function for the displacement sample given as a log-probability
        '''
        if model_uncertainty is not None:
            return self.performance_function(displacement_sample) * model_uncertainty - 1
        return self.performance_function(displacement_sample) - 1
    
    def survival_function(self, displacement_sample: float, model_uncertainty: Optional[float] = None) -> float:
        '''
        Survival function for the displacement sample given as a probability
        '''
        if model_uncertainty is not None:
            return 1 - self.performance_function(displacement_sample) * model_uncertainty
        return 1 - self.performance_function(displacement_sample)
    
    def compute_likelihood_for_inequality_information(self, displacement_sample: float, performance_function: int) -> float:
        """
        Compute likelihood for a displacement sample.
        """
        # Check Mark vd Krogt dissertation
        # return 1 - performance_function(displacement_sample)
        pass
    
    def compute_likelihood_for_parameters(self, parameters: List[float]) -> tuple:
        """
        Compute likelihood for model parameters.
        
        Args:
            parameters (List[float]): The model parameters
            model_function: Function that computes displacement from parameters
            
        Returns:
            tuple: (likelihood value, displacement)
        """
        self.parameter_value = self.get_displacement_from_dsheet_model(parameters)
        return self.compute_likelihood_for_equality_information(self.parameter_value)
    
    def compute_log_likelihood_for_parameters(self, parameters: List[float]) -> tuple:
        """
        Compute log likelihood for model parameters.
        
        Args:
            parameters (List[float]): The model parameters
            model_function: Function that computes displacement from parameters
            
        Returns:
            tuple: (log likelihood value, displacement)
        """
        likelihood = self.compute_likelihood_for_parameters(parameters)
        return np.log(likelihood)
    
    def fake_surrogate_function(self, parameters: list[float]):
        """
        Fake surrogate function for given parameters
        parameters: list of parameters
        parameters[0]: soil cohesion
        parameters[1]: soil phi (friction angle)
        parameters[2]: water level
        Optional: parameters[3]: corrosion
        
        # Base displacement value dependent on key parameters
        # Realistic effects:
        # - Higher cohesion reduces displacement
        # - Higher friction angle reduces displacement
        # - Higher water level (less negative) increases displacement
        # - More corrosion increases displacement
        Returns: 
            displacement: estimated displacement in cm
        """        
        # Add some noise to simulate model error
        noise = np.random.normal(0, 0.1)
        a = 12
        b = 0.8
        c = 0.5
        d = 0.2
        displacement = a + b * parameters[0] + c * parameters[1] + d * parameters[2]
        return displacement
        # return parameters[0] + parameters[1] + parameters[2]
    
    def get_displacement_from_dsheet_model(self, updated_parameters: list[float], stage_id: int = -1):
        """
        Run the Dsheet analysis for given parameters or use the surrogate model
        updated_parameters: list of parameters with
        updated_parameters[0]: soil cohesion
        updated_parameters[1]: soil phi
        updated_parameters[2]: water level
        Optional: updated_parameters[3]: corrosion
        """
        # If surrogate mode is enabled, use the fake surrogate function
        if hasattr(self, 'use_surrogate') and self.use_surrogate:
            return self.fake_surrogate_function(updated_parameters)
        
        from src.reliability_models.dsheetpiling.lsf import unpack_soil_params, unpack_water_params
        # Otherwise use the actual DSheetPiling model
        # Pair parameters with names
        params = {name: rv for (name, rv) in zip(self.parameter_names, updated_parameters)}
        
        soil_data = unpack_soil_params(params, list(self.model.soils.keys()))
        water_data = unpack_water_params(params, [lvl.name for lvl in self.model.water.water_lvls])
        
        self.model.update_soils(soil_data)
        self.model.update_water(water_data)
        self.model.execute(i_run=0)
        
        #TODO: Get the displacement
        results = self.model.results

        # return max displacement of the last stage
        return results.max_displacement[stage_id]
