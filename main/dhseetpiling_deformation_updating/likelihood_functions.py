from typing import List, Union, Optional
import numpy as np
from scipy.stats import norm
from main.case_study_2025.surrogate_dependent_gpr import DependentGPRModels, MultitaskGPModel
import torch
import gpytorch

class BaseLikelihood:
    """Base class for likelihood functions."""
    
    def __init__(self):
        """Initialize the base likelihood class."""
        self.parameter_value = None
        self.max_paramater_value = None
        self.measured_mean = None
        self.measured_sigma = None
        # pass
    


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
        torch.set_num_threads(1)
        self.use_surrogate = use_surrogate
        self.model_path = model_path
        self.parameter_names = parameter_names

    def set_measured_mean_and_sigma(self, measured_displacement_mean: np.ndarray, 
                                    measured_displacement_sigma: Union[float, List[float]]):
        self.measured_mean = measured_displacement_mean
        self.measured_sigma = measured_displacement_sigma
        return self.measured_mean, self.measured_sigma
    
    def set_max_parameter_value(self, max_parameter_value: float):
        self.max_paramater_value = max_parameter_value
        return self.max_paramater_value

    def generate_synthetic_measurement(self, parameters: np.ndarray, sigma: float = 1):
        """
        Generate a synthetic measurement for given parameters and sigma
        """
        # self.measured_mean = []
        # self.measured_sigma = []
        # for parameter in parameters:
        #     self.measured_mean.append(self.get_displacement_for_parameters(parameter))
        #     self.measured_sigma.append(sigma)
        # self.measured_mean = np.array(self.measured_mean)
        # self.measured_sigma = np.array(self.measured_sigma)
        self.measured_mean = self.get_displacement_for_parameters(parameters)
        self.measured_sigma = sigma * self.measured_mean
        return self.measured_mean, self.measured_sigma

    def compute_total_likelihood(self, data: np.ndarray) -> np.ndarray:
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


    def compute_likelihood_for_equality_information(self, displacement_samples: np.ndarray) -> np.ndarray:
        """
        Compute likelihood for a displacement sample.
        
        Args:
            displacement_sample (float): The displacement value to evaluate
            
        Returns:
            float: Likelihood value
        """
        n = self.measured_mean.shape[0]
        w = 1 / n   # uniform weights
        likelihood = np.array([sum(w * norm.pdf(displacement_samples, self.measured_mean[i], self.measured_sigma[i])
                         for i in range(n))])
        return likelihood
    
    def compute_log_likelihood_for_equality_information(self, displacement_samples: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood for a displacement sample.
        """
        return np.log(self.compute_likelihood_for_equality_information(displacement_samples))
    
    def safety_factor(self, displacement_sample: np.ndarray) -> np.ndarray:
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
    
    def compute_likelihood_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute likelihood for model parameters.
        
        Args:
            parameters (List[float]): The model parameters
            model_function: Function that computes displacement from parameters
            
        Returns:
            tuple: (likelihood value, displacement)
        """
        self.parameter_values = self.get_displacement_for_parameters(parameters)
        likelihood = self.compute_likelihood_for_equality_information(self.parameter_values)
        return likelihood
    
    def compute_log_likelihood_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
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
    
    def GPR_surrogate_function(self, parameters: np.ndarray):
        """
        GPR surrogate function for given parameters: 
        'Klei_soilphi', 'Klei_soilcohesion', 'Klei_soilcurkb1',
        'Zand_soilphi', 'Zand_soilcurkb1',
        'Zandlos_soilphi', 'Zandlos_soilcurkb1',
        'Zandvast_soilphi', 'Zandvast_soilcurkb1',
        'Wall_SheetPilingElementEI'
        parameters: list of parameters
        parameters[0]: Klei_soilcohesion
        parameters[1]: Klei_soilphi
        parameters[2]: Klei_soilcurkb1
        parameters[3]: Zand_soilphi
        parameters[4]: Zand_soilcurkb1
        parameters[5]: Zandlos_soilphi
        parameters[6]: Zandlos_soilcurkb1
        parameters[7]: Zandvast_soilphi
        parameters[8]: Zandvast_soilcurkb1
        parameters[9]: Wall_SheetPilingElementEI
        Returns:
            displacement: estimated displacement in mm
        """

        X_predict_tensor = torch.tensor(parameters, dtype=torch.float32)
        # ensure that X_predict_tensor is a 2D tensor
        if X_predict_tensor.dim() == 1:
            X_predict_tensor = X_predict_tensor.unsqueeze(0)

        
        cur_model = DependentGPRModels()
        cur_model.load(self.model_path)
        displacement, var_displacement = cur_model.predict(X_predict_tensor)
        # import matplotlib.pyplot as plt
        # n_parameters = displacement.shape[2]
        # fig, axes = plt.subplots(3, n_parameters//3, figsize=(5*n_parameters//3, 15))
        # for i in range(n_parameters):
        #     ax1 = axes[i//(n_parameters//3), i%(n_parameters//3)]
        #     ax1.hist(displacement[:, 0, i], bins=100)
        #     ax1.set_title(f'Displacement for parameter {i}')
        #     ax1.set_ylabel('Frequency')
        #     ax1.set_xlabel('Displacement (mm)')    
        # plt.tight_layout()
        # plt.savefig('displacement_histogram.png')
        # plt.show()
        return displacement[:, 0, -1] # + 1000

        # Plot the displacement
        # print(f"Displacement shape: {displacement.shape}")
        # fig, axes = plt.subplots(3, n_parameters//3, figsize=(5*n_parameters//3, 15))
        # for i in range(n_parameters):
        #     ax1 = axes[i//(n_parameters//3), i%(n_parameters//3)]
        #     ax1.hist(displacement[:, i], bins=100)
        #     ax1.set_title(f'Displacement for parameter {i}')
        #     ax1.set_ylabel('Frequency')
        #     ax1.set_xlabel('Displacement (mm)')    
        # plt.tight_layout()
        # # save figure

        # # plt.close()

        # print(f"Displacement: {displacement}")
        # return first element of displacement which has shape (1, 1, 15)
        # return displacement[:, -1]
    
    def fake_surrogate_function(self, parameters: np.ndarray):
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
        displacement = 0       
        for par in parameters:
            displacement += par
        return displacement
        # noise = np.random.normal(0, 5)
        # return 10 + noise

        # return parameters[0] + parameters[1] + parameters[2]
    
    def get_displacement_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        if self.use_surrogate == True:
            return self.GPR_surrogate_function(parameters)
            # return self.fake_surrogate_function(parameters)
        else:
            parameter_values = []
            for i in range(len(parameters)):
                parameter_values.append(self.get_displacement_from_dsheet_model(parameters[i]))
            return parameter_values

    def get_displacement_from_dsheet_model(self, updated_parameters: np.ndarray, stage_id: int = -1) -> float:
        """
        Run the Dsheet analysis for given parameters or use the surrogate model
        updated_parameters: list of parameters with
        updated_parameters[0]: soil cohesion
        updated_parameters[1]: soil phi
        updated_parameters[2]: water level
        Optional: updated_parameters[3]: corrosion
        """
        # Only import and initialize DSheetPiling if we're not using the surrogate
        from src.reliability_models.dsheetpiling.lsf import unpack_soil_params, unpack_water_params
        from src.geotechnical_models.dsheetpiling.model import DSheetPiling
        self.model = DSheetPiling(self.model_path)

        # Pair parameters with names
        params = {name: rv for (name, rv) in zip(self.parameter_names, updated_parameters)}
        # if k1 in params, add k2=0.5*k1 and k3=0.2*k1
        for key, value in params.items():
            # check if key contains 'k1' -> key of type: soilname_k1
            if 'soilcurkb1' in key:
                params[key.replace('soilcurkb1', 'soilcurkb2')] = 0.5 * value
                params[key.replace('soilcurkb1', 'soilcurkb3')] = 0.25 * value
                params[key.replace('soilcurkb1', 'soilcurko1')] = value
                params[key.replace('soilcurkb1', 'soilcurko2')] = 0.5 * value
                params[key.replace('soilcurkb1', 'soilcurko3')] = 0.25 * value

        
        soil_data = unpack_soil_params(params, list(self.model.soils.keys()))
        water_data = unpack_water_params(params, [lvl.name for lvl in self.model.water.water_lvls])
        self.model.update_soils(soil_data)
        self.model.update_water(water_data)

        self.model.execute(i_run=0)
        
        #TODO: Get the displacement
        results = self.model.results

        # return max displacement of the last stage
        return results.max_displacement[stage_id]