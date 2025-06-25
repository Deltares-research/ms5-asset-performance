from typing import List, Union, Optional
import numpy as np
from scipy.stats import norm
import torch
import gpytorch
from pathlib import Path
import sys
from src.geotechnical_models.gpr.gpr_classes import DependentGPRModels, MultitaskGPModel, load_gpr_model
from src.corrosion.corrosion_model import CorrosionModelSimple
from src.geotechnical_models.mlp.MLP_class import MLP, inference

class BaseLikelihood:
    """Base class for likelihood functions."""
    
    def __init__(self):
        """Initialize the base likelihood class."""
        self.parameter_value = None
        self.max_paramater_value = None
        self.measured_mean = None
        self.measured_sigma = None


class CorrosionLikelihood(BaseLikelihood):
    """Likelihood function for corrosion measurements."""
    def __init__(self):
        super().__init__()
        torch.set_num_threads(1)
        self.corrosion_model = CorrosionModelSimple()
        self.measured_mean = None
        self.measured_sigma = None

    def set_measured_mean_and_sigma(self, measured_corrosion_mean: np.ndarray, 
                                    measured_corrosion_sigma: Union[float, List[float]]):
        self.measured_mean = np.array([measured_corrosion_mean])
        self.measured_sigma = measured_corrosion_sigma * self.measured_mean
        # self.measured_sigma[self.measured_sigma < 0] = -self.measured_sigma[self.measured_sigma < 0]
        # return self.measured_mean, self.measured_sigma
    
    def set_max_parameter_value(self, max_parameter_value: float):
        self.max_paramater_value = max_parameter_value
        return self.max_paramater_value

    def generate_synthetic_measurement(self, corrosion_samples: np.ndarray, sigma: float = 0.1):
        """
        Generate a synthetic measurement for given parameters and sigma
        """
        self.measured_mean = corrosion_samples
        self.measured_sigma = sigma * self.measured_mean
        mask = self.measured_sigma < 0
        self.measured_sigma[mask] = -self.measured_sigma[mask]
        # return self.measured_mean, self.measured_sigma
        # self.measured_mean = self.get_displacement_for_parameters(parameters)
        # self.measured_sigma = sigma * self.measured_mean
        # mask = self.measured_sigma < 0
        # self.measured_sigma[mask] = -self.measured_sigma[mask]
        # return self.measured_mean, self.measured_sigma


    def compute_likelihood_for_equality_information(self, corrosion_samples: np.ndarray) -> np.ndarray:
        """
        Compute likelihood for a displacement sample.
        
        Args:
            displacement_sample (float): The displacement value to evaluate
            
        Returns:
            float: Likelihood value
        """
        n = self.measured_mean.shape[0]
        w = 1 / n   # uniform weights
        likelihood = np.array([sum(w * norm.pdf(corrosion_samples, self.measured_mean[i], self.measured_sigma[i])
                         for i in range(n))])
        return likelihood
    
    def compute_log_likelihood_for_equality_information(self, displacement_samples: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood for a displacement sample.
        """
        return np.log(self.compute_likelihood_for_equality_information(displacement_samples))
    
    def compute_likelihood_for_corrosion(self, corrosion_samples: np.ndarray) -> np.ndarray:
        """
        Compute likelihood for a corrosion sample.
        """
        n = self.measured_mean.shape[0]
        w = 1 / n   # uniform weights
        likelihood = np.array([sum(w * norm.pdf(corrosion_samples, self.measured_mean[i], self.measured_sigma[i])
                         for i in range(n))])
        return likelihood
    
    def compute_log_likelihood_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood for a corrosion sample.
        """
        self.parameter_values = self.corrosion_model.get_corrosion_rate_at_t(samples_a=parameters, cur_t=self.cur_t).flatten()
        return np.log(self.compute_likelihood_for_corrosion(self.parameter_values))

class SoilPropertyLikelihood(BaseLikelihood):
    """Likelihood function for soil property measurements."""
    def __init__(self):
        super().__init__()
        torch.set_num_threads(1)
        self.measured_mean = None
        self.measured_sigma = None

    def set_measured_mean_and_sigma(self, measured_soil_mean: np.ndarray, 
                                    measured_soil_sigma: Union[float, List[float]]):
        self.measured_mean = measured_soil_mean
        self.measured_sigma = measured_soil_sigma * self.measured_mean

    def compute_likelihood_for_soil_sample(self, soil_samples: np.ndarray) -> np.ndarray:
        """
        Compute likelihood for a soil sample.
        """
        n = self.measured_mean.shape[0]
        w = 1 / n   # uniform weights
        likelihood = np.array([sum(w * norm.pdf(soil_samples, self.measured_mean[i], self.measured_sigma[i])
                         for i in range(n))])
        return likelihood
    
    def compute_log_likelihood_for_soil_sample(self, soil_samples: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood for a soil sample.
        """
        return np.log(self.compute_likelihood_for_soil_sample(soil_samples))
    
    def compute_likelihood_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute likelihood for a parameters sample.
        """
        self.parameter_values = parameters.flatten()
        return self.compute_likelihood_for_soil_sample(self.parameter_values)
    
    def compute_log_likelihood_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood for a parameters sample.
        """
        return np.log(self.compute_likelihood_for_parameters(parameters))

class MomentLikelihoodSurvived(BaseLikelihood):

    """Likelihood function for moment measurements."""
    def __init__(self, model_path: str, model_type: str = "gpr"):
        super().__init__()
        torch.set_num_threads(1)
        self.moment_model = ModelBridge(model_path, model_type)
        self.max_moment = None
        self.survived_moment = None

    def set_max_parameter_value(self, max_moment: float):
        self.max_moment = max_moment
        return self.max_moment
    
    def set_survived_moment(self, survived_moment: float):
        self.measured_mean = np.array([survived_moment])
        self.measured_sigma = 0.1 * self.measured_mean
        self.survived_moment = survived_moment
        # return self.survived_moment

    def compute_likelihood_for_survived_moment(self, moment_samples: np.ndarray) -> np.ndarray:
        """
        Compute likelihood for a survived moment sample.
        """
        # difference = moment_samples - self.max_moment
        difference = self.survived_moment - moment_samples
        return difference > 0

    def compute_log_likelihood_for_survived_moment(self, moment_samples: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood for a survived moment sample.
        """
        # return np.log(self.compute_likelihood_for_survived_moment(moment_samples))
        return self.compute_likelihood_for_survived_moment(moment_samples)

    def compute_log_likelihood_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood for a survived parameters sample.
        """
        # add column of self.survived_moment to the parameters
        self.parameter_values = self.moment_model.get_moment_for_parameters(parameters)
        return self.compute_log_likelihood_for_survived_moment(self.parameter_values)  
    
class MomentLikelihood(BaseLikelihood):

    """Likelihood function for moment measurements."""
    def __init__(self, model_path: str, model_type: str = "gpr"):
        super().__init__()
        torch.set_num_threads(1)
        self.moment_model = ModelBridge(model_path, model_type)
        self.survived_moment = None

    def set_max_parameter_value(self, max_moment: float):
        self.max_moment = max_moment
        return self.max_moment
    
    def set_measured_mean_and_sigma(self, measured_moment_mean: np.ndarray, 
                                    measured_moment_sigma: Union[float, List[float]],
                                    water_levels: np.ndarray):
        self.measured_mean = measured_moment_mean
        self.measured_sigma = measured_moment_sigma * self.measured_mean
        self.measured_sigma[self.measured_sigma < 0] = -self.measured_sigma[self.measured_sigma < 0]
        self.water_levels = water_levels
        # return self.measured_mean, self.measured_sigma

    def generate_synthetic_measurement(self, parameters: np.ndarray, sigma: float = 1):
        """
        Generate a synthetic measurement for given parameters and sigma
        """
        self.measured_mean = self.moment_model.get_moment_for_parameters(parameters)
        self.measured_sigma = sigma * self.measured_mean
        mask = self.measured_sigma < 0
        self.measured_sigma[mask] = -self.measured_sigma[mask]
        # return self.measured_mean, self.measured_sigma

    def compute_likelihood_for_equality_moment(self, moment_samples: np.ndarray) -> np.ndarray:
        """
        Compute likelihood for a survived moment sample.
        """
        n = self.measured_mean.shape[0]
        w = 1 / n   # uniform weights
        likelihood = np.array([sum(w * norm.pdf(moment_samples, self.measured_mean[i], self.measured_sigma[i])
                         for i in range(n))])
        return likelihood

    def compute_log_likelihood_for_equality_moment(self, moment_samples: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood for a survived moment sample.
        """
        return np.log(self.compute_likelihood_for_equality_moment(moment_samples))

    def compute_likelihood_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute likelihood for a parameters sample.
        """
        if parameters.ndim == 1:
            parameters = parameters.reshape(-1, 1)

        cur_water_levels = self.water_levels[:len(parameters), :]
        collected_parameters = np.hstack([parameters, cur_water_levels])
        self.parameter_values = self.moment_model.get_moment_for_parameters(collected_parameters)
        return self.compute_likelihood_for_equality_moment(self.parameter_values)
    
    def compute_log_likelihood_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood for a parameters sample.
        """
        return np.log(self.compute_likelihood_for_parameters(parameters))
    
class DisplacementLikelihood(BaseLikelihood):
    """Likelihood function for displacement measurements."""
    def __init__(self,
                 model_path: str,
                 model_type: str = "gpr"):
        """
        Initialize the displacement likelihood.
        
        Args:
            measured_displacement_mean (np.ndarray): Mean values of measured displacements
            measured_displacement_sigma (Union[float, List[float]]): Standard deviations of measured displacements
        """
        super().__init__()
        torch.set_num_threads(1)
        self.displacement_model = ModelBridge(model_path, model_type)

    def set_measured_mean_and_sigma(self, measured_displacement_mean: np.ndarray, 
                                    measured_displacement_sigma: Union[float, List[float]]):
        self.measured_mean = measured_displacement_mean
        self.measured_sigma = measured_displacement_sigma * self.measured_mean
        self.measured_sigma[self.measured_sigma < 0] = -self.measured_sigma[self.measured_sigma < 0]
        # return self.measured_mean, self.measured_sigma
    
    def set_max_parameter_value(self, max_parameter_value: float):
        self.max_paramater_value = max_parameter_value
        return self.max_paramater_value

    def generate_synthetic_measurement(self, parameters: np.ndarray, sigma: float = 1):
        """
        Generate a synthetic measurement for given parameters and sigma
        """
        self.measured_mean = self.displacement_model.get_displacement_for_parameters(parameters)
        self.measured_sigma = sigma * self.measured_mean
        mask = self.measured_sigma < 0
        self.measured_sigma[mask] = -self.measured_sigma[mask]
        # return self.measured_mean, self.measured_sigma

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
    
    def compute_likelihood_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute likelihood for model parameters.
        
        Args:
            parameters (List[float]): The model parameters
            model_function: Function that computes displacement from parameters
            
        Returns:
            tuple: (likelihood value, displacement)
        """
        # self.parameter_values = self.get_displacement_for_parameters(parameters)
        self.parameter_values = self.displacement_model.get_displacement_for_parameters(parameters)
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

class ModelBridge:
    """
    ModelBridge class for the DSheetPiling model
    """
    def __init__(self, model_path: str, model_type: str = "gpr"):
        self.model_type = model_type
        self.model_path = model_path

    def get_moment_for_soil_parameters(self, soil_parameters: np.ndarray, soil_water_level: np.ndarray, canal_water_level: np.ndarray) -> np.ndarray:
        """
        Get the moment for given soil parameters
        """
        parameters = np.concatenate([soil_parameters, soil_water_level, canal_water_level])
        if self.model_type == "gpr":
            return self.GPR_surrogate_function(parameters)
        elif self.model_type == "torch":
            return self.torch_surrogate_function(parameters)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}. Not implemented yet.")

    def get_moment_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Get the moment for given parameters
        """
        if self.model_type == "gpr":
            return self.GPR_surrogate_function(parameters)
        elif self.model_type == "torch":
            return self.torch_surrogate_function(parameters)
        elif self.model_type == "dsheet":
            raise ValueError("DSheetPiling moment calculations are not implemented yet.")
        #     parameter_values = []
        #     for i in range(len(parameters)):
        #         parameter_values.append(self.get_moment_from_dsheet_model(parameters[i]))
        #     return parameter_values
        else:
            raise ValueError(f"Invalid model type: {self.model_type}. Not implemented yet.")

    def get_displacement_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        if self.model_type == "gpr":
            return self.GPR_surrogate_function(parameters)
        elif self.model_type == "torch":
            return self.torch_surrogate_function(parameters)
        elif self.model_type == "dsheet":
            parameter_values = []
            for i in range(len(parameters)):
                parameter_values.append(self.get_displacement_from_dsheet_model(parameters[i]))
            return parameter_values
        else:
            raise ValueError(f"Invalid model type: {self.model_type}. Not implemented yet.")
        
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
        parameters[10]: Water_level
        parameters[11]: Water_constant
        Returns:
            displacement: estimated displacement in mm
        """
        # Ensure model path exists
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"GPR model file not found: {self.model_path}")
            
        X_predict_tensor = torch.tensor(parameters, dtype=torch.float32)
        # ensure that X_predict_tensor is a 2D tensor
        if X_predict_tensor.dim() == 1:
            X_predict_tensor = X_predict_tensor.unsqueeze(0)

        # cur_model = DependentGPRModels()
        # cur_model.load(self.model
        cur_model = load_gpr_model(self.model_path)
        parameter_values, var_parameter_values = cur_model.predict(X_predict_tensor)
        return parameter_values[:, 10]
    
    def torch_surrogate_function(self, parameters: np.ndarray):
        """
        Torch surrogate function for given parameters
        """
        # Check if all required torch model files exist
        torch_model_path = Path(self.model_path) / "torch_model.pth"
        torch_model_params_path = Path(self.model_path) / "torch_weights.pth"
        scaler_x_path = Path(self.model_path) / "scaler_x.joblib"
        scaler_y_path = Path(self.model_path) / "scaler_y.joblib"
        
        missing_files = []
        if not torch_model_path.exists():
            missing_files.append(str(torch_model_path))
        if not scaler_x_path.exists():
            missing_files.append(str(scaler_x_path))
        if not scaler_y_path.exists():
            missing_files.append(str(scaler_y_path))
            
        if missing_files:
            raise FileNotFoundError(f"Torch model files not found: {missing_files}")
            
        X_predict_tensor = torch.tensor(parameters, dtype=torch.float32)
        # ensure that X_predict_tensor is a 2D tensor
        if X_predict_tensor.dim() == 1:
            X_predict_tensor = X_predict_tensor.unsqueeze(0)

        #TODO: unhardcode
        model = MLP(input_dim=11, hidden_dims=[1024, 512, 256, 128, 64, 32], output_dim=15)
        model_state_dict = torch.load(torch_model_params_path)
        model.load_state_dict(model_state_dict)
        # model = torch.load(torch_model_path)
        model.eval()
        import joblib
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        # return displacement[:, 0, -1]
        parameter_values = inference(model, X_predict_tensor, scaler_x, scaler_y)
        return parameter_values[:,10]

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