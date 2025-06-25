from src.reliability_models.surrogate.reliability import SurrogateReliability
from src.geotechnical_models.gpr.gpr_classes import DependentGPRModels, MultitaskGPModel, load_gpr_model
from src.geotechnical_models.mlp.MLP_class import MLP, inference
from src.bayesian_updating.ERADist import ERADist
from src.bayesian_updating.ERANataf import ERANataf
from src.bayesian_updating.BUS_SuS import BUS_SuS
from src.bayesian_updating.aBUS_SuS import aBUS_SuS
from src.bayesian_updating.iTMCMC import iTMCMC
from src.bayesian_updating.likelihood_functions import DisplacementLikelihood
from main.dhseetpiling_deformation_updating.BUS_approach import PosteriorRetainingStructure

from datetime import datetime
import pandas as pd
from typing import Annotated, List, Tuple, Dict, NamedTuple
import numpy as np
from pathlib import Path
import json
import torch




class PhysicalTwin:

    def __init__(self, model_path: str, model_type: str = "gpr", t_start: int = 0, t_end: int = 50, t_step: int = 1, max_moment: float = 40):
        self.model_path = model_path
        self.model_type = model_type
        self.start_capacity = max_moment
        self.timeframe = np.arange(t_start, t_end, t_step)
        self.surrogate_model = load_gpr_model(self.model_path)
        self.reliability_model = SurrogateReliability(max_moment=max_moment, max_displacement=10)
        self.parameter_names, self.parameter_values, self.water_parameter_names, self.water_parameter_values = self.generate_initial_state()
        self.state_history = []
        self.water_state_history = []

    def generate_state_history(self, timeframe: np.ndarray):
        """
        Get the current state
        """
        self.corrosion_rate_history, true_EIs, self.capacity_history = self._corrosion_model(timeframe, self.start_EI, self.start_capacity)
        true_WaterLvls = self._lower_water_level_model(timeframe)
        true_WaterConstants = self._water_level_model(timeframe)
        # true_WaterConstants[1] = -0.8
        # true_WaterLvls[1] = -0.1
        
        self.state_history = []
        # self.displacement_history = []
        self.moment_history = []
        self.pf_history = []
        self.beta_history = []
        self.water_level_history = []
        self.ei_history = []
        
        wall_ei_id = self.parameter_names.index('Wall_SheetPilingElementEI')
        water_lvl_id = self.water_parameter_names.index('water_lvl')
        water_constant_id = self.water_parameter_names.index('water_constant')
        for ei, water_lvl, water_constant, capacity in zip(true_EIs, true_WaterLvls, true_WaterConstants, self.capacity_history):
            cur_parameter_values = self.parameter_values.copy()
            cur_water_parameter_values = self.water_parameter_values.copy()
            cur_parameter_values[wall_ei_id] = ei
            cur_water_parameter_values[water_lvl_id] = water_lvl
            cur_water_parameter_values[water_constant_id] = water_constant
            

            # displacements = self.get_displacement_from_samples(cur_parameter_values)
            # pf = self.reliability_model.probability_of_failure_displacement(displacements)
            load_moments = self.get_moment_from_samples(cur_parameter_values, cur_water_parameter_values)
            self.parameter_values = cur_parameter_values
            self.water_parameter_values = cur_water_parameter_values
            self.state_history.append(cur_parameter_values)
            self.water_state_history.append(cur_water_parameter_values)
            self.moment_history.append(load_moments)
            self.ei_history.append(ei)
            self.pf_history.append(self.reliability_model.probability_of_failure_moment(load_moments, capacity))
            self.beta_history.append(self.reliability_model.calculate_reliability_index(self.pf_history[-1]))


    def generate_initial_state(self, scale: float = 1.):
        """
        Get the true soil parameters
        """
        with open('examples/deterministic_parameters.json', 'r') as f:
            parameters = json.load(f)
        material_properties = parameters['material_properties']
        parameter_names = []
        parameter_values = []
        for material_name, material_values in material_properties.items():
            if material_name == 'klei siltig' or material_name == 'Zand los (18)':
                continue
            if material_values['cohesion'] != 0:
                parameter_names.append(f'{material_name}_soilcohesion')
                parameter_values.append(material_values['cohesion']*scale)
            if material_values['phi'] != 0:
                parameter_names.append(f'{material_name}_soilphi')
                parameter_values.append(material_values['phi']*scale)
            if material_values['k_top']['kh1'] != 0:
                parameter_names.append(f'{material_name}_soilcurkb1')
                parameter_values.append(material_values['k_top']['kh1']*scale)

        # add wall EI
        self.start_EI = 45_000*scale
        parameter_names.append('Wall_SheetPilingElementEI')
        parameter_values.append(self.start_EI)

        water_parameter_names = []
        water_parameter_values = []
        # add water level
        water_parameter_names.append('water_lvl')
        water_parameter_values.append(-0.8*scale)
        
        # add water constant
        water_parameter_names.append('water_constant')
        water_parameter_values.append(-0.4*scale)

        return parameter_names, parameter_values, water_parameter_names, water_parameter_values

    
    def get_water_level(self, time: int):
        """
        Get the water level at a given time
        """
        return self.state_history[time][self.parameter_names.index('water_lvl')]
    
    def get_corrosion_rate(self, time: int):
        """
        Get the corrosion rate at a given time
        """
        return self.corrosion_rate_history[time]
    
    def get_displacement(self, time: int):
        """
        Get the displacement at a given time
        """
        return self.displacement_history[time]
    
    def get_moment(self, time: int):
        """
        Get the moment at a given time
        """
        return self.moment_history[time]
    
    def get_soil_sample(self):
        """
        Get the soil sample at a given time
        """
        return np.array(self.parameter_values[:-1])
    
    @staticmethod
    def _corrosion_model(timeframe: np.ndarray, start_EI: float, start_capacity: float, a: float = 0.3226, b: float = 0.57, var: float = 0.1):
        # """
        # Corrosion model based on Jongbloed curves
        # """
        # return a * timeframe ** b
        """
        Update the wall EI and current state due to corrosion.
        Returns the new pdf
        """
        corrosion_rate_mean = (a * timeframe** b) / 10
        corrosion_rate_std = var * corrosion_rate_mean
        corrosion_rate = np.random.normal(corrosion_rate_mean, corrosion_rate_std)
        ei_over_time = (1-corrosion_rate) * start_EI
        capacity_over_time = (1-corrosion_rate) * start_capacity
        return corrosion_rate, ei_over_time, capacity_over_time
    
    @staticmethod
    def _corrosion_model_old(timeframe: np.ndarray, start_EI: float):
        """
        Update the wall EI and current state due to corrosion.
        Returns the new pdf
        """
        # get corrosion rate for time passed since last update
        corrosion_rates = 1 - timeframe / 120
        # calculate new wall EIs due to corrosion
        wall_EI_over_time = corrosion_rates * start_EI
        # return wall_EI_over_time
        return np.repeat(start_EI, len(timeframe))

    @staticmethod
    def _lower_water_level_model(timeframe: np.ndarray, water_period_yr: int = 20):
        """
        Simple water level model -> -0.4 + np.random.normal(0, 0.05, len(timeframe))
        """
        return -0.4 + np.random.normal(0, 0.05, len(timeframe))
        # return np.repeat(-0.4, len(timeframe))

    @staticmethod
    def _water_level_model(timeframe: np.ndarray, water_period_yr: int = 20):
        """
        Simple water level model -> -0.4 + np.random.normal(0, 0.05, len(timeframe))
        """
        return -0.65 + np.random.normal(0, 0.1, len(timeframe))
        # return np.repeat(-0.6, len(timeframe))
    

    def get_displacement_from_samples(self, samples: np.ndarray):
        """
        Get the displacement from the samples
        """
        X_predict_tensor = torch.tensor(samples, dtype=torch.float32)
        # ensure that X_predict_tensor is a 2D tensor
        if X_predict_tensor.dim() == 1:
            X_predict_tensor = X_predict_tensor.unsqueeze(0)

        displacement, var_displacement = self.surrogate_model.predict(X_predict_tensor)
        return displacement[:, 10]

    def get_moment_from_samples(self, samples: np.ndarray, water_samples: np.ndarray):
        """
        Get the moment from the samples
        """
        collected_samples = np.concatenate([samples, water_samples])
        X_predict_tensor = torch.tensor(collected_samples, dtype=torch.float32)
        # ensure that X_predict_tensor is a 2D tensor
        if X_predict_tensor.dim() == 1:
            X_predict_tensor = X_predict_tensor.unsqueeze(0)
        moment, var_moment = self.surrogate_model.predict(X_predict_tensor)
        return moment[:, 10]

if __name__ == "__main__":
    pass