import sys
from pathlib import Path
# # # Add the project root directory to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.reliability_models.surrogate.reliability import SurrogateReliability
from src.geotechnical_models.gpr.gpr_classes import DependentGPRModels, MultitaskGPModel, load_gpr_model
from src.geotechnical_models.mlp.MLP_class import MLP, inference
from src.bayesian_updating.ERADist import ERADist
from src.bayesian_updating.ERANataf import ERANataf
from src.bayesian_updating.BUS_SuS import BUS_SuS
from src.bayesian_updating.aBUS_SuS import aBUS_SuS
from src.bayesian_updating.iTMCMC import iTMCMC
from src.bayesian_updating.likelihood_functions import DisplacementLikelihood, MomentLikelihood, MomentLikelihoodSurvived, CorrosionLikelihood, SoilPropertyLikelihood
from src.corrosion.corrosion_model import CorrosionModelSimple
from main.dhseetpiling_deformation_updating.BUS_approach import PosteriorRetainingStructure

from datetime import datetime
import pandas as pd
from typing import Annotated, List, Tuple, Dict, NamedTuple
import numpy as np
from pathlib import Path
import json
import torch


class DigitalTwin:

    def __init__(self, model_path: str,
                 timeframe: np.ndarray,
                 model_type: str = "gpr",
                 normal_distribution: bool = False,
                 max_moment: float = 40,
                 max_displacement: float = 10,
                 n_samples: int = 1_00):
        """
        Initialize the PosteriorRetainingStructure class

        Args:
            model_path (str): Path to the DSheetPiling model file
            measurement_path (str): Path to the measurement data file
            use_surrogate (bool): Whether to use the fake surrogate model instead of DSheetPiling
        """
        self.model_type = model_type
        self.n_samples = n_samples
        self.model_path = model_path
        self.timeframe = timeframe
        self.corrosion_model = CorrosionModelSimple()
        self.surrogate_model = load_gpr_model(model_path)
        self.reliability_model = SurrogateReliability(max_moment=max_moment, max_displacement=max_displacement)

        self.soil_properties = {}
        self.wall_properties = {}
        self.water_levels = {}
        self.moment_dict = {}
        self.corrosion_dict = {}
        
        self.define_initial_state(normal_distribution=normal_distribution, start_max_moment=max_moment)

        # self.l_displacement = self.create_likelihood_function(model_path, l_type='displacement')
        self.l_moment = self.create_likelihood_function(model_path, l_type='moment')
        self.l_corrosion = self.create_likelihood_function(model_path, l_type='corrosion')
        self.l_moment_survived = self.create_likelihood_function(model_path, l_type='moment_survived')
        self.l_soil_property = self.create_likelihood_function(model_path, l_type='soil_property')
        
        self.posterior_retention_structure = self.initialize_inference_model()
        self.received_displacements = []
        self.received_moments = []
        self.received_corrosions = []

    def create_likelihood_function(self, model_path: str, l_type: str = 'displacement'):
        """
        Create the likelihood function
        """
        if l_type == 'displacement':
            return DisplacementLikelihood(
                    model_path=model_path,
                    model_type=self.model_type,
                    )
        
        elif l_type == 'moment':
            return MomentLikelihood(
                model_path=model_path,
                model_type=self.model_type,
            )
        
        elif l_type == 'moment_survived':
            return MomentLikelihoodSurvived(
                model_path=model_path,
                model_type=self.model_type,
            )
        elif l_type == 'corrosion':
            return CorrosionLikelihood(
            )
        elif l_type == 'soil_property':
            return SoilPropertyLikelihood(
            )
        
    def initialize_inference_model(self):
        return PosteriorRetainingStructure(
            self.model_path, model_type=self.model_type, normal_distribution=True
        )
    
    def generate_samples(self, n_samples: int = 1_000):
        """
        Generate samples from the current state
        """
        samples = self.current_pdf.random(n_samples)
        # save the samples to a csv file
        df = pd.DataFrame(samples, columns=self.parameter_names)
        df.to_csv(f'/Users/dafydd/Deltares/ms5-asset-performance/main/case_study_2025/train/data/1M_samples_normally_distributed.csv', index=False)
        return samples

    def define_initial_state(self, normal_distribution: bool = False, start_max_moment: float = 40):
        # Calculate the corrosion rate history
        self.corrosion_dict, capacity_history, wall_state_history = self._get_predicted_corrosion_rate(timeframe=self.timeframe, start_capacity=start_max_moment)
        
        # Define initial state
        self.soil_properties = self._initial_soil_properties(normal_distribution=normal_distribution)
        self.wall_properties = self._initial_wall_properties(normal_distribution=normal_distribution, state_history=wall_state_history)
        self.water_level_properties = self._initial_water_properties(normal_distribution=normal_distribution, n_samples=self.n_samples)
        
        self.moment_dict = self._initial_moment_dict(soil_distributions=self.soil_properties['distributions'],
                                                     wall_distributions=self.wall_properties['distributions'],
                                                     water_samples=self.water_level_properties['samples'],
                                                     capacity_history=capacity_history,
                                                     n_samples=self.n_samples,
                                                     init_time=self.timeframe[0],
                                                     start_max_moment=start_max_moment)


    def _initial_moment_dict(self,
                             soil_distributions: List[ERADist], 
                             wall_distributions: List[ERADist], 
                             water_samples: np.ndarray,
                             capacity_history: Dict,
                             n_samples: int = 1_00,
                             init_time: int = 0,
                             start_max_moment: float = 40):
        """
        Initialize the moment dictionary
        """
        # Define initial pdf for calculating moment
        dimensions = len(soil_distributions+wall_distributions)
        R = np.eye(dimensions)
        # Define initial pdf for calculating moment with soil and wall properties
        moment_property_joint_pdf = ERANataf(soil_distributions+wall_distributions, R)
        # Generate samples from the initial pdf
        property_samples = moment_property_joint_pdf.random(n_samples)
        # Calculate the moment history for soil, wall and water level properties
        moment_samples = [self.get_moment_from_samples(property_samples, water_samples)]

        for cur_time in self.timeframe:
            capacity_history[cur_time]['pf'] = self.reliability_model.probability_of_failure_moment([moment_samples], [capacity_history[cur_time]['samples']])
            capacity_history[cur_time]['beta'] = self.reliability_model.calculate_reliability_index(capacity_history[cur_time]['pf'])
        
        
        moment_dict = {'R': R,
                       'joint_distribution': moment_property_joint_pdf,
                       'load_history': {init_time: {
                           'property_samples': property_samples,
                           'behavior_samples': water_samples,
                           'moment_samples': moment_samples,
                       }},
                       'start_capacity': start_max_moment,
                       'capacity_history': capacity_history,
        }
        return moment_dict

    @staticmethod
    def _initial_soil_properties(normal_distribution: bool = True):
        """
        Initialize the soil properties
        """
        # load parameters from json file
        with open('examples/deterministic_parameters.json', 'r') as f:
            parameters = json.load(f)

        material_properties = parameters['material_properties']
        distributions = []
        parameter_names = []
        scale = 1.3
        lower_bound = 0.3
        upper_bound = 1.8
        std_scale = 0.1
        mean_values = []
        std_values = []
        for material_name, material_values in material_properties.items():
            if material_name == 'klei siltig' or material_name == 'Zand los (18)':
                continue
            if material_values['cohesion'] != 0:
                if normal_distribution:
                    cohesion = ERADist('normal', 'PAR', [material_values['cohesion'], material_values['cohesion']*std_scale])
                    mean_values.append(material_values['cohesion'])
                    std_values.append(material_values['cohesion']*std_scale)
                else:
                    cohesion = ERADist('uniform', 'PAR', [material_values['cohesion']*lower_bound, material_values['cohesion']*upper_bound])
                distributions.append(cohesion)
                parameter_names.append(f'{material_name}_soilcohesion')
            if material_values['phi'] != 0:
                if normal_distribution:
                    phi = ERADist('normal', 'PAR', [material_values['phi'], material_values['phi']*std_scale])
                    mean_values.append(material_values['phi'])
                    std_values.append(material_values['phi']*std_scale)
                else:
                    phi = ERADist('uniform', 'PAR', [material_values['phi']*lower_bound, material_values['phi']*upper_bound])
                distributions.append(phi)
                parameter_names.append(f'{material_name}_soilphi')
            if material_values['k_top']['kh1'] != 0:
                if normal_distribution:
                    kh1 = ERADist('normal', 'PAR', [material_values['k_top']['kh1'], material_values['k_top']['kh1']*std_scale])
                    mean_values.append(material_values['k_top']['kh1'])
                    std_values.append(material_values['k_top']['kh1']*std_scale)
                else:
                    kh1 = ERADist('uniform', 'PAR', [material_values['k_top']['kh1']*lower_bound, material_values['k_top']['kh1']*upper_bound])
                distributions.append(kh1)
                parameter_names.append(f'{material_name}_soilcurkb1') # b van boven?
        
        soil_properties = {'distributions': distributions, 
                           'parameter_names': parameter_names, 
                           'mean': mean_values,
                           'std': std_values,
                           'state_history': {"0": {"distributions": distributions,
                                                   "mean": mean_values,
                                                   "std": std_values}}}
        return soil_properties
    
    @staticmethod
    def _initial_wall_properties(normal_distribution: bool = True, wall_EI_start: float = 45_000, variance_start: float = 0.001, state_history: Dict = None):
        """
        Initialize the wall properties
        """
        std_start = variance_start*wall_EI_start
        if normal_distribution:
            wall_eis = ERADist('normal', 'PAR', [wall_EI_start, std_start])
        else:
            wall_eis = ERADist('uniform', 'PAR', [5000, 50000])
        wall_properties = {'distributions': [wall_eis], 
                           'mean': [wall_EI_start],
                           'std': [std_start],
                           'parameter_names': ['Wall_SheetPilingElementEI'],
                           'state_history': state_history}
        return wall_properties
    
    @staticmethod
    def _initial_water_properties(normal_distribution: bool = True, n_samples: int = 1_00):
        """
        Initialize the water properties
        """
        water_distributions = []
        # # add water level
        if normal_distribution:
            mean_water_level = -0.6
            std_water_level = 0.02
            canal_water_level = ERADist('normal', 'PAR', [mean_water_level, std_water_level])
            mean_water_constant = -0.4
            std_water_constant = 0.1
            soil_water_level = ERADist('normal', 'PAR', [mean_water_constant, std_water_constant])
        else:
            lower_water_level = -1.
            upper_water_level = 0.
            soil_water_level = ERADist('uniform', 'PAR', [lower_water_level, upper_water_level])
            canal_water_level = ERADist('uniform', 'PAR', [lower_water_level, upper_water_level])
        water_distributions.append(soil_water_level)
        water_distributions.append(canal_water_level)
        water_samples = np.vstack([soil_water_level.random(n_samples), canal_water_level.random(n_samples)]).T

        water_properties = {'distributions': water_distributions, 
                           'parameter_names': ['water_lvl', 'water_constant'], 
                           'samples': water_samples,
                           'state_history': {"0": {"distributions": water_distributions,
                                                   "mean": [mean_water_level, mean_water_constant],
                                                   "std": [std_water_level, std_water_constant]}}}
        return water_properties

    def get_moment_from_samples(self, property_samples: np.ndarray, behavior_samples: np.ndarray):
        """
        Get the moment from the samples
        """
        samples = np.hstack([property_samples, behavior_samples])
        X_predict_tensor = torch.tensor(samples, dtype=torch.float32)
        # ensure that X_predict_tensor is a 2D tensor
        if X_predict_tensor.dim() == 1:
            X_predict_tensor = X_predict_tensor.unsqueeze(0)
        moment, var_moment = self.surrogate_model.predict(X_predict_tensor)
        return moment[:, 10]
    
    def get_displacement_from_samples(self, samples: np.ndarray):
        """
        Get the displacement from the samples
        """
        X_predict_tensor = torch.tensor(samples, dtype=torch.float32)
        # ensure that X_predict_tensor is a 2D tensor
        if X_predict_tensor.dim() == 1:
            X_predict_tensor = X_predict_tensor.unsqueeze(0)

        displacement, var_displacement = self.surrogate_model.predict(X_predict_tensor)
        return displacement[:, -1]
    
    def get_new_state_for_corrosion(self, N, p0, approach, max_it):
        """
        Get the new state of the posterior for corrosion
        """
        # get id for EI from parameter names
        # ei_id = self.parameter_names.index('Wall_SheetPilingElementEI')
        # get the current pdf
        # current_pdf = self.dist_parameters[ei_id]
        if approach == 'BUS':
            return BUS_SuS(N, p0, self.l_corrosion, self.corrosion_dict['parameter_distributions'][0], max_it=max_it)
        elif approach == 'aBUS':
            return aBUS_SuS(N, p0, self.l_corrosion, self.corrosion_dict['parameter_distributions'][0], max_it=max_it)
        else:
            raise ValueError(f'Invalid approach: {approach}')
        
    def get_new_state_for_survived_moment(self, N, p0, approach, max_it):
        """
        Get the new state of the posterior for survived moment
        """
        if approach == 'BUS':
            return BUS_SuS(N, p0, self.l_moment_survived, self.current_pdf, max_it=max_it)
        elif approach == 'aBUS':
            return aBUS_SuS(N, p0, self.l_moment_survived, self.current_pdf, max_it=max_it)
        else:
            raise ValueError(f'Invalid approach: {approach}')
        
    def get_new_state_for_measured_moment(self, N, p0, approach, max_it, cur_time: int = 0):
        """
        Get the new state of the posterior
        """
        if approach == 'BUS':
            return BUS_SuS(N, p0, self.l_moment, self.moment_dict['joint_distribution'], max_it=max_it)
        elif approach == 'aBUS':
            return aBUS_SuS(N, p0, self.l_moment, self.moment_dict['joint_distribution'], max_it=max_it)
        else:
            raise ValueError(f'Invalid approach: {approach}')
        
    def get_new_state_for_soil_sample(self, N, p0, approach, max_it, parameter_distribution: ERADist):
        """
        Get the new state of the posterior for soil sample
        """
        if approach == 'BUS':
            return BUS_SuS(N, p0, self.l_soil_property, parameter_distribution, max_it=max_it)
        elif approach == 'aBUS':
            return aBUS_SuS(N, p0, self.l_soil_property, parameter_distribution, max_it=max_it)
        else:
            raise ValueError(f'Invalid approach: {approach}')
        
    def set_state_from_corrosion_samples(self, corrosion_value_samples: np.ndarray, parameter_a_samples: np.ndarray, time_index: int = 0):
        """
        Set the state with new parameter values
        """
        new_corrosion_parameter_mean = np.mean(parameter_a_samples)
        new_corrosion_parameter_std = np.std(parameter_a_samples)
        self.corrosion_dict['parameter_distributions'] = [ERADist('normal', 'PAR', [new_corrosion_parameter_mean, new_corrosion_parameter_std])]

        corrosion_rate_history = self.corrosion_dict['corrosion_rate_history'].copy()
        corrosion_rate_history[self.timeframe[time_index]] = {"samples": corrosion_value_samples,
                                                             "mean": np.mean(corrosion_value_samples),
                                                             "std": np.std(corrosion_value_samples)}

        capacity_history = self.moment_dict['capacity_history'].copy()
        capacity_history[self.timeframe[time_index]] = {"samples": self.moment_dict['start_capacity'] * (1-corrosion_value_samples),
                                                        "mean": np.mean(self.moment_dict['start_capacity'] * (1-corrosion_value_samples)),
                                                        "std": np.std(self.moment_dict['start_capacity'] * (1-corrosion_value_samples))}
        
        ei_history = self.wall_properties['state_history'].copy()
        ei_history[self.timeframe[time_index]] = {"samples": self.wall_properties['state_history'][self.timeframe[time_index]]['samples'] * (1-corrosion_value_samples),
                                                  "mean": np.mean(self.wall_properties['state_history'][self.timeframe[time_index]]['samples'] * (1-corrosion_value_samples)),
                                                  "std": np.std(self.wall_properties['state_history'][self.timeframe[time_index]]['samples'] * (1-corrosion_value_samples))}

        for cur_t in self.timeframe[time_index+1:]:
            cur_corrosion_rate = self.corrosion_model.get_corrosion_rate_at_t(cur_t=cur_t, samples_a=parameter_a_samples)
            corrosion_rate_history[cur_t] = {"samples": cur_corrosion_rate,
                                             "mean": np.mean(cur_corrosion_rate),
                                             "std": np.std(cur_corrosion_rate)}
            cur_capacity = self.moment_dict['start_capacity'] * (1-cur_corrosion_rate)
            capacity_history[cur_t] = {"samples": cur_capacity,
                                       "mean": np.mean(cur_capacity),
                                       "std": np.std(cur_capacity)}
            cur_ei = self.wall_properties['state_history'][self.timeframe[time_index]]['samples'] * (1-cur_corrosion_rate)
            ei_history[cur_t] = {"samples": cur_ei,
                                 "mean": np.mean(cur_ei),
                                 "std": np.std(cur_ei)}
        
        self.corrosion_dict['corrosion_rate_history'] = corrosion_rate_history
        self.moment_dict['capacity_history'] = capacity_history
        self.wall_properties['state_history'] = ei_history


    def update_state_for_new_corrosion(self, measured_corrosion: float, measured_sigma: float = 0.1, time_index: int = 0):
        """
        Update the state with new corrosion data
        """
        self.received_corrosions.append((measured_corrosion, measured_sigma))
        self.l_corrosion.cur_t = self.timeframe[time_index]
        self.l_corrosion.set_measured_mean_and_sigma(measured_corrosion, measured_sigma)
    
        # Initialize with surrogate model
        h, samplesU, samplesX, logcE, c, sigma, parameter_values = self.get_new_state_for_corrosion(N=self.n_samples, p0=0.1, approach='aBUS', max_it=10)

        self.set_state_from_corrosion_samples(corrosion_value_samples=parameter_values[-1], parameter_a_samples=samplesX[-1][:,0], time_index=time_index)
 
    def update_state_for_new_displacement(self, measured_displacement: float, measured_sigma: float = 0.1):
        """
        Update the state with new displacement data
        """
        self.received_displacements.append((measured_displacement, measured_sigma))
        self.l_displacement.set_measured_mean_and_sigma(measured_displacement, measured_sigma)
        # Initialize with surrogate model
        h, samplesU, samplesX, logcE, c, sigma, parameter_values = self.get_new_state(N=self.n_samples, p0=0.1, approach='aBUS', max_it=10)
        self.current_pdf = self.set_state_from_samples(samplesX[-1])
        self.displacement_history.append(parameter_values[-1])  
        self.pf_history.append(self.reliability_model.probability_of_failure_displacement(parameter_values[-1]))
        self.beta_history.append(self.reliability_model.calculate_reliability_index(self.pf_history[-1]))

    def update_state_for_new_moment(self, measured_moment: float, measured_sigma: float = 0.1, cur_time: int = 0):
        """
        Update the state with new moment data
        """
        self.received_moments.append((measured_moment, measured_sigma))
        self.l_moment.set_measured_mean_and_sigma(measured_moment, measured_sigma, self.water_level_properties['samples'])
        
        # Initialize with surrogate model
        h, samplesU, samplesX, logcE, c, sigma, parameter_values = self.get_new_state_for_measured_moment(N=self.n_samples, p0=0.1, approach='aBUS', max_it=10)
        
        nr_soil_parameters = len(self.soil_properties['parameter_names'])
        nr_wall_parameters = len(self.wall_properties['parameter_names'])
        new_soil_parameter_distributions = []
        new_soil_parameter_means = []
        new_soil_parameter_stds = []
        for i_par, parameter_name in enumerate(self.soil_properties['parameter_names']):
            new_soil_parameter_distributions.append(ERADist('normal', 'DATA', [samplesX[-1][:,i_par]]))
            new_soil_parameter_means.append(np.mean(samplesX[-1][:,i_par]))
            new_soil_parameter_stds.append(np.std(samplesX[-1][:,i_par]))
        self.soil_properties['distributions'] = new_soil_parameter_distributions
        self.soil_properties['mean'] = new_soil_parameter_means
        self.soil_properties['std'] = new_soil_parameter_stds

        new_wall_parameter_distributions = []
        new_wall_parameter_means = []
        new_wall_parameter_stds = []
        for j_par, parameter_name in enumerate(self.wall_properties['parameter_names']):
            new_wall_parameter_distributions.append(ERADist('normal', 'DATA', [samplesX[-1][:,nr_soil_parameters+j_par]]))
            new_wall_parameter_means.append(np.mean(samplesX[-1][:,nr_soil_parameters+j_par]))
            new_wall_parameter_stds.append(np.std(samplesX[-1][:,nr_soil_parameters+j_par]))
        self.wall_properties['distributions'] = new_wall_parameter_distributions
        self.wall_properties['mean'] = new_wall_parameter_means
        self.wall_properties['std'] = new_wall_parameter_stds

        new_joint_distribution = ERANataf(new_soil_parameter_distributions + new_wall_parameter_distributions, self.moment_dict['R'])
        self.moment_dict['joint_distribution'] = new_joint_distribution
        return samplesX[-1], parameter_values[-1]

    def update_state(self, cur_time_index: int, property_samples: np.ndarray = None, moment_samples: np.ndarray = None):
        self.soil_properties['state_history'][cur_time_index] = {"distributions": self.soil_properties['distributions'],
                                                                 "mean": self.soil_properties['mean'],
                                                                 "std": self.soil_properties['std']}
        self.wall_properties['state_history'][cur_time_index] = {"distributions": self.wall_properties['distributions'],
                                                                 "mean": self.wall_properties['mean'],
                                                                 "std": self.wall_properties['std']}
        self.water_level_properties['state_history'][cur_time_index] = {"distributions": self.water_level_properties['distributions'],
                                                                 "mean": [self.water_level_properties['distributions'][0].Par["mu"], 
                                                                          self.water_level_properties['distributions'][1].Par["mu"]],
                                                                 "std": [self.water_level_properties['distributions'][0].Par["sigma"], 
                                                                         self.water_level_properties['distributions'][1].Par["sigma"]]}

        # get new property samples if not provided
        if property_samples is None:
            property_samples = self.moment_dict['joint_distribution'].random(self.n_samples)

        # get new moment samples if not provided
        if moment_samples is None:
            moment_samples = self.get_moment_from_samples(property_samples, self.water_level_properties['samples'])

        for i_cur_time, cur_time in enumerate(self.timeframe[cur_time_index:]):
            self.moment_dict['load_history'][i_cur_time+cur_time_index] = {"property_samples": property_samples,
                                                          "behavior_samples": self.water_level_properties['samples'],
                                                          "moment_samples": moment_samples}

            self.moment_dict['capacity_history'][cur_time]['pf'] = self.reliability_model.probability_of_failure_moment([moment_samples], [self.moment_dict['capacity_history'][cur_time]['samples']])
            self.moment_dict['capacity_history'][cur_time]['beta'] = self.reliability_model.calculate_reliability_index(self.moment_dict['capacity_history'][cur_time]['pf'])
    
    def update_state_for_survived_load(self, survived_load: float, survived_sigma: float = 0.1):
        """
        Update the state with new load data
        """
        # self.received_loads.append((survived_load, survived_sigma))
        self.l_moment_survived.set_survived_moment(survived_load)
        
        # Initialize with surrogate model
        h, samplesU, samplesX, logcE, c, sigma, parameter_values = self.get_new_state_for_survived_moment(N=self.n_samples, p0=0.1, approach='aBUS', max_it=10)
        self.current_pdf = self.set_state_from_samples(samplesX[-1])
        self.moment_history.append(parameter_values[-1]) 
        self.pf_history.append(self.reliability_model.probability_of_failure_moment(parameter_values[-1]))
        self.beta_history.append(self.reliability_model.calculate_reliability_index(self.pf_history[-1]))

    def update_state_for_new_soil_sample(self, measured_soil_sample: List[float], measured_sigma: List[float]):
        """
        Update the state with new soil sample data
        """
        soil_names = self.soil_properties['parameter_names']
        # self.received_soil_samples.append((measured_soil_sample[i_soil_type], measured_sigma[i_soil_type]))
        soil_property_distributions = []
        soil_property_means = []
        soil_property_stds = []
        for i_soil_type, soil_type in enumerate(soil_names):
            self.l_soil_property.set_measured_mean_and_sigma(np.array([measured_soil_sample[i_soil_type]]), np.array([measured_sigma[i_soil_type]]))
            # Initialize with surrogate model
            cur_parameter_distribution = self.soil_properties['distributions'][i_soil_type]
            h, samplesU, samplesX, logcE, c, sigma, parameter_values = self.get_new_state_for_soil_sample(N=self.n_samples, p0=0.1, approach='aBUS', max_it=10, parameter_distribution=cur_parameter_distribution)
            soil_property_distributions.append(ERADist('normal', 'DATA', [samplesX[-1][:,0]]))
            soil_property_means.append(np.mean(samplesX[-1][:,0]))
            soil_property_stds.append(np.std(samplesX[-1][:,0]))
        self.soil_properties['distributions'] = soil_property_distributions
        self.soil_properties['mean'] = soil_property_means
        self.soil_properties['std'] = soil_property_stds       

        new_joint_distribution = ERANataf(soil_property_distributions + self.wall_properties['distributions'], self.moment_dict['R'])
        self.moment_dict['joint_distribution'] = new_joint_distribution 
        


    def _get_predicted_corrosion_rate(self, timeframe: np.ndarray, 
                                      corrosion_parameter_mean: float = 0.3226, corrosion_parameter_std: float = 0.1, 
                                      start_capacity: float = 40, start_ei: float = 45_000):
        """
        Get the predicted corrosion rate with time-dependent uncertainty.
        
        The uncertainty increases over time to reflect growing prediction uncertainty
        for longer time horizons.
        
        Parameters:
        -----------
        timeframe : np.ndarray
            Time points for prediction
        a, b : float
            Corrosion model parameters (Jongbloed curves)
        base_sigma : float
            Base uncertainty (coefficient of variation) at t=0
        sigma_growth_rate : float
            Rate at which uncertainty grows per year (default: 1% per year)
        """
        # Generate samples from the initial pdf
        parameter_a_distribution = ERADist('normal', 'PAR', [corrosion_parameter_mean, corrosion_parameter_std])
        samples_a = parameter_a_distribution.random(self.n_samples)
        capacity_history = {}
        ei_history = {}
        corrosion_rate_dict = {}
        for cur_t in timeframe:
            cur_corrosion_rate = self.corrosion_model.get_corrosion_rate_at_t(cur_t=cur_t, samples_a=samples_a)
            corrosion_rate_dict[cur_t] = {"samples": cur_corrosion_rate,
                                      "mean": np.mean(cur_corrosion_rate),
                                      "std": np.std(cur_corrosion_rate)}
            capacity_history[cur_t] = {"samples": start_capacity * (1-cur_corrosion_rate),
                                       "mean": np.mean(start_capacity * (1-cur_corrosion_rate)),
                                       "std": np.std(start_capacity * (1-cur_corrosion_rate))}
            ei_history[cur_t] = {"samples": start_ei * (1-cur_corrosion_rate),
                                 "mean": np.mean(start_ei * (1-cur_corrosion_rate)),
                                 "std": np.std(start_ei * (1-cur_corrosion_rate))}

        corrosion_dict ={'parameter_distributions': [parameter_a_distribution],
                        'parameter_names': ['parameter_a'],
                        'parameter_state_history': {"0": {"parameter_distributions": [parameter_a_distribution],
                                                "mean": [corrosion_parameter_mean],
                                                "std": [corrosion_parameter_std],
                                                "parameter_samples": [samples_a]}},
                        'corrosion_rate_history': corrosion_rate_dict,
                        }
        return corrosion_dict, capacity_history, ei_history

    
if __name__ == "__main__":
    dt = DigitalTwin(model_path='main/case_study_2025/train/results/srg/gpr/lr_1.0e-02_epochs_1000_rank_1/model_params.pkl', model_type='gpr', normal_distribution=False)
    dt.generate_samples(n_samples=1_000_000)
    # pass
