from datetime import datetime
import pandas as pd
from typing import Annotated, List, Tuple, Dict, NamedTuple
import numpy as np
from pathlib import Path
import json
import sys
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
from src.bayesian_updating.likelihood_functions import DisplacementLikelihood
from main.dhseetpiling_deformation_updating.BUS_approach import PosteriorRetainingStructure
from main.case_study_prototype.DigitalTwin import DigitalTwin
from main.case_study_prototype.PhysicalTwin import PhysicalTwin
from main.case_study_prototype.PlotUtils import plot_state_history

def simple_case_study():
    """
    Simple case study
    """
    # General case study parameters
    t_start = 0
    t_step = 1 #year
    t_project = 20 #years
    timeframe = np.arange(t_start, t_project+1, t_step)
    water_measurements_are_available = False

    model_path = 'main/case_study_2025/train/results_moments/srg/gpr/lr_1.0e-02_epochs_1000_rank_4/model_params.pkl'
    # model_path = 'main/case_study_2025/train/results_moments/srg_moments_20250617_101504/gpr/lr_1.0e-02_epochs_100_rank_1/model_params.pkl'
    model_type = 'gpr'
    normal_distribution = True
    max_moment = 30

    # Initialize digital twin and physical twin
    print("Initializing digital twin...")
    pdt_prototype = DigitalTwin(model_path=model_path, model_type=model_type, normal_distribution=normal_distribution, max_moment=max_moment, n_samples=100, timeframe=timeframe)
    print("Initializing physical twin...")
    physical_twin = PhysicalTwin(model_path=model_path, model_type=model_type, t_start=t_start, t_end=t_project, t_step=t_step, max_moment=max_moment)
    print("Generating state history for physical twin...")
    # Generate state history for physical twin
    physical_twin.generate_state_history(timeframe=timeframe)

    plt_dir = 'main/case_study_prototype/plots'
    Path(plt_dir).mkdir(parents=True, exist_ok=True)
    run_id = 'run_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(plt_dir + '/' + run_id).mkdir(parents=True, exist_ok=True)
    # Loop over time
    measured_moments = {'value': [], 'sigma': [], 'time': []}
    measured_corrosions = {'value': [], 'sigma': [], 'time': []}
    measured_soil_samples = {'value': [], 'sigma': [], 'time': []}
    print("Updating state for digital twin...")

    # save true state history to csv
    df = pd.DataFrame(physical_twin.state_history, columns=physical_twin.parameter_names)
    df.to_csv(plt_dir + '/' + run_id + '/true_state_history.csv', index=False)


    # for t in range(t_start, t_project, t_step):
    for i in range(1,len(timeframe)):
        
        # Buffer for nicer plots
        if i == t_project-4:
            break
        
        posterior_moment_samples = None
        posterior_property_samples = None
        t = timeframe[i]
        print(50*'-=')
        print(f"Updating state for digital twin at time {t}...")

        ###### CORROSION UPDATE ######
        if t == 10:
            measured_corrosion_rate = physical_twin.get_corrosion_rate(i)
            print(f"Measured corrosion rate: {measured_corrosion_rate}")
            sigma = 0.1
            # update state due to corrosion measurement
            pdt_prototype.update_state_for_new_corrosion(measured_corrosion=measured_corrosion_rate, measured_sigma=sigma, time_index=i)
            measured_corrosions['value'].append(measured_corrosion_rate)
            measured_corrosions['sigma'].append(sigma)
            measured_corrosions['time'].append(i)

        ###### MOMENT UPDATE ######
        # generate synthetic measurement from physical twin and update state
        if t % 5 == 0:  
            cur_measured_moment = physical_twin.get_moment(i) - np.random.normal(0, 2)
            cur_measured_moment_sigma = 0.1
            print(f"Updating for measured moment: {cur_measured_moment}")
            posterior_property_samples, posterior_moment_samples = pdt_prototype.update_state_for_new_moment(measured_moment=cur_measured_moment, measured_sigma=cur_measured_moment_sigma)
            measured_moments['value'].append(cur_measured_moment)
            measured_moments['sigma'].append(cur_measured_moment_sigma)
            measured_moments['time'].append(i)

        pdt_prototype.update_state(cur_time_index=i, property_samples=posterior_property_samples, moment_samples=posterior_moment_samples)

                    
        if t == 8:
            measured_soil_sample = physical_twin.get_soil_sample() * 0.95
            measured_soil_sigma = np.repeat(0.1, len(measured_soil_sample))
            print(f"Updating for measured soil sample: {measured_soil_sample}")
            pdt_prototype.update_state_for_new_soil_sample(measured_soil_sample=measured_soil_sample, measured_sigma=measured_soil_sigma)
            measured_soil_samples['value'].append(measured_soil_sample)
            measured_soil_samples['sigma'].append(measured_soil_sigma)
            measured_soil_samples['time'].append(np.repeat(i, len(measured_soil_sample)))

        
        if t % 5 == 0:
            plot_state_history(i, physical_twin, pdt_prototype, 
                           timeframe, max_moment, 
                           measured_moments=measured_moments, 
                           measured_corrosions=measured_corrosions,
                           measured_soil_samples=measured_soil_samples,
                           plot_dir=plt_dir + '/' + run_id)

        # ###### SURVIVED MOMENT UPDATE ######
        # # generate synthetic measurement from physical twin and update state
        # # cur_measured_moment = 25
        # # sigma = 0.1
        # cur_measured_moment = physical_twin.get_moment(i) + np.random.normal(0, 1)
        # sigma = 0.1
        # print(f"Updating for measured moment: {cur_measured_moment}")
        # pdt_prototype.update_state_for_survived_load(survived_load=cur_measured_moment, survived_sigma=sigma)
        # measured_moments.append(cur_measured_moment)
        # measured_sigmas.append(sigma)
        # plot_state_history(t, physical_twin, pdt_prototype, timeframe, max_moment, measured_moments=measured_moments, measured_sigmas=measured_sigmas, plot_dir=plt_dir + '/' + run_id)





if __name__ == "__main__":
    simple_case_study()