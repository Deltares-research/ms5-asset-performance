from typing import Optional, Callable, List
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm


import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

import sys
import os
import json
from pathlib import Path
# # # Add the project root directory to the path
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

# Import required modules
from src.bayesian_updating.ERADist import ERADist
from src.bayesian_updating.ERANataf import ERANataf
from src.bayesian_updating.BUS_SuS import BUS_SuS
from src.bayesian_updating.aBUS_SuS import aBUS_SuS
from src.bayesian_updating.iTMCMC import iTMCMC
from datetime import datetime
import pandas as pd
from src.bayesian_updating.likelihood_functions import DisplacementLikelihood
import typer

class PosteriorRetainingStructure:
    
    def __init__(self, model_path: str, model_type: str = "gpr", normal_distribution: bool = False):
        """
        Initialize the PosteriorRetainingStructure class

        Args:
            model_path (str): Path to the DSheetPiling model file
            measurement_path (str): Path to the measurement data file
            use_surrogate (bool): Whether to use the fake surrogate model instead of DSheetPiling
        """
        self.model_type = model_type
        self.define_parameters(normal_distribution=normal_distribution)
        # self.load_synthetic_data(model_path)
        

    def load_synthetic_data(self, model_path: str):
        """
        Load the synthetic data
        """ 
        # Initialize the displacement likelihood function
        self.l_displacement = DisplacementLikelihood(
            model_path=model_path,
            model_type=self.model_type,
        )
        self.l_displacement.generate_synthetic_measurement(np.array([self.synthetic_values]), sigma=0.1)

    def generate_samples(self, n_samples: int = 1000, normal_distribution: bool = False):
        """
        Generate samples from the posterior distribution
        """
        samples = self.prior_pdf.random(n_samples)
        # remove duplicate rows
        samples = np.unique(samples, axis=0)
        
        sample_dict = {"n_samples": n_samples,
                       "samples": {}}
        for i_par, parameter_name in enumerate(self.parameter_names):
            sample_dict["samples"][parameter_name] = samples[:, i_par]
            if 'soilcurkb1' in parameter_name:
                sample_dict["samples"][parameter_name.replace('soilcurkb1', 'soilcurkb2')] = samples[:, i_par] * 0.5
                sample_dict["samples"][parameter_name.replace('soilcurkb1', 'soilcurkb3')] = samples[:, i_par] * 0.25
                sample_dict["samples"][parameter_name.replace('soilcurkb1', 'soilcurko1')] = samples[:, i_par]
                sample_dict["samples"][parameter_name.replace('soilcurkb1', 'soilcurko2')] = samples[:, i_par] * 0.5
                sample_dict["samples"][parameter_name.replace('soilcurkb1', 'soilcurko3')] = samples[:, i_par] * 0.25  
     
        # print(f"sample_dict: {sample_dict['samples'].keys()}")
        param_names = list(sample_dict['samples'].keys())
        # Create a single DataFrame with all parameters
        print("Creating dataframe with all parameters...")
        df = pd.DataFrame({param: sample_dict["samples"][param] for param in param_names})
        # print(f"df: {df}")
        print("Saving dataframe to csv file...")
        file_name = f'1M_parameter_samples_normally_distributed.csv' if normal_distribution else '1M_parameter_samples_uniformly_distributed.csv'
        df.to_csv(file_name, index=False)
        print("Done!")

        # save sample_dict to json file
        # with open('sample_dict.json', 'w') as f:
            # json.dump(sample_dict, f)

        return sample_dict
        
    
    def define_parameters(self, normal_distribution: bool = False):
        # load parameters from json file
        with open('examples/deterministic_parameters.json', 'r') as f:
            parameters = json.load(f)

        material_properties = parameters['material_properties']
        # sheet_piling_walls = parameters['sheet_piling_walls']
        # anchors = parameters['anchors']
        # water_levels = parameters['water_levels']

        distributions = []
        parameter_names = []
        self.synthetic_values = []
        scale = 1.3
        lower_bound = 0.3
        upper_bound = 1.8
        std_scale = 0.1
        for material_name, material_values in material_properties.items():
            if material_name == 'klei siltig' or material_name == 'Zand los (18)':
                continue
            if material_values['cohesion'] != 0:
                if normal_distribution:
                    cohesion = ERADist('normal', 'PAR', [material_values['cohesion'], material_values['cohesion']*std_scale])
                else:
                    cohesion = ERADist('uniform', 'PAR', [material_values['cohesion']*lower_bound, material_values['cohesion']*upper_bound])
                distributions.append(cohesion)
                parameter_names.append(f'{material_name}_soilcohesion')
                self.synthetic_values.append(material_values['cohesion']*scale)
            if material_values['phi'] != 0:
                if normal_distribution:
                    phi = ERADist('normal', 'PAR', [material_values['phi'], material_values['phi']*std_scale])
                else:
                    phi = ERADist('uniform', 'PAR', [material_values['phi']*lower_bound, material_values['phi']*upper_bound])
                distributions.append(phi)
                parameter_names.append(f'{material_name}_soilphi')
                self.synthetic_values.append(material_values['phi']*scale)
            if material_values['k_top']['kh1'] != 0:
                if normal_distribution:
                    kh1 = ERADist('normal', 'PAR', [material_values['k_top']['kh1'], material_values['k_top']['kh1']*std_scale])
                else:
                    kh1 = ERADist('uniform', 'PAR', [material_values['k_top']['kh1']*lower_bound, material_values['k_top']['kh1']*upper_bound])
                distributions.append(kh1)
                parameter_names.append(f'{material_name}_soilcurkb1') # b van boven?
                self.synthetic_values.append(material_values['k_top']['kh1']*scale)

        if normal_distribution:
            wall_eis = ERADist('normal', 'PAR', [40000, 4500])
        else:
            wall_eis = ERADist('uniform', 'PAR', [5000, 50000])
        distributions.append(wall_eis)
        parameter_names.append(f'Wall_SheetPilingElementEI')
        self.synthetic_values.append(45000)

        # # add water level
        if normal_distribution:
            mean_water_level = -0.8
            std_water_level = 0.2
            water_level = ERADist('normal', 'PAR', [mean_water_level, std_water_level])
        else:
            lower_water_level = -4.4
            upper_water_level = 0.4
            water_level = ERADist('uniform', 'PAR', [lower_water_level, upper_water_level])
        distributions.append(water_level)
        parameter_names.append('water_lvl')
        self.synthetic_values.append(mean_water_level*scale)

        self.dist_parameters = distributions
        self.parameter_names = parameter_names

        self.dimensions = len(self.dist_parameters)
        self.R = np.eye(self.dimensions)
        self.prior_pdf = ERANataf(self.dist_parameters, self.R)
        self.synthetic_values = self.prior_pdf.random(1)
    # Removed the old likelihood methods since they're now in the DisplacementLikelihood class
    
    def find_c(self, method: int = 2, l_class: DisplacementLikelihood = None):
        """
        Find the constant c
        """
        realmin = np.finfo(np.double).tiny
        # use MLE to find c
        if method == 1:
            u_start = np.log(abs(l_class.measured_mean) + realmin)
            fun     = lambda lnU: -l_class.compute_log_likelihood_for_equality_information(np.exp(lnU))
            MLE_ln  = sp.optimize.fmin(func=fun, x0=u_start)
            MLE     = np.exp(MLE_ln)   # likelihood(MLE) = 1
            if l_class.measured_mean[0] < 0:
                MLE = -MLE
            c       = 1/l_class.compute_likelihood_for_equality_information(MLE)
            return c if not np.isnan(c) else 100
        # some likelihood evaluations to find c
        elif method == 2:
            # raise NotImplementedError('This method is not implemented yet')
            K  = int(1e2)                       # number of samples      
            u  = np.random.normal(size=(self.dimensions,K))   # samples in standard space
            x  = self.prior_pdf.U2X(u.T)               # samples in physical space
            # likelihood function evaluation
            L_eval = np.zeros((K,1))
            L_eval = self.l_displacement.compute_likelihood_for_parameters(x)
            c = 1/np.max(L_eval)    # Ref. 1 Eq. 34
            
        # use approximation to find c
        elif method == 3:
            raise NotImplementedError('This method is not implemented yet')
            # print('This method requires a large number of measurements')
            # m = len(f_tilde)
            # p = 0.05
            # c = 1/np.exp(-0.5*sp.stats.chi2.ppf(p,df=m))     # Ref. 1 Eq. 38
            
        else:
            raise RuntimeError('Finding the scale constant c requires -method- 1, 2 or 3')

    def update_for_new_displacement_data(self, list_of_N: List[int] = [100, 200, 500, 1000, 2000], p0: List[float] = [0.1, 0.2, 0.25],
                                         approaches: List[str] = ['BUS', 'aBUS', 'iTMCMC'], max_it: int = 20,
                                         results_dir: str = None):
        """
        Update the posterior for new data
        """        
        # test_n = [100, 200, 500, 1000, 2000]
        self.sample_results = {}
        for cur_p0 in p0:
            # self.sample_results[f"p0{cur_p0}"] = {}
            sample_results_n = {}  
            for cur_n in list_of_N:
                sample_results_n[f"n{cur_n}"] = {}

                valid_approaches = False
                if 'BUS' in approaches:
                    # find c
                    valid_approaches = True
                    c = self.find_c(method=1, l_class=self.l_displacement)
                    
                    h, samplesU, samplesX, logcE, sigma, displacement_samples = BUS_SuS(cur_n, cur_p0, c, self.l_displacement, distr = self.prior_pdf, max_it=max_it)
                    sample_results_n[f"n{cur_n}"]["BUS"] = {
                        'h': h,
                        'samplesU': samplesU,
                        'samplesX': samplesX,
                        'logcE': logcE,
                        'sigma': sigma,
                        'displacement_samples': displacement_samples, #-1000,
                        'n': cur_n,
                        'p0': cur_p0,
                        'max_it': max_it
                    }
                
                if 'aBUS' in approaches:
                    valid_approaches = True
                    h, samplesU, samplesX, logcE, c, sigma, displacement_samples = aBUS_SuS(cur_n, cur_p0, self.l_displacement, self.prior_pdf, max_it=max_it)
                    sample_results_n[f"n{cur_n}"]["aBUS"] = {
                        'h': h,
                        'samplesU': samplesU,
                        'samplesX': samplesX,
                        'logcE': logcE,
                        'sigma': sigma,
                        'displacement_samples': displacement_samples,
                        'n': cur_n,
                        'p0': cur_p0,
                        'max_it': max_it
                    }

                if 'iTMCMC' in approaches:
                    valid_approaches = True
                    samplesU, samplesX, q, logcE, displacement_samples = iTMCMC(cur_n, int(0.1*cur_n), self.l_displacement, self.prior_pdf)
                    sample_results_n[f"n{cur_n}"]["iTMCMC"] = {
                        'samplesU': samplesU,
                        'samplesX': samplesX,
                        'q': q,
                        'logcE': logcE,
                        'displacement_samples': displacement_samples,
                        'n': cur_n,
                        'p0': cur_p0,
                        'max_it': max_it
                    }

                if not valid_approaches:
                    raise ValueError(f'No valid approaches found. Currently implemented approaches are: BUS, aBUS, iTMCMC')

            self.sample_results[f"p0{cur_p0}"] = sample_results_n
        # Store results as class attributes to make them available for plotting

        # rescale displacment samples
        # self.displacement_samples = self.displacement_samples - 1000
        # Create the results directory if it doesn't exist
        # Save the results to a JSON file
        # Create runID based on the current date and time
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")   
        result_path = f"{results_dir}/results_{'_'.join(approaches)}_p0{p0}_maxit{max_it}_runID{run_id}.npz"
        np.savez(result_path, **self.sample_results)
        # Print summary statistics
        print('\nModel evidence =', np.exp(logcE), '\n')
        for i in range(self.dimensions):
            print(f'Mean value of {self.parameter_names[i]} =', np.mean(samplesX[-1][:, i]))
            print(f'Std of {self.parameter_names[i]} =', np.std(samplesX[-1][:, i]), '\n')

    def plot_prior_and_posterior_marginal_pdfs(self, plot_dir: str = None):
        """
        Plot the prior and posterior marginal pdfs
        """
        
        # Options for font-family and font-size
        plt.rc('font', size=12)
        plt.rc('axes', titlesize=16)  # fontsize of the axes title
        plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
        plt.rc('figure', titlesize=18)  # fontsize of the figure title
        
        # Get prior samples
        prior_samples = self.prior_pdf.random(1000) 
        # Get posterior samples if available

        list_of_p0 = self.sample_results.keys()
        # list_of_N = self.sample_results.keys()
        for i_p0, cur_p0 in enumerate(list_of_p0): 
            p0_sample_results = self.sample_results[cur_p0]
            list_of_N = p0_sample_results.keys()
            for i_N, cur_N in enumerate(list_of_N):
                plt.figure(figsize=(20, 4*self.dimensions))
                # Get keys from self.sample_results and format them nicely
                keys = p0_sample_results[cur_N].keys()
                approaches = ", ".join(keys)

                plt.suptitle(f'Comparison of {approaches} approaches for p0 = {cur_p0.split("p0")[1]}\nSample Size N = {cur_N.split("n")[1]}', 
                            fontsize=18, fontweight='bold')
                nr_keys = len(keys)
                for i_key, key in enumerate(keys):
                    curN = p0_sample_results[cur_N][key]["n"]
                    nr_bins = 50 if int(curN) > 499 else 20
                    posterior_samples = p0_sample_results[cur_N][key]["samplesX"][-1]
                    displacement_samples = p0_sample_results[cur_N][key]["displacement_samples"]
                    
                    param_ranges = []
                    for i in range(self.dimensions):
                        # Determine range for each parameter
                        param_min = min(np.min(prior_samples[:, i]), np.min(posterior_samples[:, i]))
                        param_max = max(np.max(prior_samples[:, i]), np.max(posterior_samples[:, i]))
                        # Add some padding
                        padding = 0.1 * (param_max - param_min)
                        param_ranges.append([param_min - padding, param_max + padding])
                    
                    # Plot displacement samples
                    plt.subplot(self.dimensions+1, nr_keys, i_key+1)
                    
                    # Add measured displacement lines
                    for i, measured_displacement in enumerate(self.l_displacement.measured_mean):
                        # measured_displacement = measured_displacement - 1000
                        plt.axvline(measured_displacement, color='orange', linestyle='--', 
                                   label=f'Measured value: {measured_displacement:.2f}' if i == 0 else "_nolegend_")
                        

                    # For the prior (first level)
                    prior_displacements = displacement_samples[0]
                    posterior_displacements = displacement_samples[-1]
                    kde_prior = gaussian_kde(prior_displacements)
                    min_displacement = min(np.min(prior_displacements), np.min(posterior_displacements))
                    max_displacement = max(np.max(prior_displacements), np.max(posterior_displacements))
                    x_range = np.linspace(min_displacement, max_displacement, 200)
                    prior_density = kde_prior(x_range)
                    plt.plot(x_range, prior_density, 'b-', linewidth=2, label='Prior')
                    plt.fill_between(x_range, prior_density, alpha=0.1, color='blue')
                    # plt.hist(prior_displacements, density=True, bins=nr_bins, 
                    #         alpha=0.5, color='blue', label='Prior')
                    
                    # For the posterior (last level)
                    try:
                        kde_posterior = gaussian_kde(posterior_displacements)
                        posterior_density = kde_posterior(x_range)
                        plt.plot(x_range, posterior_density, 'r-', linewidth=2, label='Posterior')
                        plt.fill_between(x_range, posterior_density, alpha=0.3, color='red')
                    except:
                        plt.hist(posterior_displacements, density=True, bins=nr_bins,
                            alpha=0.5, color='red', label='Posterior')

                        # Plot likelihood as normal distribution for each measurement
                        plt.fill_between(x_range, prior_density, alpha=0.2, color='darkgoldenrod')
                
                                
                    plt.title(f'Marginal PDF of the displacement for {key}')
                    plt.xlabel('Displacement [mm]')
                    plt.ylabel('Density')
                    plt.grid(True, alpha=0.3)
                    plt.legend()

                    # true_parameters = [self.true_cohesion, self.true_phi, self.true_water_level]
                    # Plot each parameter's marginal distribution
                    for i in range(self.dimensions):
                        plt.subplot(self.dimensions+1, nr_keys, nr_keys*(i+1)+1+i_key)
                        
                        # Prior KDE
                        try:
                            kde_prior = gaussian_kde(prior_samples[:, i])
                            x_range = np.linspace(param_ranges[i][0], param_ranges[i][1], 200)
                            prior_density = kde_prior(x_range)
                            plt.plot(x_range, prior_density, 'b-', linewidth=2, label='Prior')
                            plt.fill_between(x_range, prior_density, alpha=0.1, color='blue')
                        except:
                            plt.hist(prior_samples[:, i], density=True, alpha=0.5, color='blue', label='Prior', bins=nr_bins)

                        # plt.hist(prior_samples[:, i], density=True, alpha=0.5, color='blue', label='Prior', bins=nr_bins)
                        
                        # Posterior KDE
                        try:
                            kde_post = gaussian_kde(posterior_samples[:, i])
                            post_density = kde_post(x_range)
                            plt.plot(x_range, post_density, 'r-', linewidth=2, label='Posterior')
                            plt.fill_between(x_range, post_density, alpha=0.3, color='red')
                        except:
                            plt.hist(posterior_samples[:, i], density=True, alpha=0.5, color='red', label='Posterior', bins=nr_bins)

                        # plt.hist(posterior_samples[:, i], density=True, alpha=0.5, color='red', label='Posterior', bins=nr_bins)

                        # Add mean lines
                        plt.axvline(self.synthetic_values[i], color='blue', linestyle='--', label=f'True {self.parameter_names[i]}')
                        # plt.hist(posterior_samples[:, i], density=True, alpha=0.5, color='red', label='Posterior')
                        # Add mean lines
                        # plt.axvline(np.mean(prior_samples[:, i]), color='blue', linestyle='--', 
                                #    label=f'Prior mean: {np.mean(prior_samples[:, i]):.2f}')
                        # plt.axvline(np.mean(posterior_samples[:, i]), color='red', linestyle='--', 
                                #    label=f'Posterior mean: {np.mean(posterior_samples[:, i]):.2f}')
                        
                        plt.title(f'Marginal PDF of {self.parameter_names[i]}')
                        plt.xlabel(self.parameter_names[i])
                        plt.ylabel('Density')
                        plt.grid(True, alpha=0.3)
                        plt.legend()


                plt.tight_layout()
                plt.savefig(f'{plot_dir}/marginal_pdfs_for_{cur_p0}_N{cur_N}.png', dpi=300)
        # plt.show()  

    def plot_threshold_progression(self, plot_dir: str = None):
        """
        Plot the posterior samples
        """
        
        # Options for font-family and font-size
        plt.rc('font', size=12)
        plt.rc('axes', titlesize=16)  # fontsize of the axes title
        plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
        plt.rc('figure', titlesize=18)  # fontsize of the figure title
        
        # Plot threshold progression
        plt.figure(figsize=(10, 6))
        list_of_p0 = list(self.sample_results.keys())
        list_of_N = list(self.sample_results[list_of_p0[0]].keys())
        # generate a list of N colors
        colors = plt.cm.inferno(np.linspace(0.2, 0.8, len(list_of_N)))
        list_of_N = list(list_of_N)  # Convert dict_keys to a list so it's subscriptable
        methods = self.sample_results[list_of_p0[0]][list_of_N[0]].keys()
        methods = [method for method in methods if method != "iTMCMC"]
        for i_p0, cur_p0 in enumerate(list_of_p0):
            for i_method, method in enumerate(methods):
                plt.subplot(len(list_of_p0), len(methods), i_p0*len(methods)+i_method+1)
                plt.grid(True)    
                plt.tight_layout()
                for i_N, cur_N in enumerate(list_of_N):
                    h = self.sample_results[cur_p0][cur_N][method]["h"]
                    plt.plot(range(len(h)), h, 'bo-', linewidth=2, label=f'N = {cur_N.split("n")[1]}', color=colors[i_N])
            
                plt.legend()
                plt.axhline(y=0, color='r', linestyle='--', label='Stopping Condition')
                plt.xlabel('Subset level')
                plt.ylabel('Threshold value (h)')
                plt.title(f'Progression of Intermediate Thresholds\np0 = {cur_p0.split("p0")[1]} and {method}')
        plt.savefig(f'{plot_dir}/threshold_progression.png', dpi=300)        
        # plt.show()



app = typer.Typer()
@app.command()
def run_bus_approach(modeltype: str = "torch", 
                     modelname: str = "lr_1.0e-04_epochs_10000"):
    """
    Run the BUS approach
    """
     # Run for gpr
    if modeltype == "gpr":
        from src.geotechnical_models.gpr.gpr_classes import DependentGPRModels, MultitaskGPModel
        model_path = "main/case_study_2025/train/results/srg/gpr/" + modelname + "/model_params.pkl"
    # Run for torch
    elif modeltype == "torch":
        from src.geotechnical_models.mlp.MLP_class import MLP
        model_path = "main/case_study_2025/train/results/srg/torch/" + modelname

    # Run for dsheet
    elif modeltype == "dsheet":
        model_path = "c:\\Users\\cotoarba\\OneDrive - Stichting Deltares\\Desktop\\dummy_123.shi"
    else:
        raise ValueError(f"Invalid model type: {modeltype}. Not implemented yet.")
    
    # Verify that the model file exists
    if modeltype != "dsheet" and not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Using model path: {model_path}")
    
    # Get measurement data path relative to the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    measurement_path = os.path.join(current_dir, 'data/synthetic_measurement_data.json')
    
    # Initialize with surrogate model
    posterior_retention_structure = PosteriorRetainingStructure(
        model_path, model_type=modeltype, normal_distribution=True
    )

    # # # ############################################################
    # Generate samples for surrogate model
    # posterior_retention_structure.generate_samples(n_samples=1_000_000, normal_distribution=True)
    
    ############################################################
    # CONTOUR PLOT OF THE GRADIENT
    # Run updating
    results_dir = current_dir + '/results/' + modeltype + '/' + modelname
    os.makedirs(results_dir, exist_ok=True)
    posterior_retention_structure.update_for_new_displacement_data(list_of_N=[200], p0=[0.1], approaches=['BUS', 'aBUS'], max_it=10, results_dir=results_dir)
    # make plot director
    # plot_dir = results_dir + '/plots/'
    # os.makedirs(plot_dir, exist_ok=True)

    # posterior_retention_structure.l_displacement.measured_displacement_mean = posterior_retention_structure.measured_displacement_mean - 1000
    posterior_retention_structure.plot_prior_and_posterior_marginal_pdfs(plot_dir=results_dir)
    posterior_retention_structure.plot_threshold_progression(plot_dir=results_dir)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    app()

   