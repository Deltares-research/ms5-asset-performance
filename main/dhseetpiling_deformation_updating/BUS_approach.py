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
# Add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

# Import required modules
from src.bayesian_updating.ERADist import ERADist
from src.bayesian_updating.ERANataf import ERANataf
from src.bayesian_updating.BUS_SuS import BUS_SuS
from src.bayesian_updating.aBUS_SuS import aBUS_SuS
from datetime import datetime
import pandas as pd
from main.dhseetpiling_deformation_updating.likelihood_functions import DisplacementLikelihood


class PosteriorRetainingStructure:
    
    def __init__(self, model_path: str, use_surrogate: bool = True):
        """
        Initialize the PosteriorRetainingStructure class

        Args:
            model_path (str): Path to the DSheetPiling model file
            measurement_path (str): Path to the measurement data file
            use_surrogate (bool): Whether to use the fake surrogate model instead of DSheetPiling
        """
        self.use_surrogate = use_surrogate
        
        self.define_parameters()
        self.load_synthetic_data(model_path)
        

    def load_synthetic_data(self, model_path: str):
        """
        Load the synthetic data
        """
        # if measurement_path.endswith('.json'):
        #     self.synthetic_data = pd.read_json(measurement_path)
        # elif measurement_path.endswith('.csv'):
        #     self.synthetic_data = pd.read_csv(measurement_path)
        # else:
        #     raise ValueError('Invalid file extension')
        # self.measured_displacement_mean = self.synthetic_data['displacement'].values
        # self.measured_displacement_sigma = self.synthetic_data['sigma'].values
        self.true_cohesion = 10
        self.true_phi = 25
        self.true_water_level = -0.7
        parameters = [[self.true_cohesion, self.true_phi, self.true_water_level]]
        
        # Initialize the displacement likelihood function
        self.l_displacement = DisplacementLikelihood(
            model_path,
            use_surrogate=self.use_surrogate
        )
        self.l_displacement.generate_synthetic_measurement(parameters)

    def define_parameters(self):
        # soil_layers = {"Klei": ("soilphi", "soilcohesion")}
        # rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["Klei_soilphi", "Klei_soilcohesion"])
        # rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.08]), names=["water_A"])
        # state = GaussianState(rvs=[rv_strength, rv_water])

        # soil_cohesion = ERADist('normal', 'PAR', [10, 1])
        soil_cohesion = ERADist('uniform', 'PAR', [0, 20])
        # soil_phi = ERADist('normal', 'PAR', [30, 3])
        soil_phi = ERADist('uniform', 'PAR', [10, 40])
        # water_level = ERADist('normal', 'PAR', [-0.8, 0.08])
        water_level = ERADist('uniform', 'PAR', [-1.2, -0.5])
            
        self.dist_parameters = [soil_cohesion, soil_phi, water_level]
        self.parameter_names = ['Klei_soilcohesion', 'Klei_soilphi', 'water_A']

        self.dimensions = len(self.dist_parameters)
        self.R = np.eye(self.dimensions)
        self.prior_pdf = ERANataf(self.dist_parameters, self.R)

    # Removed the old likelihood methods since they're now in the DisplacementLikelihood class
    
    # def fake_surrogate_function(self, parameters: list[float]):
    #     """
    #     Fake surrogate function for given parameters
    #     parameters: list of parameters
    #     parameters[0]: soil cohesion
    #     parameters[1]: soil phi (friction angle)
    #     parameters[2]: water level
    #     Optional: parameters[3]: corrosion
        
    #     # Base displacement value dependent on key parameters
    #     # Realistic effects:
    #     # - Higher cohesion reduces displacement
    #     # - Higher friction angle reduces displacement
    #     # - Higher water level (less negative) increases displacement
    #     # - More corrosion increases displacement
    #     Returns: 
    #         displacement: estimated displacement in cm
    #     """        
    #     # # Add some noise to simulate model error
    #     # noise = np.random.normal(0, 0.1)
    #     a = 12
    #     b = 0.6
    #     c = 0.4
    #     d = 0.2
    #     displacement = a + b * parameters[0] + c * parameters[1] + d * parameters[2]
    #     return displacement # + noise

    
    # def get_displacement_from_dsheet_model(self, updated_parameters: list[float], stage_id: int = -1):
    #     """
    #     Run the Dsheet analysis for given parameters or use the surrogate model
    #     updated_parameters: list of parameters with
    #     updated_parameters[0]: soil cohesion
    #     updated_parameters[1]: soil phi
    #     updated_parameters[2]: water level
    #     Optional: updated_parameters[3]: corrosion
    #     """
    #     # If surrogate mode is enabled, use the fake surrogate function
    #     if hasattr(self, 'use_surrogate') and self.use_surrogate:
    #         return self.fake_surrogate_function(updated_parameters)
        
    #     from src.reliability_models.dsheetpiling.lsf import unpack_soil_params, unpack_water_params
    #     # Otherwise use the actual DSheetPiling model
    #     # Pair parameters with names
    #     params = {name: rv for (name, rv) in zip(self.parameter_names, updated_parameters)}
        
    #     soil_data = unpack_soil_params(params, list(self.model.soils.keys()))
    #     water_data = unpack_water_params(params, [lvl.name for lvl in self.model.water.water_lvls])
        
    #     self.model.update_soils(soil_data)
    #     self.model.update_water(water_data)
    #     self.model.execute(i_run=0)
        
    #     #TODO: Get the displacement
    #     results = self.model.results

    #     # return max displacement of the last stage
    #     return results.max_displacement[stage_id]


    def find_c(self, method: int = 1, l_class: DisplacementLikelihood = None):
        """
        Find the constant c
        """
        realmin = np.finfo(np.double).tiny
        # use MLE to find c
        if method == 1:    
            u_start = np.log(l_class.measured_mean + realmin)
            fun     = lambda lnU: -l_class.compute_log_likelihood(np.exp(lnU)) 
            MLE_ln  = sp.optimize.fmin(func=fun, x0=u_start)
            MLE     = np.exp(MLE_ln)   # likelihood(MLE) = 1
            c       = 1/l_class.compute_likelihood_for_equality_information(MLE)
            return c
        # some likelihood evaluations to find c
        elif method == 2:
            raise NotImplementedError('This method is not implemented yet')
            # K  = int(5e3)                       # number of samples      
            # u  = np.random.normal(size=(n,K))   # samples in standard space
            # x  = prior_pdf.U2X(u)               # samples in physical space
            # # likelihood function evaluation
            # L_eval = np.zeros((K,1))
            # for i in range(K):
            #     L_eval[i] = likelihood(x[:,i])
            # c = 1/np.max(L_eval)    # Ref. 1 Eq. 34
            
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
                                         approach: str = 'BUS', max_it: int = 20):
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
                if approach == 'BUS' or approach == "both":
                    # find c
                    c = self.find_c(method=1, l_class=self.l_displacement)
                    h, samplesU, samplesX, logcE, sigma, displacement_samples = BUS_SuS(cur_n, cur_p0, c, self.l_displacement, distr = self.prior_pdf)
                    sample_results_n[f"n{cur_n}"]["BUS"] = {
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
                
                if approach == 'aBUS' or approach == "both":
                    h, samplesU, samplesX, logcE, c, sigma, displacement_samples = aBUS_SuS(cur_n, cur_p0, self.l_displacement, self.prior_pdf)
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
                if approach != 'BUS' and approach != 'aBUS' and approach != 'both':
                    raise ValueError('Invalid approach')
            self.sample_results[f"p0{cur_p0}"] = sample_results_n
        # Store results as class attributes to make them available for plotting

        # Create the results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        # Save the results to a JSON file
        # Create runID based on the current date and time
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")   
        result_path = f"results/results_{approach}_p0{p0}_maxit{max_it}_runID{run_id}.npz"
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
                plt.figure(figsize=(12, 10))
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
                    kde_posterior = gaussian_kde(posterior_displacements)
                    posterior_density = kde_posterior(x_range)
                    plt.plot(x_range, posterior_density, 'r-', linewidth=2, label='Posterior')
                    plt.fill_between(x_range, posterior_density, alpha=0.3, color='red')
                    # plt.hist(posterior_displacements, density=True, bins=nr_bins,
                    #         alpha=0.5, color='red', label='Posterior')

                                        # Plot likelihood as normal distribution for each measurement
                    for i, (measured_displacement, sigma) in enumerate(zip(self.l_displacement.measured_mean, self.l_displacement.measured_sigma)):
                        # Create x range around measurement (±4 sigma to capture full distribution)
                        x_likelihood = np.linspace(measured_displacement - 4*sigma, measured_displacement + 4*sigma, 200)
                        # Calculate normal distribution values
                        norm_pdf = norm.pdf(x_likelihood, loc=measured_displacement, scale=sigma)
                        # Scale to make it visible on the plot
                        scale_factor = 0.6 * max(prior_density.max(), posterior_density.max()) / norm_pdf.max()
                        scaled_pdf = norm_pdf * scale_factor
                        # Plot the normal distribution
                        plt.plot(x_likelihood, scaled_pdf, color='darkgoldenrod', linestyle='-', linewidth=2,
                                label=f'Likelihood N({measured_displacement:.2f}, {sigma:.2f})' if i == 0 else "_nolegend_")
                        plt.fill_between(x_likelihood, scaled_pdf, alpha=0.2, color='darkgoldenrod')
                
                                
                    plt.title(f'Marginal PDF of the displacement for {key}')
                    plt.xlabel('Displacement [cm]')
                    plt.ylabel('Density')
                    plt.grid(True, alpha=0.3)
                    plt.legend()

                    true_parameters = [self.true_cohesion, self.true_phi, self.true_water_level]
                    # Plot each parameter's marginal distribution
                    for i in range(self.dimensions):
                        plt.subplot(self.dimensions+1, nr_keys, nr_keys*(i+1)+1+i_key)
                        
                        # Prior KDE
                        kde_prior = gaussian_kde(prior_samples[:, i])
                        x_range = np.linspace(param_ranges[i][0], param_ranges[i][1], 200)
                        prior_density = kde_prior(x_range)
                        plt.plot(x_range, prior_density, 'b-', linewidth=2, label='Prior')
                        plt.fill_between(x_range, prior_density, alpha=0.1, color='blue')
                        # plt.hist(prior_samples[:, i], density=True, alpha=0.5, color='blue', label='Prior', bins=nr_bins)
                        
                        # Posterior KDE
                        kde_post = gaussian_kde(posterior_samples[:, i])
                        post_density = kde_post(x_range)
                        plt.plot(x_range, post_density, 'r-', linewidth=2, label='Posterior')
                        plt.fill_between(x_range, post_density, alpha=0.3, color='red')
                        # plt.hist(posterior_samples[:, i], density=True, alpha=0.5, color='red', label='Posterior', bins=nr_bins)

                        # Add mean lines
                        plt.axvline(true_parameters[i], color='blue', linestyle='--', label=f'True {self.parameter_names[i]}')
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

def run_bus_approach_comparison():
    """
    Run the BUS approach comparison
    """
    pass


if __name__ == '__main__':    
    # Use a dummy model path when using surrogate model
    model_path = "c:\\Users\\cotoarba\\OneDrive - Stichting Deltares\\Desktop\\dummy_123.shi"
    
    # Get measurement data path relative to the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    measurement_path = os.path.join(current_dir, 'synthetic_measurement_data.json')
    
    # Initialize with surrogate model
    posterior_retention_structure = PosteriorRetainingStructure(
        model_path, use_surrogate=True
    )
    
    posterior_retention_structure.define_parameters()
    posterior_retention_structure.update_for_new_displacement_data(list_of_N=[100, 1000], p0=[0.1, 0.2, 0.25], approach='both')
    # make plot directory
    os.makedirs('plots', exist_ok=True)
    posterior_retention_structure.plot_prior_and_posterior_marginal_pdfs(plot_dir='plots')
    posterior_retention_structure.plot_threshold_progression(plot_dir='plots')
    # plt.show()
    plt.close()
