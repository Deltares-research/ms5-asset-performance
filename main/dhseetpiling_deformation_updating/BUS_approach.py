from typing import Optional, Callable
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm

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
# from src.reliability_models.dsheetpiling.lsf import unpack_soil_params, unpack_water_params
# from src.bayesian_updating.aCS import aCS
# from src.geotechnical_models.dsheetpiling.model import DSheetPiling
import pandas as pd


class PosteriorRetainingStructure:
    
    def __init__(self, model_path: str, measurement_path: str, use_surrogate: bool = True):
        """
        Initialize the PosteriorRetainingStructure class

        Args:
            model_path (str): Path to the DSheetPiling model file
            measurement_path (str): Path to the measurement data file
            use_surrogate (bool): Whether to use the fake surrogate model instead of DSheetPiling
        """
        self.use_surrogate = use_surrogate
        
        if not use_surrogate:
            # Only import and initialize DSheetPiling if we're not using the surrogate
            from src.geotechnical_models.dsheetpiling.model import DSheetPiling
            self.model = DSheetPiling(model_path)
            result_path = r"results/results.json"
        
        self.load_synthetic_data(measurement_path)
        self.define_parameters()

    def load_synthetic_data(self, measurement_path: str):
        """
        Load the synthetic data
        """
        if measurement_path.endswith('.json'):
            self.synthetic_data = pd.read_json(measurement_path)
        elif measurement_path.endswith('.csv'):
            self.synthetic_data = pd.read_csv(measurement_path)
        else:
            raise ValueError('Invalid file extension')
        self.measured_displacement_mean = self.synthetic_data['displacement'].values
        self.measured_displacement_sigma = self.synthetic_data['sigma'].values

    def define_parameters(self):
        # soil_layers = {"Klei": ("soilphi", "soilcohesion")}
        # rv_strength = MvnRV(mus=np.array([30, 10]), stds=np.array([3, 1]), names=["Klei_soilphi", "Klei_soilcohesion"])
        # rv_water = MvnRV(mus=np.array([-0.8]), stds=np.array([0.08]), names=["water_A"])
        # state = GaussianState(rvs=[rv_strength, rv_water])

        soil_cohesion = ERADist('normal', 'PAR', [30, 3])
        soil_phi = ERADist('normal', 'PAR', [10, 1])

        water_level = ERADist('normal', 'PAR', [-0.8, 0.08])
            
        self.dist_parameters = [soil_cohesion, soil_phi, water_level]
        self.parameter_names = ['Klei_soilcohesion', 'Klei_soilphi', 'water_A']
        
        self.dimensions = len(self.dist_parameters)
        self.R = np.eye(self.dimensions)
        self.prior_pdf = ERANataf(self.dist_parameters, self.R)


    def likelihood_function_for_displacement(self, displacement_sample: float):
        """
        Likelihood function for displacement
        """
        n = self.measured_displacement_mean.shape[0]
        w = 1 / n * np.ones(n)  # uniform weights
        # return norm.pdf(displacement_sample, self.measured_displacement_mean, self.measured_displacement_sigma)    
        return float(sum(w[i]*norm.pdf(displacement_sample, self.measured_displacement_mean[i], self.measured_displacement_sigma) for i in range(n))[0])
    
    def log_likelihood_function_for_displacement(self, displacement_sample: float):
        """
        Log likelihood function for displacement
        """
        return np.log(self.likelihood_function_for_displacement(displacement_sample))
    
    def likelihood_function_for_parameters(self, parameters: list[float]):
        """
        Likelihood function for parameters
        """
        displacement = self.get_displacement_from_dsheet_model(parameters)
        return self.likelihood_function_for_displacement(displacement), displacement
    
    def log_likelihood_function_for_parameters(self, parameters: list[float]):
        """
        Log likelihood function for parameters
        """
        likelihood, displacement = self.likelihood_function_for_parameters(parameters.T)
        return np.log(likelihood), displacement

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
        # Unpack parameters
        cohesion = parameters[0]  # kPa
        phi = parameters[1]       # degrees
        water_level = parameters[2]  # m
        
        # Add corrosion factor if provided
        corrosion = 1.0
        if len(parameters) > 3:
            corrosion = parameters[3]

        # Base displacement value (cm)
        base_displacement = 5.0
        
        # Parameter influence factors
        cohesion_factor = 30.0 / (cohesion + 5.0)  # Inversely proportional to cohesion
        phi_factor = 35.0 / (phi + 2.0)            # Inversely proportional to friction angle
        water_factor = 1.5 * (water_level + 1.0)   # Directly proportional to water level
        
        # Calculate displacement with some non-linearity and parameter interaction
        displacement = base_displacement * cohesion_factor * phi_factor * water_factor * corrosion

        # Ensure displacement is positive
        # print(f"Displacement: {displacement}")
        displacement = max(0.1, displacement)
        
        # Add some noise to simulate model error
        noise = np.random.normal(0, 0.05 * displacement)
        displacement += noise
        
        return displacement

    
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


    def find_c(self, method: int = 1):
        """
        Find the constant c
        """
        realmin = np.finfo(np.double).tiny
        # use MLE to find c
        if method == 1:    
            u_start = np.log(self.measured_displacement_mean + realmin)
            fun     = lambda lnU: -self.log_likelihood_function_for_displacement(np.exp(lnU)) 
            MLE_ln  = sp.optimize.fmin(func=fun, x0=u_start)
            MLE     = np.exp(MLE_ln)   # likelihood(MLE) = 1
            c       = 1/self.likelihood_function_for_displacement(MLE)
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

    def update_for_new_displacement_data(self, N: int = 1000, p0: float = 0.1,
                                         approach: str = 'BUS'):
        """
        Update the posterior for new data
        """
        
        if approach == 'BUS':
            # find c
            c = self.find_c(method=1)
            h, samplesU, samplesX, logcE, sigma, displacement_samples = BUS_SuS(N, p0, c, self.log_likelihood_function_for_displacement, self.get_displacement_from_dsheet_model, distr = self.prior_pdf)
        elif approach == 'aBUS':
            h, samplesU, samplesX, logcE, sigma = aBUS_SuS(N, p0, self.log_likelihood_function_for_parameters, self.prior_pdf)
        else:
            raise ValueError('Invalid approach')
        
        # Store results as class attributes to make them available for plotting
        self.h = h
        self.samplesU = samplesU
        self.samplesX = samplesX
        self.logcE = logcE
        self.sigma = sigma
        self.displacement_samples = displacement_samples

        # Save all samples in a dictionary
        self.sample_results = {
            'samplesU': samplesU,
            'samplesX': samplesX,
            'logcE': logcE,
            'sigma': sigma,
            'displacement_samples': displacement_samples
        }
        # Create the results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        # Save the results to a JSON file
        result_path = r"results/results_run_n100_maxit_5.npz"
        np.savez(result_path, **self.sample_results)
        # Print summary statistics
        print('\nModel evidence =', np.exp(logcE), '\n')
        for i in range(self.dimensions):
            print(f'Mean value of {self.parameter_names[i]} =', np.mean(samplesX[-1][:, i]))
            print(f'Std of {self.parameter_names[i]} =', np.std(samplesX[-1][:, i]), '\n')

    def plot_prior_and_posterior_marginal_pdfs(self):
        """
        Plot the prior and posterior marginal pdfs
        """
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
        
        # Options for font-family and font-size
        plt.rc('font', size=12)
        plt.rc('axes', titlesize=16)  # fontsize of the axes title
        plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
        plt.rc('figure', titlesize=18)  # fontsize of the figure title
        
        plt.figure(figsize=(12, 10))
        
        # Get prior samples
        prior_samples = self.prior_pdf.random(1000)
        
        # Get posterior samples if available
        try:
            posterior_samples = self.samplesX[-1]
            has_posterior = True
        except (AttributeError, IndexError):
            has_posterior = False
            print("No posterior samples available. Run update_for_new_displacement_data first.")
            return
        
        param_ranges = []
        for i in range(self.dimensions):
            # Determine range for each parameter
            param_min = min(np.min(prior_samples[:, i]), np.min(posterior_samples[:, i]))
            param_max = max(np.max(prior_samples[:, i]), np.max(posterior_samples[:, i]))
            # Add some padding
            padding = 0.1 * (param_max - param_min)
            param_ranges.append([param_min - padding, param_max + padding])
        
        # Plot displacement samples
        plt.subplot(self.dimensions+2, 2, 1)
        # Prior KDE
        # kde_prior = gaussian_kde(self.displacement_samples[0])
        # min_displacement_0 = np.min(self.displacement_samples[0])
        # max_displacement_0 = np.max(self.displacement_samples[0])
        # min_displacement_1 = np.min(self.displacement_samples[-1])
        # max_displacement_1 = np.max(self.displacement_samples[-1])
        # min_displacement = min(min_displacement_0, min_displacement_1)
        # max_displacement = max(max_displacement_0, max_displacement_1)
        # x_range = np.linspace(min_displacement, max_displacement, 200)
        # prior_density = kde_prior(x_range)
        # plt.plot(x_range, prior_density, 'b-', linewidth=2, label='Prior')
        # plt.fill_between(x_range, prior_density, alpha=0.1, color='blue')      
        plt.hist(self.displacement_samples[0], density=True, stacked=True, alpha=0.5, color='blue', label='Prior')
        
        # Posterior KDE
        # kde_post = gaussian_kde(self.displacement_samples[-1])
        # post_density = kde_post(x_range)
        # plt.plot(x_range, post_density, 'r-', linewidth=2, label='Posterior')
        # plt.fill_between(x_range, post_density, alpha=0.3, color='red')
        plt.hist(self.displacement_samples[-1], density=True, stacked=True, alpha=0.5, color='red', label='Posterior')
        # Add measured displacement lines
        for measured_displacement in self.measured_displacement_mean:
            plt.axvline(measured_displacement, color='blue', linestyle='--', label=f'Measured displacement:{measured_displacement:.2f}')
        # plt.axvline(np.mean(posterior_samples[:, i]), color='red', linestyle='--', 
                #    label=f'Posterior mean: {np.mean(posterior_samples[:, i]):.2f}')
        
        plt.title(f'Marginal PDF of the displacement')
        plt.xlabel('Displacement [cm]')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        # plt.legend()

        # Plot each parameter's marginal distribution
        for i in range(self.dimensions):
            plt.subplot(self.dimensions+2, 2, 2*i+3)
            
            # Prior KDE
            kde_prior = gaussian_kde(prior_samples[:, i])
            x_range = np.linspace(param_ranges[i][0], param_ranges[i][1], 200)
            prior_density = kde_prior(x_range)
            plt.plot(x_range, prior_density, 'b-', linewidth=2, label='Prior')
            plt.fill_between(x_range, prior_density, alpha=0.1, color='blue')
            
            # Posterior KDE
            kde_post = gaussian_kde(posterior_samples[:, i])
            post_density = kde_post(x_range)
            plt.plot(x_range, post_density, 'r-', linewidth=2, label='Posterior')
            plt.fill_between(x_range, post_density, alpha=0.3, color='red')
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
            
            # Add cumulative distribution function
            plt.subplot(self.dimensions+2, 2, 2*i+4)
            
            # Sort samples for CDF
            prior_sorted = np.sort(prior_samples[:, i])
            posterior_sorted = np.sort(posterior_samples[:, i])
            p = np.linspace(0, 1, len(prior_sorted))
            pp = np.linspace(0, 1, len(posterior_sorted))
            
            plt.plot(prior_sorted, p, 'b-', linewidth=2, label='Prior CDF')
            plt.plot(posterior_sorted, pp, 'r-', linewidth=2, label='Posterior CDF')
            
            plt.title(f'Cumulative Distribution of {self.parameter_names[i]}')
            plt.xlabel(self.parameter_names[i])
            plt.ylabel('Probability')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('marginal_pdfs.png', dpi=300)
        plt.show()

    def plot_posterior_samples(self):
        """
        Plot the posterior samples
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from scipy.stats import gaussian_kde
        import numpy as np
        
        # Options for font-family and font-size
        plt.rc('font', size=12)
        plt.rc('axes', titlesize=16)  # fontsize of the axes title
        plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
        plt.rc('figure', titlesize=18)  # fontsize of the figure title
        
        # Check if posterior samples exist
        try:
            samplesX = self.samplesX
            h = self.h
            nsub = len(h.flatten())  # number of levels + final posterior
        except (AttributeError, IndexError):
            print("No posterior samples available. Run update_for_new_displacement_data first.")
            return
        
        # Organize samples by subset level
        param_samples = []
        for i in range(nsub):
            param_samples.append(samplesX[i][:, :self.dimensions])  # Only parameter columns, not p-value
        
        # Create titles for each subplot
        titles = ['Prior']
        for i in range(1, nsub-1):
            titles.append(f'Subset {i}: h = {h[i]:.2e}')
        titles.append('Posterior')
        
        # Plot progression of samples through BUS-SuS
        plt.figure(figsize=(15, 10))
        plt.suptitle('BUS-SuS: Parameter Estimation Progression')
        
        # Plot first two parameters (if available)
        if self.dimensions >= 2:
            for i in range(min(nsub, 6)):  # Show at most 6 subplots
                plt.subplot(2, 3, i+1)
                
                # Choose colormap
                cmap = plt.cm.viridis
                
                # Plot the samples with p-value coloring if available
                if i < len(samplesX) and samplesX[i].shape[1] > self.dimensions:
                    p_values = samplesX[i][:, -1]  # last column contains p-values
                    sc = plt.scatter(
                        param_samples[i][:, 0], 
                        param_samples[i][:, 1], 
                        c=p_values, 
                        cmap=cmap, 
                        s=20, 
                        alpha=0.6
                    )
                else:
                    # If no p-values, use default color
                    sc = plt.scatter(
                        param_samples[i][:, 0], 
                        param_samples[i][:, 1], 
                        s=20, 
                        alpha=0.6
                    )
                
                # Add measured data location if it exists
                if hasattr(self, 'measured_params') and i == nsub-1:
                    plt.scatter(
                        self.measured_params[:, 0], 
                        self.measured_params[:, 1], 
                        color='red', 
                        marker='x', 
                        s=100, 
                        label='True params', 
                        zorder=10, 
                        linewidth=2
                    )
                    plt.legend(loc='upper right')
                
                # For the posterior (last) plot, add contours if possible
                if i == nsub-1 and self.dimensions >= 2:
                    try:
                        # Create a grid for density evaluation
                        x_min, x_max = plt.xlim()
                        y_min, y_max = plt.ylim()
                        x_grid = np.linspace(x_min, x_max, 50)
                        y_grid = np.linspace(y_min, y_max, 50)
                        X, Y = np.meshgrid(x_grid, y_grid)
                        
                        # Calculate KDE for 2D posterior
                        positions = np.vstack([X.ravel(), Y.ravel()])
                        posterior_points = np.vstack([
                            param_samples[i][:, 0], 
                            param_samples[i][:, 1]
                        ])
                        kernel = gaussian_kde(posterior_points)
                        Z = np.reshape(kernel(positions), X.shape)
                        
                        # Plot contours of density
                        contour = plt.contour(X, Y, Z, colors='k', alpha=0.3, linewidths=0.5)
                    except Exception as e:
                        print(f"Could not generate contour plot: {e}")
                
                plt.title(titles[i])
                plt.xlabel(self.parameter_names[0])
                plt.ylabel(self.parameter_names[1])
                
                # Add colorbar to the last subplot
                if i == nsub-1:
                    cbar = plt.colorbar(sc)
                    cbar.set_label('p value')
        
        # Plot threshold progression
        plt.figure(figsize=(10, 6))
        plt.plot(range(nsub), h, 'bo-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', label='Failure threshold')
        plt.xlabel('Subset level')
        plt.ylabel('Threshold value (h)')
        plt.title('Progression of Intermediate Thresholds')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('threshold_progression.png', dpi=300)
        
        # If we have at least 3 parameters, create pairwise scatter plots
        if self.dimensions >= 3:
            import seaborn as sns
            
            # Create pairwise plots for posterior samples
            plt.figure(figsize=(12, 10))
            plt.suptitle('Pairwise Relationships in Posterior Distribution')
            
            # Create a dataframe for the posterior samples
            import pandas as pd
            df = pd.DataFrame(param_samples[-1], columns=self.parameter_names)
            
            # Create pairwise plots
            g = sns.pairplot(df, diag_kind='kde')
            g.fig.subplots_adjust(top=0.95)
            plt.savefig('pairwise_posterior.png', dpi=300)
        
        plt.show()


if __name__ == '__main__':    
    # Use a dummy model path when using surrogate model
    model_path = "dummy_model.shi"
    
    # Get measurement data path relative to the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    measurement_path = os.path.join(current_dir, 'synthetic_measurement_data.json')
    
    # Make sure the measurement data file exists
    if not os.path.exists(measurement_path):
        print(f"Warning: Measurement data file not found at {measurement_path}")
        print("Creating a dummy measurement data file...")
        
        # Create dummy measurement data
        import json
        dummy_data = [
            {"displacement": 7.5, "sigma": 0.1},
            # {"displacement": 2.7, "sigma": 0.15},
            # {"displacement": 2.6, "sigma": 0.12}
        ]
        os.makedirs(os.path.dirname(measurement_path), exist_ok=True)
        with open(measurement_path, 'w') as f:
            json.dump(dummy_data, f)
    
    # Initialize with surrogate model
    posterior_retention_structure = PosteriorRetainingStructure(
        model_path, measurement_path, use_surrogate=True
    )
    
    posterior_retention_structure.define_parameters()
    posterior_retention_structure.update_for_new_displacement_data(N=1000, p0=0.1, approach='BUS')
    posterior_retention_structure.plot_prior_and_posterior_marginal_pdfs()
    # posterior_retention_structure.plot_posterior_samples()
