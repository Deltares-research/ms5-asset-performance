#%%
import json
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike, NDArray


class JPDF():
    """
    class for the JPDF, specifically for the analytical example. 
    
    
    to be developed and moved to 
    
    """

    def __init__(self,name: str = "",comments: str = ""):
        self.name: str = name
        self.comments: str = comments


    def set_prior_by_input_file(self,filename):
        
        with open(filename,'r') as file:
            input = json.load(file)

        self.nvar = input.get("number_of_variables")
        self.variable_names = [v.get("name","var_"+str(i+1)) for i,v in enumerate(input["variables"])]

        self.variables = []

        for i,v in enumerate(input["variables"]):
            
            # normal distributions
            if v["distribution_type"].lower() in ["normal","norm","n","gaussian"]:
                var = st.norm(loc = v["mean"],scale = v["standard_deviation"])
            else:
                ValueError(f"{v["distribution_type"]} not recognised as a disribution type for the JPDF")
                var = None

            self.variables.append(var)

        self.correlation_matrix = input.get("correlation_in_u_space",np.eye(self.nvar))


    def initiate_samples(self,N: int, use_IS: bool = False, mean_IS: ArrayLike = [], var_IS: ArrayLike = []):
        
        # sampling weights
        if use_IS:
            self.U_samples = st.multivariate_normal(mean = np.zeros(self.nvar),cov = 1).rvs(N)
            p = 0
            q = 0
            self.W_samples = p / q

        else:
            self.Y_samples = [None]*N
            self.G_samples = [None]*N
            self.W_samples = np.ones(N)
            self.U_samples = st.multivariate_normal(mean = np.zeros(self.nvar),cov = 1).rvs(N)

        # Transform correlated 
        self.X_samples = (np.linalg.cholesky(self.correlation_matrix) @ self.U_samples.T).T
        for i in range(self.nvar):
            self.X_samples[:,i] = self.variables[i].ppf(st.norm.cdf(self.X_samples[:,i]))


    def plot_1_2(self):
        '''
        TODO: to be replaced by an appropriate visualisation function
        '''
        plt.scatter(self.X_samples[:,0],self.X_samples[:,1],marker = '.',alpha = self.W_samples/np.max(self.W_samples))



if __name__ == '__main__':
    pass
    # EXAMPLE HERE




