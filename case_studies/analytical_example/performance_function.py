#%%
import os

#dir_path = os.path.dirname(os.path.realpath(__file__))
#os.chdir(dir_path)
#os.chdir('../..')
#print(os.getcwd())

from src.performance import BasePerformance

import numpy as np


class Performance(BasePerformance):

    def __init__(self, name, parameters: dict = {}):
        super().__init__(name,parameters)
        pass

    def _lsf(self, x, t = 0):

        """
        parameter dictionary contains
            x (np.ndarray): stochastic samples in X-space (ndim x nsamples)
            t (float): time in years
            q (float): normalised load
            c0 (float): tweak parameter for reliability index. Defaults to 4.65.
            c1 (float): relative importance of strength. Defaults to 1. 
            c2 (float): relative importance of degradation. Defaults to 1. 
            c3 (float): relative importance of load. Defaults to 1. 
            exp (float): exponent in the time-dependent relationship. Defaults to 1.
            t_design (float): design lifetime. Defaults to 30.

        Returns: 
            g (np.ndarray): performance function evaluations (nsamples)
            
            
        """

        x_ = np.atleast_2d(x)

        c0 = self.parameters.get("c0",4.65)
        c1 = self.parameters.get("c1",1.0)
        c2 = self.parameters.get("c2",1.0)
        c3 = self.parameters.get("c3",1.0)
        q = self.parameters.get("q",1.0)

        t_design = self.parameters.get("t_design",30.0)
        power = self.parameters.get("power",1.0)
        delta_q = self.parameters.get("delta_q",0.3)

        # strength model
        strength = c1 * x_[:,0]
        
        # degradation model
        degradation = c2 * (np.sqrt(0.9)*x_[:,1]*(t/t_design)**power + min(c1/c2/2,np.sqrt(0.1)*(t/t_design)**power)*x_[:,0])  ; 
        
        degradation = np.maximum(degradation,0)
        
        # load model ()
        load =  c3*( q + delta_q*x_[:,2]) 

        g = c0 + strength - load - degradation

        # return performance function g (float or array) and behavioural data {} or [{}]
        return g, {}

    def _grad(self, x, t = 0): 
        pass


