#%%
import json
import sys
import matplotlib.pyplot as plt
import scipy.stats as st

sys.path.append('..')
sys.path.append('../..')

import numpy as np
from config import JPDF
from performance_function import Performance

# initiate prior JPDF
jpdf = JPDF()
jpdf.set_prior_by_input_file(r"case_study_specifications.json")
jpdf.initiate_samples(N = 1_000_000)

input = json.load(open('case_study_specifications.json','r'))

performance = Performance(name = 'analytical example',parameters = input["parameters"])


# initiate perfromance function
performance_high_q = Performance(name = 'analytical example',parameters = input["parameters"])
g_t = {}
y_t = {}

#%% A-PRIORI MCS for all timesteps t TODO: [use range t in input?]
for t in range(101):
    g_t[t],y_t[t] = performance.lsf(jpdf.X_samples,t = t)   # Type hinting to be fixed: tuple of fload and dictionary

#%%
# now added performance and behaviour to JPDF, TODO: evaluate and implement properly (also for large output etc.)

####### for dev only:
jpdf.W_samples = np.ones(jpdf.W_samples.shape)
########

jpdf.G_samples = g_t
jpdf.Y_samples = y_t

pf = [sum((g<0)*(jpdf.W_samples))/sum(jpdf.W_samples) for t,g in g_t.items()]
plt.plot(-st.norm.ppf(pf))


#% reweighting (possible because of available distribution of prior...)

w_obs_20 = st.norm.pdf(jpdf.X_samples[:,0],loc = 0.5, scale = 0.8 ) \
         / st.norm.pdf(jpdf.X_samples[:,0],loc = 0., scale = 1)


jpdf.W_samples *= w_obs_20
pf = np.array([sum((g<0)*(jpdf.W_samples))/sum(jpdf.W_samples) for t,g in g_t.items()])
pf[:20] = 0
plt.plot(-st.norm.ppf(pf))

#%
toenamefactor = 1.5

new_mean = (toenamefactor-1.0)/input["parameters"]["delta_q"]

w_load_25 = st.norm.pdf(jpdf.X_samples[:,2],loc = new_mean, scale = 1.0) / st.norm.pdf(jpdf.X_samples[:,2],loc = 0.0, scale = 1.0 ) 

jpdf.W_samples *= w_load_25

pf = np.array([sum((g<0)*(jpdf.W_samples))/sum(jpdf.W_samples) for t,g in g_t.items()])
pf[:25] = 0
plt.plot(-st.norm.ppf(pf))
