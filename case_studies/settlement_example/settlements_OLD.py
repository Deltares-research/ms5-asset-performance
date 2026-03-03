
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import pypdf
import os

data_directory = f'C:/Users/{os.getlogin()}/Stichting Deltares/SITO-IS 2025 Moonshot 5 - 02_Asset performance/02 Case studies results/Case studies 2026/settlements/CRS Delft-Schiedam example/'

def degree_of_consolidation(t,h=1.,cv = 1.,method = 'Terzaghi'):
    '''
    
    Parameters
    ----------
    t : array_like
        time [days]
    h : float or array_like, optional 
        percolation layer thickness [m]. The default is 1.0.             
    cv : float or array_like, optional
        consolidation coefficient [m2/day]. The default is 1.0.
    method : str, optional
        method to compute the consolidation curve. The default is 'Terzaghi'.

    
    Returns
    -------
    U : array_like
        mean degree of consolidation [-]

    '''


    Tv = cv *t / h**2


    if method == 'Terzaghi':
        U_start = np.sqrt( Tv * 4 / np.pi)
        U_end = 1 - 8 / np.pi**2 * np.exp( -Tv / 4 * np.pi**2  )
        
        U = np.minimum(U_start,U_end)

    else:
        ValueError(f'method {method} not implemented')

    return U


def final_settlement(sigma_0,sigma_v, Cr = 0.02,Cc = 0.2,Ca = 0.,sigma_p = 0.,e0 = 1.0, h = 1.):
    '''
    NEN-Bjerrum settlement model for primary consolidation settlement.
    
    Parameters
    ----------
    sigma_0 : float
        initial effective vertical stress [kPa]
    sigma_v : array_like
        effective vertical stress [kPa]
    Cr : float, optional
        recompression index [-]. The default is 0.02.
    Cc : float, optional    
        compression index [-]. The default is 0.2.  
    sigma_p : float, optional
        preconsolidation stress [kPa]. The default is 0.0.
    e0 : float, optional    
        initial void ratio [-]. The default is 1.0.
    h : float, optional
        layer thickness [m]. The default is 1.0.    
    
    Returns
    ------- 
    S : array_like
        final settlement [m]
    
    '''


    S = h / (1+e0) * ( Cr * np.log10(sigma_p/sigma_0) + 
                        Cc * np.log10(sigma_v/sigma_p) )

    if sigma_v < sigma_p:
        S = h / (1+e0) *  Cr * np.log10(sigma_v/sigma_0)

    return S

def lsf(Cc,Cr,e0,kn,
          depth = 2.5, 
          gamma_clay = 17.0, 
          gamma_crust = 17.0, 
          sigma_load = 60.0,
          sigma_p = 34.5,
          t_build = 1.0,
          t_end = 71.0,
          S_required = 0.05):
    
    sigma_0 = depth*(gamma_clay - 10.) + 1.0 * gamma_crust
    sigma_v = sigma_0 + sigma_load

    mv = Cc / (sigma_v * (1 + e0) * np.log(10) ) 
    cv = kn / (10.0 * mv)

    S_tot = final_settlement(sigma_0,sigma_v,Cr = Cr,Cc = Cc,Ca = 0.,sigma_p = sigma_p,e0 = e0,h = depth * 2)
    delta_U = degree_of_consolidation(t_end,cv = cv,h = depth) - degree_of_consolidation(t_build,cv = cv,h = depth) 

    S_residual = S_tot * delta_U

    g = 1000 * ( S_required - S_residual)

    return g



def logn(mu,cov):
    '''
    Wrapper for lognormal distribution for Y = exp(X) with X ~ N(mu_X,sigma_X) 
    such that Y has mean mu and coefficient of variation cov. 
    
    :param mu: mean of Y
    :param cov: coefficient of variation (std/mean)
    '''
    sigma = np.sqrt(np.log(cov**2 + 1))
    scale = mu / np.sqrt(1 + cov**2)
    return st.lognorm(s=sigma,scale=scale)

# definition of the problem: stresses evaluated at:
depth = 2.5
gamma_clay = 17.0
gamma_crust = 17.0
sigma_load = 60.0

# stochastic variables:
Cr_dist = logn(0.04,0.3)
Cc_dist = logn(0.2,0.3)
kn_dist = logn(3.e-9 * (365 * 24 * 60 * 60),1.0)    
e0_dist = logn(1.2,0.2)

sigma_0 = depth*(gamma_clay - 10.) + 1.0 * gamma_crust
sigma_v = sigma_0 + sigma_load
sigma_p = 34.5

t = np.geomspace(0.001,10,1001)

for i in range(1000):

    # sample stochastic variables
    Cr = Cr_dist.rvs(size = 1)
    Cc = Cc_dist.rvs(size = 1)
    kn = kn_dist.rvs(size = 1)
    e0 = e0_dist.rvs(size = 1)

    # evaluate dependent variables 
    mv = Cc / (sigma_v * (1 + e0) * np.log(10) ) 
    cv = kn / (10.0 * mv)

    S_tot = final_settlement(sigma_0,sigma_v,Cr = Cr,Cc = Cc,Ca = 0.,sigma_p = sigma_p,e0 = e0,h = 5.0)
    
    U = degree_of_consolidation(t,cv = cv,h = 2.5)
    U1 = degree_of_consolidation(1,cv = cv,h = 2.5)
    U71 = degree_of_consolidation(71,cv = cv,h = 2.5)

    S_res = S_tot * (U71 - U1) 

    plt.plot(t,-S_tot*U,'gray',alpha = 0.05)
    if S_res[-1] > 0.05:
        plt.plot(t,-S_tot*U,'red',alpha = 0.5)

plt.plot([],[],'gray',alpha = 0.2,label = '1000 samples')
plt.plot([],[],'red',alpha = 0.5,label = '$S_{res,1-71}$ > 0.05 m')

plt.xlim([0,10])
#plt.ylim([0,1.05])
plt.grid()
plt.xlabel('time [years]')
plt.ylabel('settlement [m]')
plt.title('consolidation model')
plt.legend()


#%%

N = 1000_000

# sample stochastic variables
Cr = Cr_dist.rvs(size = N)
Cc = Cc_dist.rvs(size = N)
kn = kn_dist.rvs(size = N)
e0 = e0_dist.rvs(size = N)

g = lsf(Cc,Cr,e0,kn)

# evaluate dependent variables
mv = Cc / (sigma_v * (1 + e0) * np.log(10) ) 
cv = kn / (10.0 * mv)

# evaluate total settlement and residual settlement
S_tot = final_settlement(sigma_0,sigma_v,sigma_p = 34.5, Cr = Cr,Cc = Cc,e0 = e0,h = 5.0)
S_res = S_tot * ( degree_of_consolidation(71,cv = cv,h = 2.5) - degree_of_consolidation(1,cv = cv,h = 2.5) ) 

S05 = S_tot * degree_of_consolidation(0.5, cv = cv, h = 2.5)

plt.figure()
plt.hist(S_res,np.linspace(0,0.1,51))
plt.hist(S_res[S_res>0.05],np.linspace(0,0.1,51),color = 'red',label = f'$P_f={np.mean(g<0):.5f}$')
plt.yticks([])
plt.ylim([0,N/10])
plt.xlabel('residual settlement after 1 year [m]')
plt.legend()

plt.figure(figsize = (4,4))

plt.plot(Cc,kn,'.',alpha = 0.1)
plt.plot(Cc[S_res>0.05],kn[S_res>0.05],'.r',alpha = 0.5,label = 'failure')

m = abs(S05-0.08)<0.005
plt.plot(Cc[m],kn[m],'.k',alpha = 0.5,label = 'observation S=8$\pm$0.5cm at 0.5 year')
plt.legend()
plt.xlabel('compression coefficient Cc [-]')
plt.ylabel('hydraulic conductivity kn [m/year]')

plt.xscale('log')
plt.yscale('log')   

#%%

plt.figure(figsize = (4,4))
plt.plot(e0,kn,'.',alpha = 0.1)
plt.plot(e0[S_res>0.05],kn[S_res>0.05],'.r',alpha = 0.5,label = 'failure')

plt.xlabel('Initial void ratio $e_0$ [-]')
plt.ylabel('permeability kn [m/year]')
plt.legend()
plt.xscale('log')
plt.yscale('log')   


#%% Implementation in PTK

from probabilistic_library import ReliabilityMethod, ReliabilityProject, DistributionType, StartMethod
project = ReliabilityProject()

project.model = lsf

project.variables['Cc'].distribution = DistributionType.log_normal
project.variables['Cc'].mean = 0.2
project.variables['Cc'].variation = 0.3


project.variables['Cr'].distribution = DistributionType.log_normal
project.variables['Cr'].mean = 0.04
project.variables['Cr'].variation = 0.3
project.variables['e0'].distribution = DistributionType.log_normal
project.variables['e0'].mean = 1.2
project.variables['e0'].variation = 0.2
project.variables['kn'].distribution = DistributionType.log_normal
project.variables['kn'].mean = 3.e-9 * (365 * 24 * 60 * 60)
project.variables['kn'].variation = 1.0

project.variables['depth'].distribution = DistributionType.deterministic
project.variables['depth'].mean = 2.5

project.settings.random_seed = np.random.choice(1000)
project.settings.reliability_method = ReliabilityMethod.crude_monte_carlo
project.settings.minimum_samples = 10_000
project.settings.maximum_samples = 5000_000
project.settings.save_realizations = True
project.settings.save_convergence = True

project.run()

project.design_point.plot_alphas()
project.design_point.print()
project.design_point.probability_failure


project.settings.random_seed = np.random.choice(1000)
project.settings.reliability_method = ReliabilityMethod.form
project.settings.save_realizations = True
project.settings.save_convergence = True

# set starting value away from u=0 to avoid zero-gradient issues
for stochast in project.settings.stochast_settings:
    stochast.start_value = -2

project.run()

project.design_point.plot_alphas()
project.design_point.print()
project.design_point.probability_failure

#%% Example of reading the CRS dataset from Delft-Schiedam and plotting the CR-parameter as a function of e0.
# May be used as site investigation data [?]


e0 = []
CR = []
Pg = []
ePg = []
Pgmax = []
ePgmax = []
Pg_H = []
ePg_H = []
Pgmax_H = []
ePgmax_H = []
grondsoort = []
for file in os.listdir(data_directory):
    if file.endswith('.pdf'):
        filename = os.fsdecode(file)
        reader = pypdf.PdfReader(os.path.join(data_directory,filename))
        L = reader.pages[0].extract_text().split('\n')
        for l in L:
            if 'e0' in l:
                e0.append(float(l.split(' ')[-1].replace(',','.')))
            if 'Grondsoort' in l:
                grondsoort.append(l.split(' Porien')[0].replace('Grondsoort ',''))
            if 'Pg ' in l:
                Pg_H.append(float(l.split(' ')[-2].replace(',','.')))
            if 'Hpg ' in l:
                ePg_H.append(float(l.split(' ')[-2].replace(',','.')))
            if 'Pg,max ' in l:
                Pgmax_H.append(float(l.split(' ')[-2].replace(',','.')))
            if 'Hpg,max ' in l:
                ePgmax_H.append(float(l.split(' ')[-2].replace(',','.')))
        L = reader.pages[1].extract_text().split('\n')
        for l in L:
            if 'CR-parameter' in l:
                CR.append(float(l.split(' ')[1].replace(',','.')))
            if 'Pg ' in l:
                Pg.append(float(l.split(' ')[-2].replace(',','.')))
            if 'Hpg ' in l:
                ePg.append(float(l.split(' ')[-2].replace(',','.')))
            if 'Pg,max ' in l:
                Pgmax.append(float(l.split(' ')[-2].replace(',','.')))
            if 'Hpg,max ' in l:
                ePgmax.append(float(l.split(' ')[-2].replace(',','.')))

Pg = np.array(Pg)
ePg = np.array(ePg)
Pgmax = np.array(Pgmax)
ePgmax = np.array(ePgmax)

Pg_H = np.array(Pg_H)
ePg_H = np.array(ePg_H)
Pgmax_H = np.array(Pgmax_H)
ePgmax_H = np.array(ePgmax_H)

e0 = np.array(e0)
CR = np.array(CR)
grondsoort = np.array(grondsoort)


mask = np.array([g.startswith('Veen') for g in grondsoort])
plt.scatter(e0[mask],CR[mask],label = 'peat',color = 'brown')

mask = np.array([g.startswith('Klei') for g in grondsoort])
plt.scatter(e0[mask],CR[mask],label = 'clay',color = 'green')
plt.xlabel('e_0')
plt.ylabel('CR')
plt.xlim([0,16])
plt.ylim([0,0.7])
plt.grid()
plt.title('Delft-Schiedam CRS dataset \n(for what its worth...)')

m = (e0<3) & mask 

a,b = np.polyfit(e0[m],CR[m],1)
e0_ = np.array([0.5,3])
plt.plot(e0_,a*e0_+b,'--k',label = f'{b:.2} + {a:.2} $\\cdot e_0$')
plt.legend()

plt.figure()

mask = np.array([g.startswith('Veen') for g in grondsoort])
plt.scatter(e0[mask],CR[mask]*(1+e0[mask]),label = 'peat',color = 'brown')

mask = np.array([g.startswith('Klei') for g in grondsoort])
plt.scatter(e0[mask],CR[mask]*(1+e0[mask]),label = 'clay',color = 'green')
plt.xlabel('e_0')
plt.ylabel('$C_c = CR\cdot(1+e_0)$')
plt.xlim([0,4])
plt.ylim([0,2])

m = (e0<3) & mask 

a,b = np.polyfit(e0[m],CR[m]*(1+e0[m]),1)
e0_ = np.array([1,3])
plt.plot(e0_,a*e0_+b,'--k',label = f'{b:.2} + {a:.2} $\\cdot e_0$')
plt.legend()

plt.grid()
plt.title('Delft-Schiedam CRS dataset \n(for what its worth...)')
