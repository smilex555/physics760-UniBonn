import numpy as np
import matplotlib.pyplot as plt
from leapfrog5 import leapfrog, ham
from tqdm import tqdm # for HMC progress bar

def HMC_alg(phi0,nmd,tau,N_samples,N_burn):
    #phi = 1
    acceptence = 0
    phi_all=np.zeros((N_samples))
    phi = phi0
    
    # burning some MC steps to get thermal equilibrium
    print('Burn-in')
    for i in tqdm(range(N_burn)):
        p0 = np.random.normal(0, 1)
        p_leap, phi_leap = leapfrog(p0, phi, nmd, tau)
        H_old = ham(p0, phi)
        H_new = ham(p_leap,phi_leap)
        delta_E = H_old - H_new
        random=np.random.rand()
        #Metropolis accept-rejection:
        if random <= np.exp(delta_E):
            phi = phi_leap
        else:
            phi = phi
    
    print('HMC-core')
    for i in tqdm(range(N_samples)):
        p0 = np.random.normal(0, 1)
        p_leap, phi_leap = leapfrog(p0, phi, nmd, tau)
        H_old = ham(p0, phi)
        H_new = ham(p_leap,phi_leap)
        delta_E = H_old - H_new
        random=np.random.rand()
        #Metropolis accept-rejection:
        if random <= np.exp(delta_E):
            phi = phi_leap
            acceptence += 1
        else:
            phi = phi
        
        phi_all[i] = phi
    
    return phi_all, acceptence/N_samples


#define the autocorrelation function
def auto_corr_func(mc,mean,N,t):
    mc_sum = 0
    for i in range(N-t):
        mc_sum += (mc[i+1]-mean)*(mc[int(i+1+t)]-mean)    
    return (1/(N-t)) * np.sum(mc_sum) 

#define the normalized autocorrelation function
def Gamma_func(mc,mean,N,t):
    c_t = auto_corr_func(mc,mean,N,t)
    c_0 = auto_corr_func(mc,mean,N,0)
    return c_t/c_0#, c_t, c_0

#get autocorrelation function with autocorr_time
def Gamma_func_2(t,tau):
    return np.exp(-t/tau)
    

# integrated autocorrelation time
def integrated_time(gam_0, gam_t):
    return 0.5*gam_0 + gam_t


phi0 = 1
nmd = 3
Tau = .1
N_config = 100000
N_burn = 10000

m_chain, accept_rate = HMC_alg(phi0, nmd, Tau, N_config, N_burn)

# the estimated mean value of the simulated markov_chain
est_mean = np.mean(m_chain)

#
Gamma = 0
Gamma_t = [] # save the autocorrelation function for different t
W = 0
for t in range(len(m_chain)):
    g_t = Gamma_func(m_chain, est_mean, len(m_chain), t+1)
    Gamma += g_t 
    Gamma[t] = g_t 
    W+=1
    if g_t >= 0:
        break

Gamma_0 = Gamma_func(m_chain, est_mean, len(m_chain), 0)    
tau_int = 0.5*Gamma_0 + Gamma
markov_time = np.arange(1,W,1)

plt.figure()
plt.plot(markov_time, Gamma_t, '.', label='hmc')
plt.plot(markov_time, Gamma_func_2(markov_time,tau_int), '.', label='hmc')
plt.xlabel('markov chain time t')
plt.ylabel(' autocorrelation $\Gamma$(t)')
plt.label()
plt.grid()
plt.show()


# Blocking / Binning the markov chain data
def blocking(mc_data, bin_width):
    if len(mc_data) % bin_width == 0:
        N_bin = len(mc_data) // bin_width
    else:
        N_bin = len(mc_data) // bin_width + 1  
    mean_block = np.zeros(N_bin)
    #idx = np.arange(0,N_bin,1)
    for j in range(N_bin):
        i = bin_width*j
        if j >= N_bin - 1:
            mean_block[j] = np.mean(mc_data[i:])
        else:
            mean_block[j] = np.mean(mc_data[i:i+bin_width])
    
    return mean_block    

 